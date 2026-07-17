"""v6 eğitimi için mevcut TRAIN çiftlerinden iki tür TÜREV kopya üreticisi.

GitHub issue #1'in iki kusuruna VERİ ile cevap verir (bkz. görev bağlamı):

1. **Kadraj-kırpma kopyaları (`{stem}_e00`)**: eğitim kompozitlerinde özne hep
   tuvalin İÇİNDE kaldığı için model "kadraja değen özne" görmedi ve bu özneleri
   SİLİYOR. Burada mevcut bir im/gt çifti, öznenin bbox'ını (gt alpha >
   `SUBJECT_ALPHA_THRESH`) rastgele bir kenardan bbox uzunluğunun %20-60'ı
   (`CUT_FRAC_LO..CUT_FRAC_HI`) kadar KESECEK şekilde kırpılır — kırpma
   penceresinin sınırı öznenin İÇİNDEN geçer, özne yeni görüntüde kadraja değer.
   4 kenardan rastgele biri seçilir; `SECOND_EDGE_PROB` olasılıkla bir KOMŞU
   (dik) kenardan ikinci bir kesik daha atılır. Pencere alanı orijinalin en az
   `MIN_KEEP_AREA`'sı (%50) kalır (aşırı küçülme yok). GT AYNI pencereyle
   kırpılır ve alpha değerleri DEĞİŞMEZ — kadraja değen kısım KATI kalır
   (ders tam bu: kenara değmek saydamlık nedeni DEĞİLDİR).

2. **Karma-opaklık kopyaları (`{stem}_m00`, `_m01`)**: saydam nesnelerin katı
   parçaları (şişe kapağı, gözlük sapı...) yarı saydamlaşıyor çünkü eğitimde
   HEM katı HEM yumuşak alpha içeren örnek az. `transparent` kategorisindeki,
   GT'si hem katı (alpha > `SOLID_ALPHA_THRESH` piksel oranı >=
   `SOLID_MIN_RATIO`) hem yumuşak (`SOFT_LO` < alpha < `SOFT_HI` oranı >=
   `SOFT_MIN_RATIO`) olan çiftlerin `MIXED_COPIES` (2) augment'li kopyası
   üretilir. Augment `bgr.compositing.augment` ile ve `flip_prob=0.0` ile
   çağrılır (imza koddan doğrulandı: flip tek geometrik dönüşümdür; renk
   jitter / blur / JPEG artifact yalnız RGB'yi etkiler) — geometri değişmez,
   alpha AYNEN korunur.

KAYNAK SEÇİMİ SÖZLEŞMELERİ:
- `_e<NN>`/`_m<NN>` türevi stem'ler KAYNAK OLARAK KULLANILMAZ (türevin türevi
  olmasın); `_o<NN>` (orijinal arka planlı make_composites kopyaları) kaynak
  OLABİLİR ve TERCİH EDİLİR — gerçek arka planlı kırpmalar en değerlisi.
- Edge-crop kaynakları kategori başına ORANTILI (largest-remainder) dağıtılır;
  kategori İÇİNDE sıralama deterministiktir: önce tercihli kaynaklar
  (`_o<NN>` son ekli stem'ler ile `ORIGINAL_BG_CATEGORIES` — kompozitsiz/
  orijinal arka planlı kategoriler, ör. camouflage), sonra kalanlar, her grup
  kendi içinde alfabetik. Böylece edge-crop kaynaklarının olabildiğince büyük
  kısmı (hedef: en az yarısı) gerçek arka planlı örneklerden gelir.
- Mixed kaynak seçimi: `transparent` stem'ler SIRALI taranır, eşik testini
  geçen ilk `mixed_cap / MIXED_COPIES` stem seçilir (deterministik). Çıktı
  kopyalarından biri diskte zaten varsa GT yeniden yüklenmeden uygun sayılır
  (resume hızlandırması — yalnız uygun kaynaklar çıktı üretebildiği için
  dosya varlığı uygunluğun kanıtıdır).

ÇIKTI SÖZLEŞMELERİ (scripts/make_textfx.py ile AYNI):
- Çıktı düzeni: `out_dir/im/{stem}.jpg` (RGB, JPEG q92) + `out_dir/gt/{stem}.png`
  (L modu 8-bit alpha) — `_save_pair` birebir aynı.
- Manifest: her çift için `{"id": yeni_stem, "category": kaynak_kategori}`
  satırı JSONL'e APPEND (`out_manifest`, varsayılan `out_dir/manifest.jsonl`).
- Determinizm: `_item_rng(seed, yeni_stem)` (make_composites.py kalıbının
  birebir kopyası) — aynı seed + aynı stem -> bit-identical çıktı, işlem
  sırasından ve atlanmış öğelerden bağımsız (resume güvenliği).
- İdempotentlik: im+gt çifti diskte zaten varsa üretim atlanır; dosya var ama
  manifest satırı eksikse yalnız satır tamamlanır, dosya yeniden ÜRETİLMEZ.
- Öznesi olmayan (gt tamamen boş) ya da kesilebilir kenarı bulunamayan kaynak
  SESSİZCE atlanır (çift üretilmez) — atlama içerikten türediği için
  deterministiktir, hedef sayı bu durumda biraz altta kalabilir ("~9.000").

Kullanım:
    uv run python scripts/make_v6_copies.py \
        --train-im-dir data/TRAIN/im --train-gt-dir data/TRAIN/gt \
        --categories-manifest train_composites_manifest.jsonl \
        --out-dir data/train_v6 --seed 42 --edge-count 9000 --mixed-cap 4000
"""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image

from bgr.compositing import augment

# TRAIN havuzunda 100MP+ kaynaklı kompozitler olabilir; PIL'in 179MP
# "decompression bomb" eşiği güvenilir kaynaklar için kaldırılır
# (bkz. scripts/make_textfx.py aynı not).
Image.MAX_IMAGE_PIXELS = None

DEFAULT_EDGE_COUNT = 9000
DEFAULT_MIXED_CAP = 4000
MIXED_COPIES = 2  # {stem}_m00 + {stem}_m01
SUBJECT_ALPHA_THRESH = 0.1  # özne bbox'ı bu eşiğin üzerindeki piksellerden
CUT_FRAC_LO, CUT_FRAC_HI = 0.2, 0.6  # bbox uzunluğunun kesilen payı
MIN_KEEP_AREA = 0.5  # pencere alanı >= orijinalin %50'si
SECOND_EDGE_PROB = 0.35  # bazen iki komşu kenardan kesme olasılığı
SOLID_ALPHA_THRESH = 0.9
SOLID_MIN_RATIO = 0.08  # katı piksel oranı eşiği (alpha > 0.9)
SOFT_LO, SOFT_HI = 0.05, 0.95
SOFT_MIN_RATIO = 0.08  # yumuşak piksel oranı eşiği (0.05 < alpha < 0.95)

# Türev son ekleri: bu scriptin kendi çıktıları (_eNN/_mNN) kaynak OLAMAZ.
_DERIVED_SUFFIX_RE = re.compile(r"_[em]\d{2}$")
# make_composites'ın orijinal-arka-plan kopyaları (_oNN) — tercihli kaynak.
_ORIGINAL_BG_SUFFIX_RE = re.compile(r"_o\d{2}$")
# Kompoziti hiç yapılmayan (orijinal arka planı hep korunan) kategoriler —
# bkz. scripts/make_composites.py NO_COMPOSE_CATEGORIES.
ORIGINAL_BG_CATEGORIES = {"camouflage"}

_EDGES = ("left", "right", "top", "bottom")


# ==========================================================================
# Ortak yardımcılar (kaynak: scripts/make_composites.py + make_textfx.py —
# aynı sözleşmeler, birebir kopyalar)
# ==========================================================================
def _item_rng(seed: int, key: str) -> np.random.Generator:
    """(global seed, öğe anahtarı) çiftinden bağımsız/deterministik rastgele akış.

    İşlem sırasından ve önceden atlanmış (zaten var olan) öğelerden ETKİLENMEZ —
    her öğe kendi id'sinden türetilen sabit bir alt-seed kullanır.
    (Kaynak: scripts/make_composites.py::_item_rng, birebir kopya.)
    """
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    entropy = [seed & 0xFFFFFFFF] + [
        int.from_bytes(digest[i : i + 4], "big") for i in range(0, 16, 4)
    ]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def _save_pair(rgb: np.ndarray, alpha: np.ndarray, img_path: Path, gt_path: Path) -> None:
    """Kaynak: scripts/make_composites.py::_save_pair — aynı kayıt sözleşmesi."""
    img_path.parent.mkdir(parents=True, exist_ok=True)
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(img_path, format="JPEG", quality=92)
    Image.fromarray(np.round(alpha.clip(0, 1) * 255).astype(np.uint8), mode="L").save(gt_path)


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _load_alpha(path: Path, target_size: tuple[int, int] | None = None) -> np.ndarray:
    """target_size = (w, h); verilirse ve boyut uyuşmuyorsa alpha yeniden
    ölçeklenir (kaynak: make_composites.py::_load_alpha)."""
    im = Image.open(path).convert("L")
    if target_size is not None and im.size != target_size:
        im = im.resize(target_size, Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def _load_manifest_ids(path: Path) -> set[str]:
    """Çıktı manifest'indeki id'ler (resume'da satır tekrarını önlemek için)."""
    ids: set[str] = set()
    if not path.exists():
        return ids
    for line in path.read_text().splitlines():
        if line.strip():
            ids.add(json.loads(line)["id"])
    return ids


def _append_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _list_pair_stems(im_dir: Path, gt_dir: Path) -> list[str]:
    """`im_dir/*.jpg` ile `gt_dir/*.png` kesişimindeki stem'ler (sıralı) —
    macOS AppleDouble artıkları (`._*`) filtrelenir (v4 hücresi kalıbı)."""
    ims = {p.stem for p in Path(im_dir).iterdir()
           if p.is_file() and p.suffix.lower() == ".jpg" and not p.name.startswith("._")}
    gts = {p.stem for p in Path(gt_dir).iterdir()
           if p.is_file() and p.suffix.lower() == ".png" and not p.name.startswith("._")}
    return sorted(ims & gts)


def _is_preferred_source(stem: str, category: str) -> bool:
    """Gerçek arka planlı kaynak mı? `_o<NN>` kopyaları + kompoziti hiç
    yapılmayan kategoriler (bkz. modül docstring'i kaynak seçimi)."""
    return category in ORIGINAL_BG_CATEGORIES or _ORIGINAL_BG_SUFFIX_RE.search(stem) is not None


# ==========================================================================
# Kadraj-kırpma (edge-crop) — pencere seçimi + üretim
# ==========================================================================
def _cut_bounds(
    b_lo: int, b_hi: int, length: int, min_keep_px: int, side: str
) -> tuple[int, int] | None:
    """Bir eksen için geçerli kesik-piksel aralığını [cut_lo, cut_hi] döndürür.

    `side='lo'`: eksenin BAŞINDAN kesilir, yeni pencere `[b_lo+cut, length)` —
    sınır bbox'ın içinden geçer. `side='hi'`: eksenin SONUNDAN kesilir, yeni
    pencere `[0, b_hi-cut)`. Kısıtlar: kesik bbox uzunluğunun %20-60'ı, kalan
    pencere uzunluğu >= `min_keep_px`, sınır bbox'ın KESİN içinde
    (1 <= cut <= b-1). Uygun aralık yoksa None."""
    b = b_hi - b_lo
    cut_lo = max(1, math.ceil(CUT_FRAC_LO * b))
    cut_hi = math.floor(CUT_FRAC_HI * b)
    if side == "lo":
        cut_hi = min(cut_hi, length - min_keep_px - b_lo, b - 1)
    else:
        cut_hi = min(cut_hi, b_hi - min_keep_px, b - 1)
    if cut_hi < cut_lo:
        return None
    return cut_lo, cut_hi


def _edge_crop_window(
    rng: np.random.Generator, alpha: np.ndarray, min_keep_area: float = MIN_KEEP_AREA
) -> tuple[int, int, int, int] | None:
    """Özneyi rastgele bir kenardan (bazen iki komşu kenardan) kesen kırpma
    penceresi `(x0, y0, x1, y1)` döndürür; uygun kesik bulunamazsa None.

    Pencere kesilmeyen yönlerde görüntünün kenarına kadar uzanır — yani kesilen
    her eksende özne yeni görüntünün TAM KENARINA değer. Toplam pencere alanı
    her zaman >= `min_keep_area` × orijinal alan (ikinci kesikte kalan pay
    `min_keep_area / k1` ile sınırlandığından çarpım korunur)."""
    h, w = alpha.shape
    win = [0, 0, w, h]  # x0, y0, x1, y1

    def _apply(edge: str, keep_frac: float) -> bool:
        x0, y0, x1, y1 = win
        ys, xs = np.nonzero(alpha[y0:y1, x0:x1] > SUBJECT_ALPHA_THRESH)
        if xs.size == 0:
            return False
        if edge in ("left", "right"):
            b_lo, b_hi = x0 + int(xs.min()), x0 + int(xs.max()) + 1
            length, min_keep_px = w, math.ceil(keep_frac * w)
            side = "lo" if edge == "left" else "hi"
        else:
            b_lo, b_hi = y0 + int(ys.min()), y0 + int(ys.max()) + 1
            length, min_keep_px = h, math.ceil(keep_frac * h)
            side = "lo" if edge == "top" else "hi"
        bounds = _cut_bounds(b_lo, b_hi, length, min_keep_px, side)
        if bounds is None:
            return False
        cut = int(rng.integers(bounds[0], bounds[1] + 1))
        if edge == "left":
            win[0] = b_lo + cut
        elif edge == "right":
            win[2] = b_hi - cut
        elif edge == "top":
            win[1] = b_lo + cut
        else:
            win[3] = b_hi - cut
        return True

    first = None
    for idx in rng.permutation(len(_EDGES)):
        edge = _EDGES[int(idx)]
        if _apply(edge, min_keep_area):
            first = edge
            break
    if first is None:
        return None

    # bazen iki KOMŞU (dik) kenardan ikinci kesik — alan garantisi için kalan
    # pay ilk kesiğin kalan oranına bölünür (k1 * k2 >= min_keep_area).
    if rng.uniform() < SECOND_EDGE_PROB:
        if first in ("left", "right"):
            k1 = (win[2] - win[0]) / w
            perp = ["top", "bottom"]
        else:
            k1 = (win[3] - win[1]) / h
            perp = ["left", "right"]
        if rng.uniform() < 0.5:
            perp.reverse()
        for edge in perp:
            if _apply(edge, min_keep_area / k1):
                break

    return tuple(win)


def select_edge_sources(
    stems: list[str], category_by_stem: dict[str, str], count: int
) -> list[tuple[str, str]]:
    """Edge-crop kaynaklarını seçer: kategori başına ORANTILI (largest
    remainder — kesirli payı en büyük kategori önce, eşitlikte alfabetik),
    kategori içinde deterministik sırada önce TERCİHLİ kaynaklar (`_o<NN>` /
    `ORIGINAL_BG_CATEGORIES`), sonra kalanlar. `(stem, category)` listesi
    döndürür. `_e/_m` türevleri ve kategorisi bilinmeyen stem'ler elenir."""
    eligible = [
        s for s in stems if s in category_by_stem and not _DERIVED_SUFFIX_RE.search(s)
    ]
    if count <= 0 or not eligible:
        return []
    by_cat: dict[str, list[str]] = {}
    for s in eligible:
        by_cat.setdefault(category_by_stem[s], []).append(s)
    for c, lst in by_cat.items():
        lst.sort(key=lambda s: (0 if _is_preferred_source(s, c) else 1, s))

    total = len(eligible)
    count = min(count, total)
    cats = sorted(by_cat)
    quotas: dict[str, int] = {}
    fracs: list[tuple[float, str]] = []
    used = 0
    for c in cats:
        exact = count * len(by_cat[c]) / total
        quotas[c] = math.floor(exact)
        used += quotas[c]
        fracs.append((-(exact - quotas[c]), c))  # kesirli pay büyük olan önce
    for _, c in sorted(fracs):
        if used >= count:
            break
        if quotas[c] < len(by_cat[c]):
            quotas[c] += 1
            used += 1
    while used < count:  # bazı kategoriler dolduysa kalanı deterministik dağıt
        progressed = False
        for c in cats:
            if used >= count:
                break
            if quotas[c] < len(by_cat[c]):
                quotas[c] += 1
                used += 1
                progressed = True
        if not progressed:
            break
    return [(s, c) for c in cats for s in by_cat[c][: quotas[c]]]


def gen_edge_crops(
    sources: list[tuple[str, str]],
    train_im_dir: Path,
    train_gt_dir: Path,
    out_im_dir: Path,
    out_gt_dir: Path,
    seed: int,
    existing_ids: set[str],
) -> tuple[list[dict], int, int]:
    """(manifest satırları, üretilen, atlanan) döndürür. Kesilebilir kenarı
    olmayan kaynaklar (boş gt / kısıtlara sığmayan bbox) sessizce atlanır."""
    new_rows: list[dict] = []
    generated = skipped = no_fit = 0
    for stem, category in sources:
        new_stem = f"{stem}_e00"
        img_path = out_im_dir / f"{new_stem}.jpg"
        gt_path = out_gt_dir / f"{new_stem}.png"
        row = {"id": new_stem, "category": category}
        if img_path.exists() and gt_path.exists():
            skipped += 1
            if new_stem not in existing_ids:
                new_rows.append(row)  # dosya var, manifest satırı eksik -> yalnız satır
            continue
        rng = _item_rng(seed, new_stem)
        rgb = _load_rgb(train_im_dir / f"{stem}.jpg")
        alpha = _load_alpha(train_gt_dir / f"{stem}.png", (rgb.shape[1], rgb.shape[0]))
        window = _edge_crop_window(rng, alpha)
        if window is None:
            no_fit += 1
            continue
        x0, y0, x1, y1 = window
        # Alpha değerleri DEĞİŞMEZ — saf dilimleme (kadraja değen kısım katı kalır).
        _save_pair(rgb[y0:y1, x0:x1], alpha[y0:y1, x0:x1], img_path, gt_path)
        new_rows.append(row)
        generated += 1
    if no_fit:
        print(f"edge-crop: {no_fit} kaynak kesilebilir kenar bulunamadığı için atlandı")
    return new_rows, generated, skipped


# ==========================================================================
# Karma-opaklık (mixed) — eşik testli seçim + augment'li kopyalar
# ==========================================================================
def is_mixed_opacity(alpha: np.ndarray) -> bool:
    """GT hem katı (alpha > 0.9 oranı >= %8) hem yumuşak (0.05 < alpha < 0.95
    oranı >= %8) piksel içeriyor mu? (Saydam nesnenin katı parçası senaryosu.)"""
    solid = float((alpha > SOLID_ALPHA_THRESH).mean())
    soft = float(((alpha > SOFT_LO) & (alpha < SOFT_HI)).mean())
    return solid >= SOLID_MIN_RATIO and soft >= SOFT_MIN_RATIO


def select_mixed_sources(
    stems: list[str],
    category_by_stem: dict[str, str],
    train_gt_dir: Path,
    out_im_dir: Path,
    out_gt_dir: Path,
    max_sources: int,
) -> list[str]:
    """`transparent` kategorisindeki stem'leri SIRALI tarar, `is_mixed_opacity`
    testini geçen ilk `max_sources` stem'i döndürür (deterministik). Çıktı
    kopyalarından biri diskte zaten varsa uygunluk GT yüklenmeden kabul edilir
    (yalnız uygun kaynaklar çıktı üretebildiği için dosya varlığı kanıttır —
    resume'da binlerce PNG'yi yeniden taramamak için)."""
    chosen: list[str] = []
    if max_sources <= 0:
        return chosen
    for stem in stems:
        if len(chosen) >= max_sources:
            break
        if category_by_stem.get(stem) != "transparent" or _DERIVED_SUFFIX_RE.search(stem):
            continue
        if any(
            (out_im_dir / f"{stem}_m{ci:02d}.jpg").exists()
            and (out_gt_dir / f"{stem}_m{ci:02d}.png").exists()
            for ci in range(MIXED_COPIES)
        ):
            chosen.append(stem)
            continue
        alpha = _load_alpha(train_gt_dir / f"{stem}.png")
        if is_mixed_opacity(alpha):
            chosen.append(stem)
    return chosen


def gen_mixed(
    sources: list[str],
    train_im_dir: Path,
    train_gt_dir: Path,
    out_im_dir: Path,
    out_gt_dir: Path,
    category_by_stem: dict[str, str],
    seed: int,
    existing_ids: set[str],
) -> tuple[list[dict], int, int]:
    """Kaynak başına `MIXED_COPIES` augment'li kopya (`_m00`, `_m01`):
    `bgr.compositing.augment(..., flip_prob=0.0)` — renk jitter / blur / JPEG
    artifact yalnız RGB'de, geometri değişmez, alpha AYNEN korunur (augment
    imzası koddan doğrulandı: flip tek geometrik dönüşüm ve kapatıldı).
    (satırlar, üretilen, atlanan) döndürür."""
    new_rows: list[dict] = []
    generated = skipped = 0
    for stem in sources:
        category = category_by_stem[stem]
        pending: list[str] = []
        for ci in range(MIXED_COPIES):
            new_stem = f"{stem}_m{ci:02d}"
            if (out_im_dir / f"{new_stem}.jpg").exists() and (out_gt_dir / f"{new_stem}.png").exists():
                skipped += 1
                if new_stem not in existing_ids:
                    new_rows.append({"id": new_stem, "category": category})
                continue
            pending.append(new_stem)
        if not pending:
            continue
        rgb = _load_rgb(train_im_dir / f"{stem}.jpg")
        alpha = _load_alpha(train_gt_dir / f"{stem}.png", (rgb.shape[1], rgb.shape[0]))
        for new_stem in pending:
            rng = _item_rng(seed, new_stem)
            out_rgb, out_alpha = augment(rgb, alpha, rng, flip_prob=0.0)
            _save_pair(out_rgb, out_alpha, out_im_dir / f"{new_stem}.jpg", out_gt_dir / f"{new_stem}.png")
            new_rows.append({"id": new_stem, "category": category})
            generated += 1
    return new_rows, generated, skipped


# ==========================================================================
# Orkestrasyon
# ==========================================================================
def run(
    train_im_dir: Path,
    train_gt_dir: Path,
    category_by_stem: dict[str, str],
    out_dir: Path,
    seed: int = 42,
    edge_count: int = DEFAULT_EDGE_COUNT,
    mixed_cap: int = DEFAULT_MIXED_CAP,
    out_manifest: Path | None = None,
    exclude_stems: set[str] | None = None,
) -> dict[str, int]:
    """İki türev üreticisini koşturur; tür -> yeni üretilen çift sayısı
    döndürür (yalnız >0 olanlar — make_textfx.run() ile aynı kalıp).

    `category_by_stem`: kaynak stem -> kategori (Drive'daki
    `train_composites_manifest.jsonl`'den, bkz. train_colab_lib.
    load_stem_categories). Haritada olmayan stem'ler kaynak havuzuna girmez.
    `exclude_stems`: kaynak olarak KULLANILMAYACAK stem'ler (VAL sızıntı
    koruması — çağıran val_stems.json'dan türetir).
    `mixed_cap`: mixed kopya TOPLAM üst sınırı (kaynak sayısı üst sınırı
    `mixed_cap / MIXED_COPIES`); mixed toplam = uygun çift sayısı × 2, üst
    sınırla kırpılmış."""
    train_im_dir, train_gt_dir = Path(train_im_dir), Path(train_gt_dir)
    out_dir = Path(out_dir)
    out_im_dir = out_dir / "im"
    out_gt_dir = out_dir / "gt"
    out_im_dir.mkdir(parents=True, exist_ok=True)
    out_gt_dir.mkdir(parents=True, exist_ok=True)
    out_manifest = Path(out_manifest) if out_manifest else out_dir / "manifest.jsonl"
    existing_ids = _load_manifest_ids(out_manifest)

    stems = _list_pair_stems(train_im_dir, train_gt_dir)
    if exclude_stems:
        stems = [s for s in stems if s not in exclude_stems]

    all_rows: list[dict] = []
    result: dict[str, int] = {}
    total_skipped = 0

    if edge_count > 0:
        sources = select_edge_sources(stems, category_by_stem, edge_count)
        n_pref = sum(1 for s, c in sources if _is_preferred_source(s, c))
        print(f"edge-crop kaynakları: {len(sources)} (tercihli/gerçek-arka-planlı: {n_pref})")
        rows, generated, skipped = gen_edge_crops(
            sources, train_im_dir, train_gt_dir, out_im_dir, out_gt_dir, seed, existing_ids
        )
        all_rows += rows
        total_skipped += skipped
        if generated:
            result["edge"] = generated

    if mixed_cap > 0:
        mixed_sources = select_mixed_sources(
            stems, category_by_stem, train_gt_dir, out_im_dir, out_gt_dir,
            max_sources=mixed_cap // MIXED_COPIES,
        )
        print(f"mixed kaynakları: {len(mixed_sources)} (kopya hedefi: {len(mixed_sources) * MIXED_COPIES})")
        rows, generated, skipped = gen_mixed(
            mixed_sources, train_im_dir, train_gt_dir, out_im_dir, out_gt_dir,
            category_by_stem, seed, existing_ids,
        )
        all_rows += rows
        total_skipped += skipped
        if generated:
            result["mixed"] = generated

    # manifest'e yalnız yeni id'ler (run içi güvenlik dedup'u dahil — make_textfx kalıbı)
    fresh: list[dict] = []
    seen = set(existing_ids)
    for row in all_rows:
        if row["id"] not in seen:
            seen.add(row["id"])
            fresh.append(row)
    if fresh:
        _append_manifest(out_manifest, fresh)

    print(f"{sum(result.values())} yeni çift yazıldı, {total_skipped} zaten vardı (atlandı)")
    for kind, n in sorted(result.items()):
        print(f"{kind}: {n}")
    return result


def _load_categories(path: Path) -> dict[str, str]:
    """JSONL manifest'ten stem -> kategori haritası (satırlarda en az
    `id` + `category` beklenir — train_composites_manifest.jsonl şeması)."""
    result: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                result[row["id"]] = row["category"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--train-im-dir", required=True, help="kaynak TRAIN im/ dizini (*.jpg)")
    parser.add_argument("--train-gt-dir", required=True, help="kaynak TRAIN gt/ dizini (*.png)")
    parser.add_argument(
        "--categories-manifest", required=True,
        help="stem->kategori JSONL'i (train_composites_manifest.jsonl)",
    )
    parser.add_argument("--out-dir", required=True, help="çıktı kökü (im/ + gt/ + manifest.jsonl)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--edge-count", type=int, default=DEFAULT_EDGE_COUNT)
    parser.add_argument("--mixed-cap", type=int, default=DEFAULT_MIXED_CAP)
    parser.add_argument("--out-manifest", default=None, help="varsayılan: <out-dir>/manifest.jsonl")
    parser.add_argument(
        "--exclude-stems-file", default=None,
        help="her satırda bir kaynak stem (VAL sızıntı koruması) — bunlar kaynak olarak kullanılmaz",
    )
    args = parser.parse_args()
    exclude_stems = None
    if args.exclude_stems_file:
        exclude_stems = {
            line.strip()
            for line in Path(args.exclude_stems_file).read_text().splitlines()
            if line.strip()
        }
    run(
        Path(args.train_im_dir),
        Path(args.train_gt_dir),
        _load_categories(Path(args.categories_manifest)),
        Path(args.out_dir),
        seed=args.seed,
        edge_count=args.edge_count,
        mixed_cap=args.mixed_cap,
        out_manifest=Path(args.out_manifest) if args.out_manifest else None,
        exclude_stems=exclude_stems,
    )


if __name__ == "__main__":
    main()
