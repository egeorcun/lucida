"""GT'li kaynak setlerden kategorili test seti örnekle.

Kullanım:
    uv run python scripts/build_testset.py                          # GT'li setleri örnekle
    uv run python scripts/build_testset.py add data/testset/incoming product  # GT'siz görsel ekle

Ham veri edinimi (data/raw/ altına, git dışı):
- P3M-500-NP: HF dataset "Rupant-ted/p3m-10k" -> data/p3m10k.zip; zip'ten yalnızca
  P3M-10k/validation/P3M-500-NP/{original_image,mask} çıkarılıp
  data/raw/p3m10k/validation/P3M-500-NP/ altına taşındı (500 çift).
- Transparent-460: HF dataset "Thinnaphat/transparent-460",
  snapshot_download(allow_patterns=["Test/*"]) -> data/raw/trans460/Test/{fg,alpha} (50 çift).
- DIS-VD: HF dataset "nobg/DIS5K" -> data/DIS_VD-*.parquet; parquet'teki (image, label)
  byte'ları PIL ile data/raw/dis5k/DIS-VD/{im,gt}/ altına dosya olarak yazıldı (470 çift,
  pyarrow gerekir). Ham dosya adları '<grupIdx>#<Grup>#<sınıfIdx>#<Sınıf>#<orijinalAd>'
  biçimindedir (ör. '1#Accessories#5#Jewelry#12836143775_...').
- CAMO (camouflage kategorisi): Faz 2 Task 1 için COD10K test split'i HF'de aranmış
  (HfApi().list_datasets/list_repo_files: "Chranos/COD10K_train", "Jrseee/COD10K" boş
  repo/LFS işaretçisiz; "chandrabhuma/animal_cod10k" gerçek COD10K-CAM-Test görsellerini
  (2026 örnek, id'ler "COD10K-CAM-..." önekli) içeriyor ama yalnız görsel+soru-cevap,
  piksel seviyeli GT maske YOK; resmi kaynak (SINet/DengPingFan) yalnız Google Drive
  üzerinden ve bu ortamda gdown/kaggle kimlik bilgisi yok). Bunun yerine HF dataset
  "nobg/camo" kullanıldı: resmi CAMO (Camouflaged Object, Le et al.) test split'i,
  image+mask parquet (250 çift, ~61MB) -> data/raw/camo_test/{im,gt}/ altına PIL ile
  dosya olarak yazıldı. CAMO, bu projenin Faz 2 planında da (Task 2) COD10K-TR'nin
  yanında kabul edilen bir kamuflaj kaynağıdır; "camouflage" kategorisi için COD10K
  yerine kullanılması raporda belgelenmiştir.

NOT (final review düzeltmesi): DIS-VD satırları ilk halde thin/complex/general'e
RASTGELE dağıtılmıştı (bkz. git geçmişi). scripts/relabel_disvd.py bunu tek seferlik
düzeltti: gerçek DIS5K sınıfı id'nin içine kodlu olduğundan (classify_disvd(), aşağıda)
her satırın kategorisi dosya adı token'ından yeniden hesaplandı; id/dosya adları
değişmedi. sample_disvd_multi() artık BAŞTAN İTİBAREN classify_disvd() kullanır, yani
gelecekteki yeniden derlemelerde rastgele dağıtım YOKTUR.
"""
import random
import re
import sys
from pathlib import Path

from PIL import Image

from benchmark.testset import append_entries

random.seed(42)
ROOT = Path(__file__).resolve().parent.parent
OUT_IMG = ROOT / "data/testset/images"
OUT_GT = ROOT / "data/testset/gt"
MANIFEST = ROOT / "data/testset/manifest.jsonl"

# (kaynak_ad, images_glob, gt_glob, kategori, adet)
# NOT: AIM-500 ve AM-2k için çalışan bir mirror bulunamadı (yalnızca Google Drive,
# klasör bazlı, binlerce dosyalık "train" ağacını da tarıyor -> pratik değil).
# Bu yüzden DIS-VD (470 çift) üç kategoriye (thin/complex/general) ayrık örneklenerek
# GT'li toplamı dengelemek için kullanıldı; bkz. sample_disvd_multi().
SOURCES: list[tuple[str, str, str, str, int]] = [
    ("p3m", "data/raw/p3m10k/validation/P3M-500-NP/original_image/*.jpg",
     "data/raw/p3m10k/validation/P3M-500-NP/mask/*.png", "hair", 40),
    ("trans460", "data/raw/trans460/Test/fg/*", "data/raw/trans460/Test/alpha/*", "transparent", 25),
    ("camo", "data/raw/camo_test/im/*", "data/raw/camo_test/gt/*", "camouflage", 25),
]

# DIS-VD tek havuzdan örneklenir; kategori dosya adı token'ından atanır (bkz. classify_disvd).
DISVD_IMG_GLOB = "data/raw/dis5k/DIS-VD/im/*"
DISVD_GT_GLOB = "data/raw/dis5k/DIS-VD/gt/*"
DISVD_N = 65  # eski thin(20)+complex(30)+general(15) toplamıyla aynı büyüklük

# DIS5K sınıf token'larından ince/karmaşık (thin/complex) sınıflandırma.
# "thin" = telli/örgü/iskelet gibi ince, delikli/örgülü geometri baskın olan sınıflar.
_THIN_DISVD_CLASSES = {
    "racket", "cable", "wire", "fence", "gate", "antenna", "jewelry", "chandelier",
    "bicycle", "tricycle", "wheel", "ladder", "windmill", "drum", "drumkit", "scaffold",
    "net", "skeleton", "umbrella", "polevault", "handrail", "floorlamp", "musicstand",
    "stand", "spider", "shrimp", "streetlamp", "shoppingcart", "seadragon", "hangglider",
    "basketballhoop", "earphone",
}


def _sanitize(stem: str) -> str:
    """URL-güvenli id: [A-Za-z0-9._-] dışındaki karakterleri '_' yap, ardışıkları tekle."""
    return re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9._-]", "_", stem))


def _copy_alpha(src: Path, dst: Path) -> None:
    """GT alpha'yı tek kanallı (L) PNG olarak normalize edip kopyala."""
    Image.open(src).convert("L").save(dst)


def _copy_image(src: Path, dst: Path) -> None:
    img = Image.open(src)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img.save(dst)


def sample_source(name: str, img_glob: str, gt_glob: str, category: str, n: int) -> list[dict]:
    imgs = sorted(ROOT.glob(img_glob))
    gts = {p.stem: p for p in ROOT.glob(gt_glob)}
    paired = [(i, gts[i.stem]) for i in imgs if i.stem in gts]
    rows = []
    for img, gt in random.sample(paired, min(n, len(paired))):
        rid = f"{name}_{_sanitize(img.stem)}"
        dst_i = OUT_IMG / f"{rid}{img.suffix}"
        dst_g = OUT_GT / f"{rid}.png"
        _copy_image(img, dst_i)
        _copy_alpha(gt, dst_g)
        rows.append({"id": rid, "image": str(dst_i.relative_to(ROOT)),
                     "category": category, "gt_alpha": str(dst_g.relative_to(ROOT))})
    return rows


def parse_disvd_class(stem: str) -> str | None:
    """DIS5K stem/id'sinden sınıf token'ını savunmacı biçimde çıkar.

    Ham dosya adları '<grupIdx>#<Grup>#<sınıfIdx>#<Sınıf>#<orijinalAd>' biçimindedir;
    '#' -> '_' sanitize edildikten (bkz. _sanitize) veya 'disvd_<eskiKategori>_' öneki
    eklendikten sonra da aynı mantıkla ayrıştırılabilir: alt çizgiyle ayrılmış token'lar
    içinde ilk iki SAF SAYISAL token grup/sınıf indeksleridir (grup adı 'Non-motor_Vehicle'
    gibi birden çok token olabilir, önemli değil); sınıf adı, ikinci sayısal token'dan hemen
    sonraki tek token'dır. Ayrıştırılamazsa None döner.
    """
    parts = stem.split("_")
    digit_idxs = [i for i, p in enumerate(parts) if p.isdigit()]
    if len(digit_idxs) < 2:
        return None
    class_idx = digit_idxs[1]
    if class_idx + 1 >= len(parts):
        return None
    return parts[class_idx + 1]


def classify_disvd(stem: str) -> str:
    """DIS5K stem/id'sinden gerçek kategoriyi (thin/complex) döndürür.

    Ayrıştırılamayan veya listede olmayan sınıflar için varsayılan 'complex'tir
    (bkz. _THIN_DISVD_CLASSES; bilinmeyen gelecekteki sınıflar için güvenli varsayılan).
    """
    cls = parse_disvd_class(stem)
    if cls is None:
        return "complex"
    return "thin" if cls.lower() in _THIN_DISVD_CLASSES else "complex"


def sample_disvd_multi(name: str, img_glob: str, gt_glob: str, n: int) -> list[dict]:
    """DIS-VD havuzundan n örnek çeker; kategori dosya adı token'ından (classify_disvd)
    atanır (rastgele dağıtım YOKTUR)."""
    imgs = sorted(ROOT.glob(img_glob))
    gts = {p.stem: p for p in ROOT.glob(gt_glob)}
    paired = [(i, gts[i.stem]) for i in imgs if i.stem in gts]
    random.shuffle(paired)

    rows = []
    for img, gt in paired[:n]:
        sanitized_stem = _sanitize(img.stem)
        category = classify_disvd(sanitized_stem)
        rid = f"{name}_{category}_{sanitized_stem}"
        dst_i = OUT_IMG / f"{rid}{img.suffix}"
        dst_g = OUT_GT / f"{rid}.png"
        _copy_image(img, dst_i)
        _copy_alpha(gt, dst_g)
        rows.append({"id": rid, "image": str(dst_i.relative_to(ROOT)),
                     "category": category, "gt_alpha": str(dst_g.relative_to(ROOT))})
    return rows


def add_unlabeled(folder: str, category: str) -> None:
    rows = []
    for img in sorted((ROOT / folder).glob("*")):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        rid = f"user_{category}_{_sanitize(img.stem)}"
        dst = OUT_IMG / f"{rid}{img.suffix}"
        _copy_image(img, dst)
        rows.append({"id": rid, "image": str(dst.relative_to(ROOT)),
                     "category": category, "gt_alpha": None})
    append_entries(str(MANIFEST), rows)
    print(f"{category}: {len(rows)} GT'siz görsel eklendi")


def build() -> None:
    OUT_IMG.mkdir(parents=True, exist_ok=True)
    OUT_GT.mkdir(parents=True, exist_ok=True)
    for src in SOURCES:
        rows = sample_source(*src)
        append_entries(str(MANIFEST), rows)
        print(f"{src[0]} ({src[3]}): {len(rows)} örnek")

    rows = sample_disvd_multi("disvd", DISVD_IMG_GLOB, DISVD_GT_GLOB, DISVD_N)
    append_entries(str(MANIFEST), rows)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
    for category, count in sorted(counts.items()):
        print(f"disvd ({category}): {count} örnek")


def add_source(name: str) -> None:
    """SOURCES'taki tek bir kaynağı örnekle ve manifest'e ekle (mevcut satırları
    yeniden eklemeden; build() sıfırdan derleme, bu ise ARTIMLI ekleme içindir)."""
    matches = [s for s in SOURCES if s[0] == name]
    if not matches:
        raise SystemExit(f"bilinmeyen kaynak: {name} (SOURCES: {[s[0] for s in SOURCES]})")
    rows = sample_source(*matches[0])
    append_entries(str(MANIFEST), rows)
    print(f"{name} ({matches[0][3]}): {len(rows)} örnek eklendi")


def main() -> None:
    OUT_IMG.mkdir(parents=True, exist_ok=True)
    OUT_GT.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) >= 4 and sys.argv[1] == "add":
        add_unlabeled(sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 3 and sys.argv[1] == "source":
        add_source(sys.argv[2])
    else:
        build()


if __name__ == "__main__":
    main()
