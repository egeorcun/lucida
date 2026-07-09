"""GT'li eğitim kaynaklarından kategorili `data/train/manifest.jsonl` üret.

Kullanım:
    uv run python scripts/build_trainset.py                # tüm SOURCES + DIS5K-TR
    uv run python scripts/build_trainset.py source camotr   # tek kaynağı ekle

Test setinden (build_testset.py) farkı: dosyalar KOPYALANMAZ, `data/train/{images,gt}/`
altına orijinal ham dosyaya işaret eden SEMBOLİK BAĞLANTI (symlink) oluşturulur — disk
bütçesi (bkz. Faz 2 planı Global Constraints) kopyalamaya izin vermez. Format dönüşümü
(örn. GT'yi tek kanallı L PNG'ye normalize etmek) bu yüzden symlink ANINDA değil, ham
veri edinimi (fetch) aşamasında yapılır; bu script yalnız zaten normalize edilmiş ham
dosyaları glob'lar ve bağlar.

Ham veri edinimi (data/raw_train/ altına, git dışı; DISK BÜTÇESİ: kaynak başına ≤300MB,
bkz. Faz 2 planı REVİZE disk kuralı — tam materyalizasyon Colab'da, bkz. data/train_sources.json):

- DIS5K-TR (kategori: dosya adı token'ından thin/complex, bkz. classify_disvd): HF
  dataset "nobg/DIS5K", data/DIS_TR-00000-of-00006-*.parquet (6 parçadan yalnız ilki;
  tam DIS_TR ~3000 çift). Parça bile ~480MB olduğundan TÜM PARQUET İNDİRİLMEDİ: pyarrow
  ParquetFile + huggingface_hub.HfFileSystem (fsspec, HTTP range request) ile YALNIZ
  0. row-group'u (100 satır, ~120MB) kısmi okundu, image/label byte'ları PIL ile
  data/raw_train/dis5k/{im,gt}/ altına dosya olarak yazıldı (100 çift; GT convert("L")
  ile tek kanala normalize edildi). Tam DIS_TR indirimi Colab'da (T5 notebook)
  data/train_sources.json'daki hf_repo üzerinden yapılacak.
- CAMO-TR (kategori: camouflage): HF dataset "nobg/camo" (resmi CAMO, Le et al. 2019;
  proje sayfası https://sites.github.com/view/ltnghia/research/camo; lisans
  CC-BY-NC-SA 4.0; train split 1000 çift = 3 row-group: 423+423+154). Yalnız 0.
  row-group'un YALNIZ image_name/image/mask kolonları (overlaid_mask_1/2 HARİÇ —
  column pruning ile ekstra indirme önlendi) kısmi okundu; ilk 100 satır
  data/raw_train/camo/{im,gt}/ altına yazıldı (~18MB toplam).
- COD10K-TR (kategori: camouflage): HF'de piksel-seviyeli GT maskeli TRAIN mirror'ı
  YOK (bkz. scripts/build_testset.py docstring — Chranos/COD10K_train ve Jrseee/COD10K
  boş/LFS işaretçisiz repo; chandrabhuma/animal_cod10k(_train) yalnız görsel+soru-cevap,
  piksel maskesi yok; bu görev sırasında ayrıca aranan SmallDoge/CoD-10K alakasız bir
  metin veri seti — "CoD" isim çakışması, "Chain of Draft" tipi bir kod/metin korpüsü).
  Resmi kaynak yalnız Google Drive üzerinden (SINet/DengPingFan reposu,
  https://github.com/DengPingFan/SINet): COD10K-train dosya id
  "1D9bf1KeeCJsxxri6d2qAC7z6O1X_fxpt" (~3040 çift). Bu ortamda gdown/Drive kimlik
  doğrulaması olmadığından LOKAL ÖRNEKLEM ATLANDI (görev talimatı: "yoksa ATLA");
  kayıt data/train_sources.json'da (drive_id + resmi URL) mevcut — Colab'da (T5)
  gdown ile tam indirilecek.
"""
import random
import sys
from pathlib import Path

from build_testset import _sanitize, classify_disvd  # noqa: E402  (aynı dizin, scripts/)

from benchmark.testset import append_entries

random.seed(42)
ROOT = Path(__file__).resolve().parent.parent
OUT_IMG = ROOT / "data/train/images"
OUT_GT = ROOT / "data/train/gt"
MANIFEST = ROOT / "data/train/manifest.jsonl"


def _link(src: Path, dst: Path) -> None:
    """dst -> src sembolik bağlantısı oluşturur (kopya YOK; disk tasarrufu)."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def sample_source(name: str, img_glob: str, gt_glob: str, category: str,
                   n: int | None = None) -> list[dict]:
    """Bir kaynaktan (img_glob/gt_glob stem'e göre eşleştirilir) örnekle ve
    `data/train/{images,gt}/` altına symlink'le. n=None -> tüm eşleşen çiftler."""
    imgs = sorted(ROOT.glob(img_glob))
    gts = {p.stem: p for p in ROOT.glob(gt_glob)}
    paired = [(i, gts[i.stem]) for i in imgs if i.stem in gts]
    if n is not None and n < len(paired):
        paired = random.sample(paired, n)

    rows = []
    for img, gt in paired:
        rid = f"{name}_{_sanitize(img.stem)}"
        dst_i = OUT_IMG / f"{rid}{img.suffix}"
        dst_g = OUT_GT / f"{rid}{gt.suffix}"
        _link(img, dst_i)
        _link(gt, dst_g)
        rows.append({"id": rid, "image": str(dst_i.relative_to(ROOT)),
                     "category": category, "gt_alpha": str(dst_g.relative_to(ROOT))})
    return rows


def sample_disvd_tokens(name: str, img_glob: str, gt_glob: str,
                         n: int | None = None) -> list[dict]:
    """DIS5K havuzundan örnekle; kategori dosya adı token'ından (classify_disvd,
    build_testset.py'den yeniden kullanılır) atanır — rastgele dağıtım YOKTUR."""
    imgs = sorted(ROOT.glob(img_glob))
    gts = {p.stem: p for p in ROOT.glob(gt_glob)}
    paired = [(i, gts[i.stem]) for i in imgs if i.stem in gts]
    random.shuffle(paired)
    if n is not None:
        paired = paired[:n]

    rows = []
    for img, gt in paired:
        sanitized_stem = _sanitize(img.stem)
        category = classify_disvd(sanitized_stem)
        rid = f"{name}_{category}_{sanitized_stem}"
        dst_i = OUT_IMG / f"{rid}{img.suffix}"
        dst_g = OUT_GT / f"{rid}{gt.suffix}"
        _link(img, dst_i)
        _link(gt, dst_g)
        rows.append({"id": rid, "image": str(dst_i.relative_to(ROOT)),
                     "category": category, "gt_alpha": str(dst_g.relative_to(ROOT))})
    return rows


# (kaynak_ad, images_glob, gt_glob, kategori, adet)
SOURCES: list[tuple[str, str, str, str, int]] = [
    ("camotr", "data/raw_train/camo/im/*", "data/raw_train/camo/gt/*", "camouflage", 100),
]

# DIS5K-TR tek havuzdan örneklenir; kategori dosya adı token'ından atanır.
DIS5KTR_IMG_GLOB = "data/raw_train/dis5k/im/*"
DIS5KTR_GT_GLOB = "data/raw_train/dis5k/gt/*"
DIS5KTR_N = 100


def build() -> None:
    OUT_IMG.mkdir(parents=True, exist_ok=True)
    OUT_GT.mkdir(parents=True, exist_ok=True)
    for src in SOURCES:
        rows = sample_source(*src)
        append_entries(str(MANIFEST), rows)
        print(f"{src[0]} ({src[3]}): {len(rows)} örnek")

    rows = sample_disvd_tokens("dis5ktr", DIS5KTR_IMG_GLOB, DIS5KTR_GT_GLOB, DIS5KTR_N)
    append_entries(str(MANIFEST), rows)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
    for category, count in sorted(counts.items()):
        print(f"dis5ktr ({category}): {count} örnek")


def add_source(name: str) -> None:
    """SOURCES'taki tek bir kaynağı örnekle ve manifest'e ekle (artımlı ekleme)."""
    matches = [s for s in SOURCES if s[0] == name]
    if not matches:
        raise SystemExit(f"bilinmeyen kaynak: {name} (SOURCES: {[s[0] for s in SOURCES]})")
    rows = sample_source(*matches[0])
    append_entries(str(MANIFEST), rows)
    print(f"{name} ({matches[0][3]}): {len(rows)} örnek eklendi")


def main() -> None:
    OUT_IMG.mkdir(parents=True, exist_ok=True)
    OUT_GT.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) >= 3 and sys.argv[1] == "source":
        add_source(sys.argv[2])
    else:
        build()


if __name__ == "__main__":
    main()
