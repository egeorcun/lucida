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
  pyarrow gerekir).
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
]

# DIS-VD tek havuzdan üç kategoriye ayrık (overlap'sız) örneklenir.
DISVD_IMG_GLOB = "data/raw/dis5k/DIS-VD/im/*"
DISVD_GT_GLOB = "data/raw/dis5k/DIS-VD/gt/*"
DISVD_SPLITS: list[tuple[str, int]] = [("thin", 20), ("complex", 30), ("general", 15)]


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


def sample_disvd_multi(name: str, img_glob: str, gt_glob: str,
                        splits: list[tuple[str, int]]) -> list[dict]:
    """Aynı havuzdan (DIS-VD) birden fazla kategoriye örtüşmeyen örnekler çeker."""
    imgs = sorted(ROOT.glob(img_glob))
    gts = {p.stem: p for p in ROOT.glob(gt_glob)}
    paired = [(i, gts[i.stem]) for i in imgs if i.stem in gts]
    random.shuffle(paired)

    rows = []
    idx = 0
    for category, n in splits:
        chunk = paired[idx: idx + n]
        idx += n
        for img, gt in chunk:
            rid = f"{name}_{category}_{_sanitize(img.stem)}"
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

    rows = sample_disvd_multi("disvd", DISVD_IMG_GLOB, DISVD_GT_GLOB, DISVD_SPLITS)
    append_entries(str(MANIFEST), rows)
    for category, n in DISVD_SPLITS:
        count = sum(1 for r in rows if r["category"] == category)
        print(f"disvd ({category}): {count} örnek")


def main() -> None:
    OUT_IMG.mkdir(parents=True, exist_ok=True)
    OUT_GT.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) >= 4 and sys.argv[1] == "add":
        add_unlabeled(sys.argv[2], sys.argv[3])
    else:
        build()


if __name__ == "__main__":
    main()
