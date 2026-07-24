"""Visual release panel: side-by-side model outputs on a dark checkerboard.

The release bar for Lucida Design (spec 2026-07-24) says the final verdict
belongs to the USER'S EYE on real-world cases — twice a numeric benchmark
missed a regression the eye caught instantly. This tool renders, for every
image in a directory, one row: [original | model A | model B | ...], each
model's RGBA composited over a dark checkerboard (light haze and kept
whites become obvious), plus a contact sheet of all rows.

Inputs live in `data/visual_panel/` (untracked — third-party artwork stays
local). Any image dropped there joins the panel.

Usage:
    uv run python scripts/make_visual_panel.py \
        --images data/visual_panel --models lucida-v7,lucida-v14 \
        --out results/visual_panel
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None

CELL = 24          # checkerboard cell size
DARK, DARKER = 70, 45
COL_W = 480        # per-column width in a row
SHEET_W = 1600


def checkerboard(w: int, h: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    cells = ((yy // CELL) + (xx // CELL)) % 2
    board = np.where(cells[..., None] == 0, DARK, DARKER).astype(np.float32)
    return np.repeat(board, 3, axis=-1) if board.shape[-1] == 1 else board


def composite_over_checker(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    h, w = alpha.shape
    board = checkerboard(w, h)
    out = alpha[..., None] * rgb.astype(np.float32) + (1 - alpha[..., None]) * board
    return out.round().clip(0, 255).astype(np.uint8)


def _fit(img: Image.Image, w: int) -> Image.Image:
    return img.resize((w, max(1, int(img.height * w / img.width))), Image.BILINEAR)


def _label(img: Image.Image, text: str) -> Image.Image:
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 8 + 7 * len(text), 22], fill=(0, 0, 0))
    d.text((4, 4), text, fill=(255, 80, 80))
    return img


def make_panel(images_dir: Path, models: list[str], out_dir: Path,
               segmenter_factory=None) -> list[Path]:
    """One labeled row image per input; returns the row paths (plus writes
    contact_sheet.jpg). `segmenter_factory(name) -> obj.predict_alpha(img)`
    defaults to the registry (injected in tests)."""
    if segmenter_factory is None:
        from bgr.registry import get_segmenter as segmenter_factory  # noqa: N816

    images_dir, out_dir = Path(images_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    segs = {m: segmenter_factory(m) for m in models}
    rows_out: list[Path] = []

    inputs = sorted(p for p in images_dir.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    for p in inputs:
        img = Image.open(p).convert("RGB")
        rgb = np.asarray(img, dtype=np.uint8)
        cols = [_label(_fit(img.copy(), COL_W), "original")]
        for m in models:
            alpha = segs[m].predict_alpha(img)
            comp = Image.fromarray(composite_over_checker(rgb, alpha))
            cols.append(_label(_fit(comp, COL_W), m))
        h = max(c.height for c in cols)
        row = Image.new("RGB", (COL_W * len(cols) + 6 * (len(cols) - 1), h), (20, 20, 20))
        x = 0
        for c in cols:
            row.paste(c, (x, 0))
            x += COL_W + 6
        dst = out_dir / f"panel_{p.stem}.jpg"
        row.save(dst, quality=90)
        rows_out.append(dst)

    if rows_out:
        rows = [_fit(Image.open(r), SHEET_W) for r in rows_out]
        sheet = Image.new("RGB", (SHEET_W, sum(r.height for r in rows) + 8 * len(rows)), (10, 10, 10))
        y = 0
        for r in rows:
            sheet.paste(r, (0, y))
            y += r.height + 8
        sheet.save(out_dir / "contact_sheet.jpg", quality=88)
    return rows_out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True)
    ap.add_argument("--models", required=True, help="comma-separated registry names")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    paths = make_panel(Path(a.images), a.models.split(","), Path(a.out))
    print(f"{len(paths)} panel rows -> {a.out}")


if __name__ == "__main__":
    main()
