"""Contact-sheet gallery for the design-expert three-way benchmark.
One JPG per category: rows = test images, columns = original | lucida-v7 |
m35 pipeline | ideogram, RGBA composited on the dark checkerboard, per-cell
MAE labels. Writes docs/assets/design-expert-bench/<cat>.jpg.

Usage: uv run python scripts/dx_gallery.py [--per-cat 4]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = Path("results/dx_bench")
DEST = Path("docs/assets/design-expert-bench")
CELL = 300
LABEL_H = 26


def dark_checker(h, w, sq=20):
    yy, xx = np.mgrid[0:h, 0:w]
    tile = ((yy // sq + xx // sq) % 2).astype(np.float32)
    return (34.0 + tile * 20.0)[..., None].repeat(3, axis=-1)


def cell_from_alpha(img_rgb: Image.Image, alpha_path: Path) -> Image.Image:
    rgb = np.asarray(img_rgb, dtype=np.float32)
    im = Image.open(alpha_path)
    a = np.asarray(im.split()[-1] if im.mode == "RGBA" else im.convert("L"),
                   dtype=np.float32) / 255.0
    if a.shape != rgb.shape[:2]:
        a = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize(
            (rgb.shape[1], rgb.shape[0]), Image.BILINEAR), dtype=np.float32) / 255.0
    if im.mode == "RGBA":  # ideogram ships its own (decontaminated) color
        fg = np.asarray(im.convert("RGB"), dtype=np.float32)
        if fg.shape != rgb.shape:
            fg = np.asarray(im.convert("RGB").resize(
                (rgb.shape[1], rgb.shape[0])), dtype=np.float32)
    else:
        fg = rgb
    bg = dark_checker(*rgb.shape[:2])
    comp = fg * a[..., None] + bg * (1 - a[..., None])
    return Image.fromarray(comp.clip(0, 255).astype(np.uint8))


def fit(im: Image.Image) -> Image.Image:
    im = im.copy()
    im.thumbnail((CELL, CELL))
    canvas = Image.new("RGB", (CELL, CELL), (24, 24, 24))
    canvas.paste(im, ((CELL - im.width) // 2, (CELL - im.height) // 2))
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cat", type=int, default=4)
    args = ap.parse_args()
    metrics = json.loads((OUT / "metrics.json").read_text())
    per = metrics["per_image"]
    manifest = {json.loads(l)["id"]: json.loads(l)
                for l in open("data/testset/manifest.jsonl")}
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 15)
    except OSError:
        font = ImageFont.load_default()
    DEST.mkdir(parents=True, exist_ok=True)
    cols = [("original", None), ("lucida-v7", OUT / "v7"),
            ("m35 pipeline", OUT / "m35pipe"),
            ("ideogram", Path("results/ideogram"))]
    for cat in ("design", "fx", "text", "complex"):
        ids = [i for i, v in per.items() if v["category"] == cat]
        # most interesting first: widest spread between the three models
        ids.sort(key=lambda i: -np.ptp([per[i].get(m, 0) for m in
                 ("lucida-v7", "m35-pipeline", "ideogram")]))
        ids = ids[:args.per_cat]
        W = CELL * 4
        H = (CELL + LABEL_H) * len(ids) + LABEL_H
        sheet = Image.new("RGB", (W, H), (16, 16, 16))
        d = ImageDraw.Draw(sheet)
        for c, (name, _) in enumerate(cols):
            d.text((c * CELL + 8, 5), name, fill=(220, 220, 220), font=font)
        for rix, iid in enumerate(ids):
            r = manifest[iid]
            img = Image.open(r["image"]).convert("RGB")
            y = LABEL_H + rix * (CELL + LABEL_H)
            for c, (name, src) in enumerate(cols):
                if src is None:
                    cellimg = fit(img)
                    label = iid[:34]
                else:
                    p = src / f"{iid}.png"
                    if not p.exists():
                        continue
                    cellimg = fit(cell_from_alpha(img, p))
                    key = "m35-pipeline" if name == "m35 pipeline" else name
                    mae = per[iid].get(key)
                    label = f"MAE {mae:.4f}" if mae is not None else ""
                sheet.paste(cellimg, (c * CELL, y))
                d.text((c * CELL + 8, y + CELL + 4), label,
                       fill=(180, 180, 180), font=font)
        dst = DEST / f"{cat}.jpg"
        sheet.save(dst, quality=88)
        print("wrote", dst)


if __name__ == "__main__":
    main()
