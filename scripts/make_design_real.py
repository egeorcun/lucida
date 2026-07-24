"""Real layered-design benchmark pairs from the Crello dataset (v14).

WHY (the v13 lesson): our design test category is a holdout of our OWN
synthetic generator, and the illustration category is ToonOut cartoons —
neither covers real-world layered artwork (posters, social-media collages,
Pinterest-style compositions). v13 regressed exactly there (it erased or
faded design elements that v7 kept) and the benchmark never saw it. This
script builds a `design_real` category with EXACT ground truth from
`cyberagent/crello` (23k real design templates whose elements ship as
individual RGBA layers): flatten the layers -> composite image; union the
FOREGROUND layers' alphas -> GT. No matting estimation anywhere.

CONTRACTS:
- Background = the leading run of bottom-of-z elements whose bbox covers
  >= BG_COVER_FRAC of the canvas (the classic full-canvas photo/color
  layer). Everything above it is foreground. Templates with no such layer
  compose over white (design-on-white is still a valid, meaningful pair).
- Element geometry is ABSOLUTE canvas pixels (left/top/width/height), the
  element bitmap is resized to that box; `angle` (radians, CCW) rotates
  around the box center; `opacity` scales the layer alpha.
- QUALITY GUARD: Crello ships a rendered `preview` per template. Our
  composite is compared against it (downscaled, mean |diff|); templates
  whose render deviates more than PREVIEW_MAX_DIFF are SKIPPED — this
  auto-filters blend modes/filters the simple renderer cannot reproduce,
  so a bad render can never become a "ground truth".
- Foreground-coverage filter: GT union must land in
  [MIN_FG_COVER, MAX_FG_COVER] — all-background and all-foreground
  templates carry no signal.
- Output follows the testset schema (id/image/category/gt_alpha JSONL,
  im/*.jpg + gt/*.png) into a SEPARATE manifest — the frozen 203-image
  set is never touched.

Usage:
    uv run python scripts/make_design_real.py \
        --out-dir data/testset_design_real --count 16 --seed 7
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

BG_COVER_FRAC = 0.92
MIN_FG_COVER, MAX_FG_COVER = 0.08, 0.92
PREVIEW_MAX_DIFF = 25.0   # mean |uint8 diff| vs the shipped preview render
MAX_LONG_SIDE = 1024
MIN_ELEMENTS = 4          # trivially simple templates carry little signal


def element_canvas_layer(
    img: Image.Image,
    left: float, top: float, width: float, height: float,
    angle: float, opacity: float,
    canvas_w: int, canvas_h: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Renders one element into canvas space; returns (rgb float, alpha float)."""
    w = max(1, int(round(width)))
    h = max(1, int(round(height)))
    layer = img.convert("RGBA").resize((w, h), Image.LANCZOS)
    if abs(angle) > 1e-4:
        layer = layer.rotate(-math.degrees(angle), expand=True,
                             resample=Image.BICUBIC, fillcolor=(0, 0, 0, 0))
        # rotation expands the box around its center — re-anchor
        left = left + (w - layer.width) / 2.0
        top = top + (h - layer.height) / 2.0
    arr = np.asarray(layer, dtype=np.float32)
    a = (arr[..., 3] / 255.0) * float(opacity)
    rgb = arr[..., :3]

    out_rgb = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    out_a = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    x0, y0 = int(round(left)), int(round(top))
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1 = min(canvas_w, x0 + layer.width)
    dy1 = min(canvas_h, y0 + layer.height)
    if dx1 <= dx0 or dy1 <= dy0:
        return out_rgb, out_a
    sw, sh = dx1 - dx0, dy1 - dy0
    out_rgb[dy0:dy1, dx0:dx1] = rgb[sy0:sy0 + sh, sx0:sx0 + sw]
    out_a[dy0:dy1, dx0:dx1] = a[sy0:sy0 + sh, sx0:sx0 + sw]
    return out_rgb, out_a


def split_background(ex: dict) -> int:
    """Index of the first FOREGROUND element: the leading bottom-of-z run of
    ~full-canvas elements is the background stack."""
    cw, ch = ex["canvas_width"], ex["canvas_height"]
    n_bg = 0
    for i in range(ex["length"]):
        cover_w = ex["width"][i] / cw
        cover_h = ex["height"][i] / ch
        if cover_w >= BG_COVER_FRAC and cover_h >= BG_COVER_FRAC:
            n_bg += 1
        else:
            break
    return n_bg


def render_template(ex: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """(composite RGB uint8, GT alpha float32) or None if unusable."""
    cw, ch = int(ex["canvas_width"]), int(ex["canvas_height"])
    if ex["length"] < MIN_ELEMENTS or cw < 200 or ch < 200 or max(cw, ch) / min(cw, ch) > 2.6:
        return None
    n_bg = split_background(ex)
    if n_bg >= ex["length"]:
        return None  # everything is background — no foreground to matte

    comp = np.full((ch, cw, 3), 255.0, dtype=np.float32)
    gt = np.zeros((ch, cw), dtype=np.float32)
    for i in range(ex["length"]):
        rgb, a = element_canvas_layer(
            ex["image"][i], ex["left"][i], ex["top"][i],
            ex["width"][i], ex["height"][i], ex["angle"][i], ex["opacity"][i],
            cw, ch,
        )
        comp = a[..., None] * rgb + (1.0 - a[..., None]) * comp
        if i >= n_bg:
            gt = 1.0 - (1.0 - gt) * (1.0 - a)

    cover = float((gt > 0.5).mean())
    if not (MIN_FG_COVER <= cover <= MAX_FG_COVER):
        return None

    # quality guard: our render must agree with Crello's own preview
    prev = ex["preview"].convert("RGB")
    ours = Image.fromarray(comp.round().clip(0, 255).astype(np.uint8)).resize(prev.size, Image.BILINEAR)
    diff = float(np.abs(np.asarray(ours, np.int16) - np.asarray(prev, np.int16)).mean())
    if diff > PREVIEW_MAX_DIFF:
        return None

    scale = min(1.0, MAX_LONG_SIDE / max(cw, ch))
    if scale < 1.0:
        nw, nh = int(cw * scale), int(ch * scale)
        comp_img = Image.fromarray(comp.round().clip(0, 255).astype(np.uint8)).resize((nw, nh), Image.LANCZOS)
        gt_img = Image.fromarray(np.round(gt * 255).astype(np.uint8)).resize((nw, nh), Image.BILINEAR)
        return np.asarray(comp_img), np.asarray(gt_img, np.float32) / 255.0
    return comp.round().clip(0, 255).astype(np.uint8), gt


def run(out_dir: Path, count: int = 16, seed: int = 7, split: str = "test",
        quality: int = 95) -> int:
    """Resume-safe: existing im+gt pairs are skipped (Colab session drops must
    not restart hours of work); the manifest is rebuilt at the end from the
    outputs actually on disk. Progress prints every 250 accepted pairs (the
    v8 lesson: a multi-hour silent stage is indistinguishable from a hung
    one)."""
    from datasets import load_dataset

    out_dir = Path(out_dir)
    (out_dir / "im").mkdir(parents=True, exist_ok=True)
    (out_dir / "gt").mkdir(parents=True, exist_ok=True)
    ds = load_dataset("cyberagent/crello", split=split)
    order = np.random.default_rng(seed).permutation(len(ds))

    rows, taken, scanned, skipped = [], 0, 0, 0
    for idx in order:
        if taken >= count:
            break
        scanned += 1
        ex = ds[int(idx)]
        stem = f"crello_{ex['id']}"
        im_p = out_dir / "im" / f"{stem}.jpg"
        gt_p = out_dir / "gt" / f"{stem}.png"
        if im_p.exists() and gt_p.exists():
            rows.append({"id": stem, "image": str(im_p), "category": "design_real",
                         "gt_alpha": str(gt_p)})
            taken += 1
            skipped += 1
            continue
        result = render_template(ex)
        if result is None:
            continue
        comp, gt = result
        Image.fromarray(comp).save(im_p, quality=quality)
        Image.fromarray(np.round(gt * 255).astype(np.uint8), mode="L").save(gt_p)
        rows.append({"id": stem, "image": str(im_p), "category": "design_real",
                     "gt_alpha": str(gt_p)})
        taken += 1
        if taken % 250 == 0:
            print(f"design_real progress: {taken}/{count} (scanned {scanned})")

    with open(out_dir / "manifest.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{taken} pairs ready ({skipped} pre-existing, scanned {scanned}) -> {out_dir}")
    return taken


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--count", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    run(Path(a.out_dir), count=a.count, seed=a.seed, split=a.split)


if __name__ == "__main__":
    main()
