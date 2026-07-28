"""Atmosphere-density training pairs (Lucida v18, spec 2026-07-29).

WHY (the I HEARD CHEESE lesson): the model prints halftone smoke/glow
fields at alpha ~1.0 while the Ideogram reference renders them as INK
DENSITY (dark smoke 0.5-0.65, page whites near 0). No local color rule can
finish the job — neutral dark smoke and dark body shading are inseparable
without object understanding — so the model must LEARN it. These pairs
teach exactly that: the image shows halftone dots / gray washes / glow
bursts on a page, and the GT alpha is the DENSITY FIELD, not the dot mask.

CONTRACTS (the make_* lineage — see scripts/make_typography.py):
- output `out_dir/im/{stem}.jpg` + `out_dir/gt/{stem}.png`, stems
  `atmo_{i:05d}`, manifest rows `{"id", "category": "atmosphere"}`;
- determinism `_item_rng(seed, stem)`; resume-safe; progress every 100.

Usage:
    uv run python scripts/make_atmosphere.py \
        --out-dir data/train_atmosphere --svg-dir /content/openclipart \
        --count 4000 --seed 88
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from make_clipart import (CANVAS_SHORT, ASPECTS, RENDER_TIMEOUT_S, _item_rng,
                          _save_pair, _svg_usable, _time_limit, render_svg_rgba)

DEFAULT_COUNT = 4000
DENSITY_GT_GAIN = 0.85     # gt in dot gaps = gain * local density
DOT_CELL_PX = (5, 11)      # halftone grid pitch
GLOW_PROB = 0.35


def _value_noise(rng: np.random.Generator, h: int, w: int, cell_px: int) -> np.ndarray:
    """Bilinear-upsampled uniform grid noise in [0,1] (make_typography kin)."""
    gh, gw = max(2, h // cell_px + 2), max(2, w // cell_px + 2)
    g = rng.uniform(0, 1, (gh, gw)).astype(np.float32)
    return np.asarray(Image.fromarray((g * 255).astype(np.uint8), mode="L")
                      .resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0


def _density_field(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    """1-3 smooth smoke blobs: radial falloffs modulated by value noise."""
    D = np.zeros((h, w), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for _ in range(int(rng.integers(1, 4))):
        cy, cx = rng.uniform(0.15, 0.85) * h, rng.uniform(0.15, 0.85) * w
        ry, rx = rng.uniform(0.15, 0.45) * h, rng.uniform(0.15, 0.45) * w
        r2 = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
        peak = rng.uniform(0.55, 1.0)
        D = np.maximum(D, peak * np.clip(1.0 - r2, 0.0, 1.0))
    noise = _value_noise(rng, h, w, int(rng.integers(24, 64)))
    return np.clip(D * (0.55 + 0.65 * noise), 0.0, 1.0)


def _halftone(rng: np.random.Generator, D: np.ndarray) -> np.ndarray:
    """Dot-mask alpha [0,1]: jittered grid, dot radius grows with density."""
    h, w = D.shape
    cell = int(rng.integers(*DOT_CELL_PX))
    mask = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(mask)
    max_r = cell * 0.62
    for gy in range(0, h, cell):
        for gx in range(0, w, cell):
            cy = gy + cell / 2 + rng.uniform(-1.2, 1.2)
            cx = gx + cell / 2 + rng.uniform(-1.2, 1.2)
            if not (0 <= int(cy) < h and 0 <= int(cx) < w):
                continue
            r = max_r * float(np.sqrt(D[int(cy), int(cx)]))
            if r < 0.7:
                continue
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
    return np.asarray(mask, dtype=np.float32) / 255.0


def render_atmosphere_sample(
    rng: np.random.Generator,
    svg_paths: list[Path],
) -> tuple[np.ndarray, np.ndarray]:
    """(RGB uint8, GT alpha float32): dark anchor art + smoke fields whose
    GT is the density, plus an optional warm glow burst."""
    short = int(rng.integers(CANVAS_SHORT[0], CANVAS_SHORT[1] + 1))
    aspect = ASPECTS[int(rng.integers(0, len(ASPECTS)))]
    w, h = (short, int(short / aspect)) if aspect <= 1.0 else (int(short * aspect), short)
    page = np.float32([250.0, 250.0, 250.0]) + rng.uniform(-4, 2, 3).astype(np.float32)
    comp = np.broadcast_to(page, (h, w, 3)).astype(np.float32).copy()
    gt = np.zeros((h, w), dtype=np.float32)

    # smoke UNDER the anchors (drawn first)
    D = _density_field(rng, h, w)
    mode = rng.uniform()
    ink_col = rng.uniform(15, 90, 3).astype(np.float32)
    ink_col[:] = ink_col.mean() + rng.uniform(-6, 6)          # near-neutral smoke
    if mode < 0.55:      # halftone dots
        dots = _halftone(rng, D)
        comp = dots[..., None] * ink_col + (1 - dots[..., None]) * comp
        gt = np.maximum(gt, np.maximum(dots, (DENSITY_GT_GAIN * D).astype(np.float32)) * (D > 0.02))
    elif mode < 0.85:    # smooth gray wash
        strength = rng.uniform(0.5, 0.9)
        wash = (D * strength)[..., None]
        comp = wash * ink_col + (1 - wash) * comp
        gt = np.maximum(gt, (D * strength).astype(np.float32))
    else:                # both: wash + sparse dots on top
        strength = rng.uniform(0.3, 0.6)
        wash = (D * strength)[..., None]
        comp = wash * ink_col + (1 - wash) * comp
        dots = _halftone(rng, np.clip(D - 0.3, 0, 1))
        comp = dots[..., None] * (ink_col * 0.5) + (1 - dots[..., None]) * comp
        gt = np.maximum(gt, np.maximum(dots, (D * (strength + 0.25)).astype(np.float32)))

    # optional glow burst: radial warm rays + hot core, gt = drawn alpha
    if rng.uniform() < GLOW_PROB:
        cy, cx = rng.uniform(0.35, 0.75) * h, rng.uniform(0.25, 0.75) * w
        rays = Image.new("L", (w, h), 0)
        rd = ImageDraw.Draw(rays)
        n_rays = int(rng.integers(40, 90))
        rmax = min(h, w) * rng.uniform(0.20, 0.40)
        for _ in range(n_rays):
            ang = rng.uniform(0, 2 * np.pi)
            ln = rmax * rng.uniform(0.35, 1.0)
            rd.line([cx, cy, cx + ln * np.cos(ang), cy + ln * np.sin(ang)],
                    fill=int(rng.integers(150, 256)), width=int(rng.integers(1, 4)))
        ray_a = np.asarray(rays, dtype=np.float32) / 255.0
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        rr = np.hypot(yy - cy, xx - cx) / (rmax * 0.45)
        core = np.clip(1.0 - rr, 0.0, 1.0) ** 1.5
        warm = np.float32([255, 225, 140]) + rng.uniform(-15, 15, 3).astype(np.float32)
        glow_a = np.clip(np.maximum(ray_a, core), 0.0, 1.0)
        comp = glow_a[..., None] * warm + (1 - glow_a[..., None]) * comp
        gt = np.maximum(gt, glow_a)

    # 1-2 dark anchor elements on top
    for _ in range(int(rng.integers(1, 3))):
        p = svg_paths[int(rng.integers(0, len(svg_paths)))]
        target = int(min(w, h) * rng.uniform(0.25, 0.55))
        try:
            with _time_limit(RENDER_TIMEOUT_S):
                rgb_el, a_el = render_svg_rgba(Path(p).read_text(errors="ignore"), out_width=target)
        except Exception:
            continue
        if a_el.shape[0] > 5000 or a_el.shape[0] > 8 * max(1, a_el.shape[1]):
            continue
        eh, ew = a_el.shape
        if ew >= w or eh >= h:
            f = min((w - 1) / ew, (h - 1) / eh)
            nw, nh = max(1, int(ew * f)), max(1, int(eh * f))
            rgb_el = np.asarray(Image.fromarray(rgb_el.astype(np.uint8)).resize((nw, nh), Image.LANCZOS), np.float32)
            a_el = np.asarray(Image.fromarray(np.round(a_el * 255).astype(np.uint8)).resize((nw, nh), Image.BILINEAR), np.float32) / 255.0
            eh, ew = nh, nw
        x0 = int(rng.integers(0, max(1, w - ew)))
        y0 = int(rng.integers(0, max(1, h - eh)))
        region = comp[y0:y0 + eh, x0:x0 + ew]
        comp[y0:y0 + eh, x0:x0 + ew] = a_el[..., None] * rgb_el + (1 - a_el[..., None]) * region
        gt[y0:y0 + eh, x0:x0 + ew] = np.maximum(gt[y0:y0 + eh, x0:x0 + ew], a_el)

    return comp.round().clip(0, 255).astype(np.uint8), np.clip(gt, 0.0, 1.0)


def run(out_dir: Path, svg_dir: Path, count: int = DEFAULT_COUNT, seed: int = 88,
        stem_prefix: str = "atmo_", category: str = "atmosphere") -> int:
    out_dir = Path(out_dir)
    (out_dir / "im").mkdir(parents=True, exist_ok=True)
    (out_dir / "gt").mkdir(parents=True, exist_ok=True)
    all_svgs = sorted(p for p in Path(svg_dir).rglob("*.svg"))
    svg_paths = [p for p in all_svgs if _svg_usable(p)]
    print(f"svg pool: {len(svg_paths)} usable / {len(all_svgs)} total")
    assert svg_paths, f"no usable SVGs in {svg_dir}"

    rows, generated, skipped = [], 0, 0
    for i in range(count):
        stem = f"{stem_prefix}{i:05d}"
        im_p = out_dir / "im" / f"{stem}.jpg"
        gt_p = out_dir / "gt" / f"{stem}.png"
        rows.append({"id": stem, "category": category})
        if im_p.exists() and gt_p.exists():
            skipped += 1
            continue
        rng = _item_rng(seed, stem)
        rgb, gt = render_atmosphere_sample(rng, svg_paths)
        _save_pair(rgb, gt, im_p, gt_p)
        generated += 1
        if generated % 100 == 0:
            print(f"atmosphere progress: {generated}/{count - skipped} generated")

    with open(out_dir / "manifest.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{generated} new pairs written, {skipped} already existed -> {out_dir}")
    return generated + skipped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--svg-dir", required=True)
    ap.add_argument("--count", type=int, default=DEFAULT_COUNT)
    ap.add_argument("--seed", type=int, default=88)
    ap.add_argument("--stem-prefix", default="atmo_")
    ap.add_argument("--category", default="atmosphere")
    a = ap.parse_args()
    run(Path(a.out_dir), Path(a.svg_dir), count=a.count, seed=a.seed,
        stem_prefix=a.stem_prefix, category=a.category)


if __name__ == "__main__":
    main()
