"""OpenClipart collage training pairs with pixel-exact SVG alpha (Lucida Design).

WHY (the Stay Fresh lesson, spec
docs/superpowers/specs/2026-07-24-lucida-design-expert.md): on sticker-style
vector art the model carves out interior whites (daisy petals, letter
fills) because nothing ever taught it that an element's white belongs to
the element. Rendering CC0 SVGs (nyuuzyou/openclipart, 178k files) gives
pixel-exact alpha for free, and composing them over WHITE backgrounds a
deliberate `white_bg_share` of the time over-represents exactly the
white-on-white decision the model keeps failing.

CONTRACTS (the make_* lineage — see scripts/make_typography.py):
- output `out_dir/im/{stem}.jpg` + `out_dir/gt/{stem}.png`, stems
  `clip_{i:05d}`, manifest rows `{"id", "category": "clipart"}`;
- determinism `_item_rng(seed, stem)`; resume-safe; progress every 250.
- Broken/unrenderable SVGs are skipped silently per element pick (the rng
  draw is consumed either way, so determinism holds).

Usage:
    DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib \
    uv run python scripts/make_clipart.py \
        --out-dir data/train_clipart --svg-dir /content/openclipart \
        --count 5000 --seed 33
"""
import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

import cairosvg

Image.MAX_IMAGE_PIXELS = None

DEFAULT_COUNT = 5000
CANVAS_SHORT = (640, 1024)
ASPECTS = (1.0, 0.75, 1.333)
ELEMENTS = (3, 6)
ELEMENT_FRAC = (0.18, 0.55)      # element long side / canvas short side
WHITE_BG_SHARE = 0.35
MARGIN_FRAC = 0.03


def _item_rng(seed: int, key: str) -> np.random.Generator:
    """Source: scripts/make_composites.py::_item_rng (exact copy)."""
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    entropy = [seed & 0xFFFFFFFF] + [
        int.from_bytes(digest[i:i + 4], "big") for i in range(0, 16, 4)]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def _save_pair(rgb: np.ndarray, alpha: np.ndarray, img_path: Path, gt_path: Path) -> None:
    img_path.parent.mkdir(parents=True, exist_ok=True)
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(img_path, format="JPEG", quality=92)
    Image.fromarray(np.round(alpha.clip(0, 1) * 255).astype(np.uint8), mode="L").save(gt_path)


def render_svg_rgba(svg_text: str, out_width: int) -> tuple[np.ndarray, np.ndarray]:
    """SVG -> (rgb float [0,255], alpha float [0,1]) at pixel-exact fidelity."""
    png = cairosvg.svg2png(bytestring=svg_text.encode("utf-8"), output_width=out_width)
    img = Image.open(io.BytesIO(png)).convert("RGBA")
    arr = np.asarray(img, dtype=np.float32)
    return arr[..., :3], arr[..., 3] / 255.0


def _background(rng: np.random.Generator, kind: str, w: int, h: int) -> np.ndarray:
    if kind == "white":
        return np.full((h, w, 3), 250.0, dtype=np.float32)
    if kind == "gradient":
        c1, c2 = rng.uniform(120, 250, 3), rng.uniform(120, 250, 3)
        t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
        return (c1 * (1 - t) + c2 * t) * np.ones((h, w, 3), dtype=np.float32)
    # flat pastel
    return np.broadcast_to(rng.uniform(150, 245, 3).astype(np.float32), (h, w, 3)).copy()


def render_clipart_sample(
    rng: np.random.Generator,
    svg_paths: list[Path],
    force_bg: str | None = None,
    n_elements: int | None = None,
    white_bg_share: float = WHITE_BG_SHARE,
) -> tuple[np.ndarray, np.ndarray]:
    """(RGB uint8, GT alpha float32) — 3-6 SVG elements over a background."""
    short = int(rng.integers(CANVAS_SHORT[0], CANVAS_SHORT[1] + 1))
    aspect = ASPECTS[int(rng.integers(0, len(ASPECTS)))]
    w, h = (short, int(short / aspect)) if aspect <= 1.0 else (int(short * aspect), short)

    if force_bg is not None:
        kind = force_bg
    else:
        u = rng.uniform()
        kind = "white" if u < white_bg_share else ("gradient" if u < white_bg_share + 0.25 else "flat")
    comp = _background(rng, kind, w, h)
    gt = np.zeros((h, w), dtype=np.float32)
    m = max(4, int(MARGIN_FRAC * min(w, h)))

    n = n_elements if n_elements is not None else int(rng.integers(ELEMENTS[0], ELEMENTS[1] + 1))
    for _ in range(n):
        p = svg_paths[int(rng.integers(0, len(svg_paths)))]
        target = int(min(w, h) * rng.uniform(*ELEMENT_FRAC))
        angle = float(rng.uniform(-25, 25))
        try:
            rgb_el, a_el = render_svg_rgba(Path(p).read_text(errors="ignore"), out_width=target)
        except Exception:
            continue  # broken SVG: rng draws already consumed -> deterministic
        el = Image.fromarray(
            np.dstack([rgb_el, a_el[..., None] * 255]).astype(np.uint8), mode="RGBA")
        el = el.rotate(angle, expand=True, resample=Image.BICUBIC, fillcolor=(0, 0, 0, 0))
        if el.width >= w - 2 * m or el.height >= h - 2 * m:
            f = min((w - 2 * m - 1) / el.width, (h - 2 * m - 1) / el.height)
            el = el.resize((max(1, int(el.width * f)), max(1, int(el.height * f))), Image.LANCZOS)
        x0 = int(rng.integers(m, max(m + 1, w - m - el.width)))
        y0 = int(rng.integers(m, max(m + 1, h - m - el.height)))
        arr = np.asarray(el, dtype=np.float32)
        a = arr[..., 3] / 255.0
        region = comp[y0:y0 + el.height, x0:x0 + el.width]
        comp[y0:y0 + el.height, x0:x0 + el.width] = a[..., None] * arr[..., :3] + (1 - a[..., None]) * region
        g = gt[y0:y0 + el.height, x0:x0 + el.width]
        gt[y0:y0 + el.height, x0:x0 + el.width] = 1.0 - (1.0 - g) * (1.0 - a)

    return comp.round().clip(0, 255).astype(np.uint8), gt


def run(out_dir: Path, svg_dir: Path, count: int = DEFAULT_COUNT, seed: int = 33,
        white_bg_share: float = WHITE_BG_SHARE) -> int:
    out_dir = Path(out_dir)
    (out_dir / "im").mkdir(parents=True, exist_ok=True)
    (out_dir / "gt").mkdir(parents=True, exist_ok=True)
    svg_paths = sorted(p for p in Path(svg_dir).rglob("*.svg"))
    assert svg_paths, f"no SVGs in {svg_dir}"

    rows, generated, skipped = [], 0, 0
    for i in range(count):
        stem = f"clip_{i:05d}"
        im_p = out_dir / "im" / f"{stem}.jpg"
        gt_p = out_dir / "gt" / f"{stem}.png"
        rows.append({"id": stem, "category": "clipart"})
        if im_p.exists() and gt_p.exists():
            skipped += 1
            continue
        rng = _item_rng(seed, stem)
        rgb, gt = render_clipart_sample(rng, svg_paths, white_bg_share=white_bg_share)
        _save_pair(rgb, gt, im_p, gt_p)
        generated += 1
        if generated % 250 == 0:
            print(f"clipart progress: {generated}/{count - skipped} generated")

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
    ap.add_argument("--seed", type=int, default=33)
    ap.add_argument("--white-bg-share", type=float, default=WHITE_BG_SHARE)
    a = ap.parse_args()
    run(Path(a.out_dir), Path(a.svg_dir), count=a.count, seed=a.seed,
        white_bg_share=a.white_bg_share)


if __name__ == "__main__":
    main()
