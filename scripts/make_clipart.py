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
import signal
from contextlib import contextmanager
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
MAX_SVG_BYTES = 300_000
RENDER_TIMEOUT_S = 10
# External refs make cairosvg do NETWORK FETCHES (no timeout — the Colab
# hang of 2026-07-24: 49 min without a single progress line); <image> tags
# embed rasters we don't want in "vector" GT anyway.
_BAD_MARKERS = (b"<image", b'href="http', b"href='http")


def _svg_usable(p: Path) -> bool:
    """Cheap, deterministic pre-filter: size cap + no external/raster refs."""
    try:
        if p.stat().st_size > MAX_SVG_BYTES:
            return False
        data = p.read_bytes().lower()
    except OSError:
        return False
    return not any(m in data for m in _BAD_MARKERS)


@contextmanager
def _time_limit(seconds: float):
    """SIGALRM-based hard timeout (main thread only; no-op elsewhere) —
    the backstop for pathological SVGs the pre-filter can't predict."""
    def _raise(signum, frame):
        raise TimeoutError("svg render timed out")
    try:
        old = signal.signal(signal.SIGALRM, _raise)
        signal.setitimer(signal.ITIMER_REAL, seconds)
    except ValueError:  # not in main thread
        yield
        return
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


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
            with _time_limit(RENDER_TIMEOUT_S):
                rgb_el, a_el = render_svg_rgba(Path(p).read_text(errors="ignore"), out_width=target)
        except Exception:
            continue  # broken/slow SVG: rng draws already consumed -> deterministic
        if a_el.shape[0] > 5000 or a_el.shape[0] > 8 * max(1, a_el.shape[1]):
            continue  # extreme aspect ratio rastered into a monster
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


OUTLINE_PX_FRAC = (0.010, 0.035)   # stroke band / element long side
FILL_EQ_BG_SHARE_CLIP = 0.5        # among outline samples: page == element color


def render_clipart_outline_sample(
    rng: np.random.Generator,
    svg_paths: list[Path],
    force_fill_eq_bg: bool | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sticker-style collage (spec 2026-07-26): every element gets an
    enclosing stroke (dilated alpha band), and half the time the page is
    painted with the FIRST element's dominant color — the element is then
    pixel-identical to the page and only its stroke separates them. GT is
    the union of element+stroke alphas."""
    from PIL import ImageFilter

    short = int(rng.integers(CANVAS_SHORT[0], CANVAS_SHORT[1] + 1))
    aspect = ASPECTS[int(rng.integers(0, len(ASPECTS)))]
    w, h = (short, int(short / aspect)) if aspect <= 1.0 else (int(short * aspect), short)
    fill_eq_bg = (rng.uniform() < FILL_EQ_BG_SHARE_CLIP
                  if force_fill_eq_bg is None else force_fill_eq_bg)

    n = int(rng.integers(2, 5))
    m = max(4, int(MARGIN_FRAC * min(w, h)))
    elements = []
    for _ in range(n):
        p = svg_paths[int(rng.integers(0, len(svg_paths)))]
        target = int(min(w, h) * rng.uniform(0.20, 0.50))
        try:
            with _time_limit(RENDER_TIMEOUT_S):
                rgb_el, a_el = render_svg_rgba(Path(p).read_text(errors="ignore"), out_width=target)
        except Exception:
            continue
        if a_el.shape[0] > 5000 or a_el.shape[0] > 8 * max(1, a_el.shape[1]):
            continue
        # stroke = dilated alpha band (PIL MaxFilter; odd kernel)
        band = max(3, int(max(a_el.shape) * rng.uniform(*OUTLINE_PX_FRAC)))
        k = band * 2 + 1
        a_img = Image.fromarray(np.round(a_el * 255).astype(np.uint8), mode="L")
        pad = band + 2
        a_pad = Image.new("L", (a_img.width + 2 * pad, a_img.height + 2 * pad), 0)
        a_pad.paste(a_img, (pad, pad))
        dil = np.asarray(a_pad.filter(ImageFilter.MaxFilter(k)), np.float32) / 255.0
        base = np.zeros_like(dil)
        base[pad:pad + a_el.shape[0], pad:pad + a_el.shape[1]] = a_el
        rgb_pad = np.zeros((*dil.shape, 3), dtype=np.float32)
        rgb_pad[pad:pad + a_el.shape[0], pad:pad + a_el.shape[1]] = rgb_el
        opaque = base > 0.9
        dominant = (np.median(rgb_pad[opaque], axis=0).astype(np.float32)
                    if opaque.sum() >= 200 else np.float32([245, 245, 245]))
        stroke_col = (rng.uniform(5, 60, 3) if dominant.mean() > 128
                      else rng.uniform(215, 255, 3)).astype(np.float32)
        elements.append((rgb_pad, base, dil, stroke_col, dominant))

    if fill_eq_bg and elements:
        page = elements[0][4]
    else:
        page = rng.uniform(150, 250, 3).astype(np.float32)
    comp = np.broadcast_to(page, (h, w, 3)).astype(np.float32).copy()
    gt = np.zeros((h, w), dtype=np.float32)

    for rgb_el, base, dil, stroke_col, _ in elements:
        eh, ew = dil.shape
        if ew >= w - 2 * m or eh >= h - 2 * m:
            f = min((w - 2 * m - 1) / ew, (h - 2 * m - 1) / eh)
            nw, nh = max(1, int(ew * f)), max(1, int(eh * f))
            rgb_el = np.asarray(Image.fromarray(rgb_el.astype(np.uint8)).resize((nw, nh), Image.LANCZOS), np.float32)
            base = np.asarray(Image.fromarray(np.round(base * 255).astype(np.uint8)).resize((nw, nh), Image.BILINEAR), np.float32) / 255.0
            dil = np.asarray(Image.fromarray(np.round(dil * 255).astype(np.uint8)).resize((nw, nh), Image.BILINEAR), np.float32) / 255.0
            eh, ew = nh, nw
        x0 = int(rng.integers(m, max(m + 1, w - m - ew)))
        y0 = int(rng.integers(m, max(m + 1, h - m - eh)))
        region = comp[y0:y0 + eh, x0:x0 + ew]
        stroke_a = np.clip(dil - base, 0.0, 1.0)
        region = stroke_a[..., None] * stroke_col + (1 - stroke_a[..., None]) * region
        comp[y0:y0 + eh, x0:x0 + ew] = base[..., None] * rgb_el + (1 - base[..., None]) * region
        g = gt[y0:y0 + eh, x0:x0 + ew]
        gt[y0:y0 + eh, x0:x0 + ew] = 1.0 - (1.0 - g) * (1.0 - dil)

    return comp.round().clip(0, 255).astype(np.uint8), gt


LIMB_WHITE_PAGE_SHARE = 0.85       # the ambiguity is white-on-white
LIMB_COUNT = (2, 4)                # limbs per body
LIMB_FRAC = (0.10, 0.22)           # limb long side / canvas short side
LIMB_OUTSIDE = (0.30, 0.70)        # fraction of the limb outside the body


def _blob_mask(rng: np.random.Generator, size: int) -> np.ndarray:
    """Rounded organic blob alpha in [0,1]: ellipse / two-lobe mitten /
    wobbled polygon — the cartoon glove/foot/ear shape family."""
    from PIL import ImageDraw, ImageFilter

    img = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(img)
    kind = rng.uniform()
    if kind < 0.4:      # ellipse
        rx, ry = rng.uniform(0.30, 0.48), rng.uniform(0.22, 0.48)
        d.ellipse([size * (0.5 - rx), size * (0.5 - ry),
                   size * (0.5 + rx), size * (0.5 + ry)], fill=255)
    elif kind < 0.7:    # two-lobe mitten: big palm + thumb lobe
        d.ellipse([size * 0.10, size * 0.22, size * 0.78, size * 0.88], fill=255)
        d.ellipse([size * 0.52, size * 0.08, size * 0.92, size * 0.48], fill=255)
    else:               # wobbled polygon
        n_pt = int(rng.integers(7, 12))
        ang = np.linspace(0, 2 * np.pi, n_pt, endpoint=False)
        rad = size * rng.uniform(0.30, 0.46, n_pt)
        pts = [(size / 2 + r * np.cos(a), size / 2 + r * np.sin(a))
               for a, r in zip(ang, rad)]
        d.polygon(pts, fill=255)
        img = img.filter(ImageFilter.MaxFilter(9))
    return np.asarray(img, dtype=np.float32) / 255.0


def render_limb_sample(
    rng: np.random.Generator,
    svg_paths: list[Path],
) -> tuple[np.ndarray, np.ndarray]:
    """The YOU'RE HAPPY gloves lesson (spec 2026-07-29): one big BODY plus
    2-4 page-colored, dark-stroked procedural limbs ATTACHED to the body's
    silhouette edge, over a (mostly white) page the limb fill matches.
    Knuckle strokes attach to the contour. GT = body + limb + strokes."""
    from PIL import ImageDraw, ImageFilter

    short = int(rng.integers(CANVAS_SHORT[0], CANVAS_SHORT[1] + 1))
    aspect = ASPECTS[int(rng.integers(0, len(ASPECTS)))]
    w, h = (short, int(short / aspect)) if aspect <= 1.0 else (int(short * aspect), short)
    if rng.uniform() < LIMB_WHITE_PAGE_SHARE:
        page = np.float32([250.0, 250.0, 250.0]) + rng.uniform(-4, 4, 3).astype(np.float32)
    else:
        page = rng.uniform(228, 250, 3).astype(np.float32)
    comp = np.broadcast_to(page, (h, w, 3)).astype(np.float32).copy()
    gt = np.zeros((h, w), dtype=np.float32)

    # BODY: one big SVG, center-biased
    body_a = None
    for _ in range(6):
        p = svg_paths[int(rng.integers(0, len(svg_paths)))]
        target = int(min(w, h) * rng.uniform(0.45, 0.70))
        try:
            with _time_limit(RENDER_TIMEOUT_S):
                rgb_el, a_el = render_svg_rgba(Path(p).read_text(errors="ignore"), out_width=target)
        except Exception:
            continue
        if a_el.shape[0] > 5000 or a_el.shape[0] > 8 * max(1, a_el.shape[1]):
            continue
        if a_el.max() < 0.5 or (a_el > 0.5).sum() < 0.05 * a_el.size:
            continue
        eh, ew = a_el.shape
        if ew >= int(w * 0.9) or eh >= int(h * 0.9):
            f = min(w * 0.9 / ew, h * 0.9 / eh)
            nw, nh = max(1, int(ew * f)), max(1, int(eh * f))
            rgb_el = np.asarray(Image.fromarray(rgb_el.astype(np.uint8)).resize((nw, nh), Image.LANCZOS), np.float32)
            a_el = np.asarray(Image.fromarray(np.round(a_el * 255).astype(np.uint8)).resize((nw, nh), Image.BILINEAR), np.float32) / 255.0
            eh, ew = nh, nw
        x0 = int((w - ew) * rng.uniform(0.30, 0.70))
        y0 = int((h - eh) * rng.uniform(0.30, 0.70))
        region = comp[y0:y0 + eh, x0:x0 + ew]
        comp[y0:y0 + eh, x0:x0 + ew] = a_el[..., None] * rgb_el + (1 - a_el[..., None]) * region
        gt[y0:y0 + eh, x0:x0 + ew] = np.maximum(gt[y0:y0 + eh, x0:x0 + ew], a_el)
        body_a = (x0, y0, ew, eh, a_el)
        break
    if body_a is None:
        return comp.round().clip(0, 255).astype(np.uint8), gt

    bx, by, bw, bh, ba = body_a
    # boundary pixels of the body mask
    bm = ba > 0.5
    inner = np.zeros_like(bm)
    inner[1:-1, 1:-1] = bm[1:-1, 1:-1] & bm[:-2, 1:-1] & bm[2:, 1:-1] & bm[1:-1, :-2] & bm[1:-1, 2:]
    bys, bxs = np.nonzero(bm & ~inner)
    if len(bys) == 0:
        return comp.round().clip(0, 255).astype(np.uint8), gt

    n_limbs = int(rng.integers(LIMB_COUNT[0], LIMB_COUNT[1] + 1))
    for _ in range(n_limbs):
        size = max(24, int(min(w, h) * rng.uniform(*LIMB_FRAC)))
        blob = _blob_mask(rng, size)
        # attach: pick a boundary point, shift the blob centre outward
        k = int(rng.integers(0, len(bys)))
        cy, cx = by + int(bys[k]), bx + int(bxs[k])
        out_frac = rng.uniform(*LIMB_OUTSIDE)
        # outward direction: away from body bbox centre
        vy, vx = cy - (by + bh / 2), cx - (bx + bw / 2)
        nv = max(1.0, float(np.hypot(vy, vx)))
        cy = int(cy + vy / nv * size * (out_frac - 0.5))
        cx = int(cx + vx / nv * size * (out_frac - 0.5))
        y0, x0 = cy - size // 2, cx - size // 2
        if y0 < 0 or x0 < 0 or y0 + size > h or x0 + size > w:
            continue
        stroke_px = max(2, int(size * rng.uniform(0.03, 0.07)))
        blob_img = Image.fromarray(np.round(blob * 255).astype(np.uint8), mode="L")
        dil = np.asarray(blob_img.filter(ImageFilter.MaxFilter(stroke_px * 2 + 1)),
                         np.float32) / 255.0
        stroke_a = np.clip(dil - blob, 0.0, 1.0)
        stroke_col = rng.uniform(5, 55, 3).astype(np.float32)
        fill_col = page + rng.uniform(-5, 3, 3).astype(np.float32)
        region = comp[y0:y0 + size, x0:x0 + size]
        region = blob[..., None] * fill_col + (1 - blob[..., None]) * region
        region = stroke_a[..., None] * stroke_col + (1 - stroke_a[..., None]) * region
        # knuckle strokes: short arcs STARTING on the contour, reaching inward
        n_kn = int(rng.integers(1, 4))
        kn = Image.fromarray(np.zeros((size, size), dtype=np.uint8), mode="L")
        kd = ImageDraw.Draw(kn)
        bys2, bxs2 = np.nonzero((blob > 0.5) & (np.asarray(
            blob_img.filter(ImageFilter.MinFilter(5)), np.float32) / 255.0 < 0.5))
        for _k in range(n_kn):
            if len(bys2) == 0:
                break
            j = int(rng.integers(0, len(bys2)))
            ky, kx = int(bys2[j]), int(bxs2[j])
            ang = rng.uniform(0, 2 * np.pi)
            ln = size * rng.uniform(0.15, 0.35)
            kd.line([kx, ky, int(kx + ln * np.cos(ang)), int(ky + ln * np.sin(ang))],
                    fill=255, width=max(1, stroke_px - 1))
        kn_a = (np.asarray(kn, np.float32) / 255.0) * blob
        region = kn_a[..., None] * stroke_col + (1 - kn_a[..., None]) * region
        comp[y0:y0 + size, x0:x0 + size] = region
        gt[y0:y0 + size, x0:x0 + size] = np.maximum(gt[y0:y0 + size, x0:x0 + size], dil)

    return comp.round().clip(0, 255).astype(np.uint8), gt


def run(out_dir: Path, svg_dir: Path, count: int = DEFAULT_COUNT, seed: int = 33,
        white_bg_share: float = WHITE_BG_SHARE, outline: bool = False,
        limb: bool = False,
        stem_prefix: str = "clip_", category: str = "clipart") -> int:
    out_dir = Path(out_dir)
    (out_dir / "im").mkdir(parents=True, exist_ok=True)
    (out_dir / "gt").mkdir(parents=True, exist_ok=True)
    all_svgs = sorted(p for p in Path(svg_dir).rglob("*.svg"))
    svg_paths = [p for p in all_svgs if _svg_usable(p)]
    print(f"svg pool: {len(svg_paths)} usable / {len(all_svgs)} total "
          f"(dropped: >|{MAX_SVG_BYTES // 1000}KB, external refs, <image>)")
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
        if limb:
            rgb, gt = render_limb_sample(rng, svg_paths)
        elif outline:
            rgb, gt = render_clipart_outline_sample(rng, svg_paths)
        else:
            rgb, gt = render_clipart_sample(rng, svg_paths, white_bg_share=white_bg_share)
        _save_pair(rgb, gt, im_p, gt_p)
        generated += 1
        if generated % 100 == 0:
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
    ap.add_argument("--limb", action="store_true",
                    help="edge-limb lesson pairs (spec 2026-07-29)")
    ap.add_argument("--stem-prefix", default="clip_")
    ap.add_argument("--category", default="clipart")
    a = ap.parse_args()
    run(Path(a.out_dir), Path(a.svg_dir), count=a.count, seed=a.seed,
        white_bg_share=a.white_bg_share, limb=a.limb,
        stem_prefix=a.stem_prefix, category=a.category)


if __name__ == "__main__":
    main()
