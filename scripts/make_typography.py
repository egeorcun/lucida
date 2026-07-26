"""Typography training pairs with EXACT counter ground truth (Lucida Design).

WHY (the CHEESE / white-on-white lessons, spec
docs/superpowers/specs/2026-07-24-lucida-design-expert.md): on real posters
the model keeps the whites between/inside letters or carves out interior
whites inconsistently — it never received a direct lesson on which white
belongs to the glyph and which to the background. Rendering display text
from real font files gives that lesson with mathematical certainty: the
glyph alpha IS the ground truth, and letter counters (the holes of O/B/e)
are exact zeros by font geometry.

Deliberate variety (each a named lesson):
- backgrounds: flat color / WHITE (white-on-white) / gradient / value-noise
  texture / photo from `bg_pool_dirs` — shares in BG_MENU;
- ink colors include white (white text on colored AND white grounds);
- distress ("print wear"): a value-noise mask fades the INK VISIBILITY in
  the RGB composite only — the GT stays glyph-exact (the alpha contract);
- 40% tight crops around the text block so glyphs are LARGE (the
  resolution lesson: at 1024 a full-poster counter is only a few pixels).

CONTRACTS (the make_* lineage — see scripts/make_bokeh_copies.py):
- output `out_dir/im/{stem}.jpg` + `out_dir/gt/{stem}.png`, stems
  `typo_{i:05d}`, manifest rows `{"id", "category": "typography"}`;
- determinism: `_item_rng(seed, stem)` split into layout/fx streams so the
  distress toggle cannot move the layout; resume: existing pairs skipped;
- progress print every 250 pairs.

Usage:
    uv run python scripts/make_typography.py \
        --out-dir data/train_typography --font-dir /content/fonts \
        --bg-pool /content/v16_train_src/im --count 6000 --seed 21
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None

DEFAULT_COUNT = 6000
CANVAS_SHORT = (640, 1024)
ASPECTS = (1.0, 0.75, 1.333)
GLYPH_FRAC = (0.25, 0.60)      # text-block height / canvas short side
BG_MENU = (("flat", 0.30), ("white", 0.20), ("gradient", 0.20),
           ("texture", 0.15), ("photo", 0.15))
WHITE_INK_PROB = 0.20
DISTRESS_PROB = 0.45
CROP_PROB = 0.40
# Outline extension (Lucida Design, the REAL Stay Fresh lesson 2026-07-25):
# sticker/poster type is usually a light fill + dark stroke sitting on a page
# whose color often EQUALS the fill (cream on cream). Without stroke samples
# the model treats fill==bg as background and hollows the letters — measured
# on the user's original: e16 fill alpha collapsed vs v7. Samples with index
# >= `outline_from` (run() param) draw from this menu; older stems consume
# NO extra rng draws and stay bit-identical.
OUTLINE_PROB = 0.75          # among outline-menu samples
FILL_EQ_BG_PROB = 0.35       # among outlined samples: fill color == flat bg color
STROKE_FRAC = (0.04, 0.14)   # stroke width / font size (220)
LINES = (1, 3)
WORDS = ["SALE", "CHEESE", "FRESH", "RETRO", "BOLD", "SUMMER", "PIZZA",
         "COFFEE", "MUSIC", "DANGER", "HAPPY", "ROCKET", "OCEAN", "BLOOM",
         "NIGHT", "POWER", "DREAM", "HELLO", "STUDIO", "VINTAGE"]


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


def _value_noise(rng: np.random.Generator, h: int, w: int, cell_px: int) -> np.ndarray:
    """Source: scripts/make_design.py::_value_noise (exact copy)."""
    gh = max(2, round(h / max(1, cell_px)))
    gw = max(2, round(w / max(1, cell_px)))
    grid = (rng.uniform(0.0, 1.0, (gh, gw)) * 255).astype(np.uint8)
    up = Image.fromarray(grid, mode="L").resize((w, h), Image.BILINEAR)
    return np.asarray(up, dtype=np.float32) / 255.0


def _make_background(rng: np.random.Generator, kind: str, w: int, h: int,
                     bg_images: list[Path]) -> np.ndarray:
    if kind == "photo" and bg_images:
        p = bg_images[int(rng.integers(0, len(bg_images)))]
        img = Image.open(p).convert("RGB").resize((w, h), Image.BILINEAR)
        return np.asarray(img, dtype=np.float32)
    if kind == "white":
        return np.full((h, w, 3), 250.0, dtype=np.float32)
    if kind == "gradient":
        c1 = rng.uniform(40, 240, 3)
        c2 = rng.uniform(40, 240, 3)
        t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
        if rng.uniform() < 0.5:
            t = np.linspace(0, 1, w, dtype=np.float32)[None, :, None]
        return (c1 * (1 - t) + c2 * t) * np.ones((h, w, 3), dtype=np.float32)
    if kind == "texture":
        base = rng.uniform(120, 245, 3)
        noise = _value_noise(rng, h, w, int(rng.integers(8, 40)))
        return np.clip(base + (noise[..., None] - 0.5) * rng.uniform(20, 70), 0, 255)
    # flat
    return np.broadcast_to(rng.uniform(30, 240, 3).astype(np.float32), (h, w, 3)).copy()


def _render_line(text: str, font: ImageFont.FreeTypeFont,
                 stroke_px: int = 0) -> tuple[Image.Image, Image.Image]:
    """One line as (fill RGBA, fill+stroke RGBA) — same canvas, same origin,
    so `union - fill` is exactly the stroke band."""
    pad = 12 + stroke_px
    bbox = font.getbbox(text, stroke_width=stroke_px)
    lw, lh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    fill_img = Image.new("RGBA", (lw + 2 * pad, lh + 2 * pad), (0, 0, 0, 0))
    ImageDraw.Draw(fill_img).text((pad - bbox[0], pad - bbox[1]), text,
                                  font=font, fill=(255, 255, 255, 255))
    if stroke_px == 0:
        return fill_img, fill_img
    union_img = Image.new("RGBA", fill_img.size, (0, 0, 0, 0))
    ImageDraw.Draw(union_img).text((pad - bbox[0], pad - bbox[1]), text,
                                   font=font, fill=(255, 255, 255, 255),
                                   stroke_width=stroke_px, stroke_fill=(255, 255, 255, 255))
    return fill_img, union_img


def _render_text_block(rng: np.random.Generator, font_paths: list[Path],
                       stroke_px: int = 0) -> tuple[Image.Image, Image.Image]:
    """1-3 stacked display lines; returns (fill block, fill+stroke block)
    in white ink (recolored later)."""
    n_lines = int(rng.integers(LINES[0], LINES[1] + 1))
    font_path = str(font_paths[int(rng.integers(0, len(font_paths)))])
    lines = []
    for _ in range(n_lines):
        text = WORDS[int(rng.integers(0, len(WORDS)))]
        font = ImageFont.truetype(font_path, 220)
        lines.append(_render_line(text, font, stroke_px))
    gap = 20
    W = max(u.width for _, u in lines)
    H = sum(u.height for _, u in lines) + gap * (len(lines) - 1)
    fill_block = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    union_block = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    y = 0
    for f, u in lines:
        x = (W - u.width) // 2
        fill_block.paste(f, (x, y), f)
        union_block.paste(u, (x, y), u)
        y += u.height + gap
    return fill_block, union_block


def render_typography_sample(
    layout_rng: np.random.Generator,
    fx_rng: np.random.Generator,
    font_paths: list[Path],
    bg_images: list[Path],
    force_text: str | None = None,
    force_bg: str | None = None,
    force_ink: tuple[int, int, int] | None = None,
    force_distress: bool | None = None,
    force_crop: bool | None = None,
    outline_menu: bool = False,
    force_outline: bool | None = None,
    force_fill_eq_bg: bool | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """(RGB uint8, GT alpha float32). layout_rng fixes geometry/colors; fx_rng
    only drives the distress mask — toggling distress cannot move the GT.
    `outline_menu=False` consumes exactly the pre-outline rng draws, so the
    original 6000 stems stay bit-identical."""
    short = int(layout_rng.integers(CANVAS_SHORT[0], CANVAS_SHORT[1] + 1))
    aspect = ASPECTS[int(layout_rng.integers(0, len(ASPECTS)))]
    w, h = (short, int(short / aspect)) if aspect <= 1.0 else (int(short * aspect), short)

    outlined = fill_eq_bg = False
    stroke_px = 0
    if outline_menu:
        outlined = (layout_rng.uniform() < OUTLINE_PROB) if force_outline is None else force_outline
        if outlined:
            stroke_px = max(2, int(220 * layout_rng.uniform(*STROKE_FRAC)))
            fill_eq_bg = ((layout_rng.uniform() < FILL_EQ_BG_PROB)
                          if force_fill_eq_bg is None else force_fill_eq_bg)

    u, kind = layout_rng.uniform(), "flat"
    acc = 0.0
    for k, p in BG_MENU:
        acc += p
        if u < acc:
            kind = k
            break
    if force_bg is not None:
        kind = force_bg
    if fill_eq_bg:
        # the page and the letter fill share ONE light color (cream-on-cream);
        # only the stroke separates the letters from the page.
        page = layout_rng.uniform(185, 252, 3).astype(np.float32)
        bg = np.broadcast_to(page, (h, w, 3)).copy()
    else:
        bg = _make_background(layout_rng, kind, w, h, bg_images)

    if force_text is not None:
        font = ImageFont.truetype(str(font_paths[0]), 220)
        fill_block, block = _render_line(force_text, font, stroke_px)
    else:
        fill_block, block = _render_text_block(layout_rng, font_paths, stroke_px)

    # two-step resize kept EXACTLY as the original (one-step LANCZOS would
    # change pixels and break bit-identity of the first 6000 stems)
    target_h = int(min(h, w) * layout_rng.uniform(*GLYPH_FRAC))
    scale = target_h / block.height
    size1 = (max(1, int(block.width * scale)), max(1, target_h))
    block = block.resize(size1, Image.LANCZOS)
    fill_block = fill_block.resize(size1, Image.LANCZOS)
    if block.width > int(w * 0.94):
        f = w * 0.94 / block.width
        size2 = (int(block.width * f), max(1, int(block.height * f)))
        block = block.resize(size2, Image.LANCZOS)
        fill_block = fill_block.resize(size2, Image.LANCZOS)

    margin = max(4, int(0.03 * min(w, h)))
    x0 = int(layout_rng.integers(margin, max(margin + 1, w - margin - block.width)))
    y0 = int(layout_rng.integers(margin, max(margin + 1, h - margin - block.height)))

    glyph = np.zeros((h, w), dtype=np.float32)      # fill + stroke (the GT)
    fill_glyph = np.zeros((h, w), dtype=np.float32)  # fill only
    glyph[y0:y0 + block.height, x0:x0 + block.width] = \
        np.asarray(block, dtype=np.float32)[..., 3] / 255.0
    fill_glyph[y0:y0 + block.height, x0:x0 + block.width] = \
        np.asarray(fill_block, dtype=np.float32)[..., 3] / 255.0

    if force_ink is not None:
        ink = np.asarray(force_ink, dtype=np.float32)
    elif fill_eq_bg:
        ink = page.copy()                            # indistinguishable from the page
    elif layout_rng.uniform() < WHITE_INK_PROB:
        ink = np.asarray((252, 250, 248), dtype=np.float32)
    else:
        ink = layout_rng.uniform(10, 235, 3).astype(np.float32)

    if outlined:
        # light fills get a dark stroke, dark fills a light one (sticker style)
        stroke_col = (layout_rng.uniform(5, 70, 3) if ink.mean() > 128
                      else layout_rng.uniform(200, 255, 3)).astype(np.float32)

    distress = DISTRESS_PROB > layout_rng.uniform() if force_distress is None else force_distress
    # ink visibility: distress fades the print in the RGB ONLY (GT untouched)
    wear = None
    if distress:
        wear = 0.45 + 0.55 * _value_noise(fx_rng, h, w, int(fx_rng.integers(3, 12)))

    if outlined:
        stroke_a = np.clip(glyph - fill_glyph, 0.0, 1.0)
        vis_s = stroke_a * wear if wear is not None else stroke_a
        vis_f = fill_glyph * wear if wear is not None else fill_glyph
        rgb = vis_s[..., None] * stroke_col + (1.0 - vis_s[..., None]) * bg
        rgb = vis_f[..., None] * ink + (1.0 - vis_f[..., None]) * rgb
    else:
        vis = glyph * wear if wear is not None else glyph
        rgb = vis[..., None] * ink + (1.0 - vis[..., None]) * bg
    gt = glyph

    do_crop = CROP_PROB > layout_rng.uniform() if force_crop is None else force_crop
    if do_crop:
        pad = int(layout_rng.uniform(0.05, 0.20) * min(w, h))
        cx0 = max(0, x0 - pad)
        cy0 = max(0, y0 - pad)
        cx1 = min(w, x0 + block.width + pad)
        cy1 = min(h, y0 + block.height + pad)
        rgb = rgb[cy0:cy1, cx0:cx1]
        gt = gt[cy0:cy1, cx0:cx1]

    return rgb.round().clip(0, 255).astype(np.uint8), gt.astype(np.float32)


def run(out_dir: Path, font_dir: Path, bg_pool_dirs: list[Path],
        count: int = DEFAULT_COUNT, seed: int = 21,
        outline_from: int | None = None,
        stem_prefix: str = "typo_", category: str = "typography") -> int:
    out_dir = Path(out_dir)
    (out_dir / "im").mkdir(parents=True, exist_ok=True)
    (out_dir / "gt").mkdir(parents=True, exist_ok=True)
    font_paths = sorted(p for p in Path(font_dir).iterdir()
                        if p.suffix.lower() in {".ttf", ".otf"})
    assert font_paths, f"no fonts in {font_dir}"
    bg_images: list[Path] = []
    for d in bg_pool_dirs:
        d = Path(d)
        if d.is_dir():
            bg_images += sorted(d.iterdir())[:20000]

    rows, generated, skipped = [], 0, 0
    for i in range(count):
        stem = f"{stem_prefix}{i:05d}"
        im_p = out_dir / "im" / f"{stem}.jpg"
        gt_p = out_dir / "gt" / f"{stem}.png"
        rows.append({"id": stem, "category": category})
        if im_p.exists() and gt_p.exists():
            skipped += 1
            continue
        base = _item_rng(seed, stem)
        layout_rng = np.random.default_rng(base.integers(0, 2**63))
        fx_rng = np.random.default_rng(base.integers(0, 2**63))
        rgb, gt = render_typography_sample(
            layout_rng, fx_rng, font_paths, bg_images,
            outline_menu=(outline_from is not None and i >= outline_from))
        _save_pair(rgb, gt, im_p, gt_p)
        generated += 1
        if generated % 250 == 0:
            print(f"typography progress: {generated}/{count - skipped} generated")

    with open(out_dir / "manifest.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{generated} new pairs written, {skipped} already existed -> {out_dir}")
    return generated + skipped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--font-dir", required=True)
    ap.add_argument("--bg-pool", action="append", default=[])
    ap.add_argument("--count", type=int, default=DEFAULT_COUNT)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--outline-from", type=int, default=None,
                    help="stems with index >= this draw from the outline style menu")
    a = ap.parse_args()
    run(Path(a.out_dir), Path(a.font_dir), [Path(p) for p in a.bg_pool],
        count=a.count, seed=a.seed, outline_from=a.outline_from)


if __name__ == "__main__":
    main()
