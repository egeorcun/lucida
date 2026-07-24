"""scripts/make_typography.py — the design-expert typography generator.

What matters (the CHEESE / white-on-white lessons):
- letter counters (the holes of O/B/e) are EXACT zeros in the GT, straight
  from font geometry — the model finally gets a direct, mathematically
  certain "between/inside letters = background" lesson,
- white text on a white background still carries full glyphs in the GT,
- the distress/print-wear effect touches only the visible RGB — the GT of a
  distressed stem is bit-identical to its clean twin,
- the make_* contracts hold: deterministic per stem, resume-safe, manifest.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy import ndimage

_SPEC = importlib.util.spec_from_file_location(
    "make_typography", Path(__file__).parent.parent / "scripts" / "make_typography.py")
mty = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("make_typography", mty)
_SPEC.loader.exec_module(mty)

_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


@pytest.fixture
def font_dir(tmp_path):
    src = next((p for p in _FONT_CANDIDATES if Path(p).exists()), None)
    if src is None:
        pytest.skip("no scalable TTF found on this system")
    d = tmp_path / "fonts"
    d.mkdir()
    (d / "test_font.ttf").write_bytes(Path(src).read_bytes())
    return d


def _enclosed_zero_regions(gt: np.ndarray) -> int:
    """Zero-regions that do NOT touch the canvas border = letter counters."""
    lab, n = ndimage.label(gt == 0)
    enclosed = 0
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if ys.min() > 0 and xs.min() > 0 and ys.max() < gt.shape[0] - 1 and xs.max() < gt.shape[1] - 1:
            enclosed += 1
    return enclosed


def test_counters_are_zero_in_gt(font_dir):
    layout_rng = np.random.default_rng(1)
    fx_rng = np.random.default_rng(2)
    rgb, gt = mty.render_typography_sample(
        layout_rng, fx_rng, [font_dir / "test_font.ttf"], bg_images=[],
        force_text="BOOB", force_bg="flat", force_distress=False, force_crop=False)
    assert gt.max() == 1.0
    assert _enclosed_zero_regions(gt) >= 2, "letter counters must be exact zeros"


def test_white_text_on_white_bg_gt_exact(font_dir):
    layout_rng = np.random.default_rng(3)
    fx_rng = np.random.default_rng(4)
    rgb, gt = mty.render_typography_sample(
        layout_rng, fx_rng, [font_dir / "test_font.ttf"], bg_images=[],
        force_text="WHITE", force_bg="white", force_ink=(255, 255, 255),
        force_distress=False, force_crop=False)
    assert float(gt.max()) == 1.0
    assert (gt > 0.5).mean() > 0.01, "glyphs must be present in the GT"
    # the image itself is nearly uniform white — that is the whole lesson
    assert float(np.asarray(rgb, np.float32).std()) < 20.0


def test_distress_never_touches_gt(font_dir):
    fonts = [font_dir / "test_font.ttf"]
    a_rgb, a_gt = mty.render_typography_sample(
        np.random.default_rng(7), np.random.default_rng(8), fonts, bg_images=[],
        force_text="WEAR", force_bg="flat", force_distress=False, force_crop=False)
    b_rgb, b_gt = mty.render_typography_sample(
        np.random.default_rng(7), np.random.default_rng(8), fonts, bg_images=[],
        force_text="WEAR", force_bg="flat", force_distress=True, force_crop=False)
    assert np.array_equal(a_gt, b_gt), "distress must not move the GT by one bit"
    assert not np.array_equal(a_rgb, b_rgb), "distress must visibly change the RGB"


@pytest.fixture
def env(tmp_path, font_dir):
    return {"out": tmp_path / "out", "fonts": font_dir}


def test_run_generates_pairs_and_manifest(env):
    n = mty.run(env["out"], env["fonts"], bg_pool_dirs=[], count=6, seed=21)
    assert n == 6
    rows = [json.loads(l) for l in (env["out"] / "manifest.jsonl").read_text().splitlines()]
    assert len(rows) == 6
    for r in rows:
        assert r["category"] == "typography"
        assert r["id"].startswith("typo_")
        assert (env["out"] / "im" / f"{r['id']}.jpg").exists()
        assert (env["out"] / "gt" / f"{r['id']}.png").exists()


def test_deterministic_and_resume(env, tmp_path):
    mty.run(env["out"], env["fonts"], bg_pool_dirs=[], count=4, seed=21)
    out2 = tmp_path / "out2"
    mty.run(out2, env["fonts"], bg_pool_dirs=[], count=4, seed=21)
    for i in range(4):
        stem = f"typo_{i:05d}"
        assert (env["out"] / "gt" / f"{stem}.png").read_bytes() == (out2 / "gt" / f"{stem}.png").read_bytes()
    # resume: rerun skips everything
    mtimes = {p: p.stat().st_mtime_ns for p in (env["out"] / "im").iterdir()}
    mty.run(env["out"], env["fonts"], bg_pool_dirs=[], count=4, seed=21)
    for p, t in mtimes.items():
        assert p.stat().st_mtime_ns == t
