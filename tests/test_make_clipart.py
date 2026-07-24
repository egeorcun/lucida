"""scripts/make_clipart.py — OpenClipart collage generator (Lucida Design).

What matters (the Stay Fresh lesson):
- SVG renders to pixel-exact alpha (interior 1, exterior 0),
- a WHITE-filled clipart over a WHITE background keeps its full shape in
  the GT — interior whites are element, not background,
- collages: GT is the union of element alphas, the background never leaks in,
- make_* contracts: per-stem determinism, resume safety, manifest schema.

Requires cairosvg + libcairo; locally run with
    DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib uv run pytest ...
On Linux/Colab libcairo is present by default.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("cairosvg")

_SPEC = importlib.util.spec_from_file_location(
    "make_clipart", Path(__file__).parent.parent / "scripts" / "make_clipart.py")
mcl = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("make_clipart", mcl)
_SPEC.loader.exec_module(mcl)

_CIRCLE = '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><circle cx="50" cy="50" r="40" fill="#cc3344"/></svg>'
_WHITE_FLOWER = ('<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">'
                 '<circle cx="50" cy="30" r="18" fill="#ffffff"/>'
                 '<circle cx="30" cy="60" r="18" fill="#ffffff"/>'
                 '<circle cx="70" cy="60" r="18" fill="#ffffff"/>'
                 '<circle cx="50" cy="50" r="10" fill="#f4c542"/></svg>')


@pytest.fixture
def svg_dir(tmp_path):
    d = tmp_path / "svgs"
    d.mkdir()
    (d / "circle.svg").write_text(_CIRCLE)
    (d / "flower.svg").write_text(_WHITE_FLOWER)
    return d


def test_svg_renders_to_exact_alpha():
    rgb, a = mcl.render_svg_rgba(_CIRCLE, out_width=200)
    assert a.shape == (200, 200)
    assert a[100, 100] == 1.0      # circle interior
    assert a[5, 5] == 0.0          # exterior
    assert rgb[100, 100, 0] > 150  # red fill


def test_white_clipart_on_white_bg_kept_in_gt(svg_dir):
    rng = np.random.default_rng(5)
    rgb, gt = mcl.render_clipart_sample(
        rng, [svg_dir / "flower.svg"], force_bg="white", n_elements=1)
    assert (gt > 0.5).mean() > 0.005, "white flower must live in the GT"
    # image is close to uniform white where the white petals sit — the lesson
    assert float(np.asarray(rgb, np.float32).std()) < 60.0


def test_collage_gt_is_union_not_background(svg_dir):
    rng = np.random.default_rng(9)
    rgb, gt = mcl.render_clipart_sample(
        rng, [svg_dir / "circle.svg", svg_dir / "flower.svg"],
        force_bg="flat", n_elements=4)
    assert 0.005 < (gt > 0.5).mean() < 0.9
    # all four corners are background
    for c in (gt[0, 0], gt[0, -1], gt[-1, 0], gt[-1, -1]):
        assert c == 0.0


def test_run_contracts_deterministic_resume(svg_dir, tmp_path):
    out1 = tmp_path / "o1"
    n = mcl.run(out1, svg_dir, count=4, seed=33)
    assert n == 4
    rows = [json.loads(l) for l in (out1 / "manifest.jsonl").read_text().splitlines()]
    assert [r["id"] for r in rows] == [f"clip_{i:05d}" for i in range(4)]
    assert all(r["category"] == "clipart" for r in rows)

    out2 = tmp_path / "o2"
    mcl.run(out2, svg_dir, count=4, seed=33)
    for i in range(4):
        s = f"clip_{i:05d}"
        assert (out1 / "gt" / f"{s}.png").read_bytes() == (out2 / "gt" / f"{s}.png").read_bytes()

    mtimes = {p: p.stat().st_mtime_ns for p in (out1 / "im").iterdir()}
    mcl.run(out1, svg_dir, count=4, seed=33)
    for p, t in mtimes.items():
        assert p.stat().st_mtime_ns == t
