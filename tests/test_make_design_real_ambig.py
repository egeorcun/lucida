"""render_template_ambig — manufactured fill==page ambiguity on REAL layouts.

The contract (spec 2026-07-26): the page is painted with the dominant color
of one FOREGROUND element, the original background stack is dropped, the GT
is the union of foreground alphas — so at least one real element is
pixel-indistinguishable from the page while its GT stays fully opaque.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_SPEC = importlib.util.spec_from_file_location(
    "make_design_real", Path(__file__).parent.parent / "scripts" / "make_design_real.py")
mdr = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("make_design_real", mdr)
_SPEC.loader.exec_module(mdr)


def _solid(w, h, rgba):
    return Image.new("RGBA", (w, h), rgba)


def _fake_ex():
    """4 elements: full-canvas bg + 3 foreground squares (teal, teal, dark)."""
    return {
        "id": "fake01",
        "canvas_width": 400,
        "canvas_height": 400,
        "length": 4,
        "image": [
            _solid(400, 400, (10, 200, 90, 255)),    # background (dropped)
            _solid(120, 120, (70, 160, 210, 255)),   # teal square
            _solid(120, 120, (70, 160, 210, 255)),   # teal square 2
            _solid(80, 80, (25, 20, 30, 255)),       # dark square
        ],
        "left": [0.0, 30.0, 240.0, 150.0],
        "top": [0.0, 40.0, 220.0, 150.0],
        "width": [400.0, 120.0, 120.0, 80.0],
        "height": [400.0, 120.0, 120.0, 80.0],
        "angle": [0.0, 0.0, 0.0, 0.0],
        "opacity": [1.0, 1.0, 1.0, 1.0],
    }


def test_ambig_page_matches_an_element_and_gt_is_opaque():
    rng = np.random.default_rng(3)
    out = mdr.render_template_ambig(_fake_ex(), rng)
    assert out is not None
    comp, gt = out
    page = comp[0, 0].astype(np.float32)          # corner = page color
    # the page color equals one element's dominant color exactly
    candidates = [np.float32([70, 160, 210]), np.float32([25, 20, 30])]
    assert any(np.abs(page - c).max() < 2 for c in candidates)
    # GT: three fg squares opaque, background stack gone
    assert 0.05 < (gt > 0.5).mean() < 0.5
    assert gt[100, 90] > 0.99                      # inside teal square 1
    # the matching element is indistinguishable from the page in RGB...
    if np.abs(page - candidates[0]).max() < 2:
        region = comp[45:155, 35:145].astype(np.float32)
        assert np.abs(region - page).max() < 3
    # ...yet its GT stays fully opaque (the whole lesson)
    assert gt[45:155, 35:145].min() > 0.99


def test_ambig_deterministic_for_same_rng():
    a = mdr.render_template_ambig(_fake_ex(), np.random.default_rng(7))
    b = mdr.render_template_ambig(_fake_ex(), np.random.default_rng(7))
    assert a is not None and b is not None
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_ambig_rejects_all_background():
    ex = _fake_ex()
    ex["length"] = 1
    ex["image"] = ex["image"][:1]
    for k in ("left", "top", "width", "height", "angle", "opacity"):
        ex[k] = ex[k][:1]
    assert mdr.render_template_ambig(ex, np.random.default_rng(1)) is None
