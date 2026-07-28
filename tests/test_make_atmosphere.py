"""scripts/make_atmosphere.py — atmosphere-density pairs (Lucida v18).

What matters (the I HEARD CHEESE lesson, spec 2026-07-29):
- GT in a halftone field is the DENSITY (mid alpha in dot gaps), never a
  binary dot mask — that IS the lesson;
- GT never exceeds 1, anchors stay solid (alpha 1) over the smoke;
- make_* contracts: per-stem determinism, resume safety, manifest schema.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

try:
    import cairosvg  # noqa: F401
except Exception:
    pytest.skip("cairosvg/libcairo unavailable on this system", allow_module_level=True)

for name, fname in [("make_clipart", "make_clipart.py"),
                    ("make_atmosphere", "make_atmosphere.py")]:
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parent.parent / "scripts" / fname)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)
matm = sys.modules["make_atmosphere"]

_CIRCLE = ('<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">'
           '<circle cx="50" cy="50" r="40" fill="#222222"/></svg>')


@pytest.fixture
def svg_dir(tmp_path):
    d = tmp_path / "svgs"
    d.mkdir()
    (d / "circle.svg").write_text(_CIRCLE)
    return d


def test_deterministic(svg_dir):
    paths = [svg_dir / "circle.svg"]
    r1 = matm.render_atmosphere_sample(matm._item_rng(994, "atmo_00000"), paths)
    r2 = matm.render_atmosphere_sample(matm._item_rng(994, "atmo_00000"), paths)
    assert np.array_equal(r1[0], r2[0]) and np.array_equal(r1[1], r2[1])


def test_gt_is_density_not_binary(svg_dir):
    """Duman alanında GT orta-alfa değerler içermeli (0/1 ikilisi değil)."""
    paths = [svg_dir / "circle.svg"]
    mids = 0
    for i in range(6):
        _, gt = matm.render_atmosphere_sample(matm._item_rng(994, f"atmo_{i:05d}"), paths)
        assert gt.min() >= 0.0 and gt.max() <= 1.0
        mids += int(((gt > 0.15) & (gt < 0.85)).sum() > 0.01 * gt.size)
    assert mids >= 4, "örneklerin çoğunda orta-alfa duman bölgesi olmalı"


def test_anchor_stays_solid(svg_dir):
    """Koyu çapa elemanın iç pikselleri GT 1.0'da kalmalı."""
    paths = [svg_dir / "circle.svg"]
    ok = 0
    for i in range(6):
        rgb, gt = matm.render_atmosphere_sample(matm._item_rng(994, f"atmo_{i:05d}"), paths)
        dark = np.asarray(rgb, np.float32).mean(-1) < 60
        if dark.sum() > 500 and float(gt[dark].mean()) > 0.95:
            ok += 1
    assert ok >= 4, "çapa elemanlar GT'de katı kalmalı"
