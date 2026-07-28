"""scripts/poster_mode.py — the decisive poster/POD policy.

Contracts (the Ideogram-policy discussion, 2026-07-28):
- nothing survives outside kept components (haze impossible);
- "open" (default): CONFIDENT-zero holes stay transparent and floating
  islands inside them (drips) are removed; HESITANT holes print solid;
- "solid": every small hole prints solid;
- large enclosed windows stay open in every mode;
- the outer edge keeps the model's soft alpha.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    "poster_mode", Path(__file__).parent.parent / "scripts" / "poster_mode.py")
pm = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("poster_mode", pm)
_SPEC.loader.exec_module(pm)


def _scene():
    """200x200: big square subject with a smoky hole, a letter-like counter
    containing a small kept drip, a big window, and outside haze."""
    a = np.zeros((200, 200), dtype=np.float32)
    a[20:180, 20:180] = 1.0
    a[40:60, 40:60] = 0.35            # smoky hole (model hesitated)
    a[80:100, 40:60] = 0.0            # counter: confident zero...
    a[88:92, 48:52] = 1.0             # ...with a floating drip inside
    a[120:170, 100:160] = 0.0         # big window (stays open everywhere)
    a[5:12, 5:12] = 0.2               # outside haze speck
    return a


def test_outside_haze_always_dies():
    for mode in ("open", "solid"):
        out = pm.poster_alpha(_scene(), counters=mode)
        assert out[5:12, 5:12].max() == 0.0


def test_open_mode_clears_counters_and_drips():
    out = pm.poster_alpha(_scene(), counters="open")
    assert out[85:95, 45:55].max() == 0.0, "counter must be fully transparent"
    assert out[89:91, 49:51].max() == 0.0, "floating drip must be removed"
    assert out[45:55, 45:55].min() > 0.99, \
        "hesitant (smoky) hole prints solid even in open mode"


def test_solid_mode_fills_small_holes():
    out = pm.poster_alpha(_scene(), counters="solid")
    assert out[45:55, 45:55].min() > 0.99, "smoky hole prints solid"
    assert out[85:95, 45:55].min() > 0.99, "counter prints solid"


def test_big_window_stays_open_in_all_modes():
    for mode in ("open", "solid"):
        out = pm.poster_alpha(_scene(), counters=mode, max_hole_frac=0.1)
        assert out[130:160, 110:150].max() == 0.0, mode


def test_invalid_mode_raises():
    import pytest
    with pytest.raises(ValueError):
        pm.poster_alpha(_scene(), counters="banana")


def test_counters_auto_resolves_by_page_color():
    import numpy as np
    a = _scene()
    white_page = np.full((200, 200, 3), 250.0, dtype=np.float32)
    cream_page = np.full((200, 200, 3), (246.0, 240.0, 214.0), dtype=np.float32)
    open_like = pm.poster_alpha(a, counters="auto", rgb=white_page, haze_matting=False)
    solid_like = pm.poster_alpha(a, counters="auto", rgb=cream_page, haze_matting=False)
    # counter bolgesi: beyaz sayfada acik, krem sayfada dolu
    assert open_like[85:95, 45:55].max() == 0.0
    assert solid_like[85:95, 45:55].min() > 0.99


def test_haze_matting_thins_draining_page_like_haze_only():
    """The CHEESE airiness lesson: page-like haze that drains to the
    silhouette edge drops to ink density; outline-locked whites survive."""
    import numpy as np
    a = np.zeros((200, 200), dtype=np.float32)
    a[20:180, 20:180] = 1.0
    rgb = np.full((200, 200, 3), 250.0, dtype=np.float32)   # page-white everywhere
    rgb[20:180, 20:180] = 240.0                              # near-page haze block
    rgb[80:120, 80:120] = (200, 30, 30)                      # a red element inside
    out = pm.poster_alpha(a, counters="open", rgb=rgb, haze_matting=True)
    assert out[30, 100] < 0.2, "draining page-like haze must thin out"
    assert out[100, 100] > 0.9, "colored content keeps full alpha"
    out_off = pm.poster_alpha(a, counters="open", rgb=rgb, haze_matting=False)
    assert out_off[30, 100] > 0.9, "matting off -> untouched"
