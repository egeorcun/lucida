"""scripts/make_visual_panel.py — the release-bar visual panel tool.

Uses a stub segmenter (no real model): the test cares about the contract —
one labeled row per input with 1+N columns, RGBA composited over a dark
checkerboard, plus a contact sheet.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_SPEC = importlib.util.spec_from_file_location(
    "make_visual_panel", Path(__file__).parent.parent / "scripts" / "make_visual_panel.py")
mvp = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("make_visual_panel", mvp)
_SPEC.loader.exec_module(mvp)


class _StubSeg:
    def __init__(self, value):
        self.value = value

    def predict_alpha(self, img):
        return np.full((img.height, img.width), self.value, dtype=np.float32)


def test_panel_rows_and_contact_sheet(tmp_path):
    src = tmp_path / "inputs"
    src.mkdir()
    for i in range(2):
        Image.new("RGB", (300, 200), (200, 60 + 40 * i, 40)).save(src / f"img{i}.jpg")

    factory = lambda name: _StubSeg(1.0 if name == "a" else 0.0)  # noqa: E731
    out = tmp_path / "out"
    rows = mvp.make_panel(src, ["a", "b"], out, segmenter_factory=factory)

    assert len(rows) == 2 and all(p.exists() for p in rows)
    row = Image.open(rows[0])
    assert row.width == mvp.COL_W * 3 + 6 * 2  # original + 2 models
    assert (out / "contact_sheet.jpg").exists()

    arr = np.asarray(row)
    col_b = arr[:, 2 * (mvp.COL_W + 6):, :]          # alpha=0 stub: pure checkerboard
    region = col_b[30:150, 30:400].astype(np.int16)  # away from the label strip
    near = (np.abs(region - mvp.DARK) <= 15) | (np.abs(region - mvp.DARKER) <= 15)
    assert near.mean() > 0.99, "alpha=0 column must be (JPEG-tolerant) pure checkerboard"


def test_checkerboard_composite_math():
    rgb = np.full((50, 50, 3), 200, dtype=np.uint8)
    alpha = np.zeros((50, 50), dtype=np.float32)
    alpha[:25] = 1.0
    out = mvp.composite_over_checker(rgb, alpha)
    assert (out[:25] == 200).all()
    assert set(np.unique(out[25:])) <= {mvp.DARK, mvp.DARKER}
