import numpy as np
import pytest
from PIL import Image

from bgr.pipeline import PipelineSegmenter


class FlatFakeSeg:
    name = "flat-fake"

    def predict_alpha(self, image):
        w, h = image.size
        a = np.full((h, w), 0.5, dtype=np.float32)
        a[: h // 4] = 0.0
        a[-h // 4 :] = 1.0
        return a


def test_name_reflects_refine_flag():
    assert PipelineSegmenter(FlatFakeSeg()).name == "flat-fake"
    assert PipelineSegmenter(FlatFakeSeg(), refine=True).name == "flat-fake+refine"


def test_predict_alpha_contract():
    p = PipelineSegmenter(FlatFakeSeg())
    a = p.predict_alpha(Image.new("RGB", (32, 40)))
    assert a.dtype == np.float32 and a.shape == (40, 32)


def test_process_returns_rgba():
    p = PipelineSegmenter(FlatFakeSeg())
    out = p.process(Image.new("RGB", (32, 32), (200, 30, 30)), decontaminate=True)
    assert out.mode == "RGBA" and out.size == (32, 32)


def test_registry_parses_refine_suffix():
    from unittest.mock import patch
    from bgr.registry import get_segmenter
    with patch("bgr.registry.BiRefNetSegmenter") as m:
        m.return_value.name = "rmbg-2.0"
        seg = get_segmenter("rmbg-2.0+refine")
    assert seg.name == "rmbg-2.0+refine"


def test_registry_unknown_base_still_raises():
    from bgr.registry import get_segmenter
    with pytest.raises(KeyError):
        get_segmenter("yok+refine")
