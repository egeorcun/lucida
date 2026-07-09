import numpy as np
import pytest
from PIL import Image, ImageDraw

from bgr.segmenter import BiRefNetSegmenter, get_device


@pytest.fixture(scope="module")
def toy_image():
    img = Image.new("RGB", (640, 480), (30, 120, 30))
    d = ImageDraw.Draw(img)
    d.ellipse([200, 100, 440, 380], fill=(220, 60, 60))
    return img


def test_get_device_is_mps():
    assert get_device().type == "mps"


@pytest.mark.slow
def test_birefnet_hr_alpha_contract(toy_image):
    seg = BiRefNetSegmenter(
        model_id="ZhengPeng7/BiRefNet_HR", input_size=2048, name="birefnet-hr"
    )
    alpha = seg.predict_alpha(toy_image)
    assert alpha.dtype == np.float32
    assert alpha.shape == (480, 640)
    assert 0.0 <= alpha.min() and alpha.max() <= 1.0
    # elipsin merkezi özne, köşe arka plan olmalı
    assert alpha[240, 320] > 0.5
    assert alpha[10, 10] < 0.5
