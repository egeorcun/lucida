import numpy as np
from unittest.mock import patch
from PIL import Image

from bgr.cli import main


class FakeSeg:
    name = "fake"

    def predict_alpha(self, image):
        w, h = image.size
        return np.ones((h, w), dtype=np.float32)


def test_remove_writes_rgba(tmp_path):
    src = tmp_path / "in.jpg"
    Image.new("RGB", (16, 16), (10, 120, 200)).save(src)
    dst = tmp_path / "out.png"
    with patch("bgr.cli.get_segmenter", return_value=FakeSeg()):
        main(["remove", str(src), "-o", str(dst), "--no-decontaminate"])
    out = Image.open(dst)
    assert out.mode == "RGBA" and out.size == (16, 16)
