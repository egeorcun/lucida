import io

import numpy as np
from unittest.mock import patch
from fastapi.testclient import TestClient
from PIL import Image


class FakeSeg:
    name = "fake"

    def predict_alpha(self, image):
        w, h = image.size
        return np.ones((h, w), dtype=np.float32)


def _client():
    from serving.app import app
    return TestClient(app)


def test_health():
    r = _client().get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_remove_returns_png():
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (50, 60, 70)).save(buf, format="JPEG")
    buf.seek(0)
    with patch("serving.app._load_segmenter", return_value=FakeSeg()):
        r = _client().post(
            "/remove?decontaminate=false",
            files={"file": ("in.jpg", buf, "image/jpeg")},
        )
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    out = Image.open(io.BytesIO(r.content))
    assert out.mode == "RGBA" and out.size == (16, 16)


def test_unknown_model_400():
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="PNG")
    buf.seek(0)
    r = _client().post("/remove?model=yok", files={"file": ("x.png", buf, "image/png")})
    assert r.status_code == 400


def test_invalid_upload_400():
    with patch("serving.app._load_segmenter", return_value=FakeSeg()):
        r = _client().post(
            "/remove",
            files={"file": ("garbage.png", io.BytesIO(b"bu bir gorsel degil"), "image/png")},
        )
    assert r.status_code == 400
