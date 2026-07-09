import json
from unittest.mock import patch

import numpy as np
from PIL import Image

from benchmark.run import run_benchmark


class FakeSeg:
    name = "fake"

    def predict_alpha(self, image):
        w, h = image.size
        return np.ones((h, w), dtype=np.float32)


def _make_testset(tmp_path):
    img = tmp_path / "a.jpg"
    Image.new("RGB", (8, 8), (10, 10, 10)).save(img)
    gt = tmp_path / "a.png"
    Image.fromarray(np.full((8, 8), 255, np.uint8)).save(gt)
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(json.dumps({
        "id": "a", "image": str(img), "category": "general", "gt_alpha": str(gt),
    }) + "\n")
    return manifest


def test_run_benchmark_outputs_and_metrics(tmp_path):
    manifest = _make_testset(tmp_path)
    with patch("benchmark.run.get_segmenter", return_value=FakeSeg()):
        result = run_benchmark(["fake"], str(manifest), str(tmp_path / "out"))
    assert (tmp_path / "out/fake/a.png").exists()
    assert result["per_image"]["fake"]["a"]["sad"] == 0.0  # tam isabet
    assert result["overall"]["fake"]["mae"] == 0.0
    assert (tmp_path / "out/metrics.json").exists()
