import json

import numpy as np
from PIL import Image

from benchmark.gallery import build_gallery


def test_gallery_contains_rows_and_images(tmp_path):
    img = tmp_path / "a.jpg"
    Image.new("RGB", (8, 8)).save(img)
    (tmp_path / "results/m1").mkdir(parents=True)
    Image.fromarray(np.full((8, 8), 200, np.uint8)).save(tmp_path / "results/m1/a.png")
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(json.dumps({
        "id": "a", "image": str(img), "category": "hair", "gt_alpha": None,
    }) + "\n")
    out = tmp_path / "results/gallery.html"
    build_gallery(str(manifest), str(tmp_path / "results"), ["m1"], str(out))
    html = out.read_text()
    assert "hair" in html and 'id="a"' in html and "m1/a" in html
