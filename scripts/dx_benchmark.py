"""Design-expert three-way benchmark: lucida-v7 vs the m35 design pipeline
(m35 weights + poster policy + SAM3 referee) vs Ideogram (cached fal.ai
outputs), on the design / fx / text / complex categories of the 203-image
testset. Writes per-image alphas, per-image MAE, and a category table.

Usage: uv run python scripts/dx_benchmark.py [--stage v7|m35|score]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CATS = ("design", "fx", "text", "complex")
OUT = Path("results/dx_bench")


def rows():
    out = []
    for line in open("data/testset/manifest.jsonl"):
        r = json.loads(line)
        if r["category"] in CATS and r.get("gt_alpha"):
            out.append(r)
    return out


def save_alpha(alpha: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(alpha, 0, 1) * 255).astype(np.uint8)).save(path)


def run_v7():
    from bgr.registry import get_segmenter
    seg = get_segmenter("lucida-v7")
    for i, r in enumerate(rows()):
        dst = OUT / "v7" / f"{r['id']}.png"
        if dst.exists():
            continue
        img = Image.open(r["image"]).convert("RGB")
        save_alpha(seg.predict_alpha(img), dst)
        print(f"[v7 {i+1}] {r['id']}", flush=True)


def run_m35():
    from safetensors.torch import load_file
    from torchvision import transforms
    from bgr.segmenter import BiRefNetSegmenter
    from poster_mode import poster_alpha
    from sam3_referee import auto_prompts, design_domain, subject_mask

    seg = None
    for i, r in enumerate(rows()):
        dst = OUT / "m35pipe" / f"{r['id']}.png"
        if dst.exists():
            continue
        img = Image.open(r["image"]).convert("RGB")
        cache = OUT / "cache" / r["id"]
        cache.mkdir(parents=True, exist_ok=True)
        if (cache / "raw.npy").exists():
            raw = np.load(cache / "raw.npy")
        else:
            if seg is None:
                seg = BiRefNetSegmenter(model_id="ZhengPeng7/BiRefNet_HR",
                                        input_size=1024, name="m35c")
                sd = load_file("/Users/egeo/Documents/comfy/ComfyUI/models/"
                               "background_removal/lucida-m35-comfy.safetensors")
                seg.model.load_state_dict(sd, strict=False)
                seg.transform = transforms.Compose([
                    transforms.Resize((1024, 1024)), transforms.ToTensor()])
            raw = seg.predict_alpha(img)
            np.save(cache / "raw.npy", raw.astype(np.float32))
        if not design_domain(img):
            save_alpha(raw, dst)
            print(f"[m35 {i+1}] {r['id']} PHOTO domain -> raw pass-through",
                  flush=True)
            continue
        if (cache / "sm.npy").exists():
            sm = np.load(cache / "sm.npy")
            prompts = ("cached",)
        else:
            prompts = auto_prompts(img)
            sm = subject_mask(img, prompts=prompts)
            np.save(cache / "sm.npy", sm)
        rgb = np.asarray(img, dtype=np.float32)
        ap = poster_alpha(raw, counters="auto", rgb=rgb, haze_matting="auto",
                          subject_mask=sm)
        save_alpha(ap, dst)
        print(f"[m35 {i+1}] {r['id']} prompts={prompts}", flush=True)


def score():
    table = {}
    per_image = {}
    for r in rows():
        gt = np.asarray(Image.open(r["gt_alpha"]).convert("L"),
                        dtype=np.float32) / 255.0
        row = {}
        for model, path in (
            ("lucida-v7", OUT / "v7" / f"{r['id']}.png"),
            ("m35-pipeline", OUT / "m35pipe" / f"{r['id']}.png"),
            ("ideogram", Path("results/ideogram") / f"{r['id']}.png"),
        ):
            if not path.exists():
                print(f"WARNING missing {model} {r['id']}")
                continue
            im = Image.open(path)
            a = np.asarray(im.split()[-1] if im.mode == "RGBA" else im.convert("L"),
                           dtype=np.float32) / 255.0
            if a.shape != gt.shape:
                a = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize(
                    (gt.shape[1], gt.shape[0]), Image.BILINEAR),
                    dtype=np.float32) / 255.0
            row[model] = float(np.abs(a - gt).mean())
        per_image[r["id"]] = {"category": r["category"], **row}
        table.setdefault(r["category"], []).append(row)
    summary = {}
    for cat, rws in table.items():
        summary[cat] = {m: round(float(np.mean([x[m] for x in rws if m in x])), 4)
                        for m in ("lucida-v7", "m35-pipeline", "ideogram")}
    (OUT / "metrics.json").write_text(json.dumps(
        {"summary": summary, "per_image": per_image}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    a = ap.parse_args()
    if a.stage in ("v7", "all"):
        run_v7()
    if a.stage in ("m35", "all"):
        run_m35()
    if a.stage in ("score", "all"):
        score()
