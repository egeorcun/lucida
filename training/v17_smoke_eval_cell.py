"""V17 SMOKE-GATE EVAL — single A100 cell (~5 min). Runs AFTER
v17_smoke_driver_cell.py finishes and prints a PASS/FAIL verdict.

Measures epoch_16 (baseline) vs the smoke checkpoint on:
- the 24-pair fill_eq_bg probe (fill_alpha must RISE — the whole point),
- design_real + typography (bg_mae must stay within +0.002 — no smear
  regression bought with the fg hinge).

PASS unlocks the full ~46-unit epoch (v17_full_driver_cell.py).
Needs HF_TOKEN in Colab Secrets (probe set lives in egeorcun/lucida-eval).

Paste-run: exec(open('/content/my-bg-remover/training/v17_smoke_eval_cell.py').read())
"""
import json
import os
import shutil
import subprocess
import sys
import time

REPO = "/content/my-bg-remover"
os.chdir(REPO)
from google.colab import userdata  # noqa: E402

os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
for m in [m for m in list(sys.modules) if m.startswith(("bgr", "benchmark"))]:
    del sys.modules[m]
import torch  # noqa: E402

assert torch.cuda.is_available()

from huggingface_hub import hf_hub_download, snapshot_download  # noqa: E402

# eval setleri: probe + design_real + typography
snapshot_download("egeorcun/lucida-eval", repo_type="dataset",
                  allow_patterns=["probe/testset_fill_eq_bg/**"], local_dir="/content/_probe")
if not os.path.isdir(f"{REPO}/data/testset_fill_eq_bg"):
    shutil.copytree("/content/_probe/probe/testset_fill_eq_bg", f"{REPO}/data/testset_fill_eq_bg")
if not os.path.exists(f"{REPO}/data/testset_design_real/manifest.jsonl"):
    import tarfile
    tar_p = hf_hub_download("egeorcun/lucida-eval", "lucida_eval_sets.tar", repo_type="dataset")
    with tarfile.open(tar_p) as tf:
        tf.extractall(f"{REPO}/data", filter="data")
import glob  # noqa: E402

for f in glob.glob(f"{REPO}/data/**/._*", recursive=True):
    os.remove(f)
subprocess.run(["pip", "install", "-q", "--no-deps", "-e", REPO], check=True)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

CKPT_DRIVE = "/content/drive/MyDrive/bg-remover-checkpoints"
CKPT_LOCAL = "/content/ckpts"
os.makedirs(CKPT_LOCAL, exist_ok=True)
for name in ("epoch_16.pth", "epoch_17.pth"):  # epoch_17 = the smoke probe here
    src, dst = f"{CKPT_DRIVE}/{name}", f"{CKPT_LOCAL}/{name}"
    if not (os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src)):
        t0 = time.time()
        shutil.copy2(src, dst)
        print(f"{name} -> local ({time.time() - t0:.0f}s)")

import bgr.registry as reg  # noqa: E402
from benchmark.run import run_benchmark  # noqa: E402

reg.MODEL_SPECS["lucida-e16"] = {"ckpt": f"{CKPT_LOCAL}/epoch_16.pth",
                                 "arch_id": "ZhengPeng7/BiRefNet_HR", "input_size": 1024}
reg.MODEL_SPECS["lucida-smoke"] = {"ckpt": f"{CKPT_LOCAL}/epoch_17.pth",
                                   "arch_id": "ZhengPeng7/BiRefNet_HR", "input_size": 1024}
MODELS = ["lucida-e16", "lucida-smoke"]
os.makedirs("results", exist_ok=True)

res = {}
for tag, manifest in [("probe", "data/testset_fill_eq_bg/manifest.jsonl"),
                      ("design_real", "data/testset_design_real/manifest.jsonl"),
                      ("typography", "data/testset_typography/manifest.jsonl")]:
    r = run_benchmark(MODELS, manifest, f"results/smoke_{tag}.json")
    res[tag] = r["per_category"]
    brief = {m: {c: {k: round(v[k], 5) for k in ("mae", "bg_mae", "fill_alpha", "fill_hole") if k in v}
                 for c, v in r["per_category"][m].items()} for m in MODELS}
    print(f"== {tag} ==")
    print(json.dumps(brief, indent=1))

p16 = res["probe"]["lucida-e16"]["fill_eq_bg"]
psm = res["probe"]["lucida-smoke"]["fill_eq_bg"]
d16 = res["design_real"]["lucida-e16"]["design_real"]
dsm = res["design_real"]["lucida-smoke"]["design_real"]
t16 = res["typography"]["lucida-e16"]["typography"]
tsm = res["typography"]["lucida-smoke"]["typography"]

fill_up = psm["fill_alpha"] > p16["fill_alpha"] + 0.01
bg_ok = (dsm["bg_mae"] <= d16["bg_mae"] + 0.002) and (tsm["bg_mae"] <= t16["bg_mae"] + 0.002)
print()
print(f"probe fill_alpha : e16={p16['fill_alpha']:.4f} -> smoke={psm['fill_alpha']:.4f} "
      f"({'YUKARI ✓' if fill_up else 'hareket yok ✗'})")
print(f"design_real bg   : e16={d16['bg_mae']:.4f} -> smoke={dsm['bg_mae']:.4f}")
print(f"typography bg    : e16={t16['bg_mae']:.4f} -> smoke={tsm['bg_mae']:.4f}")
print()
if fill_up and bg_ok:
    print("SMOKE GATE: PASS — tam epoch harcaması onaylanabilir (v17_full_driver_cell.py).")
else:
    print("SMOKE GATE: FAIL — tam harcama YAPMA; bulgularla masaya dönülecek.")
