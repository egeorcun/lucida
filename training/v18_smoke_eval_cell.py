"""V18 SMOKE-GATE EVAL — single A100 cell (~6 min). Runs AFTER
v18_smoke_driver_cell.py and prints a PASS/FAIL verdict.

Measures epoch_17 (baseline) vs the smoke checkpoint (epoch_18) on:
- testset_limb (24 pairs): fill_alpha must RISE (gloves lesson),
- testset_atmosphere (24 pairs): mae must FALL (density lesson),
- design_real + typography: bg_mae within +0.002 (no smear regression).

PASS unlocks the full ~48-unit epoch (v18_full_driver_cell.py).
Needs HF_TOKEN in Colab Secrets (probe sets live in egeorcun/lucida-eval).

Paste-run: exec(open('/content/my-bg-remover/training/v18_smoke_eval_cell.py').read())
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

snapshot_download("egeorcun/lucida-eval", repo_type="dataset",
                  allow_patterns=["probe/testset_limb/**", "probe/testset_atmosphere/**"],
                  local_dir="/content/_probe18")
for name in ("testset_limb", "testset_atmosphere"):
    if not os.path.isdir(f"{REPO}/data/{name}"):
        shutil.copytree(f"/content/_probe18/probe/{name}", f"{REPO}/data/{name}")
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
for name in ("epoch_17.pth", "epoch_18.pth"):  # epoch_18 = the smoke probe
    src, dst = f"{CKPT_DRIVE}/{name}", f"{CKPT_LOCAL}/{name}"
    if not (os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src)):
        t0 = time.time()
        shutil.copy2(src, dst)
        print(f"{name} -> local ({time.time() - t0:.0f}s)")

import bgr.registry as reg  # noqa: E402
from benchmark.run import run_benchmark  # noqa: E402

reg.MODEL_SPECS["lucida-e17"] = {"ckpt": f"{CKPT_LOCAL}/epoch_17.pth",
                                 "arch_id": "ZhengPeng7/BiRefNet_HR", "input_size": 1024}
reg.MODEL_SPECS["lucida-smoke18"] = {"ckpt": f"{CKPT_LOCAL}/epoch_18.pth",
                                     "arch_id": "ZhengPeng7/BiRefNet_HR", "input_size": 1024}
MODELS = ["lucida-e17", "lucida-smoke18"]
os.makedirs("results", exist_ok=True)

res = {}
for tag, manifest in [("limb", "data/testset_limb/manifest.jsonl"),
                      ("atmosphere", "data/testset_atmosphere/manifest.jsonl"),
                      ("design_real", "data/testset_design_real/manifest.jsonl"),
                      ("typography", "data/testset_typography/manifest.jsonl")]:
    r = run_benchmark(MODELS, manifest, f"results/smoke18_{tag}.json")
    res[tag] = r["per_category"]
    brief = {m: {c: {k: round(v[k], 5) for k in ("mae", "bg_mae", "fill_alpha", "fill_hole") if k in v}
                 for c, v in r["per_category"][m].items()} for m in MODELS}
    print(f"== {tag} ==")
    print(json.dumps(brief, indent=1))

l17 = res["limb"]["lucida-e17"]["limb"]
lsm = res["limb"]["lucida-smoke18"]["limb"]
a17 = res["atmosphere"]["lucida-e17"]["atmosphere"]
asm = res["atmosphere"]["lucida-smoke18"]["atmosphere"]
d17 = res["design_real"]["lucida-e17"]["design_real"]
dsm = res["design_real"]["lucida-smoke18"]["design_real"]
t17 = res["typography"]["lucida-e17"]["typography"]
tsm = res["typography"]["lucida-smoke18"]["typography"]

limb_up = lsm["fill_alpha"] > l17["fill_alpha"] + 0.01
atmo_down = asm["mae"] < a17["mae"] - 0.005
bg_ok = (dsm["bg_mae"] <= d17["bg_mae"] + 0.002) and (tsm["bg_mae"] <= t17["bg_mae"] + 0.002)
print()
print(f"limb fill_alpha : e17={l17['fill_alpha']:.4f} -> smoke={lsm['fill_alpha']:.4f} "
      f"({'YUKARI ✓' if limb_up else 'hareket yok ✗'})")
print(f"atmosphere mae  : e17={a17['mae']:.4f} -> smoke={asm['mae']:.4f} "
      f"({'AŞAĞI ✓' if atmo_down else 'hareket yok ✗'})")
print(f"design_real bg  : e17={d17['bg_mae']:.4f} -> smoke={dsm['bg_mae']:.4f}")
print(f"typography bg   : e17={t17['bg_mae']:.4f} -> smoke={tsm['bg_mae']:.4f}")
print()
if limb_up and atmo_down and bg_ok:
    print("SMOKE GATE: PASS — tam epoch harcaması onaylanabilir (v18_full_driver_cell.py).")
else:
    print("SMOKE GATE: FAIL — tam harcama YAPMA; bulgularla masaya dönülecek.")
