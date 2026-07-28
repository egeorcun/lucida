"""V18 FULL DRIVER — the real spend (~3.5h @1024, ~48 units). Run ONLY
after v18_smoke_eval_cell.py printed "SMOKE GATE: PASS".

Full v18 recipe from epoch_17: v17 losses (fg hinge 3.0, LR_SCALE 1.0) +
v18 sampler (limb .05 + atmosphere .06), one full epoch at 1024. The smoke
probe's throwaway epoch_18.pth is deleted first so auto-resume starts from
epoch_17 again.

Paste-run: exec(open('/content/my-bg-remover/training/v18_full_driver_cell.py').read())
"""
import json
import os
import urllib.request

CKPT_DIR = "/content/drive/MyDrive/bg-remover-checkpoints"
assert os.path.exists(f"{CKPT_DIR}/epoch_17.pth"), "epoch_17.pth yok"
if os.path.exists(f"{CKPT_DIR}/epoch_18.pth"):
    os.remove(f"{CKPT_DIR}/epoch_18.pth")
    print("smoke probe epoch_18.pth removed — full run starts from epoch_17")

_url = "https://raw.githubusercontent.com/egeorcun/lucida/design-expert/training/train_colab.ipynb"
_nb = json.loads(urllib.request.urlopen(_url, timeout=60).read())
_cells = ["".join(c["source"]) for c in _nb["cells"] if c["cell_type"] == "code"]
_code = "\n\n# ==== NOTEBOOK CELL BREAK ====\n\n".join(_cells)

_subs = [
    ('REPO_GIT_URL = ""', 'REPO_GIT_URL = "https://github.com/egeorcun/lucida.git"'),
    ('REPO_BRANCH = ""', 'REPO_BRANCH = "design-expert"'),
    ('SAMPLER_PRESET = "v16"', 'SAMPLER_PRESET = "v18"'),
    ('EPOCHS = 16', 'EPOCHS = 18'),
    ('LR_SCALE = 0.5', 'LR_SCALE = 1.0'),
    ('FG_HINGE_LAMBDA = 0.0', 'FG_HINGE_LAMBDA = 3.0'),
]
for old, new in _subs:
    _before = _code
    _code = _code.replace(old, new, 1)
    assert _code != _before, f"injection failed: {old!r}"
assert 'EPOCH_NUM_SAMPLES = 27715' in _code and 'TRAIN_SIZE = 1024' in _code

print(f"FULL V18 RUN: 1 epoch @1024 from epoch_17 (limb + atmosphere) — {len(_cells)} cells.")
exec(_code, globals())
