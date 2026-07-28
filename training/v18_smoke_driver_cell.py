"""V18 SMOKE-GATE DRIVER — single A100 cell (~15 min, ~3 units).

The PROOF step of spec docs/superpowers/specs/2026-07-29-v18-limb-
atmosphere.md: a 600-step mini-run (EPOCH_NUM_SAMPLES=1200) with the full
v18 recipe (v17 losses + limb/atmosphere data) resumed from epoch_17. It
answers ONE question cheaply: do the two lessons move their probes (limb
fill_alpha UP, atmosphere mae DOWN) without breaking bg cleanliness?
Only a PASS (see v18_smoke_eval_cell.py) unlocks the ~48-unit full epoch.

RESUME="auto" picks the latest checkpoint = epoch_17.pth (the v17 result —
exactly the intended base). The smoke run writes a throwaway epoch_18.pth.

Paste-run: exec(open('/content/my-bg-remover/training/v18_smoke_driver_cell.py').read())
"""
import json
import os
import urllib.request

CKPT_DIR = "/content/drive/MyDrive/bg-remover-checkpoints"
assert os.path.exists(f"{CKPT_DIR}/epoch_17.pth"), \
    "epoch_17.pth (v17 sonucu) yok — v18 ondan devam etmeli"
_old18 = f"{CKPT_DIR}/epoch_18.pth"
if os.path.exists(_old18):
    os.remove(_old18)  # stale probe from a previous attempt
    print("stale probe epoch_18.pth removed")

_url = "https://raw.githubusercontent.com/egeorcun/lucida/design-expert/training/train_colab.ipynb"
_nb = json.loads(urllib.request.urlopen(_url, timeout=60).read())
_cells = ["".join(c["source"]) for c in _nb["cells"] if c["cell_type"] == "code"]
_code = "\n\n# ==== NOTEBOOK CELL BREAK ====\n\n".join(_cells)

_subs = [
    ('REPO_GIT_URL = ""', 'REPO_GIT_URL = "https://github.com/egeorcun/lucida.git"'),
    ('REPO_BRANCH = ""', 'REPO_BRANCH = "design-expert"'),
    ('SAMPLER_PRESET = "v16"', 'SAMPLER_PRESET = "v18"'),
    ('EPOCHS = 16', 'EPOCHS = 18'),
    ('EPOCH_NUM_SAMPLES = 27715', 'EPOCH_NUM_SAMPLES = 1200'),
    ('LR_SCALE = 0.5', 'LR_SCALE = 1.0'),
    ('FG_HINGE_LAMBDA = 0.0', 'FG_HINGE_LAMBDA = 3.0'),
]
for old, new in _subs:
    _before = _code
    _code = _code.replace(old, new, 1)
    assert _code != _before, f"injection failed: {old!r}"
assert 'TRAIN_SIZE = 1024' in _code, "smoke must run at 1024"

print(f"V18 SMOKE RUN: 1200 samples (~600 opt steps) from epoch_17 — {len(_cells)} cells.")
exec(_code, globals())
