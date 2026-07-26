"""V17 SMOKE-GATE DRIVER — single A100 cell (~15 min, ~3 units).

The user directive behind this file: "no more blind experiments". This is
the PROOF step of spec docs/superpowers/specs/2026-07-26-fill-eq-bg-
definitive-fix.md: a 600-step mini-run (EPOCH_NUM_SAMPLES=1200) with the
FULL v17 recipe (fg hinge + v17 sampler + LR_SCALE 1.0) resumed from
epoch_16. It exists to answer ONE question cheaply: does the symmetric
objective push probe fill_alpha UP without breaking bg cleanliness?
Only a PASS (see v17_smoke_eval_cell.py) unlocks the ~46-unit full epoch.

Safety rename: RESUME="auto" picks the LATEST epoch checkpoint, which is
epoch_17.pth (the stage-2 1536 polish — a candidate we must not lose or
resume from). It is renamed to epoch_17_1536.pth ONCE, so auto-resume finds
epoch_16. The smoke run then writes its own throwaway epoch_17.pth.

Paste-run: exec(open('/content/my-bg-remover/training/v17_smoke_driver_cell.py').read())
"""
import json
import os
import urllib.request

CKPT_DIR = "/content/drive/MyDrive/bg-remover-checkpoints"
_old17 = f"{CKPT_DIR}/epoch_17.pth"
_saved17 = f"{CKPT_DIR}/epoch_17_1536.pth"
if os.path.exists(_old17) and not os.path.exists(_saved17):
    _sz = os.path.getsize(_old17)
    if _sz > 2_000_000_000:  # the full 1536 training checkpoint, not a smoke leftover
        os.rename(_old17, _saved17)
        print(f"epoch_17.pth ({_sz / 1e9:.2f} GB) -> epoch_17_1536.pth (stage-2 candidate preserved)")
if os.path.exists(_old17) and os.path.exists(_saved17):
    os.remove(_old17)  # smoke/probe leftover from a previous attempt
    print("stale probe epoch_17.pth removed")

_url = "https://raw.githubusercontent.com/egeorcun/lucida/design-expert/training/train_colab.ipynb"
_nb = json.loads(urllib.request.urlopen(_url, timeout=60).read())
_cells = ["".join(c["source"]) for c in _nb["cells"] if c["cell_type"] == "code"]
_code = "\n\n# ==== NOTEBOOK CELL BREAK ====\n\n".join(_cells)

_subs = [
    ('REPO_GIT_URL = ""', 'REPO_GIT_URL = "https://github.com/egeorcun/lucida.git"'),
    ('REPO_BRANCH = ""', 'REPO_BRANCH = "design-expert"'),
    ('SAMPLER_PRESET = "v16"', 'SAMPLER_PRESET = "v17"'),
    ('EPOCHS = 16', 'EPOCHS = 17'),
    ('EPOCH_NUM_SAMPLES = 27715', 'EPOCH_NUM_SAMPLES = 1200'),
    ('LR_SCALE = 0.5', 'LR_SCALE = 1.0'),
    ('FG_HINGE_LAMBDA = 0.0', 'FG_HINGE_LAMBDA = 3.0'),
]
for old, new in _subs:
    _before = _code
    _code = _code.replace(old, new, 1)
    assert _code != _before, f"injection failed: {old!r}"
assert 'TRAIN_SIZE = 1024' in _code, "smoke must run at 1024"

print(f"SMOKE RUN: v17 recipe, 1200 samples (~600 opt steps), from epoch_16 — {len(_cells)} cells.")
exec(_code, globals())
