"""V14 DATA UPDATE CELL — in a fresh (FREE, CPU is enough) Colab session,
generates the `design_real` training pairs from the Crello TRAIN split and
merges them into the Drive dataset.

WHY (the v13 lesson): real layered-design artwork is the category where
every measured model collapses (design_real MAE: our best 0.32, birefnet
0.39, inspyrenet 0.40, the commercial reference 0.41) and where v13
regressed visibly versus v7. `scripts/make_design_real.py` composites
`cyberagent/crello` templates (elements ship as individual RGBA layers)
into image/GT pairs with EXACT ground truth — flatten the layers for the
image, union the FOREGROUND layers' alphas for the GT, with a per-template
quality guard against Crello's own preview render. The TRAIN split
(~18k templates) is disjoint from the `design_real` benchmark category
(test split) by construction.

Flow pattern (report()/log.txt, Errno 5 retry, module-cache purge,
copy_pairs merge, flush at the end) is the v8/v9 cell lineage.

PREREQUISITES: repo cloned at /content/my-bg-remover with `pip install -e .`;
Drive with bg-remover-data/TRAIN/{im,gt} and train_composites_manifest.jsonl.
NO tar fetch needed (sources come from the HF hub, not from Drive).
"""

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import PIL.Image

PIL.Image.MAX_IMAGE_PIXELS = None

WORKDIR = "/content/my-bg-remover"
DRIVE_ROOT = "/content/drive/MyDrive"
DRIVE_OUTPUT_SUBDIR = "bg-remover-data"
DRIVE_STATUS_SUBDIR = "bg-remover-status"
SEED = 7
DESIGN_REAL_COUNT = 8000
OUT_DIR = Path("data/train_design_real")          # generator output (relative to WORKDIR)
EXPORT_DIR = "/content/birefnet_format_v14"

STATUS_DIR = Path(DRIVE_ROOT) / DRIVE_STATUS_SUBDIR
LOG_PATH = STATUS_DIR / "log.txt"
STATUS_PATH = STATUS_DIR / "status.json"

SCRIPTS_DIR = str(Path(WORKDIR) / "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import training.train_colab_lib as tcl  # noqa: E402


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def report(stage: str, status: str, **extra) -> None:
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    ts = _now()
    line = f"[{ts}] stage={stage} status={status}"
    if extra:
        line += " " + json.dumps(extra, ensure_ascii=False, default=str)
    print(line)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")
    history = []
    if STATUS_PATH.exists():
        try:
            history = json.loads(STATUS_PATH.read_text()).get("history", [])
        except Exception:
            history = []
    history.append({"stage": stage, "status": status, "time": ts, "detail": extra})
    STATUS_PATH.write_text(json.dumps({"stage": stage, "status": status, "time": ts,
                                       "detail": extra, "history": history},
                                      ensure_ascii=False, indent=2, default=str))


def _listdir_retry(d: Path, attempts: int = 5, wait_s: int = 30) -> list[Path]:
    for i in range(attempts):
        try:
            return list(d.iterdir())
        except OSError as e:
            if i == attempts - 1:
                raise
            print(f"WARNING: {e} while listing {d} — waiting {wait_s}s ({i + 1}/{attempts - 1}).")
            time.sleep(wait_s)
    raise AssertionError("unreachable")


def _git_pull_idempotent() -> None:
    try:
        r = subprocess.run(["git", "-C", WORKDIR, "pull", "--ff-only"],
                           capture_output=True, text=True, timeout=180)
        print(f"git pull: rc={r.returncode} {r.stdout.strip() or r.stderr.strip()}")
    except Exception as e:
        print(f"WARNING: git pull failed ({e}) — continuing with the existing clone.")


def _purge_script_module_cache() -> None:
    """2026-07-21 lesson: sys.modules keeps pre-pull modules across cell runs."""
    import importlib
    for name in ("make_design_real", "make_bokeh_copies", "make_design", "make_textfx",
                 "make_v6_copies", "export_birefnet", "build_testset", "build_trainset"):
        sys.modules.pop(name, None)
    importlib.reload(tcl)


def stage0_env() -> None:
    from google.colab import drive
    drive.mount("/content/drive")
    assert Path(DRIVE_ROOT).is_dir()
    report("env", "running")
    os.chdir(WORKDIR)
    _git_pull_idempotent()
    _purge_script_module_cache()
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "datasets"], check=True)
    free_gb = shutil.disk_usage("/content").free / 1e9
    print(f"local free disk: {free_gb:.0f} GB (~25 GB needed: Crello download + pairs)")
    report("env", "done", cwd=str(Path.cwd()), free_gb=round(free_gb, 1))


def stage_generate() -> int:
    report("design_real", "running")
    import make_design_real as mdr
    if not hasattr(mdr, "run"):
        raise RuntimeError("scripts/make_design_real.py missing run() — repo stale, git pull.")
    n = mdr.run(OUT_DIR, count=DESIGN_REAL_COUNT, seed=SEED, split="train", quality=92)
    if n < 500:
        raise RuntimeError(f"only {n} pairs generated — inspect the logs.")
    report("design_real", "done", pairs=n)
    return n


def stage_export() -> None:
    report("export", "running")
    import export_birefnet as eb
    stats = eb.export(manifest_path=str(OUT_DIR / "manifest.jsonl"),
                      out_dir=EXPORT_DIR, split_name="TRAIN")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    report("export", "done", stats=stats)


def stage_drive_copy() -> None:
    """Fresh-stem merge with size-checked copy_pairs (threaded, resumable)."""
    report("drive_copy", "running")
    src = Path(EXPORT_DIR)
    dst = Path(DRIVE_ROOT) / DRIVE_OUTPUT_SUBDIR
    dst_im, dst_gt = dst / "TRAIN" / "im", dst / "TRAIN" / "gt"
    assert dst_im.is_dir() and dst_gt.is_dir()

    src_im_files = list((src / "TRAIN" / "im").iterdir())
    src_gt_files = list((src / "TRAIN" / "gt").iterdir())
    assert len(src_im_files) == len(src_gt_files)
    pre_im = len(_listdir_retry(dst_im))
    stems = sorted(p.stem for p in src_im_files)
    n_copied = tcl.copy_pairs(stems, src / "TRAIN" / "im", src / "TRAIN" / "gt", dst_im, dst_gt)
    post_im, post_gt = len(_listdir_retry(dst_im)), len(_listdir_retry(dst_gt))
    print(f"copy_pairs: {n_copied} copied; Drive TRAIN im {pre_im} -> {post_im}")
    assert post_im == post_gt, f"im/gt mismatch after merge: {post_im} != {post_gt}"
    assert pre_im <= post_im <= pre_im + len(stems)

    n_appended = tcl.merge_composite_manifest(OUT_DIR / "manifest.jsonl",
                                              dst / "train_composites_manifest.jsonl")
    print(f"train_composites_manifest.jsonl: {n_appended} new rows appended.")
    report("drive_copy", "done", copied=n_copied, manifest_rows=n_appended, total_im=post_im)


def main() -> None:
    stage0_env()
    stage_generate()
    stage_export()
    stage_drive_copy()
    report("ALL", "done")
    print("Flushing Drive (waiting for async writes to land in the cloud)...")
    from google.colab import drive as _gdrive
    _gdrive.flush_and_unmount()
    print("Drive flush COMPLETE — the VM can now be safely shut down/swapped.")


try:
    main()
except Exception:
    tb = traceback.format_exc()
    report("FATAL", "error", traceback=tb)
    raise
