"""V9 DATA UPDATE CELL — in a fresh (FREE, CPU is enough) Colab session,
REGENERATES the 9,000 bokeh copies (`{stem}_k00`) on Drive with the FIXED
compositor and archives the discarded v8 checkpoint. Much lighter than the
v8 cell: no downloads, no design work.

WHY (the v8 alpha^2 lesson, measured on the epoch-8 run): the v8 bokeh
copies re-composited each photo with its own alpha (`a*photo + (1-a)*bokeh`),
dimming every soft wisp to alpha^2*F while the GT kept saying alpha. The
model learned "keep faint fuzzy regions" — the exact OPPOSITE of the
intended counter-lesson — and the hair category regressed (MAE
0.0093 -> 0.0176, 37/40 test images worse; overall 0.0257 -> 0.0287).
`scripts/make_bokeh_copies.py` now composites `a*F_ext + (1-a)*bokeh` with
an alpha-weighted foreground-color extension (validated: soft-band error vs
pymatting-reference optics -40..45% on real P3M portraits; synthetic
regression test in tests/test_make_bokeh_copies.py).

WHAT THIS CELL DOES:
1. bokeh regen: the SAME 9,000 `_k00` ids are re-rendered into a FRESH local
   dir (make_bokeh_copies skips existing files — a fresh dir forces new
   renders) and merged over Drive TRAIN with size-checked copy_pairs
   (overwrite-in-place; the manifest does not change).
2. refresh marker: the k00 stems are added to tar/_refresh_stems.json so
   train_colab.ipynb cell (c) size-revalidates any stale local copies.
3. checkpoint archive: bg-remover-checkpoints/epoch_8.pth (the alpha^2
   epoch) is renamed to epoch_8_v8_alpha2bug.pth so the next training run
   (RESUME='auto', EPOCHS=8) resumes from epoch_7 and writes a fresh
   epoch_8.pth. Idempotent: skipped if already renamed.

Everything else (flow pattern, report(), Errno 5 retry, flush at the end)
is the v8 cell verbatim — see training/v8_data_update_cell.py.

PREREQUISITES: repo cloned at /content/my-bg-remover with `pip install -e .`
(or at least importable), Drive with TRAIN/{im,gt}, tar/_manifest.json and
train_composites_manifest.jsonl. The env stage git-pulls and purges the
module cache, so a pre-existing clone is fine.
"""

import json
import os
import shutil
import subprocess
import sys
import tarfile
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
DRIVE_CKPT_SUBDIR = "bg-remover-checkpoints"
TAR_SUBDIR = "tar"
SEED = 42

LOCAL_TRAIN_ROOT = Path("/content/v9_train_src")
TAR_CACHE = Path("/content/tar_cache_v9")
BOKEH_OUT_DIR = Path("data/train_v9_bokeh")   # FRESH dir — forces re-renders of the same _k00 ids
BOKEH_COUNT = 9000
EXPORT_DIR_BOKEH = "/content/birefnet_format_v9_bokeh"
BAD_CKPT = "epoch_8.pth"
BAD_CKPT_ARCHIVED = "epoch_8_v8_alpha2bug.pth"

STATUS_DIR = Path(DRIVE_ROOT) / DRIVE_STATUS_SUBDIR
LOG_PATH = STATUS_DIR / "log.txt"
STATUS_PATH = STATUS_DIR / "status.json"

SCRIPTS_DIR = str(Path(WORKDIR) / "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import training.train_colab_lib as tcl  # noqa: E402


# ==========================================================================
# report / retry / env — v8 cell verbatim (see training/v8_data_update_cell.py)
# ==========================================================================
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
    payload = {"stage": stage, "status": status, "time": ts, "detail": extra, "history": history}
    STATUS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def _listdir_retry(d: Path, attempts: int = 5, wait_s: int = 30) -> list[Path]:
    for i in range(attempts):
        try:
            return list(d.iterdir())
        except OSError as e:
            if i == attempts - 1:
                raise
            print(f"WARNING: {e} while listing {d} — waiting {wait_s}s before retrying "
                  f"({i + 1}/{attempts - 1}).")
            time.sleep(wait_s)
    raise AssertionError("unreachable")


def _n_files(d: Path) -> int:
    return sum(1 for p in d.iterdir() if p.is_file()) if d.is_dir() else 0


def _git_pull_idempotent() -> None:
    try:
        r = subprocess.run(["git", "-C", WORKDIR, "pull", "--ff-only"],
                           capture_output=True, text=True, timeout=180)
        print(f"git pull: rc={r.returncode} {r.stdout.strip() or r.stderr.strip()}")
    except Exception as e:
        print(f"WARNING: could not run git pull ({e}) — continuing with the existing clone.")


def _purge_script_module_cache() -> None:
    """The 2026-07-21 lesson (v8 cell): sys.modules caches pre-pull modules."""
    import importlib
    for name in ("make_bokeh_copies", "make_design", "make_textfx",
                 "make_v6_copies", "export_birefnet", "build_testset", "build_trainset"):
        sys.modules.pop(name, None)
    importlib.reload(tcl)


def stage0_env() -> None:
    from google.colab import drive
    drive.mount("/content/drive")
    assert Path(DRIVE_ROOT).is_dir(), f"Drive could not be mounted: {DRIVE_ROOT} missing"
    report("env", "running")
    os.chdir(WORKDIR)
    _git_pull_idempotent()
    _purge_script_module_cache()
    free_gb = shutil.disk_usage("/content").free / 1e9
    print(f"local free disk: {free_gb:.0f} GB (~40 GB needed: tar extraction + bokeh output)")
    report("env", "done", cwd=str(Path.cwd()), free_gb=round(free_gb, 1))


# ==========================================================================
# tar_fetch + categories — v8 cell verbatim (bokeh sources live in the tars)
# ==========================================================================
def stage_tar_fetch() -> int:
    report("tar_fetch", "running")
    tar_dir = Path(DRIVE_ROOT) / DRIVE_OUTPUT_SUBDIR / TAR_SUBDIR
    manifest_path = tar_dir / "_manifest.json"
    assert manifest_path.exists(), f"{manifest_path} missing — run the tar packing cell first."
    manifest = json.loads(manifest_path.read_text())
    total_pairs = tcl.validate_tar_manifest(manifest)

    local_im, local_gt = LOCAL_TRAIN_ROOT / "im", LOCAL_TRAIN_ROOT / "gt"
    n_im, n_gt = _n_files(local_im), _n_files(local_gt)
    if n_im >= total_pairs and n_im == n_gt:
        print(f"Tar download/extract SKIPPED: {n_im} pairs already local.")
    else:
        LOCAL_TRAIN_ROOT.mkdir(parents=True, exist_ok=True)
        TAR_CACHE.mkdir(parents=True, exist_ok=True)
        for sh in manifest["shards"]:
            src, dst = tar_dir / sh["name"], TAR_CACHE / sh["name"]
            if not (dst.exists() and dst.stat().st_size == sh["bytes"]):
                shutil.copy2(src, dst)
                if dst.stat().st_size != sh["bytes"]:
                    raise RuntimeError(f"{sh['name']}: size mismatch after copy — re-run the cell.")
            with tarfile.open(dst) as tf:
                tf.extractall(LOCAL_TRAIN_ROOT, filter="data")
            dst.unlink()
            print(f"{sh['name']}: copied + extracted ({sh['pairs']} pairs).")
        n_im, n_gt = _n_files(local_im), _n_files(local_gt)
        if n_im != n_gt or n_im < total_pairs:
            raise RuntimeError(f"tar extraction mismatch: im={n_im}, gt={n_gt}, expected {total_pairs}.")
    report("tar_fetch", "done", pairs=n_im)
    return n_im


def stage_categories() -> tuple[dict[str, str], set[str]]:
    report("categories", "running")
    drive_manifest = Path(DRIVE_ROOT) / DRIVE_OUTPUT_SUBDIR / "train_composites_manifest.jsonl"
    assert drive_manifest.exists(), f"{drive_manifest} missing"
    category_by_stem = tcl.load_stem_categories(drive_manifest)
    exclude_stems: set[str] = set()
    val_json = STATUS_DIR / "val_stems.json"
    if val_json.exists():
        val_stems = json.loads(val_json.read_text())["val_stems"]
        excluded_ids, _ = tcl.derive_val_excluded_source_ids(val_stems)
        exclude_stems = set(val_stems) | {
            s for s in category_by_stem if tcl.strip_composite_copy_suffix(s) in excluded_ids
        }
        print(f"VAL leak guard: {len(val_stems)} val stems -> {len(exclude_stems)} excluded.")
    report("categories", "done", stems=len(category_by_stem), excluded=len(exclude_stems))
    return category_by_stem, exclude_stems


# ==========================================================================
# bokeh regen — SAME ids, FIXED renders (fresh out dir forces regeneration)
# ==========================================================================
def stage_bokeh_regen(category_by_stem: dict[str, str], exclude_stems: set[str]) -> None:
    report("bokeh_regen", "running")
    import make_bokeh_copies as mbc  # scripts/ on sys.path; cache purged in env

    # The whole point of the regen is the alpha^2 fix — refuse a stale clone.
    if not hasattr(mbc, "_estimate_foreground"):
        raise RuntimeError(
            "scripts/make_bokeh_copies.py does NOT contain the alpha^2 fix "
            f"(_estimate_foreground missing) — run 'git -C {WORKDIR} pull' and re-run."
        )

    counts = mbc.run(
        train_im_dir=LOCAL_TRAIN_ROOT / "im",
        train_gt_dir=LOCAL_TRAIN_ROOT / "gt",
        category_by_stem=category_by_stem,
        out_dir=BOKEH_OUT_DIR,
        seed=SEED,
        count=BOKEH_COUNT,
        categories={"hair"},
        exclude_stems=exclude_stems,
    )
    print("make_bokeh_copies.run() production:", counts)

    out_manifest = BOKEH_OUT_DIR / "manifest.jsonl"
    rows = [json.loads(line) for line in out_manifest.read_text().splitlines() if line.strip()]
    if len(rows) < 100:
        raise RuntimeError(f"only {len(rows)} bokeh pairs regenerated — inspect the logs.")
    full_manifest = BOKEH_OUT_DIR / "manifest_full.jsonl"
    with open(full_manifest, "w") as f:
        for r in rows:
            im_p = BOKEH_OUT_DIR / "im" / f"{r['id']}.jpg"
            gt_p = BOKEH_OUT_DIR / "gt" / f"{r['id']}.png"
            if not (im_p.exists() and gt_p.exists()):
                raise RuntimeError(f"file missing for manifest row: {r['id']}")
            f.write(json.dumps({"id": r["id"], "image": str(im_p),
                                "category": r["category"], "gt_alpha": str(gt_p)},
                               ensure_ascii=False) + "\n")
    report("bokeh_regen", "done", total_pairs=len(rows))


def stage_export() -> None:
    report("export", "running")
    import export_birefnet as eb
    stats = eb.export(manifest_path=str(BOKEH_OUT_DIR / "manifest_full.jsonl"),
                      out_dir=EXPORT_DIR_BOKEH, split_name="TRAIN")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    report("export", "done", stats=stats)


def stage_drive_copy() -> None:
    """Overwrite-in-place merge with size-checked copy_pairs (the v8 cell's
    copytree replacement — threaded x16, resumable, prints progress). All
    9,000 pairs already exist on Drive with v8 renders; the new file sizes
    differ, so copy_pairs re-copies them all."""
    report("drive_copy", "running")
    src = Path(EXPORT_DIR_BOKEH)
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
    print(f"copy_pairs: {n_copied} pairs copied/overwritten; Drive TRAIN im {pre_im} -> {post_im}")
    assert post_im == post_gt, f"im/gt counts differ after merge: {post_im} != {post_gt}"
    # Growth is allowed up to len(stems): the 2026-07-22 recovery path DELETES
    # the stale k00 targets first (Drive FUSE overwrites crawled at ~0.5
    # pairs/s that night; fresh writes are ~25x faster), so the merge
    # legitimately restores what was deleted. Shrinkage or growth beyond the
    # k00 set still fails loudly.
    assert pre_im <= post_im <= pre_im + len(stems), (
        f"unexpected TRAIN count change ({pre_im} -> {post_im}, k00 set={len(stems)})"
    )
    report("drive_copy", "done", overwritten=n_copied, total_im=post_im)


def stage_refresh_marker() -> None:
    report("refresh_marker", "running")
    stems = sorted(
        json.loads(line)["id"]
        for line in (BOKEH_OUT_DIR / "manifest_full.jsonl").read_text().splitlines() if line.strip()
    )
    marker = Path(DRIVE_ROOT) / DRIVE_OUTPUT_SUBDIR / TAR_SUBDIR / "_refresh_stems.json"
    existing: list[str] = []
    if marker.exists():
        try:
            existing = json.loads(marker.read_text()).get("stems", [])
        except Exception:
            existing = []
    merged = sorted(set(existing) | set(stems))
    marker.write_text(json.dumps(
        {"stems": merged, "written": _now(),
         "reason": "v9 bokeh regeneration (alpha^2 fix) — size-revalidate local copies"},
        ensure_ascii=False,
    ))
    print(f"{marker}: {len(merged)} refresh stems recorded ({len(stems)} from this run).")
    report("refresh_marker", "done", stems=len(merged))


def stage_archive_bad_checkpoint() -> None:
    """Renames the discarded alpha^2 epoch so RESUME='auto' (EPOCHS=8) resumes
    from epoch_7 and the retrain writes a fresh epoch_8.pth. Idempotent."""
    report("ckpt_archive", "running")
    ckpt_dir = Path(DRIVE_ROOT) / DRIVE_CKPT_SUBDIR
    bad, archived = ckpt_dir / BAD_CKPT, ckpt_dir / BAD_CKPT_ARCHIVED
    if archived.exists():
        print(f"{archived.name} already exists — archive step previously done.")
        if bad.exists():
            raise RuntimeError(
                f"BOTH {bad.name} and {archived.name} exist — a retrain may have already "
                f"written a new epoch_8.pth; NOT touching anything, resolve manually."
            )
    elif bad.exists():
        bad.rename(archived)
        print(f"{bad.name} -> {archived.name} (the alpha^2 epoch is out of RESUME's sight).")
    else:
        print(f"{bad.name} not found — nothing to archive (fresh setup?).")
    report("ckpt_archive", "done")


def main() -> None:
    stage0_env()
    stage_tar_fetch()
    category_by_stem, exclude_stems = stage_categories()
    stage_bokeh_regen(category_by_stem, exclude_stems)
    stage_export()
    stage_drive_copy()
    stage_refresh_marker()
    stage_archive_bad_checkpoint()
    report("ALL", "done")
    # CRITICAL (2026-07-12 lesson): Drive writes are buffered asynchronously.
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
