"""V16b DATA CELL — outlined-typography extension (3000 new pairs) for the
Lucida Design stage-2 run. Single cell on a FREE CPU Colab session.

WHY (2026-07-25, the real Stay Fresh artwork): when a letter's fill color
EQUALS the page color (cream on cream), every stage-1 candidate hollows the
fill — no training sample ever taught "dark stroke + fill==page -> keep the
fill". scripts/make_typography.py gained an outline menu (commit 3c415f3);
this cell extends data/train_typography from 6000 to 9000 pairs where stems
>= 6000 draw from that menu, then ships ONLY the 3000 new pairs to Drive
(delta tar shard + loose copy + manifest merge).

Scaffolding (report/log, tar fetch, fonts, flush) follows
training/v16_data_update_cell.py line-for-line; stages already covered by
the v16 run (clipart, design_real, openclipart) are deliberately absent.
The first 6000 stems regenerate bit-identically on this fresh VM (outline
menu off consumes no extra rng draws) and are NOT re-uploaded.
"""
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import PIL.Image

PIL.Image.MAX_IMAGE_PIXELS = None

WORKDIR = "/content/my-bg-remover"
DRIVE_ROOT = "/content/drive/MyDrive"
DRIVE_OUTPUT_SUBDIR = "bg-remover-data"
DRIVE_STATUS_SUBDIR = "bg-remover-status"
TAR_SUBDIR = "tar"

LOCAL_TRAIN_ROOT = Path("/content/v16_train_src")
TAR_CACHE = Path("/content/tar_cache_v16")
FONT_DIR = Path("/content/fonts")
TYPO_OUT = Path("data/train_typography")
TYPO_COUNT = 9000
TYPO_SEED = 21
OUTLINE_FROM = 6000
EXPORT_DIR_B = "/content/birefnet_format_v16b"

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


def _n_files(d: Path) -> int:
    return sum(1 for p in d.iterdir() if p.is_file()) if d.is_dir() else 0


def stage0_env() -> None:
    from google.colab import drive
    drive.mount("/content/drive")
    assert Path(DRIVE_ROOT).is_dir()
    report("env", "running")
    os.chdir(WORKDIR)
    r = subprocess.run(["git", "-C", WORKDIR, "pull", "--ff-only"],
                       capture_output=True, text=True, timeout=180)
    print(f"git pull: rc={r.returncode} {r.stdout.strip() or r.stderr.strip()}")
    for name in ("make_typography", "export_birefnet"):
        sys.modules.pop(name, None)
    import importlib
    importlib.reload(tcl)
    free_gb = shutil.disk_usage("/content").free / 1e9
    print(f"local free disk: {free_gb:.0f} GB (~45 GB needed: tar extract + pairs)")
    report("env", "done", free_gb=round(free_gb, 1))


def stage_tar_fetch() -> int:
    """Local TRAIN extraction — the typography generator's photo-bg pool."""
    report("tar_fetch", "running")
    tar_dir = Path(DRIVE_ROOT) / DRIVE_OUTPUT_SUBDIR / TAR_SUBDIR
    manifest = json.loads((tar_dir / "_manifest.json").read_text())
    total = tcl.validate_tar_manifest(manifest)
    local_im, local_gt = LOCAL_TRAIN_ROOT / "im", LOCAL_TRAIN_ROOT / "gt"
    n_im = _n_files(local_im)
    if n_im >= total and n_im == _n_files(local_gt):
        print(f"tar SKIPPED: {n_im} pairs local.")
    else:
        LOCAL_TRAIN_ROOT.mkdir(parents=True, exist_ok=True)
        TAR_CACHE.mkdir(parents=True, exist_ok=True)
        for sh in manifest["shards"]:
            src, dst = tar_dir / sh["name"], TAR_CACHE / sh["name"]
            if not (dst.exists() and dst.stat().st_size == sh["bytes"]):
                shutil.copy2(src, dst)
                assert dst.stat().st_size == sh["bytes"], f"{sh['name']} short copy"
            with tarfile.open(dst) as tf:
                tf.extractall(LOCAL_TRAIN_ROOT, filter="data")
            dst.unlink()
            print(f"{sh['name']}: extracted ({sh['pairs']} pairs).")
        n_im = _n_files(local_im)
    report("tar_fetch", "done", pairs=n_im)
    return n_im


# Same 60-font OFL list as v16 (404s tolerated); reuse the v16 module's list
# by importing it WITHOUT running main() is not possible (exec-style file),
# so the list is fetched from the v16 file at runtime — single source.
def stage_fonts() -> int:
    report("fonts", "running")
    FONT_DIR.mkdir(parents=True, exist_ok=True)
    v16 = Path(WORKDIR) / "training" / "v16_data_update_cell.py"
    ns: dict = {}
    src = v16.read_text()
    marker = "GOOGLE_FONT_PATHS = ["
    start = src.index(marker)
    end = src.index("]", start) + 1
    exec(src[start:end], ns)  # only the list literal
    paths = ns["GOOGLE_FONT_PATHS"]
    raw = "https://raw.githubusercontent.com/google/fonts/main/"
    done = failed = 0
    for rel in paths:
        dst = FONT_DIR / rel.split("/")[-1]
        if dst.exists():
            done += 1
            continue
        try:
            urllib.request.urlretrieve(raw + rel, dst)
            done += 1
        except Exception as e:
            failed += 1
            print(f"WARNING: font {rel}: {e}")
    print(f"{FONT_DIR}: {done} fonts ({failed} failed).")
    report("fonts", "done", total=done, failed=failed)
    return done


def stage_typography() -> int:
    report("typography", "running")
    import make_typography as mty
    n = mty.run(TYPO_OUT, FONT_DIR, [LOCAL_TRAIN_ROOT / "im"],
                count=TYPO_COUNT, seed=TYPO_SEED, outline_from=OUTLINE_FROM)
    assert n >= TYPO_COUNT, f"only {n} typography pairs"
    report("typography", "done", pairs=n)
    return n


def _full_manifest(out_dir: Path) -> Path:
    rows = [json.loads(l) for l in (out_dir / "manifest.jsonl").read_text().splitlines() if l.strip()]
    full = out_dir / "manifest_full.jsonl"
    with open(full, "w") as f:
        for r in rows:
            im_p = out_dir / "im" / f"{r['id']}.jpg"
            gt_p = out_dir / "gt" / f"{r['id']}.png"
            assert im_p.exists() and gt_p.exists(), r["id"]
            f.write(json.dumps({"id": r["id"], "image": str(im_p),
                                "category": r["category"], "gt_alpha": str(gt_p)},
                               ensure_ascii=False) + "\n")
    return full


def stage_export_and_delta() -> None:
    report("export", "running")
    import export_birefnet as eb
    full = _full_manifest(TYPO_OUT)
    stats = eb.export(manifest_path=str(full), out_dir=EXPORT_DIR_B, split_name="TRAIN")
    print("typography ->", stats["total"])
    report("export", "done")

    src_im = Path(EXPORT_DIR_B) / "TRAIN" / "im"
    src_gt = Path(EXPORT_DIR_B) / "TRAIN" / "gt"
    new_stems = sorted(p.stem for p in src_im.iterdir()
                       if int(p.stem.split("_")[1]) >= OUTLINE_FROM)
    assert len(new_stems) == TYPO_COUNT - OUTLINE_FROM, f"{len(new_stems)} new stems?"

    # 1) delta tar shard FIRST (large sequential writes are the reliable op)
    report("tar_pack", "running")
    tar_dir = Path(DRIVE_ROOT) / DRIVE_OUTPUT_SUBDIR / TAR_SUBDIR
    mp = tar_dir / "_manifest.json"
    manifest = json.loads(mp.read_text())
    name = tcl.tar_shard_name(len(manifest["shards"]))
    if name in {sh["name"] for sh in manifest["shards"]}:
        print(f"{name}: already in manifest, skipped.")
    else:
        local_tar = Path("/content") / name
        with tarfile.open(local_tar, "w") as tf:
            for s in new_stems:
                tf.add(src_im / f"{s}.jpg", arcname=f"im/{s}.jpg")
                tf.add(src_gt / f"{s}.png", arcname=f"gt/{s}.png")
        nbytes = local_tar.stat().st_size
        shutil.copy2(local_tar, tar_dir / name)
        assert (tar_dir / name).stat().st_size == nbytes, f"{name}: short copy"
        local_tar.unlink()
        manifest["shards"].append({"name": name, "pairs": len(new_stems), "bytes": nbytes})
        manifest["total_pairs"] = sum(sh["pairs"] for sh in manifest["shards"])
        mp.write_text(json.dumps(manifest, indent=2))
        print(f"{name}: {len(new_stems)} pairs, {nbytes / 1e9:.2f} GB -> Drive")
    tcl.validate_tar_manifest(manifest)
    report("tar_pack", "done", total_pairs=manifest["total_pairs"])

    # 2) loose copy of the NEW stems only (the notebook's all_stems listing
    #    is fed by the loose dir)
    report("drive_copy", "running")
    dst = Path(DRIVE_ROOT) / DRIVE_OUTPUT_SUBDIR
    n_copied = tcl.copy_pairs(new_stems, src_im, src_gt,
                              dst / "TRAIN" / "im", dst / "TRAIN" / "gt")
    n_rows = tcl.merge_composite_manifest(TYPO_OUT / "manifest_full.jsonl",
                                          dst / "train_composites_manifest.jsonl")
    print(f"copy_pairs: {n_copied} copied; manifest +{n_rows} rows")
    report("drive_copy", "done", copied=n_copied, manifest_rows=n_rows)


def main() -> None:
    stage0_env()
    stage_tar_fetch()
    stage_fonts()
    stage_typography()
    stage_export_and_delta()
    report("ALL", "done")
    print("Flushing Drive (waiting for async writes)...")
    from google.colab import drive as _gdrive
    _gdrive.flush_and_unmount()
    print("Drive flush COMPLETE — the VM can now be safely shut down/swapped.")


try:
    main()
except KeyboardInterrupt:
    raise
