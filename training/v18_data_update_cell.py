"""V18 DATA CELL — limb + atmosphere lesson pools (spec
docs/superpowers/specs/2026-07-29-v18-limb-atmosphere.md). Single cell on a
FREE CPU Colab session (~1.5h; no font pool and no train-tar extraction —
neither generator needs backgrounds).

Pools produced (deltas shipped tar-first, then loose):
- data/train_limb        4000  limb_*  (page-colored stroked limbs attached
  to a body's silhouette edge — the YH gloves lesson)
- data/train_atmosphere  4000  atmo_*  (halftone/wash smoke + glow bursts,
  GT = DENSITY, not the dot mask — the CHEESE lesson)
- data/testset_limb        24  probe seeds 993 (held-out)
- data/testset_atmosphere  24  probe seeds 994 (held-out)
  (both probe sets also uploaded to HF egeorcun/lucida-eval probe/)

Rerun-safe by content: sidecar `tar/_v18_packed.json`.

Paste-run: exec(open('/content/my-bg-remover/training/v18_data_update_cell.py').read())
"""
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request  # noqa: F401  (kept for parity with sibling cells)
from datetime import datetime, timezone
from pathlib import Path

import PIL.Image

PIL.Image.MAX_IMAGE_PIXELS = None

WORKDIR = "/content/my-bg-remover"
DRIVE_ROOT = "/content/drive/MyDrive"
DRIVE_OUTPUT_SUBDIR = "bg-remover-data"
DRIVE_STATUS_SUBDIR = "bg-remover-status"
TAR_SUBDIR = "tar"

SVG_DIR = Path("/content/openclipart_svgs")
SVG_SAMPLE = 20000

LIMB_OUT = Path("data/train_limb");             LIMB_COUNT = 4000; LIMB_SEED = 111
ATMO_OUT = Path("data/train_atmosphere");       ATMO_COUNT = 4000; ATMO_SEED = 88
PROBE_LIMB = Path("data/testset_limb")
PROBE_ATMO = Path("data/testset_atmosphere")
EXPORT_DIR = "/content/birefnet_format_v18"

STATUS_DIR = Path(DRIVE_ROOT) / DRIVE_STATUS_SUBDIR
LOG_PATH = STATUS_DIR / "log.txt"

SCRIPTS_DIR = str(Path(WORKDIR) / "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import training.train_colab_lib as tcl  # noqa: E402


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def report(stage: str, status: str, **extra) -> None:
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{_now()}] stage={stage} status={status}"
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
    for name in ("make_clipart", "make_atmosphere", "export_birefnet"):
        sys.modules.pop(name, None)
    import importlib
    importlib.reload(tcl)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "datasets", "cairosvg"], check=True)
    import cairosvg  # noqa: F401
    free_gb = shutil.disk_usage("/content").free / 1e9
    print(f"local free disk: {free_gb:.0f} GB (~15 GB needed)")
    report("env", "done", free_gb=round(free_gb, 1))


def stage_openclipart() -> int:
    report("openclipart", "running")
    if SVG_DIR.is_dir() and _n_files(SVG_DIR) >= SVG_SAMPLE * 0.5:
        print(f"openclipart SKIPPED: {_n_files(SVG_DIR)} svgs local.")
        report("openclipart", "done", svgs=_n_files(SVG_DIR), skipped=True)
        return _n_files(SVG_DIR)
    from datasets import load_dataset
    SVG_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("nyuuzyou/openclipart", split="train", streaming=True)
    svg_field, written = None, 0
    for i, ex in enumerate(ds):
        if svg_field is None:
            for k, v in ex.items():
                s = v.decode("utf-8", "ignore") if isinstance(v, bytes) else v if isinstance(v, str) else ""
                if "<svg" in s[:4000]:
                    svg_field = k
                    print(f"svg field detected: {k!r}")
                    break
            assert svg_field is not None, f"no svg field in keys {list(ex.keys())}"
        v = ex[svg_field]
        s = v.decode("utf-8", "ignore") if isinstance(v, bytes) else v
        if "<svg" not in s[:4000]:
            continue
        (SVG_DIR / f"oc_{i:06d}.svg").write_text(s)
        written += 1
        if written % 2500 == 0:
            print(f"openclipart: {written}/{SVG_SAMPLE}")
        if written >= SVG_SAMPLE:
            break
    assert written >= 1000
    report("openclipart", "done", svgs=written)
    return written


def stage_limb() -> int:
    report("limb", "running")
    import make_clipart as mcl
    n = mcl.run(LIMB_OUT, SVG_DIR, count=LIMB_COUNT, seed=LIMB_SEED,
                limb=True, stem_prefix="limb_", category="limb")
    assert n >= 500
    report("limb", "done", pairs=n)
    return n


def stage_atmosphere() -> int:
    report("atmosphere", "running")
    import make_atmosphere as matm
    n = matm.run(ATMO_OUT, SVG_DIR, count=ATMO_COUNT, seed=ATMO_SEED)
    assert n >= 500
    report("atmosphere", "done", pairs=n)
    return n


def stage_probe_sets() -> int:
    """2x24 held-out pairs (seeds 993/994) — the v18 smoke-gate metric sets;
    uploaded to HF (egeorcun/lucida-eval, probe/) when the token exists."""
    report("probe", "running")
    import make_clipart as mcl
    import make_atmosphere as matm
    svg_paths = [p for p in sorted(SVG_DIR.rglob("*.svg"))[:2000] if mcl._svg_usable(p)]
    total = 0
    for out_dir, prefix, seed, fn, cat in [
            (PROBE_LIMB, "probe_limb_", 993, mcl.render_limb_sample, "limb"),
            (PROBE_ATMO, "probe_atmo_", 994, matm.render_atmosphere_sample, "atmosphere")]:
        (out_dir / "im").mkdir(parents=True, exist_ok=True)
        (out_dir / "gt").mkdir(parents=True, exist_ok=True)
        rows = []
        made = i = 0
        while made < 24 and i < 200:
            stem = f"{prefix}{i:03d}"
            i += 1
            im_p, gt_p = out_dir / "im" / f"{stem}.jpg", out_dir / "gt" / f"{stem}.png"
            if im_p.exists() and gt_p.exists():
                rows.append({"id": stem, "image": str(im_p), "category": cat, "gt_alpha": str(gt_p)})
                made += 1
                continue
            rng = mcl._item_rng(seed, stem)
            rgb, gt = fn(rng, svg_paths)
            if (gt > 0.5).mean() < 0.01:
                continue
            mcl._save_pair(rgb, gt, im_p, gt_p)
            rows.append({"id": stem, "image": str(im_p), "category": cat, "gt_alpha": str(gt_p)})
            made += 1
        with open(out_dir / "manifest.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        total += len(rows)
        try:
            from google.colab import userdata
            os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
            from huggingface_hub import HfApi
            HfApi().upload_folder(folder_path=str(out_dir),
                                  path_in_repo=f"probe/{out_dir.name}",
                                  repo_id="egeorcun/lucida-eval", repo_type="dataset")
            print(f"{out_dir.name} uploaded to HF")
        except Exception as e:
            print(f"WARNING: {out_dir.name} HF upload skipped ({e})")
    report("probe", "done", pairs=total)
    return total


def _full_manifest(out_dir: Path) -> Path:
    rows = [json.loads(l) for l in (out_dir / "manifest.jsonl").read_text().splitlines() if l.strip()]
    full = out_dir / "manifest_full.jsonl"
    with open(full, "w") as f:
        for r in rows:
            if "image" in r:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                continue
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
    pools = [LIMB_OUT, ATMO_OUT]
    for pool in pools:
        full = _full_manifest(pool)
        stats = eb.export(manifest_path=str(full), out_dir=EXPORT_DIR, split_name="TRAIN")
        print(pool.name, "->", stats["total"])
    report("export", "done")

    src_im = Path(EXPORT_DIR) / "TRAIN" / "im"
    src_gt = Path(EXPORT_DIR) / "TRAIN" / "gt"
    stems = sorted(p.stem for p in src_im.iterdir())

    report("tar_pack", "running")
    tar_dir = Path(DRIVE_ROOT) / DRIVE_OUTPUT_SUBDIR / TAR_SUBDIR
    mp = tar_dir / "_manifest.json"
    marker_p = tar_dir / "_v18_packed.json"
    packed = json.loads(marker_p.read_text()) if marker_p.exists() else {}
    manifest = json.loads(mp.read_text())
    if packed.get("v18_pools"):
        print(f"tar_pack SKIPPED (marker): {packed}")
    else:
        SHARD = 7000
        for j in range(0, len(stems), SHARD):
            chunk = stems[j:j + SHARD]
            name = tcl.tar_shard_name(len(manifest["shards"]))
            local_tar = Path("/content") / name
            with tarfile.open(local_tar, "w") as tf:
                for s in chunk:
                    tf.add(src_im / f"{s}.jpg", arcname=f"im/{s}.jpg")
                    tf.add(src_gt / f"{s}.png", arcname=f"gt/{s}.png")
            nbytes = local_tar.stat().st_size
            shutil.copy2(local_tar, tar_dir / name)
            assert (tar_dir / name).stat().st_size == nbytes, f"{name}: short copy"
            local_tar.unlink()
            manifest["shards"].append({"name": name, "pairs": len(chunk), "bytes": nbytes})
            manifest["total_pairs"] = sum(sh["pairs"] for sh in manifest["shards"])
            mp.write_text(json.dumps(manifest, indent=2))
            print(f"{name}: {len(chunk)} pairs, {nbytes / 1e9:.2f} GB -> Drive")
        marker_p.write_text(json.dumps({"v18_pools": [p.name for p in pools],
                                        "stems": len(stems), "time": _now()}))
    tcl.validate_tar_manifest(manifest)
    report("tar_pack", "done", total_pairs=manifest["total_pairs"])

    report("drive_copy", "running")
    dst = Path(DRIVE_ROOT) / DRIVE_OUTPUT_SUBDIR
    n_copied = tcl.copy_pairs(stems, src_im, src_gt,
                              dst / "TRAIN" / "im", dst / "TRAIN" / "gt")
    n_rows = 0
    for pool in pools:
        n_rows += tcl.merge_composite_manifest(pool / "manifest_full.jsonl",
                                               dst / "train_composites_manifest.jsonl")
    print(f"copy_pairs: {n_copied} copied; manifest +{n_rows} rows")
    report("drive_copy", "done", copied=n_copied, manifest_rows=n_rows)


def main() -> None:
    stage0_env()
    stage_openclipart()
    stage_limb()
    stage_atmosphere()
    stage_probe_sets()
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
