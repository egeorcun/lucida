"""V17 DATA CELL — manufactured fill==background ambiguity pools (spec
docs/superpowers/specs/2026-07-26-fill-eq-bg-definitive-fix.md). Single cell
on a FREE CPU Colab session (~3.5h).

Pools produced (all deltas shipped tar-first, then loose):
- data/train_typography_outline  6000  typo2_*   (outline menu from index 0)
- data/train_clipart2            3000  clip2_*   (stroked stickers, fill==page .5)
- data/train_design_real_ambig   4000  crello_ambig_* (REAL templates over
  element-colored pages — the closest generator to the failing artwork)
- data/testset_fill_eq_bg          24  probe_*   (held-out seeds; the
  fill_alpha/fill_hole probe; uploaded to HF egeorcun/lucida-eval as well)

Duplicate-shard protection: v16b's rerun packed the same stems twice because
resume keyed on the shard NAME. Here a sidecar `tar/_v17_packed.json` lists
the pools already packed — rerun-safe by content, not by name.
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
SVG_DIR = Path("/content/openclipart_svgs")
SVG_SAMPLE = 20000

TYPO2_OUT = Path("data/train_typography_outline"); TYPO2_COUNT = 6000; TYPO2_SEED = 55
CLIP2_OUT = Path("data/train_clipart2");           CLIP2_COUNT = 3000; CLIP2_SEED = 66
DRA_OUT = Path("data/train_design_real_ambig");    DRA_COUNT = 4000;   DRA_SEED = 77
PROBE_OUT = Path("data/testset_fill_eq_bg")
EXPORT_DIR = "/content/birefnet_format_v17"

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
    for name in ("make_typography", "make_clipart", "make_design_real", "export_birefnet"):
        sys.modules.pop(name, None)
    import importlib
    importlib.reload(tcl)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "datasets", "cairosvg"], check=True)
    import cairosvg  # noqa: F401
    free_gb = shutil.disk_usage("/content").free / 1e9
    print(f"local free disk: {free_gb:.0f} GB (~55 GB needed)")
    report("env", "done", free_gb=round(free_gb, 1))


def stage_tar_fetch() -> int:
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


def stage_fonts() -> int:
    report("fonts", "running")
    import ast
    FONT_DIR.mkdir(parents=True, exist_ok=True)
    v16 = Path(WORKDIR) / "training" / "v16_data_update_cell.py"
    paths = None
    for node in ast.walk(ast.parse(v16.read_text())):
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "GOOGLE_FONT_PATHS":
            paths = ast.literal_eval(node.value)
            break
    assert paths, "GOOGLE_FONT_PATHS not found"
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


def stage_typography_outline() -> int:
    report("typography_outline", "running")
    import make_typography as mty
    n = mty.run(TYPO2_OUT, FONT_DIR, [LOCAL_TRAIN_ROOT / "im"],
                count=TYPO2_COUNT, seed=TYPO2_SEED, outline_from=0,
                stem_prefix="typo2_", category="typography_outline")
    assert n >= TYPO2_COUNT
    report("typography_outline", "done", pairs=n)
    return n


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


def stage_clipart2() -> int:
    report("clipart2", "running")
    import make_clipart as mcl
    n = mcl.run(CLIP2_OUT, SVG_DIR, count=CLIP2_COUNT, seed=CLIP2_SEED,
                outline=True, stem_prefix="clip2_", category="clipart2")
    assert n >= 500
    report("clipart2", "done", pairs=n)
    return n


def stage_design_real_ambig() -> int:
    report("design_real_ambig", "running")
    import make_design_real as mdr
    n = mdr.run(DRA_OUT, count=DRA_COUNT, seed=DRA_SEED, split="train", ambig=True)
    assert n >= 500
    report("design_real_ambig", "done", pairs=n)
    return n


def stage_probe_set() -> int:
    """24 held-out fill==bg pairs, category fill_eq_bg — the smoke-gate
    metric set. Also uploaded to HF (egeorcun/lucida-eval, probe/) when the
    HF token is available (Colab Secrets); benchmark cells pull it there."""
    report("probe", "running")
    import numpy as np
    import make_clipart as mcl
    import make_typography as mty
    (PROBE_OUT / "im").mkdir(parents=True, exist_ok=True)
    (PROBE_OUT / "gt").mkdir(parents=True, exist_ok=True)
    font_paths = sorted(p for p in FONT_DIR.iterdir() if p.suffix.lower() in (".ttf", ".otf"))
    svg_paths = sorted(SVG_DIR.rglob("*.svg"))[:500]
    rows = []
    for i in range(12):
        stem = f"probe_typo_{i:02d}"
        im_p, gt_p = PROBE_OUT / "im" / f"{stem}.jpg", PROBE_OUT / "gt" / f"{stem}.png"
        rows.append({"id": stem, "image": str(im_p), "category": "fill_eq_bg", "gt_alpha": str(gt_p)})
        if im_p.exists() and gt_p.exists():
            continue
        base = mty._item_rng(991, stem)
        lr = np.random.default_rng(base.integers(0, 2**63))
        fr = np.random.default_rng(base.integers(0, 2**63))
        rgb, gt = mty.render_typography_sample(
            lr, fr, font_paths, bg_images=[], outline_menu=True,
            force_outline=True, force_fill_eq_bg=True, force_crop=False)
        mty._save_pair(rgb, gt, im_p, gt_p)
    for i in range(12):
        stem = f"probe_clip_{i:02d}"
        im_p, gt_p = PROBE_OUT / "im" / f"{stem}.jpg", PROBE_OUT / "gt" / f"{stem}.png"
        rows.append({"id": stem, "image": str(im_p), "category": "fill_eq_bg", "gt_alpha": str(gt_p)})
        if im_p.exists() and gt_p.exists():
            continue
        rng = mcl._item_rng(992, stem)
        rgb, gt = mcl.render_clipart_outline_sample(rng, svg_paths, force_fill_eq_bg=True)
        mcl._save_pair(rgb, gt, im_p, gt_p)
    with open(PROBE_OUT / "manifest.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    try:
        from google.colab import userdata
        os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
        from huggingface_hub import HfApi
        HfApi().upload_folder(folder_path=str(PROBE_OUT), path_in_repo="probe/testset_fill_eq_bg",
                              repo_id="egeorcun/lucida-eval", repo_type="dataset")
        print("probe set uploaded to HF")
    except Exception as e:
        print(f"WARNING: probe HF upload skipped ({e})")
    report("probe", "done", pairs=len(rows))
    return len(rows)


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
    pools = [TYPO2_OUT, CLIP2_OUT, DRA_OUT]
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
    marker_p = tar_dir / "_v17_packed.json"
    packed = json.loads(marker_p.read_text()) if marker_p.exists() else {}
    manifest = json.loads(mp.read_text())
    if packed.get("v17_pools"):
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
        marker_p.write_text(json.dumps({"v17_pools": [p.name for p in pools],
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
    stage_tar_fetch()
    stage_fonts()
    stage_typography_outline()
    stage_openclipart()
    stage_clipart2()
    stage_design_real_ambig()
    stage_probe_set()
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
