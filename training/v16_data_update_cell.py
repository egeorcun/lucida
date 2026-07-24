"""V16 DATA UPDATE CELL — Lucida Design training data, in a fresh (FREE, CPU)
Colab session. Spec: docs/superpowers/specs/2026-07-24-lucida-design-expert.md.

Builds and merges into Drive:
1. **typography (~6k)** — scripts/make_typography.py: real-font display text,
   letter counters exact GT=0, white-on-white backgrounds included; photo
   backgrounds come from the tar-extracted local TRAIN pool.
2. **clipart (~5k)** — scripts/make_clipart.py: CC0 OpenClipart SVGs rendered
   to pixel-exact alpha, white backgrounds over-represented.
3. **design_real expansion** — Crello train split to 10,500 pairs (the 8,000
   already on Drive re-render bit-identically and are skipped by the
   size-checked merge) + validation split up to 1,000 more.

Flow contracts are the v14 cell lineage verbatim (report()/log.txt, Errno 5
retry, module-cache purge after git pull, threaded copy_pairs merge, flush
at the very end). NO tar repacking; the training notebook's delta path
fetches the new stems.
"""

import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
import urllib.parse
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

LOCAL_TRAIN_ROOT = Path("/content/v16_train_src")   # tar extraction (typography bg pool)
TAR_CACHE = Path("/content/tar_cache_v16")
FONT_DIR = Path("/content/fonts")
SVG_DIR = Path("/content/openclipart_svgs")
TYPO_OUT = Path("data/train_typography");   TYPO_COUNT = 6000;  TYPO_SEED = 21
CLIP_OUT = Path("data/train_clipart");      CLIP_COUNT = 5000;  CLIP_SEED = 33
DR_OUT = Path("data/train_design_real");    DR_COUNT = 10500;   DR_SEED = 7
DRV_OUT = Path("data/train_design_real_val"); DRV_COUNT = 1000
EXPORT_DIR = "/content/birefnet_format_v16"
SVG_SAMPLE = 20000

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


def _n_files(d: Path) -> int:
    return sum(1 for p in d.iterdir() if p.is_file()) if d.is_dir() else 0


def _git_pull_idempotent() -> None:
    try:
        r = subprocess.run(["git", "-C", WORKDIR, "pull", "--ff-only"],
                           capture_output=True, text=True, timeout=180)
        print(f"git pull: rc={r.returncode} {r.stdout.strip() or r.stderr.strip()}")
    except Exception as e:
        print(f"WARNING: git pull failed ({e}).")


def _purge_script_module_cache() -> None:
    """2026-07-21 lesson: sys.modules keeps pre-pull modules across cell runs."""
    import importlib
    for name in ("make_typography", "make_clipart", "make_design_real",
                 "make_bokeh_copies", "make_design", "make_textfx",
                 "export_birefnet", "build_testset", "build_trainset"):
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
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "datasets", "cairosvg"], check=True)
    import cairosvg  # noqa: F401  (fails loudly here, not mid-generation)
    free_gb = shutil.disk_usage("/content").free / 1e9
    print(f"local free disk: {free_gb:.0f} GB (~60 GB needed: tar + Crello + pairs)")
    report("env", "done", free_gb=round(free_gb, 1))


def stage_tar_fetch() -> int:
    """Local TRAIN extraction — the typography generator's photo-background pool."""
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


# ~60 OFL display fonts; 404s are tolerated (warn + continue, DejaVu fallback).
_GF_RAW = "https://raw.githubusercontent.com/google/fonts/main/"
GOOGLE_FONT_PATHS = [
    "ofl/anton/Anton-Regular.ttf", "ofl/bebasneue/BebasNeue-Regular.ttf",
    "ofl/lobster/Lobster-Regular.ttf", "ofl/pacifico/Pacifico-Regular.ttf",
    "ofl/bangers/Bangers-Regular.ttf", "ofl/righteous/Righteous-Regular.ttf",
    "ofl/abrilfatface/AbrilFatface-Regular.ttf", "ofl/alfaslabone/AlfaSlabOne-Regular.ttf",
    "ofl/archivoblack/ArchivoBlack-Regular.ttf", "ofl/shrikhand/Shrikhand-Regular.ttf",
    "ofl/staatliches/Staatliches-Regular.ttf", "ofl/monoton/Monoton-Regular.ttf",
    "ofl/pressstart2p/PressStart2P-Regular.ttf", "ofl/caveat/Caveat[wght].ttf",
    "ofl/dancingscript/DancingScript[wght].ttf", "ofl/oswald/Oswald[wght].ttf",
    "ofl/montserrat/Montserrat[wght].ttf", "ofl/playfairdisplay/PlayfairDisplay[wght].ttf",
    "ofl/orbitron/Orbitron[wght].ttf", "ofl/bungee/Bungee-Regular.ttf",
    "ofl/bungeeshade/BungeeShade-Regular.ttf", "ofl/blackopsone/BlackOpsOne-Regular.ttf",
    "ofl/titanone/TitanOne-Regular.ttf", "ofl/luckiestguy/LuckiestGuy-Regular.ttf",
    "ofl/fredoka/Fredoka[wdth,wght].ttf", "ofl/lilitaone/LilitaOne-Regular.ttf",
    "ofl/passionone/PassionOne-Regular.ttf", "ofl/rubikmonoone/RubikMonoOne-Regular.ttf",
    "ofl/sigmarone/SigmarOne-Regular.ttf", "ofl/ultra/Ultra-Regular.ttf",
    "ofl/carterone/CarterOne-Regular.ttf", "ofl/changaone/ChangaOne-Regular.ttf",
    "ofl/chewy/Chewy-Regular.ttf", "ofl/concertone/ConcertOne-Regular.ttf",
    "ofl/fugazone/FugazOne-Regular.ttf", "ofl/kalam/Kalam-Bold.ttf",
    "ofl/knewave/Knewave-Regular.ttf", "ofl/modak/Modak-Regular.ttf",
    "ofl/paytoneone/PaytoneOne-Regular.ttf", "ofl/ranchers/Ranchers-Regular.ttf",
    "ofl/rammettoone/RammettoOne-Regular.ttf", "ofl/secularone/SecularOne-Regular.ttf",
    "ofl/sniglet/Sniglet-Regular.ttf", "ofl/sniglet/Sniglet-ExtraBold.ttf",
    "ofl/spicyrice/SpicyRice-Regular.ttf", "ofl/squadaone/SquadaOne-Regular.ttf",
    "apache/oleoscript/OleoScript-Regular.ttf", "ofl/yellowtail/Yellowtail-Regular.ttf",
    "ofl/lobstertwo/LobsterTwo-Bold.ttf", "ofl/kaushanscript/KaushanScript-Regular.ttf",
    "ofl/permanentmarker/PermanentMarker.ttf", "ofl/creepster/Creepster-Regular.ttf",
    "ofl/nosifer/Nosifer-Regular.ttf", "ofl/frijole/Frijole-Regular.ttf",
    "ofl/graduate/Graduate-Regular.ttf", "ofl/bowlbyone/BowlbyOne-Regular.ttf",
    "ofl/bowlbyonesc/BowlbyOneSC-Regular.ttf", "ofl/racingsansone/RacingSansOne-Regular.ttf",
    "ofl/fasterone/FasterOne-Regular.ttf", "ofl/ericaone/EricaOne-Regular.ttf",
]
DEJAVU_GLOBS = ["/usr/share/fonts/truetype/dejavu/DejaVu*.ttf", "/usr/share/fonts/TTF/DejaVu*.ttf"]


def stage_fonts() -> int:
    report("fonts", "running")
    FONT_DIR.mkdir(parents=True, exist_ok=True)
    ok, failed = 0, 0
    for rel in GOOGLE_FONT_PATHS:
        target = FONT_DIR / Path(rel).name.replace("[", "_").replace("]", "_")
        if target.exists() and target.stat().st_size > 0:
            ok += 1
            continue
        try:
            with urllib.request.urlopen(_GF_RAW + urllib.parse.quote(rel), timeout=60) as resp:
                data = resp.read()
            assert data[:4] in (b"\x00\x01\x00\x00", b"OTTO", b"true"), "not a font"
            target.write_bytes(data)
            ok += 1
        except Exception as e:
            failed += 1
            print(f"WARNING: font {rel}: {e}")
    if ok < 5:
        import glob as _glob
        for pattern in DEJAVU_GLOBS:
            for p in _glob.glob(pattern):
                dst = FONT_DIR / Path(p).name
                if not dst.exists():
                    shutil.copy2(p, dst)
    total = len([p for p in FONT_DIR.iterdir() if p.suffix.lower() in {".ttf", ".otf"}])
    assert total > 0, "no fonts at all"
    print(f"/content/fonts: {total} fonts ({ok} downloaded, {failed} failed).")
    report("fonts", "done", total=total, failed=failed)
    return total


def stage_typography() -> int:
    report("typography", "running")
    import make_typography as mty
    n = mty.run(TYPO_OUT, FONT_DIR, bg_pool_dirs=[LOCAL_TRAIN_ROOT / "im"],
                count=TYPO_COUNT, seed=TYPO_SEED)
    assert n >= 500, f"only {n} typography pairs"
    report("typography", "done", pairs=n)
    return n


def stage_openclipart() -> int:
    """Samples SVG_SAMPLE files from nyuuzyou/openclipart into SVG_DIR.
    Schema-defensive: takes the first field whose value contains '<svg'."""
    report("openclipart", "running")
    if _n_files(SVG_DIR) >= SVG_SAMPLE * 0.5:
        print(f"openclipart SKIPPED: {_n_files(SVG_DIR)} svgs local.")
        report("openclipart", "done", svgs=_n_files(SVG_DIR), skipped=True)
        return _n_files(SVG_DIR)
    from datasets import load_dataset
    SVG_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("nyuuzyou/openclipart", split="train", streaming=True)
    svg_field = None
    written = 0
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
    assert written >= 1000, f"only {written} svgs"
    report("openclipart", "done", svgs=written)
    return written


def stage_clipart() -> int:
    report("clipart", "running")
    import make_clipart as mcl
    n = mcl.run(CLIP_OUT, SVG_DIR, count=CLIP_COUNT, seed=CLIP_SEED)
    assert n >= 500, f"only {n} clipart pairs"
    report("clipart", "done", pairs=n)
    return n


def stage_design_real() -> tuple[int, int]:
    report("design_real", "running")
    import make_design_real as mdr
    n_train = mdr.run(DR_OUT, count=DR_COUNT, seed=DR_SEED, split="train", quality=92)
    n_val = mdr.run(DRV_OUT, count=DRV_COUNT, seed=DR_SEED, split="validation", quality=92)
    assert n_train >= 8000, f"train expansion too small: {n_train}"
    report("design_real", "done", train=n_train, val=n_val)
    return n_train, n_val


def _full_manifest(out_dir: Path) -> Path:
    """Raw {id,category} manifest -> testset-schema manifest_full.jsonl."""
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


def stage_export_and_merge() -> None:
    report("export", "running")
    import export_birefnet as eb
    pools = [TYPO_OUT, CLIP_OUT, DR_OUT, DRV_OUT]
    for pool in pools:
        # design_real dirs already carry full-schema manifests; normalize all
        rows0 = [json.loads(l) for l in (pool / "manifest.jsonl").read_text().splitlines() if l.strip()]
        if "image" in rows0[0]:
            full = pool / "manifest_full.jsonl"
            shutil.copy(pool / "manifest.jsonl", full)
        else:
            full = _full_manifest(pool)
        stats = eb.export(manifest_path=str(full), out_dir=EXPORT_DIR, split_name="TRAIN")
        print(pool.name, "->", stats["total"])
    report("export", "done")

    report("drive_copy", "running")
    src = Path(EXPORT_DIR)
    dst = Path(DRIVE_ROOT) / DRIVE_OUTPUT_SUBDIR
    dst_im, dst_gt = dst / "TRAIN" / "im", dst / "TRAIN" / "gt"
    stems = sorted(p.stem for p in (src / "TRAIN" / "im").iterdir())
    pre = len(_listdir_retry(dst_im))
    n_copied = tcl.copy_pairs(stems, src / "TRAIN" / "im", src / "TRAIN" / "gt", dst_im, dst_gt)
    post_im, post_gt = len(_listdir_retry(dst_im)), len(_listdir_retry(dst_gt))
    print(f"copy_pairs: {n_copied} copied; Drive TRAIN {pre} -> {post_im}")
    assert post_im == post_gt
    assert pre <= post_im <= pre + len(stems)
    n_rows = 0
    for pool in pools:
        n_rows += tcl.merge_composite_manifest(pool / "manifest_full.jsonl",
                                               dst / "train_composites_manifest.jsonl")
    print(f"manifest: +{n_rows} rows")
    report("drive_copy", "done", copied=n_copied, manifest_rows=n_rows, total_im=post_im)


def main() -> None:
    stage0_env()
    stage_tar_fetch()
    stage_fonts()
    stage_typography()
    stage_openclipart()
    stage_clipart()
    stage_design_real()
    stage_export_and_merge()
    report("ALL", "done")
    print("Flushing Drive (waiting for async writes)...")
    from google.colab import drive as _gdrive
    _gdrive.flush_and_unmount()
    print("Drive flush COMPLETE — the VM can now be safely shut down/swapped.")


try:
    main()
except Exception:
    report("FATAL", "error", traceback=traceback.format_exc())
    raise
