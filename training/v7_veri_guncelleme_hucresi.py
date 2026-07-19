"""V7 VERİ GÜNCELLEME HÜCRESİ — taze bir (ÜCRETSİZ, CPU yeterli — GPU GEREKMEZ)
Colab oturumunda, mevcut Drive veri setine (`bg-remover-data/TRAIN`) yalnız YENİ
`design` kategorisini ekler (GitHub issue #2: baskı-tasarımı/sticker tarzı
görsellerde — halftone, mürekkep dokusu, dumanlı kenarlar, beyaza eriyen
ışımalar; tişört tasarımları — model özneyi siliyor ya da hayaletleştiriyor).
Üretim mantığının tamamı `scripts/make_design.py`'de (birim testli); bu hücre
yalnız Colab akışını yönetir. HİÇBİR mevcut dosyayı silmez/üzerine yazmaz.

KAYNAK / ATIF: akış kalıbı (Drive mount HERŞEYDEN önce → `report()` stage
takibi → üretim → export → TRAIN-only Drive merge → `drive.flush_and_unmount()`)
`training/v6_veri_guncelleme_hucresi.py`'den; indirme fonksiyonları
(`_download_trans460` / `_gdown_extract` / `discover_him2k_dirs` /
`merge_him2k` / `_download_toonout` / `stage_fonts`) `training/
v4_veri_guncelleme_hucresi.py`'den KOPYALANDI (o dosyalar paste-run tasarımı
gereği import edilirken `main()` çalıştırdığından modül olarak import
EDİLEMEZ — drift görürseniz oradan güncelleyin). 2026-07-12 dersi AYNEN
geçerli: Drive yazımları asenkron tamponlanır, flush olmadan VM kapatılırsa
dosyalar SESSİZCE kaybolur.

V6'DAN FARKI — TAR FETCH TAMAMEN ATLANIR: design üretiminin öznesi tar'lardaki
KOMPOZİT (arka planlı) görüntüler DEĞİL, gerçek alpha'lı ham kesitlerdir
(tar'daki görseller kompozit — fg kaynağı olamaz). Bu yüzden hücre KÜÇÜK bir
indirme yapar (~3GB): Transparent-460 (fg/alpha) + HIM2K (merge) + ToonOut
(train split) + fontlar. Zemin tamamen sentetik (kağıt/pastel) olduğundan
BG-20k havuzu da İNDİRİLMEZ (v4'ten fark).

TRANS460 NORMALİZASYONU: Transparent-460 diskte `fg/` + `alpha/` düzeninde;
make_design (make_textfx kalıbıyla) `im/` + `gt/` bekler — `_normalize_
trans460_pairs` stem-eşleşmeli SYMLINK'lerle `trans460_pairs/{im,gt}` üretir
(kopya yok, disk maliyeti sıfır, idempotent).

VAL SIZINTI KORUMASI (v3 dersi, fg SEÇİMİNDE): yeni `design_*` stem'leri
tamamen sentetik olduğundan sızıntı riski yalnız fg kaynaklarındadır. VAL
stem'leri kompozit türevleri (`<kaynak_id>_v/oNN`) olduğundan v6'daki stem
bazlı exclude fg havuzuna uygulanAMAZ; bunun yerine v3/v4 kalıbı
(`tcl.derive_val_excluded_source_ids`) ile VAL kaynak id'leri türetilir ve
kompozit id sözleşmesi `f"{kaynak_adı}_{_sanitize(stem)}"` (bkz. scripts/
build_trainset.py) üzerinden ham fg stem'lerine geri eşlenir — eşleşen stem'ler
`make_design.run(exclude_fg_stems=...)` ile havuzdan çıkarılır. ToonOut
kaynakları eğitime yalnız `illustration_{idx}_c{NN}` indeks-stem'leriyle girdi
ve VAL stem'lerinden kaynak dosyaya geri eşlenemez; ToonOut test split'ine
zaten hiç dokunulmadığı için (v4 kuralı) ek koruma gerekmez.

ÖN KOŞULLAR: repo `/content/my-bg-remover`'da klonlanmış ve `pip install -e .`
yapılmış olmalı; Drive'da `bg-remover-data/TRAIN/{im,gt}` mevcut olmalı. Repo
GÜNCEL olmalı (env aşaması idempotent `git pull` dener): `scripts/
make_design.py` bu hücreyle aynı çalışmada eklendi — eski bir klonla koşarsanız
`stage_design` net bir Türkçe hata mesajıyla durur.

Durum takibi v6 hücresiyle AYNI mekanizma (`report()` ->
`bg-remover-status/log.txt` + `status.json`) — aşamalar: env, downloads,
fonts, val_guard, design, export, drive_copy, (bitişte) ALL.
"""

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import PIL.Image

# Transparent-460/HIM2K'da 100MP+ görseller var; PIL'in 179MP "decompression
# bomb" hata eşiği kaldırılır (bkz. v4 hücresi aynı satır).
PIL.Image.MAX_IMAGE_PIXELS = None

import numpy as np  # noqa: E402  (MAX_IMAGE_PIXELS PIL importundan SONRA gelmeli)
from PIL import Image  # noqa: E402

# --- Sabitler (v4/v6 hücreleriyle AYNI Drive yerleşimi) ---
WORKDIR = "/content/my-bg-remover"
DRIVE_ROOT = "/content/drive/MyDrive"
DRIVE_OUTPUT_SUBDIR = "bg-remover-data"
DRIVE_STATUS_SUBDIR = "bg-remover-status"
SEED = 42

# --- v7'ye özgü sabitler ---
RAW = Path("data/raw_train")
TOONOUT_HF_REPO = "joelseytre/toonout"
TOONOUT_DIR = Path("/content/downloads/toonout")   # normalize edilmiş im/ gt/ buraya
FONT_DIR = Path("/content/fonts")
TRANS460_PAIRS = RAW / "trans460_pairs"            # fg/alpha -> im/gt symlink köprüsü
DESIGN_OUT_DIR = Path("data/train_design")         # make_design.run() çıktısı (WORKDIR'e göre)
EXPORT_DIR = "/content/birefnet_format_design"     # export_birefnet.export() çıktısı
DESIGN_COUNT = 6000                                # design hedefi (~6k)

# kompozit id sözleşmesi: f"{kaynak_adı}_{_sanitize(ham_stem)}" (build_trainset)
# — VAL kaynak id'lerini ham fg stem'lerine geri eşlemek için havuz -> önek.
FG_SOURCE_PREFIXES = {
    "trans460_pairs": "transparent_460_train",
    "him2k_merged": "him2k",
}

STATUS_DIR = Path(DRIVE_ROOT) / DRIVE_STATUS_SUBDIR
LOG_PATH = STATUS_DIR / "log.txt"
STATUS_PATH = STATUS_DIR / "status.json"

# scripts/ bir paket değil — make_design/export_birefnet/build_testset'i import
# edebilmek için mutlak yolu sys.path'e ekliyoruz (bkz. v4/v6 hücreleri).
SCRIPTS_DIR = str(Path(WORKDIR) / "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from benchmark.testset import CATEGORIES  # noqa: E402  (pip install -e . ile kurulu paket)
import training.train_colab_lib as tcl  # noqa: E402


# ==========================================================================
# Durum raporlama — `v6_veri_guncelleme_hucresi.py::report`'la BİREBİR AYNI.
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


# ==========================================================================
# Drive FUSE Errno 5 koruması — v6 hücresindeki _listdir_retry kalıbının kopyası.
# ==========================================================================
def _listdir_retry(d: Path, attempts: int = 5, wait_s: int = 30) -> list[Path]:
    """Drive FUSE 50k+ dosyalı dizinlerde ara sıra geçici 'Errno 5 I/O error'
    verir (v3/v4 koşularında görüldü — tekrar denemek yetti); bekleyip yeniden
    dener, son denemede hatayı olduğu gibi yükseltir."""
    for i in range(attempts):
        try:
            return list(d.iterdir())
        except OSError as e:
            if i == attempts - 1:
                raise
            print(f"UYARI: {d} listelenirken {e} — {wait_s}s bekleyip yeniden denenecek "
                  f"({i + 1}/{attempts - 1}).")
            time.sleep(wait_s)
    raise AssertionError("unreachable")


def _count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*") if p.is_file())


# ==========================================================================
# Stage "env" — Drive bağlama (HERŞEYDEN önce, STATUS_DIR Drive'da!) + repo
# git pull (idempotent). Kaynak: v6 hücresi stage0_env — make_design bu
# hücreyle aynı çalışmada eklendiği için eski klon en olası hata kaynağı.
# ==========================================================================
def _git_pull_idempotent() -> None:
    """Repo'yu günceller — `git pull --ff-only` zaten günceldeyse no-op
    (idempotent); ağ yoksa/çakışma varsa UYARI verip devam eder (make_design
    eksikse stage_design zaten net mesajla durduracak)."""
    try:
        r = subprocess.run(
            ["git", "-C", WORKDIR, "pull", "--ff-only"],
            capture_output=True, text=True, timeout=180,
        )
        print(f"git pull: rc={r.returncode} {r.stdout.strip() or r.stderr.strip()}")
        if r.returncode != 0:
            print("UYARI: git pull başarısız — repo eski kalmış olabilir; make_design.py "
                  "eksikse aşağıda net hatayla durulacak.")
    except Exception as e:
        print(f"UYARI: git pull çalıştırılamadı ({e}) — mevcut klonla devam ediliyor.")


def _setup_hf_env() -> None:
    """Kaynak: v4 hücresi aynı fonksiyon (HF indirmeleri için)."""
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
    try:
        from google.colab import userdata

        token = userdata.get("HF_TOKEN")
        if token:
            os.environ["HF_TOKEN"] = token
            print("HF_TOKEN Colab Secrets'tan alındı.")
    except Exception as e:
        print(f"HF_TOKEN alınamadı (Secrets'ta yok veya erişim izni verilmedi): {e}")


def stage0_env() -> None:
    # Drive HERŞEYDEN ÖNCE bağlanır (report() dahil — STATUS_DIR Drive'da!);
    # drive.mount idempotenttir. Kaynak: v6 hücresi aynı aşama.
    from google.colab import drive

    drive.mount("/content/drive")
    assert Path(DRIVE_ROOT).is_dir(), f"Drive bağlanamadı: {DRIVE_ROOT} yok"

    report("env", "running")
    os.chdir(WORKDIR)
    _git_pull_idempotent()
    _setup_hf_env()

    free_gb = shutil.disk_usage("/content").free / 1e9
    print(f"lokal boş disk: {free_gb:.0f} GB (gerekli ~10 GB: ~3GB indirme + design çıktısı — "
          f"TAR FETCH YOK, v6'dan fark)")
    report("env", "done", cwd=str(Path.cwd()), free_gb=round(free_gb, 1))


# ==========================================================================
# Stage "downloads" — YALNIZ design'ın gerektirdiği fg kaynakları, İDEMPOTENT.
# Kaynak: v4_veri_guncelleme_hucresi.py (kopya; kökeni v3 hücresi). BG-20k
# İNDİRİLMEZ (zemin sentetik); tar fetch TAMAMEN ATLANIR (tar'lar kompozit —
# fg kaynağı olamaz, bkz. modül docstring'i).
# ==========================================================================
def _load_source_defs() -> dict:
    with open("data/train_sources.json") as f:
        return {s["name"]: s for s in json.load(f)["sources"]}


def _download_trans460(source_defs: dict) -> int:
    """Kaynak: v4 hücresi::_download_trans460 (kopya) — design fg kaynağı:
    fg/ + alpha/ (saydam objeler, gerçek alpha'lı kesitler)."""
    from huggingface_hub import snapshot_download

    spec = source_defs["transparent_460_train"]
    trans_out = RAW / "trans460_train"
    existing = len(list((trans_out / "fg").iterdir())) if (trans_out / "fg").exists() else 0
    expected = spec.get("full_pair_count") or 0
    if expected and existing >= 0.9 * expected:
        print(f"trans460_train: diskte zaten {existing} görsel (>= %90 x {expected}); indirme atlanıyor.")
        return existing

    trans_dir = snapshot_download(repo_id=spec["hf_repo"], repo_type="dataset", allow_patterns=["Train/*"])
    if trans_out.exists():
        shutil.rmtree(trans_out)
    shutil.copytree(Path(trans_dir) / "Train" / "fg", trans_out / "fg")
    shutil.copytree(Path(trans_dir) / "Train" / "alpha", trans_out / "alpha")
    total = len(list((trans_out / "fg").iterdir()))
    print(f"transparent_460_train: {total} görsel -> {trans_out}")
    return total


def _normalize_trans460_pairs() -> int:
    """Transparent-460'ın `fg/` + `alpha/` düzenini make_design'ın beklediği
    `im/` + `gt/` düzenine STEM-eşleşmeli SYMLINK'lerle köprüler (v7'ye özgü —
    v4'te bu köprü yoktu ve trans460 `_pairs_from_dir`'de sessizce boş
    kalıyordu). İdempotent: mevcut linkler yeniden kurulmaz."""
    src_fg = RAW / "trans460_train" / "fg"
    src_alpha = RAW / "trans460_train" / "alpha"
    if not (src_fg.is_dir() and src_alpha.is_dir()):
        print("trans460_pairs: kaynak fg/alpha yok — köprü atlanıyor.")
        return 0
    out_im = TRANS460_PAIRS / "im"
    out_gt = TRANS460_PAIRS / "gt"
    out_im.mkdir(parents=True, exist_ok=True)
    out_gt.mkdir(parents=True, exist_ok=True)
    alphas = {p.stem: p for p in src_alpha.iterdir()
              if p.is_file() and not p.name.startswith("._")}
    n = 0
    for img in sorted(src_fg.iterdir()):
        if not img.is_file() or img.name.startswith("._"):
            continue
        gt = alphas.get(img.stem)
        if gt is None:
            continue
        dst_i = out_im / img.name
        dst_g = out_gt / gt.name
        if not dst_i.exists():
            dst_i.symlink_to(img.resolve())
        if not dst_g.exists():
            dst_g.symlink_to(gt.resolve())
        n += 1
    print(f"trans460_pairs: {n} çift im/gt symlink köprüsü hazır -> {TRANS460_PAIRS}")
    return n


def _ensure_gdown() -> None:
    """Kaynak: v4 hücresi::_ensure_gdown (kopya)."""
    try:
        import gdown  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "gdown", "-q"], check=True)


def _gdown_extract(drive_id: str, out_dir: Path, label: str) -> bool:
    """Kaynak: v4 hücresi::_gdown_extract (kopya) — başarısızlıkta False döner
    (pipeline'ı durdurmaz), out_dir doluysa atlar."""
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"{label}: {out_dir} zaten dolu; indirme atlanıyor.")
        return True
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir.parent / f"{out_dir.name}.zip"
    try:
        import gdown

        gdown.download(id=drive_id, output=str(zip_path), quiet=False)
        import zipfile

        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out_dir)
        print(f"{label}: indirildi ve açıldı -> {out_dir}")
        return True
    except Exception as e:
        print(f"UYARI: {label} indirilemedi ({e}) — bu kaynak ATLANACAK.")
        return False


def _walk_dirs(root: Path, max_depth: int = 4) -> list[dict]:
    """Kaynak: v4 hücresi::_walk_dirs (kopya)."""
    root = Path(root)
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel = Path(dirpath).relative_to(root)
        depth = 0 if str(rel) == "." else len(rel.parts)
        if depth >= max_depth:
            dirnames[:] = []
        jpgs = [f for f in filenames if f.lower().endswith((".jpg", ".jpeg"))]
        pngs = [f for f in filenames if f.lower().endswith(".png")]
        out.append({
            "path": Path(dirpath),
            "jpg_count": len(jpgs),
            "png_count": len(pngs),
            "jpg_stems": {Path(f).stem for f in jpgs},
            "png_stems": {Path(f).stem for f in pngs},
            "subdirs": list(dirnames),
        })
    return out


def discover_him2k_dirs(raw_dir: Path) -> tuple[Path, Path] | None:
    """Kaynak: v4 hücresi::discover_him2k_dirs (kopya)."""
    if not raw_dir.exists():
        return None

    images_dir = None
    alphas_dir = None
    for dirpath, _dirnames, _filenames in os.walk(raw_dir):
        p = Path(dirpath)
        if p.name.lower() == "train" and p.parent.name.lower() == "images":
            images_dir = p
        if p.name.lower() == "train" and p.parent.name.lower() == "alphas":
            alphas_dir = p
    if images_dir and alphas_dir:
        return images_dir, alphas_dir

    dirs = _walk_dirs(raw_dir, max_depth=4)
    img_cands = [d for d in dirs if d["jpg_count"] >= 10]
    if not img_cands:
        return None
    img_best = max(img_cands, key=lambda d: d["jpg_count"])

    alpha_best = None
    best_score = -1
    for d in dirs:
        if d["path"] == img_best["path"]:
            continue
        score = len(d["subdirs"]) if d["subdirs"] else d["png_count"]
        if score > best_score and score > 0:
            best_score = score
            alpha_best = d["path"]
    if alpha_best is None:
        return None
    return img_best["path"], alpha_best


def merge_him2k(images_dir: Path, alphas_dir: Path, out_root: Path) -> int:
    """Kaynak: v4 hücresi::merge_him2k (kopya) — instance alfalarını
    max-birleştirip {im,gt} çiftleri üretir (design general fg kaynağı)."""
    out_im = out_root / "im"
    out_gt = out_root / "gt"
    out_im.mkdir(parents=True, exist_ok=True)
    out_gt.mkdir(parents=True, exist_ok=True)

    images = {p.stem: p for p in images_dir.iterdir()
              if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    count = 0
    for stem, img_path in sorted(images.items()):
        inst_dir = alphas_dir / stem
        merged = None
        if inst_dir.is_dir():
            insts = sorted(list(inst_dir.glob("*.png")) + list(inst_dir.glob("*.jpg")))
            for ip in insts:
                arr = np.asarray(Image.open(ip).convert("L"), dtype=np.uint8)
                merged = arr if merged is None else np.maximum(merged, arr)
        else:
            flat = None
            for ext in (".png", ".jpg", ".jpeg"):
                cand = alphas_dir / f"{stem}{ext}"
                if cand.exists():
                    flat = cand
                    break
            if flat is not None:
                merged = np.asarray(Image.open(flat).convert("L"), dtype=np.uint8)

        if merged is None:
            continue
        Image.fromarray(merged, mode="L").save(out_gt / f"{stem}.png")
        shutil.copy2(img_path, out_im / img_path.name)
        count += 1
    return count


def _ensure_him2k_merged(source_defs: dict) -> int:
    """gdown ile HIM2K'yı indirip images/alphas'ı {im,gt} olarak birleştirir —
    idempotent (merged zaten doluysa atlar). İnmezse UYARI verilip yalnız
    trans460 + ToonOut ile devam edilir (make_design'a yalnız var olan fg
    dizinleri geçilir). Kaynak: v4 hücresi (kopya)."""
    _ensure_gdown()
    ok = _gdown_extract(source_defs["him2k"]["drive_id"], RAW / "him2k_raw", "HIM2K")
    if not ok:
        return 0
    out_root = RAW / "him2k_merged"
    existing_gt = len(list((out_root / "gt").iterdir())) if (out_root / "gt").exists() else 0
    existing_im = len(list((out_root / "im").iterdir())) if (out_root / "im").exists() else 0
    if existing_gt > 0 and existing_gt == existing_im:
        print(f"{out_root} zaten {existing_gt} çift içeriyor; birleştirme atlanıyor.")
        return existing_gt
    dirs = discover_him2k_dirs(RAW / "him2k_raw")
    if dirs is None:
        print("HIM2K images/alphas dizin çifti bulunamadı — general fg ATLANACAK.")
        return 0
    n = merge_him2k(dirs[0], dirs[1], out_root)
    print(f"HIM2K birleştirildi: {n} çift -> {out_root}")
    return n


def _download_toonout() -> int:
    """Kaynak: v4 hücresi::_download_toonout (kopya, ToonOut tar yapısı
    düzeltmesi DAHİL): HuggingFace `joelseytre/toonout` deposunun YALNIZ train
    split'ini indirir (data/train_*.tar arşivleri; test split'i BİLEREK hiç
    indirilmez — illustration benchmark'ı için ayrıldı) ve `/content/downloads/
    toonout/{im,gt}` olarak normalize eder. İdempotent."""
    from huggingface_hub import snapshot_download

    out_im = TOONOUT_DIR / "im"
    out_gt = TOONOUT_DIR / "gt"
    existing_im = len(list(out_im.iterdir())) if out_im.exists() else 0
    existing_gt = len(list(out_gt.iterdir())) if out_gt.exists() else 0
    if existing_im > 0 and existing_im == existing_gt:
        print(f"toonout: {TOONOUT_DIR} zaten {existing_im} çift içeriyor; indirme atlanıyor.")
        return existing_im

    # Depo yapısı (2026-07 itibarıyla doğrulandı): split'ler klasör DEĞİL,
    # `data/{train,validation,test}_generations_*.tar` arşivleri. Her tar'ın
    # içinde `<generation_adı>/{im,gt,an}` var. Yalnız train tar'ları indirilir.
    import tarfile

    snap = Path(snapshot_download(repo_id=TOONOUT_HF_REPO, repo_type="dataset",
                                  allow_patterns=["data/train_*.tar"]))
    tars = sorted((snap / "data").glob("train_*.tar"))
    assert tars, (
        f"ToonOut snapshot'ında data/train_*.tar bulunamadı: {snap} — repo yapısı "
        f"değişmiş olabilir (beklenen: data/train_generations_*.tar arşivleri)."
    )
    extract_root = TOONOUT_DIR / "_extract"
    extract_root.mkdir(parents=True, exist_ok=True)
    for t in tars:
        with tarfile.open(t) as tf:
            tf.extractall(extract_root, filter="data")

    out_im.mkdir(parents=True, exist_ok=True)
    out_gt.mkdir(parents=True, exist_ok=True)
    copied = 0
    for gen_dir in sorted(p for p in extract_root.iterdir() if p.is_dir()):
        src_im, src_gt = gen_dir / "im", gen_dir / "gt"
        if not (src_im.is_dir() and src_gt.is_dir()):
            continue
        # macOS AppleDouble artıkları (`._*`) görüntü değildir — filtrele.
        gt_by_stem = {p.stem: p for p in src_gt.iterdir()
                      if p.is_file() and not p.name.startswith("._")}
        for img in sorted(p for p in src_im.iterdir()
                          if p.is_file() and not p.name.startswith("._")):
            gt = gt_by_stem.get(img.stem)
            if gt is None:
                continue  # gt'siz görsel kaynak olamaz
            # generation klasörleri arasında ad çakışması olabilir -> öneksle
            stem = f"{gen_dir.name}_{img.stem}"
            dst_i = out_im / f"{stem}{img.suffix}"
            dst_g = out_gt / f"{stem}{gt.suffix}"
            if dst_i.exists() and dst_g.exists():
                copied += 1
                continue
            shutil.copy2(img, dst_i)
            shutil.copy2(gt, dst_g)
            copied += 1
    shutil.rmtree(extract_root, ignore_errors=True)
    assert copied > 0, "ToonOut train tar'larından hiç im/gt çifti çıkarılamadı."
    print(f"toonout (train split): {copied} im/gt çifti -> {TOONOUT_DIR} (test split'e DOKUNULMADI).")
    return copied


def stage_downloads() -> dict:
    report("downloads", "running")
    RAW.mkdir(parents=True, exist_ok=True)
    source_defs = _load_source_defs()
    results: dict = {}

    try:
        results["trans460"] = _download_trans460(source_defs)
    except Exception as e:
        print(f"UYARI: transparent_460 indirilemedi ({e}); mevcutsa diskteki kullanılacak.")
        results["trans460"] = -1
    results["trans460_pairs"] = _normalize_trans460_pairs()

    results["him2k_merged"] = _ensure_him2k_merged(source_defs)
    results["toonout"] = _download_toonout()

    # En az bir fg kaynağı ŞART (zemin/yazı/dekor sentetik ama özne değil).
    assert (results["trans460_pairs"] > 0 or results["him2k_merged"] > 0
            or results["toonout"] > 0), (
        "Hiçbir fg kaynağı hazırlanamadı (trans460_pairs / him2k_merged / toonout) — "
        "design üretimi öznesiz olamaz; indirme loglarını inceleyin."
    )
    report("downloads", "done", results=results)
    return results


# ==========================================================================
# Stage "fonts" — v4 hücresi::stage_fonts (kopya): Google Fonts deposundan
# ~20 OFL TTF -> /content/fonts; hiçbiri inmezse DejaVu fallback'i.
# ==========================================================================
_GF_RAW = "https://raw.githubusercontent.com/google/fonts/main/"
GOOGLE_FONT_PATHS = [
    "ofl/anton/Anton-Regular.ttf",
    "ofl/bebasneue/BebasNeue-Regular.ttf",
    "ofl/lobster/Lobster-Regular.ttf",
    "ofl/pacifico/Pacifico-Regular.ttf",
    "ofl/permanentmarker/PermanentMarker-Regular.ttf",
    "ofl/bangers/Bangers-Regular.ttf",
    "ofl/righteous/Righteous-Regular.ttf",
    "ofl/satisfy/Satisfy-Regular.ttf",
    "ofl/abrilfatface/AbrilFatface-Regular.ttf",
    "ofl/alfaslabone/AlfaSlabOne-Regular.ttf",
    "ofl/archivoblack/ArchivoBlack-Regular.ttf",
    "ofl/shrikhand/Shrikhand-Regular.ttf",
    "ofl/staatliches/Staatliches-Regular.ttf",
    "ofl/monoton/Monoton-Regular.ttf",
    "ofl/pressstart2p/PressStart2P-Regular.ttf",
    "ofl/caveat/Caveat[wght].ttf",
    "ofl/dancingscript/DancingScript[wght].ttf",
    "ofl/oswald/Oswald[wght].ttf",
    "ofl/montserrat/Montserrat[wght].ttf",
    "ofl/playfairdisplay/PlayfairDisplay[wght].ttf",
    "ofl/orbitron/Orbitron[wght].ttf",
]
DEJAVU_GLOBS = [
    "/usr/share/fonts/truetype/dejavu/DejaVu*.ttf",  # Colab/Ubuntu standart yolu
    "/usr/share/fonts/TTF/DejaVu*.ttf",
]


def stage_fonts() -> int:
    report("fonts", "running")
    FONT_DIR.mkdir(parents=True, exist_ok=True)

    ok, failed = 0, []
    for rel in GOOGLE_FONT_PATHS:
        # Dosya adındaki [wght] köşeli ayraçları URL'de yüzde-kodlanmalı;
        # yerelde ayraçsız sade ad (glob desenleriyle çakışmasın diye).
        target = FONT_DIR / Path(rel).name.replace("[", "_").replace("]", "_")
        if target.exists() and target.stat().st_size > 0:
            ok += 1
            continue
        url = _GF_RAW + urllib.parse.quote(rel)
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                data = resp.read()
            assert data[:4] in (b"\x00\x01\x00\x00", b"OTTO", b"true"), "TTF/OTF imzası değil"
            target.write_bytes(data)
            ok += 1
        except Exception as e:
            failed.append((rel, str(e)))
            print(f"UYARI: font indirilemedi ({rel}): {e}")

    if ok < 5:
        print(f"Yalnız {ok} Google Fonts fontu inebildi — DejaVu fallback'ine düşülüyor.")
        import glob as _glob

        for pattern in DEJAVU_GLOBS:
            for p in _glob.glob(pattern):
                dst = FONT_DIR / Path(p).name
                if not dst.exists():
                    shutil.copy2(p, dst)

    total = len([p for p in FONT_DIR.iterdir() if p.suffix.lower() in {".ttf", ".otf"}])
    print(f"/content/fonts: {total} font hazır ({ok} Google Fonts, {len(failed)} başarısız).")
    if total == 0:
        raise RuntimeError(
            "Hiç font indirilemedi ve DejaVu fallback'i de bulunamadı — design yazı blokları "
            "üretilemez. Ağ bağlantısını kontrol edin veya /content/fonts'a elle TTF koyun."
        )
    report("fonts", "done", downloaded=ok, failed=len(failed), total=total)
    return total


# ==========================================================================
# Stage "val_guard" — VAL sızıntı koruması fg SEÇİMİNDE (v3/v4 kalıbı):
# val_stems.json -> tcl.derive_val_excluded_source_ids -> kompozit id
# sözleşmesi f"{kaynak_adı}_{_sanitize(stem)}" üzerinden ham fg stem'lerine
# geri eşleme (bkz. modül docstring'i).
# ==========================================================================
def stage_val_guard() -> set[str]:
    report("val_guard", "running")
    val_json = STATUS_DIR / "val_stems.json"
    if not val_json.exists():
        print(f"NOT: {val_json} yok (henüz hiç eğitim koşulmamış olabilir) — VAL hariç tutma "
              f"atlanıyor; yeni design stem'leri zaten her zaman TRAIN'e gider.")
        report("val_guard", "done", excluded=0)
        return set()

    from build_testset import _sanitize  # scripts/ sys.path'te (build_trainset id sözleşmesi)

    val_stems = json.loads(val_json.read_text())["val_stems"]
    excluded_ids, unmatched = tcl.derive_val_excluded_source_ids(val_stems)
    if unmatched:
        print(f"UYARI: {len(unmatched)} val stem'i `_v/_o<NN>` son ek desenine uymuyor "
              f"(ör. {unmatched[:5]}) — bunlar kaynak-id düzeyinde eşlenemez (v3 dersi); "
              f"design fg havuzu için risk yalnız trans460/him2k kaynaklı id'lerdedir.")

    exclude_fg_stems: set[str] = set()
    per_pool: dict[str, int] = {}
    for pool_dirname, prefix in FG_SOURCE_PREFIXES.items():
        im_dir = RAW / pool_dirname / "im"
        if not im_dir.is_dir():
            continue
        n = 0
        for p in im_dir.iterdir():
            if not p.is_file() or p.name.startswith("._"):
                continue
            if f"{prefix}_{_sanitize(p.stem)}" in excluded_ids:
                exclude_fg_stems.add(p.stem)
                n += 1
        per_pool[pool_dirname] = n
    print(f"VAL sızıntı koruması: {len(val_stems)} val stem'i -> {len(exclude_fg_stems)} "
          f"ham fg stem'i havuzdan hariç tutulacak (havuz bazında: {per_pool}).")
    report("val_guard", "done", excluded=len(exclude_fg_stems), per_pool=per_pool)
    return exclude_fg_stems


# ==========================================================================
# Stage "design" — ÜRETİM: scripts/make_design.py (birim testli). İmza/import
# uyuşmazlığında NET Türkçe hata mesajıyla durulur (v6 stage_v6 kalıbı),
# sessizce yarım veri üretilmez.
# ==========================================================================
def stage_design(exclude_fg_stems: set[str]) -> dict[str, int]:
    report("design", "running")

    if "design" not in CATEGORIES:
        raise RuntimeError(
            f"benchmark.testset.CATEGORIES 'design' kategorisini tanımıyor — repo klonunuz "
            f"eski görünüyor. 'git -C {WORKDIR} pull' çalıştırıp hücreyi yeniden koşun."
        )

    try:
        import make_design as mdz  # scripts/ sys.path'te
    except ImportError as e:
        raise RuntimeError(
            f"scripts/make_design.py import edilemedi ({e}) — repo'nuz güncel mi? "
            f"'git -C {WORKDIR} pull' deneyin (script bu hücreyle aynı çalışmada eklendi)."
        ) from e

    fg_dirs = [d for d in (TRANS460_PAIRS, RAW / "him2k_merged") if (d / "im").is_dir()]
    toon_dir = TOONOUT_DIR if (TOONOUT_DIR / "im").is_dir() else None

    try:
        counts = mdz.run(
            out_dir=DESIGN_OUT_DIR,
            bg_dir=None,  # zemin sentetik — kullanılmaz
            fg_dirs=fg_dirs,
            toonout_dir=toon_dir,
            font_dir=FONT_DIR,
            seed=SEED,
            count=DESIGN_COUNT,
            exclude_fg_stems=exclude_fg_stems,
        )
    except TypeError as e:
        raise RuntimeError(
            f"make_design.run() beklenen imzayla çağrılamadı ({e}) — bu hücre "
            f"run(out_dir, bg_dir, fg_dirs, toonout_dir, font_dir, seed, count, "
            f"exclude_fg_stems) imzasını varsayar; scripts/make_design.py'nin güncel "
            f"imzasına bakıp çağrıyı uyarlayın."
        ) from e

    print("make_design.run() üretim:", counts)

    # Manifest guard (v3 dersi): boş/eksik manifest'le export'a GEÇME.
    # make_design'ın çıktı manifest'i {"id","category"} satırlarıdır — export
    # TAM testset şeması (image + gt_alpha) istediği için manifest_full'e
    # dönüştürülür (v4/v6 hücreleri aynı kalıp).
    out_manifest = DESIGN_OUT_DIR / "manifest.jsonl"
    if not out_manifest.exists():
        raise RuntimeError(f"{out_manifest} yok — make_design üretimi başarısız olmuş olmalı.")
    rows = [json.loads(line) for line in out_manifest.read_text().splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"{out_manifest} boş — export'a geçilmiyor (v3 dersi).")

    full_manifest = DESIGN_OUT_DIR / "manifest_full.jsonl"
    with open(full_manifest, "w") as f:
        for r in rows:
            im_p = DESIGN_OUT_DIR / "im" / f"{r['id']}.jpg"
            gt_p = DESIGN_OUT_DIR / "gt" / f"{r['id']}.png"
            if not (im_p.exists() and gt_p.exists()):
                raise RuntimeError(f"manifest satırının dosyası eksik: {r['id']} — üretim yarım kalmış olabilir.")
            f.write(json.dumps({"id": r["id"], "image": str(im_p),
                                "category": r["category"], "gt_alpha": str(gt_p)},
                               ensure_ascii=False) + "\n")

    print(f"PRE-FLIGHT — {out_manifest}: toplam {len(rows)} design çifti.")
    if len(rows) < 100:
        print(f"UYARI: design sayısı çok düşük ({len(rows)} < 100) — fg kaynakları eksik "
              f"olabilir, logları inceleyin.")

    report("design", "done", counts=counts, total_pairs=len(rows))
    return counts


# ==========================================================================
# Stage "export" — v6 kalıbı: export_birefnet.export() taze/boş bir yerel
# dizine karşı çalışır. split_name="TRAIN": yeni stemler HER ZAMAN TRAIN'e.
# ==========================================================================
def stage_export_design() -> dict:
    report("export", "running")
    import export_birefnet as eb  # scripts/ sys.path'te

    stats = eb.export(
        manifest_path=str(DESIGN_OUT_DIR / "manifest_full.jsonl"),
        out_dir=EXPORT_DIR,
        split_name="TRAIN",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    report("export", "done", stats=stats)
    return stats


# ==========================================================================
# Stage "drive_copy" — v6 kalıbı: var olan Drive TRAIN'e MERGE (dirs_exist_ok=
# True, silme/üzerine yazma yok; im/gt AYRI sayaçlı — 2026-07-12 dersi) +
# kompozit manifest'e APPEND (tcl.merge_composite_manifest, dedupe'lu).
# ==========================================================================
def stage_drive_copy_design() -> None:
    report("drive_copy", "running")
    src = Path(EXPORT_DIR)
    dst = Path(DRIVE_ROOT) / DRIVE_OUTPUT_SUBDIR
    dst_train_im = dst / "TRAIN" / "im"
    dst_train_gt = dst / "TRAIN" / "gt"
    assert dst_train_im.is_dir() and dst_train_gt.is_dir(), (
        f"Drive'da beklenen TRAIN verisi bulunamadı: {dst_train_im} / {dst_train_gt} — "
        f"bu hücre yalnız MEVCUT bir veri setine design kategorisini EKLEMEK içindir."
    )

    src_im_files = list((src / "TRAIN" / "im").iterdir())
    src_gt_files = list((src / "TRAIN" / "gt").iterdir())
    assert len(src_im_files) == len(src_gt_files), "yerel design export'unda im/gt sayıları uyuşmuyor!"

    # im ve gt AYRI sayılır (v4/v6 hücreleri / 2026-07-12 dersi).
    existing_dst_im_stems = {p.stem for p in _listdir_retry(dst_train_im)}
    existing_dst_gt_stems = {p.stem for p in _listdir_retry(dst_train_gt)}
    growth_im = len({p.stem for p in src_im_files} - existing_dst_im_stems)
    growth_gt = len({p.stem for p in src_gt_files} - existing_dst_gt_stems)

    pre_im, pre_gt = len(existing_dst_im_stems), len(existing_dst_gt_stems)
    print(f"Merge öncesi Drive TRAIN: im={pre_im}, gt={pre_gt} — beklenen artış: "
          f"im +{growth_im}, gt +{growth_gt}")

    # YALNIZ TRAIN/ alt ağacı kopyalanır — src kökündeki KISMİ stats.json
    # Drive'daki otoriter TAM stats.json'u ezmesin diye KOPYALANMAZ (v3 fix'i).
    print(f"Kopyalanıyor (MERGE, silme yok, yalnız TRAIN/): {src / 'TRAIN'} -> {dst / 'TRAIN'}")
    shutil.copytree(src / "TRAIN", dst / "TRAIN", dirs_exist_ok=True)

    post_im, post_gt = len(_listdir_retry(dst_train_im)), len(_listdir_retry(dst_train_gt))
    print(f"Merge sonrası Drive TRAIN: im={post_im}, gt={post_gt}")

    assert post_im - pre_im == growth_im, (
        f"im/ büyümesi beklenenle uyuşmuyor: {post_im - pre_im} != {growth_im}"
    )
    assert post_gt - pre_gt == growth_gt, (
        f"gt/ büyümesi beklenenle uyuşmuyor: {post_gt - pre_gt} != {growth_gt}"
    )
    # ASIL bütünlük şartı: merge sonrası im/gt stem sayıları eşit.
    assert post_im == post_gt, f"Drive TRAIN im/gt sayıları eşit değil: {post_im} != {post_gt}"

    # manifest_full.jsonl: merge_composite_manifest içindeki load_manifest
    # doğrulaması TAM şema (image+gt_alpha) istediği için ham manifest verilemez.
    comp_manifest_local = DESIGN_OUT_DIR / "manifest_full.jsonl"
    comp_manifest_drive = dst / "train_composites_manifest.jsonl"
    n_appended = tcl.merge_composite_manifest(comp_manifest_local, comp_manifest_drive)
    print(f"train_composites_manifest.jsonl: {n_appended} yeni satır eklendi (mevcut satırlar "
          f"KORUNDU, üzerine yazılmadı). Onarım koşusunda 0 olabilir — hata değil (v4 dersi).")

    print("\nBÜTÜNLÜK KONTROLÜ BAŞARILI — design verisi Drive'a MERGE edildi.")
    report(
        "drive_copy", "done",
        added_im=growth_im, added_gt=growth_gt, added_manifest_rows=n_appended,
        total_im=post_im, total_gt=post_gt,
    )


# ==========================================================================
# Orkestrasyon — üst düzeyde koşar (hücre yapıştırılıp çalıştırıldığında).
# ==========================================================================
def main() -> None:
    stage0_env()                       # Drive mount + git pull — her şeyden önce
    stage_downloads()                  # trans460 + HIM2K + ToonOut (~3GB; TAR FETCH YOK)
    stage_fonts()                      # ~20 OFL Google Fonts -> /content/fonts (DejaVu fallback)
    exclude_fg_stems = stage_val_guard()
    stage_design(exclude_fg_stems)     # make_design.run(count=6000) + manifest guard
    stage_export_design()
    stage_drive_copy_design()
    report("ALL", "done")
    print(
        "\nNOT: tar shard'ları YENİDEN PAKETLENMEDİ — bir sonraki eğitim koşusunda "
        "train_colab.ipynb hücre (c), tar'ları açtıktan sonra yeni ~6k çifti delta olarak "
        "copy_pairs ile Drive'dan tamamlayacak (birkaç dk sürer). İstersen "
        "training/veri_tar_paketleme_hucresi.py'yi yeniden koşup delta'yı sıfırlayabilirsin "
        "(DİKKAT: çoğu shard yeniden paketlenir — ~1 saatlik ücretsiz CPU koşusu; delta "
        "copy_pairs genelde daha ucuz)."
    )
    # KRİTİK (2026-07-12 dersi): Drive yazımları ASENKRON tamponlanır — VM bu
    # flush bitmeden kapatılırsa dosyalar SESSİZCE kaybolur. flush_and_unmount()
    # tamponu boşaltmayı ZORLAR ve bitene kadar bloklar. Drive'a yazan HER
    # ŞEYDEN (report dahil) SONRA çağrılır.
    print("Drive flush ediliyor (asenkron yazımların buluta inmesi bekleniyor)...")
    from google.colab import drive as _gdrive
    _gdrive.flush_and_unmount()
    print("Drive flush TAMAM — VM artık güvenle kapatılabilir/değiştirilebilir.")


try:
    main()
except Exception:
    tb = traceback.format_exc()
    report("FATAL", "error", traceback=tb)
    raise
