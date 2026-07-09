"""DEVAM HÜCRESİ — training/prepare_data_colab.ipynb'nin kalan tüm adımlarını
(arka plan havuzu indirme, COD10K/HIM2K yapı keşfi + HIM2K birleştirme, tam
manifest, kompozit üretimi, BiRefNet export, Drive'a kopyalama + bütünlük
kontrolü) TEK bir hücrede baştan sona koşturur.

KULLANIM: Bu dosyanın TÜM içeriğini canlı Colab runtime'ında (repo zaten
/content/my-bg-remover'da açılmış, Drive bağlı, `pip install -e .` yapılmış,
data/raw_train/{dis5k,camo,p3m,trans460_train} + data/raw_train/{cod10k_raw,
him2k_raw} zaten mevcut) yeni bir hücreye YAPIŞTIRIP çalıştırın. Argparse yok,
`if __name__` bloğu yok — dosya import edilmeden doğrudan üst düzeyde koşar.

Kernel'de önceki hücrelerden kalan değişkenlere GÜVENİLMEZ: bu dosya kendi
başına, sıfırdan tüm durumu tanımlar (idempotent olduğu yerlerde — bkz. her
aşamanın docstring'i).

Durum takibi: her aşama başlangıcında/sonunda `report()` çağrılır; bu hem
konsola yazar hem de Drive'a (`/content/drive/MyDrive/bg-remover-status/`)
`log.txt` (append) ve `status.json` (üzerine yaz, `history` biriktirir) olarak
kaydeder — dışarıdan (bu Colab oturumunun dışından) ilerlemeyi izlemek için.
Beklenmeyen bir hata olursa `stage="FATAL"` ile tam traceback raporlanır ve
hata TEKRAR FIRLATILIR (sessizce yutulmaz).
"""

import io
import json
import os
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

# --- Sabitler -----------------------------------------------------------
WORKDIR = "/content/my-bg-remover"
DRIVE_ROOT = "/content/drive/MyDrive"
DRIVE_OUTPUT_SUBDIR = "bg-remover-data"
SEED = 42
BG_POOL_SIZE = 5000

STATUS_DIR = Path(DRIVE_ROOT) / "bg-remover-status"
LOG_PATH = STATUS_DIR / "log.txt"
STATUS_PATH = STATUS_DIR / "status.json"

# scripts/ bir paket değil (üstü __init__.py yok) — build_trainset/make_composites/
# export_birefnet'i import edebilmek için mutlak yolu sys.path'e ekliyoruz (cwd henüz
# değişmemiş olsa bile çalışsın diye mutlak yol kullanıyoruz, os.chdir'e bağımlı değil).
SCRIPTS_DIR = str(Path(WORKDIR) / "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from benchmark.testset import append_entries  # noqa: E402  (pip install -e . ile kurulu paket)


# ==========================================================================
# Durum raporlama — controller bu dosyaları DIŞARIDAN izler, kritik.
# ==========================================================================
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def report(stage: str, status: str, **extra) -> None:
    """log.txt'e satır ekler + status.json'u (history biriktirerek) yeniden yazar.

    Her çağrıda status.json'un mevcut `history`si okunur (varsa) ve yeni girdi
    eklenir — böylece script kesintiye uğrayıp yeniden koşturulsa bile Drive'daki
    geçmiş kaybolmaz."""
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
# Stage 0 — ortam sağlık kontrolü
# ==========================================================================
RAW_DIR_CHECKS = {
    "dis5k": "data/raw_train/dis5k/im",
    "camo": "data/raw_train/camo/im",
    "p3m": "data/raw_train/p3m/im",
    "trans460_train": "data/raw_train/trans460_train/fg",
    "cod10k_raw": "data/raw_train/cod10k_raw",
    "him2k_raw": "data/raw_train/him2k_raw",
    "backgrounds": "data/backgrounds",
}


def _count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*") if p.is_file())


def _setup_hf_env() -> None:
    """HF indirme zaman aşımı (takılan indirme dersinden) + Colab Secrets'tan
    HF_TOKEN (varsa) — bulunamazsa sessizce devam (çoğu kaynak anonim erişimle çalışır)."""
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
    try:
        from google.colab import userdata

        token = userdata.get("HF_TOKEN")
        if token:
            os.environ["HF_TOKEN"] = token
            print("HF_TOKEN Colab Secrets'tan alındı.")
    except Exception as e:
        print(f"HF_TOKEN alınamadı (Secrets'ta yok veya erişim izni verilmedi): {e}")


def stage0_env_sanity() -> dict:
    report("env", "running")
    os.chdir(WORKDIR)
    _setup_hf_env()

    counts = {name: _count_files(Path(rel)) for name, rel in RAW_DIR_CHECKS.items()}
    for name, c in counts.items():
        print(f"{name}: {c} dosya")

    report("env", "done", cwd=str(Path.cwd()), counts=counts)
    return counts


# ==========================================================================
# Stage 1 — arka plan havuzu (BG-20k)
# ==========================================================================
def stage1_bg_pool() -> int:
    report("bg_pool", "running")
    bg_dir = Path("data/backgrounds")
    bg_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(bg_dir.iterdir()))
    if existing >= BG_POOL_SIZE:
        print(f"data/backgrounds zaten {existing} görsel içeriyor (>= {BG_POOL_SIZE}); indirme atlanıyor.")
        report("bg_pool", "done", count=existing, skipped=True)
        return existing

    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    with open("data/train_sources.json") as f:
        source_defs = {s["name"]: s for s in json.load(f)["sources"]}
    bg_spec = source_defs["bg_20k"]

    fs = HfFileSystem()
    pattern = bg_spec["split_patterns"][0]  # "data/train-*-of-00022.parquet"
    parts = sorted(fs.glob(f"datasets/{bg_spec['hf_repo']}/{pattern}"))

    written = existing  # KÜMÜLATİF sayaç — parça sınırlarında sıfırlanmaz (bkz. notebook cell (c) notu)
    for part in parts:
        if written >= BG_POOL_SIZE:
            break
        with fs.open(part, "rb") as fh:
            table = pq.read_table(fh, columns=["image"])
        for i in range(table.num_rows):
            if written >= BG_POOL_SIZE:
                break
            out_path = bg_dir / f"bg20k_{written:06d}.jpg"
            if out_path.exists():
                written += 1
                continue
            img_bytes = table["image"][i].as_py()["bytes"]
            im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            im.thumbnail((1024, 1024))
            im.save(out_path, format="JPEG", quality=88)
            written += 1

    print(f"data/backgrounds: {written} arka plan görseli.")
    report("bg_pool", "done", count=written)
    return written


# ==========================================================================
# Stage 2 — COD10K/HIM2K gerçek klasör yapısı keşfi
# ==========================================================================
def _walk_dirs(root: Path, max_depth: int = 4) -> list[dict]:
    """root altındaki her dizin için (derinlik <= max_depth) jpg/png sayıları
    ve stem kümelerini döndürür — img/gt dizin çiftlerini eşleştirmek için."""
    root = Path(root)
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel = Path(dirpath).relative_to(root)
        depth = 0 if str(rel) == "." else len(rel.parts)
        if depth >= max_depth:
            dirnames[:] = []  # daha derine inme (ama bu dizinin kendisi işlenir)
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


def discover_cod10k(raw_dir: Path) -> dict | None:
    """COD10K-v3 zip'inin gerçek iç yapısını keşfeder: çok sayıda .jpg içeren bir
    dizin ile aynı stem'lerin çoğunu paylaşan .png dizinini eşleştirir (stem
    örtüşmesi = asıl doğruluk sinyali; isim tercihi (Image/GT/Train) yalnız
    eşit derecede örtüşen adaylar arasında tie-break için kullanılır)."""
    if not raw_dir.exists():
        return None
    dirs = _walk_dirs(raw_dir, max_depth=4)
    img_candidates = [d for d in dirs if d["jpg_count"] >= 10]
    gt_candidates = [d for d in dirs if d["png_count"] >= 10]

    scored = []
    for ic in img_candidates:
        for gc in gt_candidates:
            if ic["path"] == gc["path"]:
                continue
            overlap = len(ic["jpg_stems"] & gc["png_stems"])
            if overlap == 0:
                continue
            name_bonus = 0
            if "image" in ic["path"].name.lower():
                name_bonus += 2
            if "gt" in gc["path"].name.lower():
                name_bonus += 2
            if "train" in str(ic["path"]).lower():
                name_bonus += 1
            scored.append({
                "img_dir": ic["path"], "gt_dir": gc["path"], "overlap": overlap,
                "score": (overlap, name_bonus),
            })
    if not scored:
        return None
    scored.sort(key=lambda s: s["score"], reverse=True)
    best = scored[0]
    ambiguous = len(scored) > 1 and scored[0]["score"] == scored[1]["score"]
    return {
        "img_dir": best["img_dir"], "gt_dir": best["gt_dir"], "overlap": best["overlap"],
        "ambiguous": ambiguous,
        "candidates": [(str(s["img_dir"]), str(s["gt_dir"]), s["overlap"]) for s in scored[:5]],
    }


def stage2_discover_structure() -> dict | None:
    report("discover_cod10k", "running")
    raw_dir = Path("data/raw_train/cod10k_raw")
    if not raw_dir.exists():
        print("data/raw_train/cod10k_raw yok — COD10K atlanıyor.")
        report("discover_cod10k", "skipped", reason="dizin yok")
        return None

    info = discover_cod10k(raw_dir)
    if info is None:
        print("COD10K için örtüşen img/gt dizin çifti bulunamadı.")
        report("discover_cod10k", "skipped", reason="eşleşme yok")
        return None

    print(f"COD10K seçilen çift: img={info['img_dir']}  gt={info['gt_dir']}  "
          f"örtüşen stem={info['overlap']}  belirsiz={info['ambiguous']}")
    if info["ambiguous"]:
        print(f"UYARI: birden çok aday eşit skorlu — en iyi tahmin seçildi. Adaylar: {info['candidates']}")
    report("discover_cod10k", "done", img_dir=str(info["img_dir"]), gt_dir=str(info["gt_dir"]),
           overlap=info["overlap"], ambiguous=info["ambiguous"], candidates=info["candidates"])
    return info


# ==========================================================================
# Stage 3 — HIM2K instance-matting birleştirme
# ==========================================================================
def discover_him2k_dirs(raw_dir: Path) -> tuple[Path, Path] | None:
    """images/train ve alphas/train dizinlerini bulur. Önce isimle (tam
    'images/train' + 'alphas/train' yol örüntüsü) dener; bulamazsa sayaç
    bazlı fallback (en çok .jpg içeren dizin = images; en çok alt-dizin
    barındıran ayrı bir dizin = alphas, instance klasörleri varsayımıyla)."""
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

    # Fallback: isimle bulunamadı — sayaç bazlı en iyi tahmin.
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
    """Her görsel için alphas_dir/<stem>/ bir dizinse (instance PNG'leri) hepsini
    piksel-bazında max ile birleştirir; alphas_dir/<stem>.{png,jpg} düz bir
    dosyaysa doğrudan onu kullanır. Görseller kopyalanır (Drive taşımasında
    symlink kırılma riski yok, bkz. scripts/build_trainset.py _link notu)."""
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


def stage3_merge_him2k() -> int:
    report("him2k_merge", "running")
    raw_dir = Path("data/raw_train/him2k_raw")
    if not raw_dir.exists():
        print("data/raw_train/him2k_raw yok — HIM2K atlanıyor (general kategorisi opsiyonel).")
        report("him2k_merge", "skipped", reason="dizin yok")
        return 0

    dirs = discover_him2k_dirs(raw_dir)
    if dirs is None:
        print("HIM2K images/alphas dizin çifti bulunamadı — atlanıyor.")
        report("him2k_merge", "skipped", reason="images/alphas bulunamadı")
        return 0
    images_dir, alphas_dir = dirs
    print(f"HIM2K: images_dir={images_dir}  alphas_dir={alphas_dir}")

    out_root = Path("data/raw_train/him2k_merged")
    existing_gt = len(list((out_root / "gt").iterdir())) if (out_root / "gt").exists() else 0
    existing_im = len(list((out_root / "im").iterdir())) if (out_root / "im").exists() else 0
    if existing_gt > 0 and existing_gt == existing_im:
        print(f"data/raw_train/him2k_merged zaten {existing_gt} çift içeriyor; birleştirme atlanıyor (idempotent).")
        report("him2k_merge", "done", count=existing_gt, skipped=True)
        return existing_gt

    count = merge_him2k(images_dir, alphas_dir, out_root)
    print(f"HIM2K birleştirildi: {count} çift -> {out_root}")
    report("him2k_merge", "done", count=count)
    return count


# ==========================================================================
# Stage 4 — tam manifest (build_trainset.py mantığıyla, n=None + copy=True)
# ==========================================================================
def stage4_build_manifest(cod10k_info: dict | None, him2k_count: int) -> dict:
    report("manifest", "running")
    import build_trainset as bt  # scripts/ sys.path'te (dosya başında eklendi)

    # Her koşuda temiz baştan — deterministik (spec madde 10).
    if bt.MANIFEST.exists():
        bt.MANIFEST.unlink()
    for d in (bt.OUT_IMG, bt.OUT_GT):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    counts: dict = {}

    def _run(name: str, img_glob: str, gt_glob: str, category: str, **kw) -> int:
        # sample_source(n=None, ...) TÜM eşleşen çiftleri döndürür (kaynak koda
        # göre doğrulandı: n is not None kontrolüyle sample sadece n verilince
        # devreye girer) — huge-int hilesine gerek yok.
        rows = bt.sample_source(name, img_glob, gt_glob, category, n=None, copy=True, **kw)
        append_entries(str(bt.MANIFEST), rows)
        counts[name] = len(rows)
        print(f"{name} ({category}): {len(rows)} çift")
        return len(rows)

    # camotr / p3m / trans460tr — SOURCE_SPECS TEK doğruluk kaynağından (disvd_tokens hariç).
    for name, spec in bt.SOURCE_SPECS.items():
        if spec["category"] == "disvd_tokens":
            continue
        _run(name, spec["img_glob"], spec["gt_glob"], spec["category"])

    # dis5ktr — kategori dosya adı token'ından atanır (thin/complex).
    rows = bt.sample_disvd_tokens("dis5ktr", bt.DIS5KTR_IMG_GLOB, bt.DIS5KTR_GT_GLOB, n=None, copy=True)
    append_entries(str(bt.MANIFEST), rows)
    dis_counts: dict = {}
    for r in rows:
        dis_counts[r["category"]] = dis_counts.get(r["category"], 0) + 1
    counts["dis5ktr"] = dis_counts
    for category, c in sorted(dis_counts.items()):
        print(f"dis5ktr ({category}): {c} çift")

    # cod10ktr — Stage 2'de keşfedilen gerçek img/gt dizinlerinden.
    if cod10k_info:
        img_glob = str(cod10k_info["img_dir"].relative_to(bt.ROOT)) + "/*"
        gt_glob = str(cod10k_info["gt_dir"].relative_to(bt.ROOT)) + "/*"
        _run("cod10ktr", img_glob, gt_glob, "camouflage")
    else:
        counts["cod10ktr"] = 0
        print("cod10ktr: atlandı (Stage 2'de dizin bulunamadı)")

    # him2k — Stage 3'te birleştirilmiş him2k_merged/{im,gt}'den (general kategorisi, opsiyonel).
    if him2k_count > 0:
        _run("him2k", "data/raw_train/him2k_merged/im/*", "data/raw_train/him2k_merged/gt/*", "general")
    else:
        counts["him2k"] = 0
        print("him2k: atlandı (Stage 3'te birleştirme yapılamadı)")

    report("manifest", "done", counts=counts)
    return counts


# ==========================================================================
# Stage 5 — kompozit + augmentasyon üretimi (make_composites.run)
# ==========================================================================
def stage5_make_composites() -> dict:
    report("composites", "running")
    import make_composites as mc  # scripts/ sys.path'te

    # per_image=1 + script varsayılan CATEGORY_MULTIPLIER (transparent x10,
    # camouflage x2) — override YOK, drift önleme (spec: script defaults kullanılır).
    counts = mc.run(
        manifest_path=Path("data/train/manifest.jsonl"),
        backgrounds_dir=Path("data/backgrounds"),
        per_image=1,
        seed=SEED,
        out_dir=Path("data/train_composites"),
    )
    print("Kategori bazlı üretilen kompozit sayısı:", counts)
    report("composites", "done", counts=counts)
    return counts


# ==========================================================================
# Stage 6 — BiRefNet formatına export
# ==========================================================================
def stage6_export() -> dict:
    report("export", "running")
    import export_birefnet as eb  # scripts/ sys.path'te

    stats = eb.export(
        manifest_path="data/train_composites/manifest.jsonl",
        out_dir="/content/birefnet_format",
        split_name="TRAIN",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    report("export", "done", stats=stats)
    return stats


# ==========================================================================
# Stage 7 — Drive'a kopyala + bütünlük kontrolü
# ==========================================================================
def stage7_drive_copy(stats: dict) -> None:
    report("drive_copy", "running")
    src = Path("/content/birefnet_format")
    dst = Path(DRIVE_ROOT) / DRIVE_OUTPUT_SUBDIR
    dst.mkdir(parents=True, exist_ok=True)

    print(f"Kopyalanıyor: {src} -> {dst}")
    shutil.copytree(src, dst, dirs_exist_ok=True)

    comp_manifest = Path("data/train_composites/manifest.jsonl")
    if comp_manifest.exists():
        shutil.copy2(comp_manifest, dst / "train_composites_manifest.jsonl")
        print(f"Kompozit manifest de kopyalandı: {dst / 'train_composites_manifest.jsonl'}")

    src_im = list((src / "TRAIN" / "im").iterdir())
    src_gt = list((src / "TRAIN" / "gt").iterdir())
    dst_im = list((dst / "TRAIN" / "im").iterdir())
    dst_gt = list((dst / "TRAIN" / "gt").iterdir())

    with open(src / "stats.json") as f:
        stats_on_disk = json.load(f)

    print(f"im/: kaynak={len(src_im)}, hedef={len(dst_im)}")
    print(f"gt/: kaynak={len(src_gt)}, hedef={len(dst_gt)}")
    print(f"stats.json total: {stats_on_disk['total']}")

    assert len(src_im) == len(dst_im), "im/ dosya sayısı Drive kopyasında eşleşmiyor!"
    assert len(src_gt) == len(dst_gt), "gt/ dosya sayısı Drive kopyasında eşleşmiyor!"
    assert len(dst_im) == len(dst_gt) == stats_on_disk["total"], "im/gt/stats.json toplam sayıları tutarsız!"

    print("\nBÜTÜNLÜK KONTROLÜ BAŞARILI — veri Drive'da hazır.")
    report("drive_copy", "done", im=len(dst_im), gt=len(dst_gt), total=stats_on_disk["total"])


# ==========================================================================
# Orkestrasyon — üst düzeyde koşar (hücre yapıştırılıp çalıştırıldığında).
# ==========================================================================
def main() -> None:
    stage0_env_sanity()
    stage1_bg_pool()
    cod10k_info = stage2_discover_structure()
    him2k_count = stage3_merge_him2k()
    stage4_build_manifest(cod10k_info, him2k_count)
    stage5_make_composites()
    stats = stage6_export()
    stage7_drive_copy(stats)
    report("ALL", "done")


try:
    main()
except Exception:
    tb = traceback.format_exc()
    report("FATAL", "error", traceback=tb)
    raise
