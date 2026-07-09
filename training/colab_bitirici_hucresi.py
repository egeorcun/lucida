"""BİTİRİCİ HÜCRE — `colab_devam_hucresi.py`'nin Stage 5'i (kompozit üretimi)
28.281 hedefin ~26-27 bininde, dev boyutlu (Transparent-460 ×10 kopya ve/veya
HIM2K genel görselleri, 100-246MP) foreground'lar Colab'ın 12GB RAM'ini
tıkadığı ve kullanıcı hücreyi kesintiye uğrattığı için TAMAMLANAMADI.

KULLANIM: Bu dosyanın TÜM içeriğini canlı Colab runtime'ında (repo zaten
/content/my-bg-remover'da açılmış, Drive bağlı, `pip install -e .` yapılmış,
data/train/manifest.jsonl ve data/backgrounds zaten hazır — Stage 0-4 daha
önce `colab_devam_hucresi.py` tarafından tamamlanmış) yeni bir hücreye
YAPIŞTIRIP çalıştırın.

Neden `scripts/make_composites.py::run()` yeniden çağrılmıyor: `run()` yeni
girdileri BELLEKTE `new_entries` listesinde biriktirip TEK bir toplu
`append_entries` çağrısıyla (tüm kaynak satırlar işlendikten SONRA) dosyaya
yazıyor — kesinti olursa o ana kadar diske YAZILMIŞ onbinlerce görsel/gt
dosyası için manifest satırı asla eklenmez. `run()`'u aynen yeniden çağırmak,
disk üzerinde zaten var olan dosyalar için (id manifest'te YOK diye)
gereksiz yeniden üretime yol açar — hem zaman kaybı hem de aynı dev
görsellerde tekrar takılma riski. Bu yüzden aşağıdaki "bitirici döngü" DOSYA
varlığını (yalnız manifest'i değil) kontrol eder ve HER öğeden SONRA
`append_entries` çağırır (kesintiye dayanıklı).

Kritik değişmez (invariant): dev-boyut OLMAYAN öğeler için üretim yolu
`make_composites.run()` ile BİREBİR aynı olmalı (aynı `_item_rng` alt-akışı,
aynı `compose`/`augment` çağrı sırası) — aksi halde önceden üretilmiş ile
yeni üretilen öğeler arasında istatistiksel tutarsızlık oluşur. Bunu sağlamak
için tüm yardımcı fonksiyonlar `scripts/make_composites.py`'den import
edilir, KOPYALANMAZ. Yalnız dev görseller (uzun kenar > 2048px) compose/augment
ÖNCESİNDE küçültülür — bu öğeler için `run()` ile bit-birebir aynılık zaten
İSTENMİYOR (asıl amaç budur: dev canvas'ları küçültüp RAM patlamasını önlemek).

Durum takibi: `colab_devam_hucresi.py` ile AYNI `report()` mekanizması
kullanılır (`/content/drive/MyDrive/bg-remover-status/log.txt` + `status.json`).
Aşama adları: `finisher` -> `export` -> `drive_copy` -> `ALL`. Beklenmeyen
hata `stage="FATAL"` ile tam traceback raporlanır ve tekrar fırlatılır.
"""

import json
import os
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import PIL.Image

# Transparent-460/HIM2K'da 100MP+ görseller var; PIL'in 179MP "decompression
# bomb" hata eşiğini aşabiliyor. Veri güvenilir akademik setlerden geldiği
# için limit kaldırılıyor (colab_devam_hucresi.py ile AYNI satır).
PIL.Image.MAX_IMAGE_PIXELS = None

import numpy as np
from PIL import Image

# --- Sabitler (colab_devam_hucresi.py ile AYNI) --------------------------
WORKDIR = "/content/my-bg-remover"
DRIVE_ROOT = "/content/drive/MyDrive"
DRIVE_OUTPUT_SUBDIR = "bg-remover-data"
SEED = 42
PER_IMAGE = 1  # stage5_make_composites'teki per_image=1 ile AYNI (drift önleme)
MAX_LONG_SIDE = 2048  # bu kenardan uzun fg'ler compose/augment ÖNCESİ küçültülür

STATUS_DIR = Path(DRIVE_ROOT) / "bg-remover-status"
LOG_PATH = STATUS_DIR / "log.txt"
STATUS_PATH = STATUS_DIR / "status.json"

# scripts/ bir paket değil — mutlak yolu sys.path'e ekliyoruz (colab_devam_hucresi.py
# ile AYNI mantık, os.chdir'e bağımlı değil).
SCRIPTS_DIR = str(Path(WORKDIR) / "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from benchmark.testset import append_entries, load_manifest  # noqa: E402


# ==========================================================================
# Durum raporlama — colab_devam_hucresi.py'den VERBATIM (aynı mekanizma).
# ==========================================================================
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def report(stage: str, status: str, **extra) -> None:
    """log.txt'e satır ekler + status.json'u (history biriktirerek) yeniden yazar."""
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
# Dev görsel küçültme — compose/augment ÖNCESİ, yalnız uzun kenar > 2048px.
# ==========================================================================
def downscale_giant(
    rgb: np.ndarray, alpha: np.ndarray, max_long_side: int = MAX_LONG_SIDE
) -> tuple[np.ndarray, np.ndarray, bool]:
    """rgb+alpha'yı BİRLİKTE (aynı yeni boyuta) küçültür — rgb LANCZOS (kalite),
    alpha `bgr/compositing.py::_resize_alpha` ile AYNI mode='F' + BILINEAR
    deseni (float32 hassasiyeti korunur). Uzun kenar zaten <= max_long_side ise
    hiçbir şey yapmadan (aynı nesneleri) döndürür -> dev-olmayan öğeler için
    `make_composites.run()` ile bit-birebir aynılık BOZULMAZ."""
    h, w = rgb.shape[:2]
    long_side = max(h, w)
    if long_side <= max_long_side:
        return rgb, alpha, False

    scale = max_long_side / long_side
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))

    rgb_ds = np.asarray(
        Image.fromarray(rgb, mode="RGB").resize((new_w, new_h), Image.LANCZOS), dtype=np.uint8
    )
    alpha_ds = np.asarray(
        Image.fromarray(alpha.astype(np.float32), mode="F").resize((new_w, new_h), Image.BILINEAR),
        dtype=np.float32,
    ).clip(0, 1)
    return rgb_ds, alpha_ds, True


# ==========================================================================
# Bitirici aşama — make_composites.run()'un kaldığı yerden, dev görselleri
# küçülterek tamamlar. Yardımcılar mc modülünden import edilir (KOPYALANMAZ).
# ==========================================================================
def stage_finisher() -> dict:
    report("finisher", "running")
    os.chdir(WORKDIR)

    import make_composites as mc  # scripts/ sys.path'te
    from bgr.compositing import augment, compose

    manifest_path = Path("data/train/manifest.jsonl")
    backgrounds_dir = Path("data/backgrounds")
    out_dir = Path("data/train_composites")
    out_img_dir = out_dir / "images"
    out_gt_dir = out_dir / "gt"
    out_manifest = out_dir / "manifest.jsonl"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_gt_dir.mkdir(parents=True, exist_ok=True)

    # run() ile AYNI filtre/sıra (id/copy/seed sözleşmesi için şart).
    rows = [r for r in load_manifest(str(manifest_path)) if r.get("gt_alpha")]

    bg_paths = sorted(p for p in backgrounds_dir.iterdir() if p.suffix.lower() in mc.IMG_EXTS)
    if not bg_paths:
        raise SystemExit(f"arka plan bulunamadı: {backgrounds_dir}")

    manifest_ids: set[str] = set()
    if out_manifest.exists():
        manifest_ids = {r["id"] for r in load_manifest(str(out_manifest))}

    # beklenen toplam: kaynak manifestteki her satır için kategori çarpanına göre kopya sayısı.
    expected_total = sum(PER_IMAGE * mc.multiplier(r["category"]) for r in rows)

    counts: dict[str, int] = {}
    produced = 0
    reconciled = 0
    skipped = 0
    downscaled_ids: list[str] = []

    for row in rows:
        category = row["category"]
        n_copies = PER_IMAGE * mc.multiplier(category)
        out_ids = [f"{row['id']}_v{ci:02d}" for ci in range(n_copies)]

        # önce DİSK durumuna bak (yalnız manifest'e değil) — run()'un toplu-append
        # açığını (dosya var, satır yok) burada onarıyoruz.
        pending: list[tuple[str, Path, Path]] = []
        for out_id in out_ids:
            img_path = out_img_dir / f"{out_id}.jpg"
            gt_path = out_gt_dir / f"{out_id}.png"
            files_ok = img_path.exists() and gt_path.exists()
            row_exists = out_id in manifest_ids

            if files_ok and row_exists:
                skipped += 1
                continue
            if files_ok and not row_exists:
                # dosyalar zaten yazılmış (önceki koşu kesintiye uğramış) -> yalnız satırı ekle.
                entry = {"id": out_id, "image": str(img_path), "category": category, "gt_alpha": str(gt_path)}
                append_entries(str(out_manifest), [entry])
                manifest_ids.add(out_id)
                reconciled += 1
                continue
            # dosyalar eksik -> üretilecek (satır olsun ya da olmasın; duplicate satır
            # asla eklenmeyecek, aşağıda tekrar kontrol ediliyor).
            pending.append((out_id, img_path, gt_path))

        if not pending:
            continue

        fg_rgb = mc._load_rgb(Path(row["image"]))
        alpha = mc._load_alpha(Path(row["gt_alpha"]), target_size=(fg_rgb.shape[1], fg_rgb.shape[0]))

        for out_id, img_path, gt_path in pending:
            rng = mc._item_rng(SEED, out_id)  # run() ile AYNI alt-akış türetimi

            item_fg_rgb, item_alpha, was_ds = downscale_giant(fg_rgb, alpha)
            if was_ds:
                downscaled_ids.append(out_id)

            if category in mc.NO_COMPOSE_CATEGORIES:
                out_rgb, out_alpha = item_fg_rgb, item_alpha
            else:
                bg_idx = int(rng.integers(0, len(bg_paths)))
                bg_rgb = mc._load_rgb(bg_paths[bg_idx])
                out_rgb, out_alpha = compose(item_fg_rgb, item_alpha, bg_rgb, rng)

            out_rgb, out_alpha = augment(out_rgb, out_alpha, rng)
            mc._save_pair(out_rgb, out_alpha, img_path, gt_path)

            if out_id not in manifest_ids:
                entry = {"id": out_id, "image": str(img_path), "category": category, "gt_alpha": str(gt_path)}
                append_entries(str(out_manifest), [entry])  # HER ÖGEDEN SONRA -> kesintiye dayanıklı
                manifest_ids.add(out_id)

            counts[category] = counts.get(category, 0) + 1
            produced += 1

            if produced % 100 == 0:
                report("finisher", "progress", produced=produced, downscaled=len(downscaled_ids))

    actual_total = len(list(out_img_dir.glob("*.jpg")))
    per_category_actual: dict[str, int] = {}
    if out_manifest.exists():
        for r in load_manifest(str(out_manifest)):
            per_category_actual[r["category"]] = per_category_actual.get(r["category"], 0) + 1

    ok = actual_total == expected_total
    print(f"Bitirici: {produced} yeni üretildi, {reconciled} satır onarıldı, {skipped} zaten tamdı.")
    print(f"Dev boyut nedeniyle küçültülen öğe sayısı: {len(downscaled_ids)}")
    print(f"Beklenen toplam: {expected_total}  Gerçek toplam (images/): {actual_total}  Eşleşiyor mu: {ok}")
    for cat, c in sorted(per_category_actual.items()):
        print(f"  {cat}: {c}")
    if not ok:
        print("UYARI: beklenen ve gerçek toplam eşleşmiyor — export'a geçmeden önce incele.")

    report(
        "finisher", "done",
        produced=produced, reconciled=reconciled, skipped=skipped,
        downscaled_count=len(downscaled_ids), expected_total=expected_total,
        actual_total=actual_total, counts_match=ok, per_category=per_category_actual,
    )
    return {
        "counts": counts, "expected_total": expected_total, "actual_total": actual_total, "ok": ok,
        "downscaled_ids": downscaled_ids,
    }


# ==========================================================================
# Export + Drive kopyalama — colab_devam_hucresi.py Stage 6/7'den VERBATIM.
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
    os.chdir(WORKDIR)
    stage_finisher()
    stats = stage6_export()
    stage7_drive_copy(stats)
    report("ALL", "done")


try:
    main()
except Exception:
    tb = traceback.format_exc()
    report("FATAL", "error", traceback=tb)
    raise
