"""V6 VERİ GÜNCELLEME HÜCRESİ — taze bir (ÜCRETSİZ, CPU yeterli — GPU GEREKMEZ)
Colab oturumunda, mevcut Drive veri setine (`bg-remover-data/TRAIN`) yalnız YENİ
v6 türev kopyalarını ekler (GitHub issue #1'in iki kusuru için veri düzeltmesi):
- **kadraj-kırpma** (`{stem}_e00`, ~9.000): kadraja değen özneler silinmesin —
  özne bbox'ını bir kenardan %20-60 kesen kırpmalar (GT alpha'sı DEĞİŞMEZ,
  kadraja değen kısım katı kalır),
- **karma-opaklık** (`{stem}_m00`/`_m01`, <= 4.000): saydam nesnelerin katı
  parçaları yarı saydamlaşmasın — hem katı hem yumuşak alpha'lı transparent
  çiftlerin augment'li kopyaları.
Üretim mantığının tamamı `scripts/make_v6_copies.py`'de (birim testli); bu
hücre yalnız Colab akışını (mount → kaynak → üretim → export → Drive merge)
yönetir. HİÇBİR mevcut dosyayı silmez/üzerine yazmaz.

KAYNAK / ATIF: akış kalıbı (Drive mount HERŞEYDEN önce → `report()` stage
takibi → `_listdir_retry` Errno 5 koruması → TRAIN-only merge → iş sonunda
`drive.flush_and_unmount()`) `training/v4_veri_guncelleme_hucresi.py`'den
alındı — 2026-07-12 dersi AYNEN geçerli: Drive yazımları asenkron tamponlanır,
flush olmadan VM kapatılırsa dosyalar SESSİZCE kaybolur.

V4'TEN FARKI — İNDİRME YOK: v6 üretiminin kaynağı dış veri setleri değil,
Drive'daki MEVCUT TRAIN'in kendisi. 52k+ küçük dosyayı Drive FUSE'dan tek tek
kopyalamak yerine `training/veri_tar_paketleme_hucresi.py`'nin ürettiği tar
shard'ları (`bg-remover-data/tar/TRAIN_shard_XX.tar`, içinde `im/<dosya>` +
`gt/<dosya>`; `_manifest.json`) lokale kopyalanıp açılır — `train_colab.ipynb`
hücre (c)'deki tar yolu kalıbının aynısı (byte doğrulama dahil). Bu lokal
TRAIN, üretimin kaynağıdır.

VAL SIZINTI KORUMASI (v3 dersi): tar shard'ları Drive TRAIN'in TAMAMINI içerir
— eğitim tarafında VAL'e ayrılan stem'ler de içinde. Bir VAL stem'inin (ya da
AYNI kaynak görselin başka bir kopyasının) `_e00`/`_m00` türevini TRAIN'e
eklemek, o görselin hem TRAIN hem VAL'de görülmesi demek olurdu.
`bg-remover-status/val_stems.json` varsa: val stem'lerinin kendileri + kaynak
id'si (`tcl.strip_composite_copy_suffix`) bir val stem'inin kaynağıyla eşleşen
TÜM stem'ler kaynak havuzundan çıkarılır (`tcl.derive_val_excluded_source_ids`).

TAR'LAR YENİDEN PAKETLENMEZ: eğitim tarafı (`train_colab.ipynb` hücre (c))
tar'ları açtıktan sonra Drive'a sonradan eklenen çiftleri (delta) `copy_pairs`
ile zaten tamamlıyor — hücre sonunda kullanıcıya not basılır.

ÖN KOŞULLAR: repo `/content/my-bg-remover`'da klonlanmış ve `pip install -e .`
yapılmış olmalı; Drive'da `bg-remover-data/TRAIN/{im,gt}`, `bg-remover-data/
tar/_manifest.json` (paketleme hücresi koşmuş olmalı) ve `bg-remover-data/
train_composites_manifest.jsonl` mevcut olmalı. Repo GÜNCEL olmalı (env
aşaması idempotent `git pull` dener): `scripts/make_v6_copies.py` bu hücreyle
aynı çalışmada eklendi — eski bir klonla koşarsanız `stage_v6` net bir Türkçe
hata mesajıyla durur.

Durum takibi v4 hücresiyle AYNI mekanizma (`report()` ->
`bg-remover-status/log.txt` + `status.json`) — aşamalar: env, tar_fetch,
categories, v6, export, drive_copy, (bitişte) ALL.
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

# TRAIN havuzunda 100MP+ kaynaklı kompozitler olabilir; PIL'in 179MP
# "decompression bomb" hata eşiği kaldırılır (bkz. v4 hücresi aynı satır).
PIL.Image.MAX_IMAGE_PIXELS = None

# --- Sabitler (v4_veri_guncelleme_hucresi.py ile AYNI Drive yerleşimi) ---
WORKDIR = "/content/my-bg-remover"
DRIVE_ROOT = "/content/drive/MyDrive"
DRIVE_OUTPUT_SUBDIR = "bg-remover-data"
DRIVE_STATUS_SUBDIR = "bg-remover-status"
TAR_SUBDIR = "tar"
SEED = 42

# --- v6'ya özgü sabitler ---
LOCAL_TRAIN_ROOT = Path("/content/v6_train_src")  # tar'ların açıldığı lokal TRAIN (im/ + gt/)
TAR_CACHE = Path("/content/tar_cache_v6")         # shard'ların geçici lokal kopyası (tek tek silinir)
V6_OUT_DIR = Path("data/train_v6")                # make_v6_copies.run() çıktısı (yerel, WORKDIR'e göre)
EXPORT_DIR = "/content/birefnet_format_v6"        # export_birefnet.export() çıktısı
EDGE_COUNT = 9000                                 # kadraj-kırpma hedefi (~9k)
MIXED_CAP = 4000                                  # karma-opaklık kopya üst sınırı (kaynak x 2)

STATUS_DIR = Path(DRIVE_ROOT) / DRIVE_STATUS_SUBDIR
LOG_PATH = STATUS_DIR / "log.txt"
STATUS_PATH = STATUS_DIR / "status.json"

# scripts/ bir paket değil — make_v6_copies/export_birefnet'i import edebilmek
# için mutlak yolu sys.path'e ekliyoruz (bkz. v4_veri_guncelleme_hucresi.py).
SCRIPTS_DIR = str(Path(WORKDIR) / "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import training.train_colab_lib as tcl  # noqa: E402  (pip install -e . ile kurulu paket)


# ==========================================================================
# Durum raporlama — `v4_veri_guncelleme_hucresi.py::report`'la BİREBİR AYNI.
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
# Drive FUSE Errno 5 koruması — v4 hücresindeki _listdir_retry kalıbının kopyası.
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


def _n_files(d: Path) -> int:
    return sum(1 for p in d.iterdir() if p.is_file()) if d.is_dir() else 0


# ==========================================================================
# Stage "env" — Drive bağlama (HERŞEYDEN önce, STATUS_DIR Drive'da!) + repo
# git pull (idempotent). Kaynak: v4 hücresi stage0_env_sanity — make_v6_copies
# bu hücreyle aynı çalışmada eklendiği için eski klon en olası hata kaynağı.
# ==========================================================================
def _git_pull_idempotent() -> None:
    """Repo'yu günceller — `git pull --ff-only` zaten günceldeyse no-op
    (idempotent); ağ yoksa/çakışma varsa UYARI verip devam eder (make_v6_copies
    eksikse stage_v6 zaten net mesajla durduracak)."""
    try:
        r = subprocess.run(
            ["git", "-C", WORKDIR, "pull", "--ff-only"],
            capture_output=True, text=True, timeout=180,
        )
        print(f"git pull: rc={r.returncode} {r.stdout.strip() or r.stderr.strip()}")
        if r.returncode != 0:
            print("UYARI: git pull başarısız — repo eski kalmış olabilir; make_v6_copies.py "
                  "eksikse aşağıda net hatayla durulacak.")
    except Exception as e:
        print(f"UYARI: git pull çalıştırılamadı ({e}) — mevcut klonla devam ediliyor.")


def stage0_env() -> None:
    # Drive HERŞEYDEN ÖNCE bağlanır (report() dahil — STATUS_DIR Drive'da!);
    # drive.mount idempotenttir. Kaynak: v4 hücresi aynı aşama.
    from google.colab import drive

    drive.mount("/content/drive")
    assert Path(DRIVE_ROOT).is_dir(), f"Drive bağlanamadı: {DRIVE_ROOT} yok"

    report("env", "running")
    os.chdir(WORKDIR)
    _git_pull_idempotent()

    free_gb = shutil.disk_usage("/content").free / 1e9
    print(f"lokal boş disk: {free_gb:.0f} GB (gerekli ~35 GB: tar açılımı + v6 çıktısı)")
    report("env", "done", cwd=str(Path.cwd()), free_gb=round(free_gb, 1))


# ==========================================================================
# Stage "tar_fetch" — Drive'daki tar shard'larını lokale kopyala + aç.
# Kaynak kalıp: train_colab.ipynb hücre (c) "hızlı yol" (byte doğrulama,
# shard başına tek lokal tar, açtıktan sonra silme) — İNDİRME YOK, kaynak
# veri Drive'daki MEVCUT TRAIN'in tar paketi.
# ==========================================================================
def stage_tar_fetch() -> int:
    report("tar_fetch", "running")
    tar_dir = Path(DRIVE_ROOT) / DRIVE_OUTPUT_SUBDIR / TAR_SUBDIR
    manifest_path = tar_dir / "_manifest.json"
    assert manifest_path.exists(), (
        f"{manifest_path} yok — önce training/veri_tar_paketleme_hucresi.py'yi (ücretsiz CPU "
        f"Colab) koşup TRAIN'i tar shard'larına paketleyin; v6 üretiminin kaynağı bu shard'lardır "
        f"(52k+ küçük dosyayı Drive FUSE'dan tek tek kopyalamak ~75 dk sürerdi)."
    )
    manifest = json.loads(manifest_path.read_text())
    total_pairs = tcl.validate_tar_manifest(manifest)  # iç tutarlılık: shard toplamı == total_pairs

    local_im = LOCAL_TRAIN_ROOT / "im"
    local_gt = LOCAL_TRAIN_ROOT / "gt"
    n_im, n_gt = _n_files(local_im), _n_files(local_gt)
    if n_im >= total_pairs and n_im == n_gt:
        print(f"Tar indirme/açma ATLANDI: lokalde zaten {n_im} çift var (>= manifest {total_pairs}).")
    else:
        LOCAL_TRAIN_ROOT.mkdir(parents=True, exist_ok=True)
        TAR_CACHE.mkdir(parents=True, exist_ok=True)
        for sh in manifest["shards"]:
            src, dst = tar_dir / sh["name"], TAR_CACHE / sh["name"]
            if not (dst.exists() and dst.stat().st_size == sh["bytes"]):
                shutil.copy2(src, dst)  # tek BÜYÜK dosya — Drive FUSE'da hızlı
                if dst.stat().st_size != sh["bytes"]:
                    raise RuntimeError(
                        f"{sh['name']}: Drive'dan kopyalanan boyut ({dst.stat().st_size}) "
                        f"manifest'tekiyle ({sh['bytes']}) uyuşmuyor — aktarım yarım kalmış "
                        f"olabilir, hücreyi yeniden koşun."
                    )
            with tarfile.open(dst) as tf:
                tf.extractall(LOCAL_TRAIN_ROOT, filter="data")  # üyeler: im/<dosya> + gt/<dosya>
            dst.unlink()  # açılan shard'ın lokal tar'ı hemen silinir (disk güvenliği)
            print(f"{sh['name']}: kopyalandı + açıldı ({sh['pairs']} çift, {sh['bytes'] / 1e9:.2f} GB).")
        n_im, n_gt = _n_files(local_im), _n_files(local_gt)
        if n_im != n_gt or n_im < total_pairs:
            raise RuntimeError(
                f"tar açılımı manifest'le uyuşmuyor: im={n_im}, gt={n_gt}, beklenen en az "
                f"{total_pairs} (ve im == gt) — shard'lar eksik/bozuk olabilir; paketleme "
                f"hücresini yeniden koşun."
            )

    print(f"Lokal TRAIN kaynağı hazır: {n_im} çift -> {LOCAL_TRAIN_ROOT}")
    report("tar_fetch", "done", pairs=n_im, total_pairs_manifest=total_pairs)
    return n_im


# ==========================================================================
# Stage "categories" — Drive'daki train_composites_manifest.jsonl'den
# stem -> kategori haritası (tcl.load_stem_categories — id/category okur) +
# VAL sızıntı koruması için hariç tutulacak kaynak stem kümesi.
# ==========================================================================
def stage_categories() -> tuple[dict[str, str], set[str]]:
    report("categories", "running")
    drive_manifest = Path(DRIVE_ROOT) / DRIVE_OUTPUT_SUBDIR / "train_composites_manifest.jsonl"
    assert drive_manifest.exists(), (
        f"{drive_manifest} yok — kategori haritası olmadan v6 kaynak seçimi (kategori başına "
        f"orantılı + transparent filtresi) yapılamaz; Faz 2 / v3 / v4 hücreleri koşmuş olmalı."
    )
    category_by_stem = tcl.load_stem_categories(drive_manifest)
    print(f"Kategori haritası: {len(category_by_stem)} stem.")

    # VAL sızıntı koruması (bkz. modül docstring'i): val stem'lerinin kendileri
    # + aynı KAYNAK görselin diğer kopyaları kaynak havuzundan çıkarılır.
    exclude_stems: set[str] = set()
    val_json = STATUS_DIR / "val_stems.json"
    if val_json.exists():
        val_stems = json.loads(val_json.read_text())["val_stems"]
        excluded_ids, unmatched = tcl.derive_val_excluded_source_ids(val_stems)
        if unmatched:
            print(f"UYARI: {len(unmatched)} val stem'i `_v/_o<NN>` son ek desenine uymuyor "
                  f"(ör. {unmatched[:5]}) — bunlar yalnız kendi stem'leriyle hariç tutulur, "
                  f"kaynak-id düzeyinde koruma o stem'ler için uygulanamaz (v3 dersi).")
        exclude_stems = set(val_stems) | {
            s for s in category_by_stem
            if tcl.strip_composite_copy_suffix(s) in excluded_ids
        }
        print(f"VAL sızıntı koruması: {len(val_stems)} val stem'i -> {len(exclude_stems)} "
              f"stem kaynak havuzundan hariç tutulacak.")
    else:
        print(f"NOT: {val_json} yok (henüz hiç eğitim koşulmamış olabilir) — VAL hariç tutma "
              f"atlanıyor; yeni stem'ler zaten her zaman TRAIN'e gider.")

    report("categories", "done", stems=len(category_by_stem), excluded=len(exclude_stems))
    return category_by_stem, exclude_stems


# ==========================================================================
# Stage "v6" — ÜRETİM: scripts/make_v6_copies.py (birim testli). İmza/import
# uyuşmazlığında NET Türkçe hata mesajıyla durulur (v4 stage_textfx kalıbı),
# sessizce yarım veri üretilmez.
# ==========================================================================
def stage_v6(category_by_stem: dict[str, str], exclude_stems: set[str]) -> dict[str, int]:
    report("v6", "running")

    try:
        import make_v6_copies as mv6  # scripts/ sys.path'te
    except ImportError as e:
        raise RuntimeError(
            f"scripts/make_v6_copies.py import edilemedi ({e}) — repo'nuz güncel mi? "
            f"'git -C {WORKDIR} pull' deneyin (script bu hücreyle aynı çalışmada eklendi)."
        ) from e

    try:
        counts = mv6.run(
            train_im_dir=LOCAL_TRAIN_ROOT / "im",
            train_gt_dir=LOCAL_TRAIN_ROOT / "gt",
            category_by_stem=category_by_stem,
            out_dir=V6_OUT_DIR,
            seed=SEED,
            edge_count=EDGE_COUNT,
            mixed_cap=MIXED_CAP,
            exclude_stems=exclude_stems,
        )
    except TypeError as e:
        raise RuntimeError(
            f"make_v6_copies.run() beklenen imzayla çağrılamadı ({e}) — bu hücre "
            f"run(train_im_dir, train_gt_dir, category_by_stem, out_dir, seed, edge_count, "
            f"mixed_cap, exclude_stems) imzasını varsayar; scripts/make_v6_copies.py'nin "
            f"güncel imzasına bakıp çağrıyı uyarlayın."
        ) from e

    print("make_v6_copies.run() üretim:", counts)

    # Manifest guard (v3 dersi): boş/eksik manifest'le export'a GEÇME.
    # make_v6_copies'ın çıktı manifest'i {"id","category"} satırlarıdır —
    # export_birefnet TAM testset şeması (image + gt_alpha) istediği için
    # manifest_full'e dönüştürülür (v4 hücresi stage_textfx kalıbının aynısı).
    out_manifest = V6_OUT_DIR / "manifest.jsonl"
    if not out_manifest.exists():
        raise RuntimeError(f"{out_manifest} yok — make_v6_copies üretimi başarısız olmuş olmalı.")
    rows = [json.loads(line) for line in out_manifest.read_text().splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"{out_manifest} boş — export'a geçilmiyor (v3 dersi).")

    full_manifest = V6_OUT_DIR / "manifest_full.jsonl"
    with open(full_manifest, "w") as f:
        for r in rows:
            im_p = V6_OUT_DIR / "im" / f"{r['id']}.jpg"
            gt_p = V6_OUT_DIR / "gt" / f"{r['id']}.png"
            if not (im_p.exists() and gt_p.exists()):
                raise RuntimeError(f"manifest satırının dosyası eksik: {r['id']} — üretim yarım kalmış olabilir.")
            f.write(json.dumps({"id": r["id"], "image": str(im_p),
                                "category": r["category"], "gt_alpha": str(gt_p)},
                               ensure_ascii=False) + "\n")

    n_edge = sum(1 for r in rows if r["id"].endswith("_e00"))
    n_mixed = len(rows) - n_edge
    by_cat: dict[str, int] = {}
    for r in rows:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1
    print(f"PRE-FLIGHT — {out_manifest}: toplam {len(rows)} çift "
          f"(edge-crop: {n_edge}, mixed: {n_mixed}); kategori bazında:")
    for cat, n in sorted(by_cat.items(), key=lambda kv: -kv[1]):
        print(f"  {cat}: {n}")
    if n_edge < 100:
        print(f"UYARI: edge-crop sayısı çok düşük ({n_edge} < 100) — kaynak havuzu/kategori "
              f"haritası eksik olabilir, logları inceleyin.")

    report("v6", "done", counts=counts, total_pairs=len(rows), edge=n_edge, mixed=n_mixed,
           by_category=by_cat)
    return by_cat


# ==========================================================================
# Stage "export" — v4 kalıbı: export_birefnet.export() taze/boş bir yerel
# dizine karşı çalışır, diskte yalnız yeni v6 dosyaları oluşur.
# split_name="TRAIN": yeni stemler HER ZAMAN TRAIN'e gider (mevcut kural).
# ==========================================================================
def stage_export_v6() -> dict:
    report("export", "running")
    import export_birefnet as eb  # scripts/ sys.path'te

    stats = eb.export(
        manifest_path=str(V6_OUT_DIR / "manifest_full.jsonl"),
        out_dir=EXPORT_DIR,
        split_name="TRAIN",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    report("export", "done", stats=stats)
    return stats


# ==========================================================================
# Stage "drive_copy" — v4 kalıbı: var olan Drive TRAIN'e MERGE (dirs_exist_ok=
# True, silme/üzerine yazma yok; im/gt AYRI sayaçlı — 2026-07-12 dersi: yarım
# kalmış önceki bir yükleme im'i ulaşmış gt'si ulaşmamış çiftler bırakabilir)
# + kompozit manifest'e APPEND (tcl.merge_composite_manifest, dedupe'lu).
# ==========================================================================
def stage_drive_copy_v6() -> None:
    report("drive_copy", "running")
    src = Path(EXPORT_DIR)
    dst = Path(DRIVE_ROOT) / DRIVE_OUTPUT_SUBDIR
    dst_train_im = dst / "TRAIN" / "im"
    dst_train_gt = dst / "TRAIN" / "gt"
    assert dst_train_im.is_dir() and dst_train_gt.is_dir(), (
        f"Drive'da beklenen TRAIN verisi bulunamadı: {dst_train_im} / {dst_train_gt} — "
        f"bu hücre yalnız MEVCUT bir veri setine v6 türevlerini EKLEMEK içindir."
    )

    src_im_files = list((src / "TRAIN" / "im").iterdir())
    src_gt_files = list((src / "TRAIN" / "gt").iterdir())
    assert len(src_im_files) == len(src_gt_files), "yerel v6 export'unda im/gt sayıları uyuşmuyor!"

    # im ve gt AYRI sayılır (v4 hücresi / 2026-07-12 dersi — bkz. aşama yorumu).
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
    comp_manifest_local = V6_OUT_DIR / "manifest_full.jsonl"
    comp_manifest_drive = dst / "train_composites_manifest.jsonl"
    n_appended = tcl.merge_composite_manifest(comp_manifest_local, comp_manifest_drive)
    print(f"train_composites_manifest.jsonl: {n_appended} yeni satır eklendi (mevcut satırlar "
          f"KORUNDU, üzerine yazılmadı). Onarım koşusunda 0 olabilir — hata değil (v4 dersi).")

    print("\nBÜTÜNLÜK KONTROLÜ BAŞARILI — v6 (edge-crop + mixed) verisi Drive'a MERGE edildi.")
    report(
        "drive_copy", "done",
        added_im=growth_im, added_gt=growth_gt, added_manifest_rows=n_appended,
        total_im=post_im, total_gt=post_gt,
    )


# ==========================================================================
# Orkestrasyon — üst düzeyde koşar (hücre yapıştırılıp çalıştırıldığında).
# ==========================================================================
def main() -> None:
    stage0_env()                                   # Drive mount + git pull — her şeyden önce
    stage_tar_fetch()                              # tar shard'ları -> lokal TRAIN (üretim kaynağı)
    category_by_stem, exclude_stems = stage_categories()
    stage_v6(category_by_stem, exclude_stems)      # make_v6_copies.run() + manifest guard
    stage_export_v6()
    stage_drive_copy_v6()
    report("ALL", "done")
    print(
        "\nNOT: tar shard'ları YENİDEN PAKETLENMEDİ — bir sonraki eğitim koşusunda "
        "train_colab.ipynb hücre (c), tar'ları açtıktan sonra yeni ~13k çifti delta olarak "
        "copy_pairs ile Drive'dan tamamlayacak (birkaç dk sürer). İstersen "
        "training/veri_tar_paketleme_hucresi.py'yi yeniden koşup delta'yı sıfırlayabilirsin "
        "(DİKKAT: yeni stem'ler sıralamada araya girdiği için çoğu shard değişir ve yeniden "
        "paketlenir — ~1 saatlik ücretsiz CPU koşusu; delta copy_pairs genelde daha ucuz)."
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
