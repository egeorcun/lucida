"""VERİ TAR PAKETLEME HÜCRESİ — taze bir (ÜCRETSİZ, CPU yeterli — GPU GEREKMEZ)
Colab oturumunda TEK hücre olarak yapıştırılıp koşulur; Drive'daki
`bg-remover-data/TRAIN/{im,gt}` çiftlerini (52.882+52.882 küçük dosya) az
sayıda BÜYÜK tar shard'ına paketleyip `bg-remover-data/tar/` altına koyar.

NEDEN: her eğitim koşusunda `training/train_colab.ipynb` hücre (c) bu küçük
dosyaları Drive FUSE üzerinden VM'e TEK TEK kopyalıyor (~75 dk, ara sıra geçici
'Errno 5' hatalarıyla). Bu hücre BİR KEZ koşulduktan sonra eğitim tarafı
manifest'i görüp shard indirme+açma yoluna geçer (~10 dk, ~7x hızlanma).

KAYNAK / ATIF: akış kalıbı (Drive mount HERŞEYDEN önce → `report()` stage
takibi → `_listdir_retry` Errno 5 koruması → iş sonunda `drive.flush_and_
unmount()`) `training/v4_veri_guncelleme_hucresi.py`'den alındı. 2026-07-12
dersi AYNEN geçerli: Drive yazımları asenkron tamponlanır — flush olmadan VM
kapatılırsa dosyalar SESSİZCE kaybolur. `tar_shard_name` / `split_stems_to_
shards` / `validate_tar_manifest` fonksiyonları `training/train_colab_lib.py`
'den BİREBİR KOPYADIR (bu hücre paste-run tasarımı gereği repo klonu + `pip
install -e .` GEREKTİRMESİN diye) — TEK DOĞRULUK KAYNAĞI o dosyadır; kopya
saparsa `tests/test_train_colab_lib.py`'deki AST karşılaştırma testi kırmızıya
döner, drift görürseniz oradan güncelleyin.

AKIŞ:
1. Drive mount → `TRAIN/{im,gt}` listele (retry'lı) → im/gt stem eşleşmesi
   doğrula (eşleşmeyen varsa RuntimeError — yarım çift tar'a girmez).
2. Stem'ler SIRALI ve DETERMİNİSTİK şekilde `SHARD_SIZE`'lık dilimlere bölünür
   (52.882 çift, SHARD_SIZE=7000 -> 8 shard, shard başına ~6-7k çift). Her
   shard `TRAIN_shard_{k:02d}.tar` içinde `im/<dosya>` + `gt/<dosya>` yolları.
3. Her tar önce VM LOKAL diskinde oluşturulur (Drive FUSE'a doğrudan tar yazmak
   YAVAŞ), üye sayısı doğrulanır, Drive'a kopyalanıp boyutu doğrulanır ve
   lokal tar HEMEN silinir (VM disk güvenliği: disk ~100GB, veri ~30GB — yine
   de shard'lar tek tek işlenir, diskte aynı anda en fazla 1 shard durur).
4. İDEMPOTENT: Drive'da zaten var olan ve önceki manifest'te (final `_manifest
   .json` ya da ara `_manifest_partial.json`) beklenen çift sayısı + byte
   boyutuyla eşleşen shard ATLANIR — yarım kalmış bir koşu güvenle devam eder.
5. Nihai `bg-remover-data/tar/_manifest.json` yalnız TÜM shard'lar
   doğrulandıktan SONRA yazılır (toplam çift sayısı TRAIN listelemesiyle
   eşleşmezse RuntimeError); her shard sonrası `_manifest_partial.json`
   güncellenir. Eğitim notebook'u YALNIZ final manifest'e bakar — yarım
   paketleme, manifest yokluğundan anlaşılır ve eski copy_pairs yoluna düşülür.
6. Rapor + `drive.flush_and_unmount()`.
"""

import io
import json
import shutil
import tarfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# --- Sabitler (Drive yerleşimi v4_veri_guncelleme_hucresi.py ile AYNI) ---
DRIVE_ROOT = "/content/drive/MyDrive"
DRIVE_OUTPUT_SUBDIR = "bg-remover-data"
DRIVE_STATUS_SUBDIR = "bg-remover-status"
TAR_SUBDIR = "tar"                      # shard'lar + manifest buraya: bg-remover-data/tar/
SHARD_SIZE = 7000                       # 52.882 çift -> 8 shard (7x7000 + 1x3882), shard ~3-4GB
LOCAL_TAR_DIR = Path("/content/tar_build")  # tar'lar önce burada (lokal disk) oluşturulur

MANIFEST_NAME = "_manifest.json"            # FINAL — yalnız tüm shard'lar doğrulanınca yazılır
MANIFEST_PARTIAL_NAME = "_manifest_partial.json"  # her shard sonrası güncellenir (güvenli devam)

STATUS_DIR = Path(DRIVE_ROOT) / DRIVE_STATUS_SUBDIR
LOG_PATH = STATUS_DIR / "log.txt"
STATUS_PATH = STATUS_DIR / "status.json"


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
# Drive FUSE Errno 5 korumaları — `v4_veri_guncelleme_hucresi.py::
# stage_drive_copy_textfx içindeki _listdir_retry kalıbının kopyası (listeleme)
# + aynı kalıbın dosya OKUMA'ya uyarlanmış hali (_read_with_retry): tar'a
# eklerken 52k dosya tek tek okunur, geçici I/O hatası tüm shard'ı düşürmesin.
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


def _read_with_retry(p: Path, attempts: int = 4, wait_s: int = 15) -> bytes:
    """Tek dosyayı Drive FUSE'dan okur; geçici OSError'da bekleyip yeniden
    dener. Tar'a doğrudan `tf.add(p)` ile eklemek yerine önce belleğe okuyoruz:
    `tf.add` okuma ORTASINDA hata verirse tar akışı yarım üye ile bozulurdu —
    burada okuma tamamlanmadan tar'a tek bayt yazılmaz."""
    for i in range(attempts):
        try:
            return p.read_bytes()
        except OSError as e:
            if i == attempts - 1:
                raise
            print(f"UYARI: {p} okunurken {e} — {wait_s}s bekleyip yeniden denenecek "
                  f"({i + 1}/{attempts - 1}).")
            time.sleep(wait_s)
    raise AssertionError("unreachable")


# ==========================================================================
# training/train_colab_lib.py'den BİREBİR KOPYA (bkz. modül docstring'i:
# paste-run hücresi repo klonu gerektirmesin diye; drift AST testiyle yakalanır).
# ==========================================================================
def tar_shard_name(index: int) -> str:
    """`index` (0 tabanlı) için shard tar dosya adı: `TRAIN_shard_{index:02d}.tar`.
    Adlandırma sözleşmesinin TEK kaynağı — paketleyen hücre bu adla yazar,
    notebook tarafı manifest'teki `name` alanları üzerinden okur."""
    if index < 0:
        raise ValueError(f"index >= 0 olmalı: {index}")
    return f"TRAIN_shard_{index:02d}.tar"


def split_stems_to_shards(stems: list[str], shard_size: int) -> list[list[str]]:
    """`stems`'i SIRALI ve DETERMİNİSTİK shard'lara böler: önce `sorted()`,
    sonra ardışık `shard_size`'lık dilimler — sonuç girdi (dosya sistemi
    listeleme) sırasından BAĞIMSIZ, yeniden koşularda AYNIDIR (idempotent
    shard atlama ancak böyle mümkün: aynı stem kümesi her koşuda aynı shard'a
    düşer). Toplam KORUNUR: dilimlerin ardışık birleşimi `sorted(stems)`'in
    kendisidir (kayıp/tekrar yok); son dilim `shard_size`'dan kısa olabilir.
    Boş liste -> boş liste. `shard_size <= 0` -> ValueError."""
    if shard_size <= 0:
        raise ValueError(f"shard_size > 0 olmalı: {shard_size}")
    stems_sorted = sorted(stems)
    return [stems_sorted[i : i + shard_size] for i in range(0, len(stems_sorted), shard_size)]


def validate_tar_manifest(manifest: dict, expected_total: int | None = None) -> int:
    """Tar manifest'inin (`bg-remover-data/tar/_manifest.json`) iç tutarlılığını
    doğrular ve `total_pairs`'ı döndürür; her tutarsızlıkta NET bir RuntimeError
    fırlatır (sessizce devam etmek = eksik/bozuk veriyle eğitime girme riski):
    - `shards` boş olmayan bir liste, `total_pairs` pozitif bir tamsayı olmalı;
    - her shard girdisinde `name`/`pairs`/`bytes` bulunmalı ve `pairs`/`bytes` > 0;
    - shard adları benzersiz olmalı (aynı tar iki kez sayılmasın);
    - shard `pairs` toplamı `total_pairs`'a eşit olmalı;
    - `expected_total` verilmişse `total_pairs` ona da eşit olmalı (paketleyen
      hücre kaynak TRAIN listeleme uzunluğunu geçirir — Drive listelemesiyle
      manifest'in aynı veri setini anlattığı garanti edilir)."""
    shards = manifest.get("shards")
    total = manifest.get("total_pairs")
    if not isinstance(shards, list) or not shards:
        raise RuntimeError(
            f"tar manifest'inde boş olmayan bir 'shards' listesi yok (paketleme hücresi "
            f"hiç koşmamış ya da yarım kalmış olabilir): {shards!r}"
        )
    if not isinstance(total, int) or total <= 0:
        raise RuntimeError(f"tar manifest'inde pozitif bir 'total_pairs' alanı yok: {total!r}")
    names: list[str] = []
    total_from_shards = 0
    for entry in shards:
        name, pairs, n_bytes = entry.get("name"), entry.get("pairs"), entry.get("bytes")
        if not name or not isinstance(pairs, int) or pairs <= 0 or not isinstance(n_bytes, int) or n_bytes <= 0:
            raise RuntimeError(f"bozuk shard girdisi (name/pairs/bytes eksik ya da <= 0): {entry!r}")
        names.append(name)
        total_from_shards += pairs
    if len(set(names)) != len(names):
        raise RuntimeError(f"tar manifest'inde tekrar eden shard adları var: {names}")
    if total_from_shards != total:
        raise RuntimeError(
            f"shard 'pairs' toplamı ({total_from_shards}) manifest'in 'total_pairs' değeriyle "
            f"({total}) uyuşmuyor — manifest bozuk, paketleme hücresi yeniden koşulmalı."
        )
    if expected_total is not None and total != expected_total:
        raise RuntimeError(
            f"manifest'in 'total_pairs' değeri ({total}) beklenen kaynak çift sayısıyla "
            f"({expected_total}) uyuşmuyor."
        )
    return total


# ==========================================================================
# Stage "env" — Drive mount (HERŞEYDEN önce: STATUS_DIR Drive'da!) + kaynak
# dizin kontrolü. Kaynak kalıp: v4_veri_guncelleme_hucresi.py::stage0_env_sanity.
# ==========================================================================
def stage0_env() -> tuple[Path, Path, Path]:
    from google.colab import drive

    drive.mount("/content/drive")
    assert Path(DRIVE_ROOT).is_dir(), f"Drive bağlanamadı: {DRIVE_ROOT} yok"

    report("env", "running")
    data_dir = Path(DRIVE_ROOT) / DRIVE_OUTPUT_SUBDIR
    train_im = data_dir / "TRAIN" / "im"
    train_gt = data_dir / "TRAIN" / "gt"
    assert train_im.is_dir() and train_gt.is_dir(), (
        f"Drive'da beklenen veri bulunamadı: {train_im} / {train_gt} — bu hücre MEVCUT bir "
        f"TRAIN veri setini paketlemek içindir (Faz 2 / v4 hücreleri önce koşmuş olmalı)."
    )
    tar_dir = data_dir / TAR_SUBDIR
    report("env", "done", train_im=str(train_im), tar_dir=str(tar_dir))
    return train_im, train_gt, tar_dir


# ==========================================================================
# Stage "list" — TRAIN/{im,gt} listeleme (retry'lı) + çift doğrulama.
# ==========================================================================
def stage_list(train_im: Path, train_gt: Path) -> tuple[dict[str, Path], dict[str, Path], list[str]]:
    report("list", "running")

    def _by_stem(d: Path, label: str) -> dict[str, Path]:
        # macOS AppleDouble artıkları (`._*`) görüntü değildir — filtrele (v4 kalıbı).
        files = [p for p in _listdir_retry(d) if p.is_file() and not p.name.startswith("._")]
        by_stem: dict[str, Path] = {}
        dupes: list[str] = []
        for p in sorted(files):
            if p.stem in by_stem:
                dupes.append(f"{by_stem[p.stem].name} <-> {p.name}")
            by_stem[p.stem] = p
        if dupes:
            raise RuntimeError(
                f"{label} dizininde aynı stem'e sahip birden çok dosya var (hangisi çiftin "
                f"parçası belirsiz — tar'a hangisinin gireceği tanımsız olurdu): "
                f"{dupes[:10]}{' ...' if len(dupes) > 10 else ''}"
            )
        return by_stem

    im_by_stem = _by_stem(train_im, "TRAIN/im")
    gt_by_stem = _by_stem(train_gt, "TRAIN/gt")

    im_only = sorted(set(im_by_stem) - set(gt_by_stem))
    gt_only = sorted(set(gt_by_stem) - set(im_by_stem))
    if im_only or gt_only:
        raise RuntimeError(
            f"TRAIN im/gt stem eşleşmesi BOZUK — yarım çiftler tar'a giremez (eğitim tarafı "
            f"bu dosyalarla çökerdi): gt'siz im={len(im_only)} (ör. {im_only[:5]}), "
            f"im'siz gt={len(gt_only)} (ör. {gt_only[:5]}). Önce veri setini onarın "
            f"(2026-07-12 dersi: yarım kalan Drive flush'ı böyle kırık çiftler bırakabilir)."
        )

    stems = sorted(im_by_stem)
    assert stems, "TRAIN/im boş — paketlenecek veri yok."
    print(f"TRAIN: {len(stems)} çift doğrulandı (im={len(im_by_stem)}, gt={len(gt_by_stem)}).")
    report("list", "done", pairs=len(stems))
    return im_by_stem, gt_by_stem, stems


# ==========================================================================
# Stage "pack" — shard'ları lokalde oluştur, Drive'a kopyala (boyut doğrulamalı),
# lokal tar'ı sil. İDEMPOTENT: önceki (final ya da partial) manifest'le eşleşen
# ve Drive'da doğru boyutta duran shard atlanır.
# ==========================================================================
def _load_previous_entries(tar_dir: Path) -> dict[str, dict]:
    prev: dict[str, dict] = {}
    # final önce, partial SONRA okunur (partial daha güncel bir yarım koşuya ait olabilir).
    for name in (MANIFEST_NAME, MANIFEST_PARTIAL_NAME):
        p = tar_dir / name
        if not p.exists():
            continue
        try:
            for e in json.loads(p.read_text()).get("shards", []):
                if e.get("name"):
                    prev[e["name"]] = e
        except Exception as exc:
            print(f"UYARI: {p} okunamadı ({exc}) — bu manifest atlama için KULLANILMAYACAK.")
    return prev


def _write_partial_manifest(tar_dir: Path, entries: list[dict], n_source_pairs: int) -> None:
    (tar_dir / MANIFEST_PARTIAL_NAME).write_text(json.dumps({
        "note": "YARIM koşu ara durumu — eğitim tarafı YALNIZ _manifest.json'a bakar.",
        "updated_at": _now(),
        "shard_size": SHARD_SIZE,
        "source_pairs": n_source_pairs,
        "shards": entries,
    }, ensure_ascii=False, indent=2))


def stage_pack(im_by_stem: dict[str, Path], gt_by_stem: dict[str, Path],
               stems: list[str], tar_dir: Path) -> list[dict]:
    report("pack", "running")
    shards = split_stems_to_shards(stems, SHARD_SIZE)
    print(f"{len(stems)} çift -> {len(shards)} shard (SHARD_SIZE={SHARD_SIZE}).")
    tar_dir.mkdir(parents=True, exist_ok=True)
    LOCAL_TAR_DIR.mkdir(parents=True, exist_ok=True)
    prev_entries = _load_previous_entries(tar_dir)

    entries: list[dict] = []
    for k, shard in enumerate(shards):
        name = tar_shard_name(k)
        drive_tar = tar_dir / name

        # İdempotent atlama: önceki manifest girdisi bu shard'ın beklenen çift
        # sayısıyla eşleşiyor VE Drive'daki dosya o girdinin byte boyutunda.
        prev = prev_entries.get(name)
        if (
            prev
            and prev.get("pairs") == len(shard)
            and isinstance(prev.get("bytes"), int) and prev["bytes"] > 0
            and drive_tar.exists() and drive_tar.stat().st_size == prev["bytes"]
        ):
            print(f"{name}: Drive'da mevcut ve manifest'le eşleşiyor "
                  f"({prev['pairs']} çift, {prev['bytes'] / 1e9:.2f} GB) — ATLANDI.")
            entries.append(prev)
            _write_partial_manifest(tar_dir, entries, len(stems))
            continue

        t0 = time.time()
        local_tar = LOCAL_TAR_DIR / name
        if local_tar.exists():
            local_tar.unlink()  # önceki koşudan yarım kalmış lokal tar — baştan oluştur
        with tarfile.open(local_tar, "w") as tf:
            for i, stem in enumerate(shard, start=1):
                for src, arc_prefix in ((im_by_stem[stem], "im"), (gt_by_stem[stem], "gt")):
                    data = _read_with_retry(src)
                    info = tarfile.TarInfo(name=f"{arc_prefix}/{src.name}")
                    info.size = len(data)
                    info.mtime = int(time.time())
                    tf.addfile(info, io.BytesIO(data))
                if i % 1000 == 0:
                    rate = i / max(time.time() - t0, 1e-9)
                    print(f"  {name}: {i}/{len(shard)} çift eklendi "
                          f"({rate:.1f} çift/sn, ETA {(len(shard) - i) / rate:.0f} sn)")

        # Üye sayısı doğrulaması (lokal disk — hızlı): her çift 2 üye (im+gt).
        with tarfile.open(local_tar) as tf:
            n_members = len(tf.getnames())
        if n_members != 2 * len(shard):
            raise RuntimeError(
                f"{name}: tar üye sayısı beklenenle uyuşmuyor: {n_members} != {2 * len(shard)} "
                f"— lokal tar bozuk, hücreyi yeniden koşun (shard baştan oluşturulur)."
            )

        n_bytes = local_tar.stat().st_size
        print(f"{name}: lokal tar hazır ({len(shard)} çift, {n_bytes / 1e9:.2f} GB, "
              f"{time.time() - t0:.0f} sn) — Drive'a kopyalanıyor...")
        shutil.copy2(local_tar, drive_tar)
        drive_size = drive_tar.stat().st_size
        if drive_size != n_bytes:
            raise RuntimeError(
                f"{name}: Drive kopyasının boyutu ({drive_size}) lokal tar'la ({n_bytes}) "
                f"uyuşmuyor — aktarım yarım kalmış olabilir, hücreyi yeniden koşun."
            )
        local_tar.unlink()  # VM disk güvenliği: diskte aynı anda en fazla 1 shard durur

        entry = {"name": name, "pairs": len(shard), "files": 2 * len(shard), "bytes": n_bytes}
        entries.append(entry)
        _write_partial_manifest(tar_dir, entries, len(stems))  # güvenli devam noktası
        print(f"{name}: Drive'a yazıldı ve doğrulandı ({time.time() - t0:.0f} sn toplam).")

    report("pack", "done", shards=len(entries))
    return entries


# ==========================================================================
# Stage "manifest" — nihai _manifest.json (yalnız TÜM shard'lar doğrulanınca).
# ==========================================================================
def stage_manifest(entries: list[dict], stems: list[str],
                   im_by_stem: dict[str, Path], gt_by_stem: dict[str, Path], tar_dir: Path) -> dict:
    report("manifest", "running")
    manifest = {
        "created_at": _now(),
        "shard_size": SHARD_SIZE,
        "total_pairs": sum(e["pairs"] for e in entries),
        "source_counts": {"im": len(im_by_stem), "gt": len(gt_by_stem)},
        "shards": entries,
    }
    # Toplam çift sayısı TRAIN listelemesiyle eşleşmiyorsa RuntimeError (görev şartı).
    validate_tar_manifest(manifest, expected_total=len(stems))
    (tar_dir / MANIFEST_NAME).write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    partial = tar_dir / MANIFEST_PARTIAL_NAME
    if partial.exists():
        partial.unlink()  # iş bitti — ara durum dosyası kafa karıştırmasın
    print(f"{tar_dir / MANIFEST_NAME}: {len(entries)} shard, toplam {manifest['total_pairs']} çift.")
    report("manifest", "done", total_pairs=manifest["total_pairs"], shards=len(entries))
    return manifest


# ==========================================================================
# Orkestrasyon — üst düzeyde koşar (hücre yapıştırılıp çalıştırıldığında).
# ==========================================================================
def main() -> None:
    train_im, train_gt, tar_dir = stage0_env()  # Drive mount BURADA — her şeyden önce
    im_by_stem, gt_by_stem, stems = stage_list(train_im, train_gt)
    entries = stage_pack(im_by_stem, gt_by_stem, stems, tar_dir)
    stage_manifest(entries, stems, im_by_stem, gt_by_stem, tar_dir)
    report("ALL", "done")
    # KRİTİK (2026-07-12 dersi): Drive yazımları ASENKRON tamponlanır — VM bu
    # flush bitmeden kapatılırsa dosyalar (tar shard'ları dahil!) SESSİZCE
    # kaybolur. flush_and_unmount() tamponu boşaltmayı ZORLAR ve bitene kadar
    # bloklar. Drive'a yazan HER ŞEYDEN (report dahil) SONRA çağrılır.
    print("Drive flush ediliyor (asenkron yazımların buluta inmesi bekleniyor)...")
    from google.colab import drive as _gdrive
    _gdrive.flush_and_unmount()
    print("Drive flush TAMAM — VM artık güvenle kapatılabilir. Bundan sonraki eğitim "
          "koşularında train_colab.ipynb hücre (c) tar yolunu otomatik kullanacak.")


try:
    main()
except Exception:
    tb = traceback.format_exc()
    report("FATAL", "error", traceback=tb)
    raise
