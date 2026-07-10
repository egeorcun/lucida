"""Faz 3 (`training/train_colab.ipynb`) için saf-Python, bağımlılıksız yardımcı
mantık. Bilinçli olarak `torch`/`PIL` içe aktarmaz — böylece hem Colab'da (BiRefNet
eğitim döngüsü içinde) hem de bu depoda GPU/torch olmadan `pytest` ile test
edilebilir (bkz. `tests/test_train_colab_lib.py`). Notebook bu dosyayı repo'yu
klonladıktan sonra `sys.path`'e ekleyip import eder (`scripts/` için kullanılan
"TEK DOĞRULUK KAYNAĞI" deseniyle aynı — bkz. `training/prepare_data_colab.ipynb`
hücre (d) notu) — mantık notebook içine kopyalanıp tekrar yazılmaz, drift riski
böylece ortadan kalkar.

Altı bağımsız endişe kapsar:
1. Kategori ağırlıklı örnekleme (`compute_sample_weights` / `compute_expected_shares`)
   — `torch.utils.data.WeightedRandomSampler`'a beslenecek ağırlıkları hesaplar.
2. Checkpoint keşfi/budama (`find_latest_checkpoint` / `prune_old_checkpoints`)
   — Colab oturum kopması sonrası otomatik devam + Drive disk kotasını sınırlama.
3. Deterministik + KALICI TRAIN/VAL bölünmesi (`deterministic_val_split` /
   `load_or_create_val_split`) + sabit hızlı-değerlendirme alt kümesi
   (`fixed_eval_subset`).
4. BiRefNet resmi `train.py`/`config.py` mantığının iki küçük parçasının yeniden
   üretimi (`should_apply_finetune_reweight`, `effective_lr`) — bkz. modül
   içindeki fonksiyon docstring'lerinde satır bazlı referanslar.
5. BiRefNet `config.py` metin yaması (`apply_config_patches`) — İDEMPOTENT
   (aynı VM'de notebook'un yeniden koşturulması patlamamalı).
6. Drive→yerel disk veri kopyalama (`copy_pairs`) — hem im hem gt dosya boyutu
   doğrulamalı (yarım kalmış/kesilmiş kopyalar onarılır).
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

CKPT_FILENAME_RE = re.compile(r"^epoch_(\d+)\.pth$")


# ============================================================================
# 1) Kategori ağırlıklı örnekleme
# ============================================================================
def load_stem_categories(manifest_path: str | Path) -> dict[str, str]:
    """Kompozit manifest'ini (`benchmark.testset` formatı: id/image/category/gt_alpha
    JSONL satırları — bkz. `scripts/export_birefnet.py` docstring'i, export sırasında
    stem = row['id']) okuyup `{stem: category}` sözlüğü döndürür.

    Notebook bu dosyayı `Drive'daki bg-remover-data/train_composites_manifest.jsonl`
    kopyasından okur (bkz. `training/colab_devam_hucresi.py` `stage7_drive_copy`)."""
    result: dict[str, str] = {}
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            result[row["id"]] = row["category"]
    return result


def compute_sample_weights(
    stems: list[str],
    stem_category: dict[str, str],
    target_share: dict[str, float],
    default_category: str = "_other",
) -> list[float]:
    """`stems` (MyData.image_paths ile AYNI SIRADA olmalı — WeightedRandomSampler
    ağırlıkları dataset indeksleriyle hizalanmak zorunda) için, `target_share`'de
    adı geçen kategorilerin epoch-içi BEKLENEN payını `target_share`'e sabitleyen,
    geri kalan kategorilerin KENDİ ARALARINDAKİ göreli oranını (ham sayılarıyla
    orantılı) koruyan ağırlıklar üretir.

    Algoritma: hedefi olan bir kategori c için örnek başına ağırlık =
    target_share[c] / count(c) (kategori toplamda tam olarak target_share[c]
    payını alır, kategori-içi örnekler eşit ağırlıklı). Hedefsiz kategoriler için
    örnek başına ağırlık = (1 - sum(target_share)) / toplam_hedefsiz_sayı — TÜM
    hedefsiz örnekler için AYNI sabit değer, bu da onların birbirlerine göre
    payının (ağırlıksız haldeki gibi) ham sayılarıyla orantılı kalmasını sağlar.

    Bu, fiziksel oversampling'e (kompozit dosyalarını fazladan üretmek) göre EN AZ
    MÜDAHALECİ mekanizma: `scripts/make_composites.py`'nin `CATEGORY_MULTIPLIER`
    çarpanlarına (transparent×10, camouflage×2 — bkz. o dosyanın docstring'i)
    dokunmadan, yalnız `DataLoader`'ın sampler'ını değiştirerek çalışır; resmi
    `train.py`nin `prepare_dataloader`'ı (`shuffle=is_train, sampler=None`)
    üzerine YALNIZ bir `sampler=` argümanı eklenir (bkz. notebook eğitim hücresi).
    """
    categories = [stem_category.get(s, default_category) for s in stems]
    counts = Counter(categories)

    targeted = {c: share for c, share in target_share.items() if counts.get(c, 0) > 0}
    sum_targeted = sum(targeted.values())
    if sum_targeted >= 1.0:
        raise ValueError(f"target_share toplamı >= 1.0 olamaz (mevcut kategorilerde): {targeted}")
    remaining_mass = 1.0 - sum_targeted

    other_categories = [c for c in counts if c not in targeted]
    n_other_total = sum(counts[c] for c in other_categories)

    per_category_weight: dict[str, float] = {}
    for c, share in targeted.items():
        per_category_weight[c] = share / counts[c]
    other_weight = (remaining_mass / n_other_total) if n_other_total > 0 else 0.0
    for c in other_categories:
        per_category_weight[c] = other_weight

    return [per_category_weight[c] for c in categories]


def compute_expected_shares(
    weights: list[float], stems: list[str], stem_category: dict[str, str], default_category: str = "_other"
) -> dict[str, float]:
    """Tanılama: verilen ağırlıklarla (normalize edilmemiş de olsa) her kategorinin
    epoch-içi BEKLENEN örnekleme payını hesaplar (`sum(weights in cat) / sum(all weights)`).
    Notebook bu fonksiyonu, sampler kurulduktan hemen sonra hedefin (≥%20) gerçekten
    tutturulduğunu konsola yazdırmak için çağırır."""
    total = sum(weights)
    if total <= 0:
        return {}
    sums: dict[str, float] = {}
    for w, s in zip(weights, stems):
        c = stem_category.get(s, default_category)
        sums[c] = sums.get(c, 0.0) + w
    return {c: v / total for c, v in sums.items()}


# ============================================================================
# 2) Checkpoint keşfi / budama (resume + Drive disk kotası)
# ============================================================================
def find_latest_checkpoint(ckpt_dir: str | Path, pattern: re.Pattern = CKPT_FILENAME_RE) -> tuple[str, int] | None:
    """`ckpt_dir` altında `epoch_<N>.pth` desenine uyan dosyaları tarar, en büyük
    N'e sahip olanı `(yol, epoch)` olarak döndürür; hiç yoksa `None` (ilk koşu —
    `BiRefNet.from_pretrained(HF_MODEL_ID)` ile sıfırdan başlanır)."""
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        return None
    best: tuple[str, int] | None = None
    for p in ckpt_dir.iterdir():
        m = pattern.match(p.name)
        if not m:
            continue
        epoch = int(m.group(1))
        if best is None or epoch > best[1]:
            best = (str(p), epoch)
    return best


def prune_old_checkpoints(
    ckpt_dir: str | Path, keep_last_n: int, pattern: re.Pattern = CKPT_FILENAME_RE
) -> list[str]:
    """`ckpt_dir`'de yalnız en son `keep_last_n` epoch'un checkpoint'ini bırakır,
    gerisini SİLER; silinen dosya yollarını döndürür. Hem lokal Colab diskinde hem
    de Drive'da çağrılır (100 epoch × ~güncel BiRefNet checkpoint boyutu Drive
    kotasını hızla doldurur — bkz. notebook parametre hücresi `KEEP_LAST_N_CHECKPOINTS`)."""
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        return []
    entries: list[tuple[int, Path]] = []
    for p in ckpt_dir.iterdir():
        m = pattern.match(p.name)
        if m:
            entries.append((int(m.group(1)), p))
    entries.sort(key=lambda t: t[0], reverse=True)
    removed = []
    for _, p in entries[keep_last_n:]:
        p.unlink()
        removed.append(str(p))
    return removed


# ============================================================================
# 3) Deterministik TRAIN/VAL bölünmesi + sabit hızlı-değerlendirme alt kümesi
# ============================================================================
def deterministic_val_split(all_stems: list[str], seed: int, val_fraction: float) -> tuple[list[str], list[str]]:
    """`all_stems`'i (girdi sırası ÖNEMSİZ — önce sıralanır, sonra tohumlu
    karıştırılır, böylece dosya sistemi listeleme sırasından bağımsız aynı sonuç
    üretir) deterministik olarak (train_stems, val_stems) ikilisine böler.
    Yeniden koşularda (idempotentlik, görev madde 6) AYNI val kümesini üretir —
    fiziksel taşıma YOK, yalnız notebook bu listeye göre hangi dosyaları
    TRAIN/ vs VAL/ alt dizinine kopyalayacağına karar verir."""
    import random

    stems_sorted = sorted(all_stems)
    rng = random.Random(seed)
    shuffled = stems_sorted[:]
    rng.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_fraction)) if shuffled else 0
    val = sorted(shuffled[:n_val])
    train = sorted(shuffled[n_val:])
    return train, val


def load_or_create_val_split(
    all_stems: list[str], seed: int, val_fraction: float, persist_path: str | Path
) -> tuple[list[str], list[str]]:
    """`deterministic_val_split`'in KALICI hali: ilk koşuda bölünmeyi yapıp
    val listesini `persist_path`'e (JSON) yazar; sonraki koşularda dosyadan okur.

    Neden gerekli: Drive'daki veri seti sonradan BÜYÜYEBİLİR (Faz 2 pipeline'ı
    idempotent — yeni bir koşu yeni çiftler ekleyebilir). Salt-deterministik
    bölünme, girdi listesi değişince FARKLI bir val kümesi üretirdi — önceki
    epoch'larda eğitimde görülmüş görseller val'e sızardı. Kalıcı listeyle val
    kümesi İLK koşuda ne seçildiyse o kalır; SONRADAN eklenen stem'lerin TAMAMI
    TRAIN'e gider (bilinçli, basit tercih: val kümesinin epoch'lar arası
    karşılaştırılabilirliği, oransal val büyütmekten daha değerli — val payı
    zamanla %2'nin biraz altına düşebilir, hızlı-değerlendirme zaten sabit
    `n=24`'lük bir alt küme kullandığı için pratik etkisi yok).

    Dosyada kayıtlı olup artık diskte OLMAYAN stem'ler sessizce val'den düşürülür
    (veri silinmişse bölünme yine tutarlı kalır)."""
    persist_path = Path(persist_path)
    if persist_path.exists():
        saved = json.loads(persist_path.read_text())
        saved_val = set(saved["val_stems"])
        all_set = set(all_stems)
        val = sorted(saved_val & all_set)
        train = sorted(all_set - saved_val)
        return train, val

    train, val = deterministic_val_split(all_stems, seed=seed, val_fraction=val_fraction)
    persist_path.parent.mkdir(parents=True, exist_ok=True)
    persist_path.write_text(
        json.dumps({"seed": seed, "val_fraction": val_fraction, "val_stems": val}, ensure_ascii=False, indent=1)
    )
    return train, val


def fixed_eval_subset(val_stems: list[str], seed: int, n: int) -> list[str]:
    """VAL kümesinden (2% — yüzlerce görsel olabilir) HER epoch aynı, sabit
    `n` (varsayılan 24) görsellik bir alt küme seçer — periyodik hızlı-değerlendirmenin
    epoch'lar arasında karşılaştırılabilir olması için (her seferinde farklı
    rastgele görsellerle ölçülen MAE'nin gürültüsü epoch-to-epoch trendini
    gizleyebilir)."""
    import random

    stems_sorted = sorted(val_stems)
    rng = random.Random(seed)
    shuffled = stems_sorted[:]
    rng.shuffle(shuffled)
    return sorted(shuffled[: min(n, len(shuffled))])


# ============================================================================
# 4) BiRefNet resmi train.py/config.py mantığının küçük parçaları
# ============================================================================
def should_apply_finetune_reweight(epoch: int, total_epochs: int, finetune_last_epochs: int) -> bool:
    """BiRefNet resmi `train.py::Trainer.train_epoch` içindeki koşul:
    `if epoch > args.epochs + config.finetune_last_epochs:` (kaynak:
    ZhengPeng7/BiRefNet `train.py`, GitHub `main` dalı, fonksiyon `train_epoch`,
    ~satır 195 civarı — bu depoya `curl` ile çekilip incelendi, bkz. Faz 3 raporu).
    `finetune_last_epochs` NEGATİF bir sayıdır (`config.py`'de `Matting` görevi
    için `-10` — son 10 epoch'ta pixel loss ağırlıkları kademeli değiştirilir,
    "belgelenen fine-tune hilesi"). `total_epochs`, o Colab OTURUMUNUN DEĞİL,
    eğitimin NİHAİ HEDEF epoch sayısıdır (`EPOCHS` parametresi — resume'lerde
    HER OTURUMDA AYNI değer verilmeli, aksi halde bu eşik oturumdan oturuma kayar).

    Resmi koşula EK iki koruma (kısa koşular için — resmi kod EPOCHS>=150
    varsaydığından bu durumu hiç ele almıyor):
    - `finetune_last_epochs == 0` -> hep False (`config.py` yorumu: "choose 0 to skip").
    - `total_epochs <= |finetune_last_epochs|` (ör. EPOCHS=6, ft=-10) -> hep False:
      pencere başlangıcı (total+ft+1) epoch 1'in ÖNCESİNE düşerdi ve decay üssü
      daha ilk epoch'ta n>1 olurdu (ör. 0.9^5) — kısa keşif koşularında loss
      ağırlıklarını daha eğitim başlamadan bozmak anlamsız, hile tamamen atlanır.
      Bu koruma sayesinde hile uygulandığında üs her zaman n>=1'den başlar
      (epoch > total+ft >= 1 -> n = epoch-(total+ft) >= 1)."""
    if finetune_last_epochs == 0:
        return False
    if finetune_last_epochs < 0 and total_epochs <= -finetune_last_epochs:
        return False
    return epoch > total_epochs + finetune_last_epochs


def effective_lr(task: str, batch_size: int, accum_steps: int, base_lr_override: float | None = None) -> float:
    """BiRefNet resmi `config.py`'deki formülün ADAPTE edilmiş hali:
    `self.lr = (1e-4 if 'DIS5K' in self.task else 1e-5) * math.sqrt(self.batch_size / 4)`
    (kaynak: `config.py`, GitHub `main`, `Config.__init__`). Resmi kodda gradient
    accumulation YOK (`train.py`'de `accelerator.gradient_accumulation_steps=1`
    SABİT kodlanmış, `accelerator.accumulate(...)` context'i YORUM SATIRINDA
    bırakılmış — kullanılmıyor); bu notebook gradient accumulation EKLEDİĞİ için
    formüldeki `batch_size`'ı, optimizer adımı başına GERÇEK (efektif) batch'e
    (`batch_size * accum_steps`) genişletiyoruz — resmi formülün "efektif batch
    büyüdükçe lr'yi karekök oranında büyüt" mantığını, artık iki eksende (fiziksel
    batch + accumulation) büyüyen efektif batch'e doğru şekilde uygulamak için.
    `base_lr_override` doluysa (parametre hücresinde `LR` elle ayarlanmışsa) bu
    hesap tamamen atlanır."""
    if base_lr_override is not None:
        return float(base_lr_override)
    base = 1e-4 if "DIS5K" in task else 1e-5
    effective_batch = batch_size * accum_steps
    return base * math.sqrt(effective_batch / 4)


# ============================================================================
# 5) BiRefNet config.py metin yaması (İDEMPOTENT)
# ============================================================================
_TASK_LIST = ["DIS5K", "COD", "HRSOD", "General", "General-2K", "Matting"]
_TASK_LINE_RE = re.compile(
    r"self\.task = (\['DIS5K', 'COD', 'HRSOD', 'General', 'General-2K', 'Matting'\])\[\d+\]"
)
_HOME_LINE_RE = re.compile(r"self\.sys_home_dir = \[os\.path\.expanduser\('~'\), '[^']*'\]\[1\]")
_BS_LINE_RE = re.compile(r"self\.batch_size = \d+")


def apply_config_patches(src: str, task: str, sys_home_dir: str, batch_size: int) -> str:
    """BiRefNet `config.py` kaynağına üç yamayı uygular: (1) `self.task` seçili
    indeksi, (2) `self.sys_home_dir` ikinci elemanı (data_root_dir'ın kökü),
    (3) `self.batch_size`. Satır desenleri GitHub `main` dalındaki `Config.__init__`
    ile doğrulandı (bkz. Faz 3 raporu).

    İDEMPOTENT ve yeniden-parametrelenebilir: regex, satırın HEM orijinal
    (yamasız) HEM önceden yamalanmış halini eşler — aynı VM'de notebook'un
    (aynı ya da FARKLI parametre değerleriyle) yeniden koşturulması hata vermez,
    `apply(apply(src)) == apply(src)`. Desen hiç eşleşmezse (repo `main` dalı
    değişmişse) SESSİZCE geçmek yerine net bir ValueError fırlatılır."""
    if task not in _TASK_LIST:
        raise ValueError(f"bilinmeyen görev: {task!r} (geçerli: {_TASK_LIST})")
    idx = _TASK_LIST.index(task)

    patched, n = _TASK_LINE_RE.subn(rf"self.task = \1[{idx}]", src, count=1)
    if n == 0:
        raise ValueError(
            "config.py'de beklenen `self.task = [...][N]` satırı bulunamadı — "
            "BiRefNet main dalı değişmiş olabilir, config.py'yi elle kontrol edin."
        )
    patched, n = _HOME_LINE_RE.subn(
        f"self.sys_home_dir = [os.path.expanduser('~'), '{sys_home_dir}'][1]", patched, count=1
    )
    if n == 0:
        raise ValueError("config.py'de beklenen `self.sys_home_dir = [...]` satırı bulunamadı.")
    patched, n = _BS_LINE_RE.subn(f"self.batch_size = {batch_size}", patched, count=1)
    if n == 0:
        raise ValueError("config.py'de beklenen `self.batch_size = N` satırı bulunamadı.")
    return patched


# ============================================================================
# 6) Drive -> yerel disk veri kopyalama (boyut doğrulamalı, idempotent)
# ============================================================================
def copy_pairs(
    stems: list[str],
    src_im_dir: str | Path,
    src_gt_dir: str | Path,
    dst_im_dir: str | Path,
    dst_gt_dir: str | Path,
    im_ext: str = ".jpg",
    gt_ext: str = ".png",
    max_workers: int = 16,
) -> int:
    """`stems` listesindeki (im, gt) çiftlerini kaynaktan hedefe kopyalar;
    kopyalanan çift sayısını döndürür. İdempotent: hedefte HEM im HEM gt varsa
    VE her ikisinin de dosya boyutu kaynakla birebir eşleşiyorsa atlanır —
    yalnız im boyutuna bakmak yetmez, yarım kalmış bir Colab kopyalamasında
    gt dosyası kesik (truncated) kalmış olabilir; o durumda çift YENİDEN
    kopyalanır (onarım).

    Drive FUSE bağlantısı üzerinden onbinlerce küçük dosyayı TEK İŞ PARÇACIĞIYLA
    kopyalamak saatler sürüyor (canlı bir Colab oturumunda ölçüldü); bu yüzden
    her çift bağımsız bir iş birimi olarak `ThreadPoolExecutor` ile `max_workers`
    kadar iş parçacığına dağıtılır (I/O-bound kopyalama — GIL burada engel
    değil). Her çiftin hedef dosyaları kendine özgü olduğundan iş parçacıkları
    arasında paylaşılan durum YOK (yarış koşulu riski yok); sonuç (kopyalanan
    sayı, atlanan/onarılan çiftler) sıralamadan BAĞIMSIZ, seri koşumla birebir
    aynıdır. Tek tek çiftlerdeki hatalar ANINDA fırlatılmaz — TÜMÜ toplanır,
    geri kalan tüm çiftler işlenir (kısmi ilerleme kaybolmaz), sonda İLK hata
    toplam hata sayısıyla birlikte yeniden fırlatılır. Her 2000 tamamlanan
    çiftte bir ilerleme (hız + ETA) konsola yazdırılır."""
    import shutil
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    src_im_dir, src_gt_dir = Path(src_im_dir), Path(src_gt_dir)
    dst_im_dir, dst_gt_dir = Path(dst_im_dir), Path(dst_gt_dir)

    def _copy_one(stem: str) -> bool:
        src_im, src_gt = src_im_dir / f"{stem}{im_ext}", src_gt_dir / f"{stem}{gt_ext}"
        dst_im, dst_gt = dst_im_dir / f"{stem}{im_ext}", dst_gt_dir / f"{stem}{gt_ext}"
        if (
            dst_im.exists()
            and dst_gt.exists()
            and dst_im.stat().st_size == src_im.stat().st_size
            and dst_gt.stat().st_size == src_gt.stat().st_size
        ):
            return False
        shutil.copy2(src_im, dst_im)
        shutil.copy2(src_gt, dst_gt)
        return True

    total = len(stems)
    copied = 0
    completed = 0
    errors: list[tuple[str, BaseException]] = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_stem = {executor.submit(_copy_one, stem): stem for stem in stems}
        for future in as_completed(future_to_stem):
            stem = future_to_stem[future]
            completed += 1
            try:
                if future.result():
                    copied += 1
            except Exception as exc:  # per-item hata: topla, işlemeye devam et
                errors.append((stem, exc))
            if completed % 2000 == 0:
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0.0
                eta = (total - completed) / rate if rate > 0 else float("inf")
                print(
                    f"copy_pairs: {completed}/{total} tamamlandı "
                    f"({rate:.1f} çift/sn, ETA {eta:.0f}sn)"
                )

    if errors:
        first_stem, first_exc = errors[0]
        raise RuntimeError(
            f"copy_pairs: {len(errors)}/{total} çift kopyalanamadı "
            f"(ilk hata, stem={first_stem!r}: {first_exc!r})"
        ) from first_exc

    return copied
