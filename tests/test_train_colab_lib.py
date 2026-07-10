"""`training/train_colab_lib.py` için saf-Python simülasyon testleri (görev
madde 6: sampler/oversampling + resume-tespiti mantığının GPU/Colab olmadan
lokal doğrulanması). Gerçek Colab/torch/Drive ortamı gerektirmez, `slow`
değildir."""
import json

import pytest

from training.train_colab_lib import (
    SAMPLER_PRESET_V1,
    SAMPLER_PRESET_V2,
    SAMPLER_PRESETS,
    apply_config_patches,
    compute_expected_shares,
    compute_sample_weights,
    copy_pairs,
    deterministic_val_split,
    effective_lr,
    find_latest_checkpoint,
    fixed_eval_subset,
    load_or_create_val_split,
    load_stem_categories,
    prune_old_checkpoints,
    should_apply_finetune_reweight,
)


def _synthetic_stems(counts: dict[str, int]) -> tuple[list[str], dict[str, str]]:
    stems: list[str] = []
    stem_category: dict[str, str] = {}
    for category, n in counts.items():
        for i in range(n):
            stem = f"{category}_{i:04d}"
            stems.append(stem)
            stem_category[stem] = category
    return stems, stem_category


# ============================================================================
# 1) Kategori ağırlıklı örnekleme
# ============================================================================
def test_compute_sample_weights_hits_target_share():
    counts = {"transparent": 50, "camouflage": 80, "hair": 9000, "general": 3000, "thin": 800, "complex": 2000}
    stems, stem_category = _synthetic_stems(counts)
    target = {"transparent": 0.20, "camouflage": 0.20}

    weights = compute_sample_weights(stems, stem_category, target)
    assert len(weights) == len(stems)

    shares = compute_expected_shares(weights, stems, stem_category)
    assert shares["transparent"] == pytest.approx(0.20, abs=1e-9)
    assert shares["camouflage"] == pytest.approx(0.20, abs=1e-9)
    # Kalan %60'lık pay, hedefsiz kategoriler arasında HAM sayılarıyla orantılı kalmalı.
    remaining = {c: shares[c] for c in ("hair", "general", "thin", "complex")}
    assert sum(remaining.values()) == pytest.approx(0.60, abs=1e-9)
    total_other = counts["hair"] + counts["general"] + counts["thin"] + counts["complex"]
    for cat in remaining:
        expected = 0.60 * counts[cat] / total_other
        assert remaining[cat] == pytest.approx(expected, abs=1e-9)


def test_compute_sample_weights_missing_target_category_is_ignored():
    # transparent hiç yoksa (bu batch'te 0 örnek), hedef payı sessizce düşürülmeli
    # (ValueError FIRLATILMAMALI) — kalan tüm pay diğer kategorilere gider.
    counts = {"camouflage": 10, "hair": 90}
    stems, stem_category = _synthetic_stems(counts)
    weights = compute_sample_weights(stems, stem_category, {"transparent": 0.20, "camouflage": 0.20})
    shares = compute_expected_shares(weights, stems, stem_category)
    assert "transparent" not in shares
    assert shares["camouflage"] == pytest.approx(0.20, abs=1e-9)
    assert shares["hair"] == pytest.approx(0.80, abs=1e-9)


def test_compute_sample_weights_rejects_impossible_target():
    stems, stem_category = _synthetic_stems({"a": 5, "b": 5})
    with pytest.raises(ValueError):
        compute_sample_weights(stems, stem_category, {"a": 0.6, "b": 0.6})


# ============================================================================
# 1b) v2 sampler preset (rebalancing — v1'deki catastrophic forgetting fix)
# ============================================================================
def test_sampler_preset_v1_matches_default_target_share():
    # target_share=None (varsayılan) v1 fine-tune koşusunun (epoch_1.pth)
    # davranışıyla BİREBİR aynı olmalı — geriye dönük uyumluluk.
    counts = {"transparent": 4100, "camouflage": 8080, "hair": 9422, "complex": 2190, "thin": 810}
    stems, stem_category = _synthetic_stems(counts)
    weights_default = compute_sample_weights(stems, stem_category, None)
    weights_explicit = compute_sample_weights(stems, stem_category, SAMPLER_PRESET_V1)
    assert weights_default == weights_explicit
    assert SAMPLER_PRESET_V1 == {"transparent": 0.20, "camouflage": 0.20}


def test_sampler_presets_registry_has_v1_and_v2():
    assert set(SAMPLER_PRESETS) == {"v1", "v2"}
    assert SAMPLER_PRESETS["v1"] is SAMPLER_PRESET_V1
    assert SAMPLER_PRESETS["v2"] is SAMPLER_PRESET_V2
    # Her iki preset de <1.0 toplamalı — compute_sample_weights sum(target_share) >= 1.0
    # durumunda ValueError fırlatır (kalan pay hep bir "diğer/_other" payına ayrılmalı).
    for preset in SAMPLER_PRESETS.values():
        assert sum(preset.values()) < 1.0


def test_sampler_preset_v2_hits_target_shares_within_one_percent():
    # `docs/reports/2026-07-faz2-veri.md` §2'nin belgelediği ham/materyalize
    # sayılara yakın, TÜM 6 kategorinin de mevcut olduğu bir dağılım (camouflage
    # ×2, transparent ×10 fiziksel çarpanlarıyla materyalize edilmiş; general=4000
    # senaryosu — doc §2 tablosu): camouflage doğal olarak en büyük paylardan biri
    # (~%28), complex/thin ise v1'de neredeyse hiç pay alamayan küçük kategoriler
    # (bkz. v1-entegrasyon + bgr-v1-comparison raporlarındaki catastrophic
    # forgetting bulgusu). Tüm hedefli kategoriler mevcut olduğunda preset'in
    # kendi %99 toplamı (kasıtlı <1.0, kalan %1 boş "_other" payı) yalnız ~%1'lik
    # bir renormalizasyona yol açar — bu yüzden "within 1%" toleransı anlamlı.
    counts = {
        "camouflage": 8080,   # 4040 ham × 2
        "hair": 9422,
        "transparent": 4100,  # 410 ham × 10
        "complex": 2190,
        "thin": 810,
        "general": 4000,
    }
    stems, stem_category = _synthetic_stems(counts)
    raw_total = sum(counts.values())
    raw_share = {c: n / raw_total for c, n in counts.items()}

    weights = compute_sample_weights(stems, stem_category, SAMPLER_PRESET_V2)
    achieved = compute_expected_shares(weights, stems, stem_category)

    for cat, target in SAMPLER_PRESET_V2.items():
        if cat not in achieved:
            continue
        assert achieved[cat] == pytest.approx(target, abs=0.01), (
            f"{cat}: hedef %{target * 100:.1f}, hesaplanan %{achieved[cat] * 100:.1f}"
        )

    # v1'in kök nedenini (complex/thin'in ham paydan çok daha düşük efektif pay
    # alması) v2'nin düzelttiğini doğrula: complex/thin artık ham paylarından
    # AÇIKÇA daha yüksek örnekleniyor, camouflage ise ham payından DÜŞÜK
    # (background: "camo downweighted from its raw ~%28-36 share").
    assert achieved["complex"] > raw_share["complex"]
    assert achieved["thin"] > raw_share["thin"]
    assert achieved["camouflage"] < raw_share["camouflage"]


def test_sampler_preset_v2_includes_general_when_present():
    counts = {
        "camouflage": 8080,
        "hair": 9422,
        "transparent": 4100,
        "complex": 2190,
        "thin": 810,
        "general": 4000,
    }
    stems, stem_category = _synthetic_stems(counts)
    weights = compute_sample_weights(stems, stem_category, SAMPLER_PRESET_V2)
    achieved = compute_expected_shares(weights, stems, stem_category)
    # Preset toplamı %99 (kasıtlı <1.0) VE tüm 6 kategori de mevcut/hedefli
    # olduğundan, "_other" payını alacak hiçbir örnek yok — gerçek WeightedRandomSampler
    # yalnız GÖRECELİ ağırlıklarla çalıştığından bu, ~%1'lik zararsız bir
    # renormalizasyona yol açar (0.09 hedefi -> 0.09/0.99 ≈ 0.0909 gerçekleşen).
    assert achieved["general"] == pytest.approx(0.09, abs=0.01)
    assert sum(achieved.values()) == pytest.approx(1.0, abs=1e-9)


def test_load_stem_categories(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        {"id": "x_v00", "image": "im/x_v00.jpg", "category": "transparent", "gt_alpha": "gt/x_v00.png"},
        {"id": "y_v00", "image": "im/y_v00.jpg", "category": "camouflage", "gt_alpha": "gt/y_v00.png"},
    ]
    manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    result = load_stem_categories(manifest)
    assert result == {"x_v00": "transparent", "y_v00": "camouflage"}


# ============================================================================
# 2) Checkpoint keşfi / budama
# ============================================================================
def test_find_latest_checkpoint_picks_max_epoch(tmp_path):
    for name in ("epoch_3.pth", "epoch_10.pth", "epoch_1.pth", "garbage.txt", "epoch_x.pth"):
        (tmp_path / name).write_text("x")
    result = find_latest_checkpoint(tmp_path)
    assert result is not None
    path, epoch = result
    assert epoch == 10
    assert path.endswith("epoch_10.pth")


def test_find_latest_checkpoint_empty_dir_returns_none(tmp_path):
    assert find_latest_checkpoint(tmp_path) is None
    assert find_latest_checkpoint(tmp_path / "does-not-exist") is None


def test_prune_old_checkpoints_keeps_only_last_n(tmp_path):
    for n in (1, 2, 3, 4, 5):
        (tmp_path / f"epoch_{n}.pth").write_text("x")
    removed = prune_old_checkpoints(tmp_path, keep_last_n=2)
    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert remaining == ["epoch_4.pth", "epoch_5.pth"]
    assert sorted(removed) == [str(tmp_path / "epoch_1.pth"), str(tmp_path / "epoch_2.pth"), str(tmp_path / "epoch_3.pth")]


def test_prune_old_checkpoints_noop_when_fewer_than_keep_n(tmp_path):
    (tmp_path / "epoch_1.pth").write_text("x")
    removed = prune_old_checkpoints(tmp_path, keep_last_n=5)
    assert removed == []
    assert (tmp_path / "epoch_1.pth").exists()


# ============================================================================
# 3) Deterministik TRAIN/VAL bölünmesi + sabit hızlı-değerlendirme alt kümesi
# ============================================================================
def test_deterministic_val_split_is_reproducible_and_covers_all():
    stems = [f"id_{i:05d}" for i in range(2000)]
    train_a, val_a = deterministic_val_split(stems, seed=42, val_fraction=0.02)
    train_b, val_b = deterministic_val_split(list(reversed(stems)), seed=42, val_fraction=0.02)

    assert train_a == train_b
    assert val_a == val_b
    assert len(val_a) == 40  # 2000 * 0.02
    assert set(train_a) | set(val_a) == set(stems)
    assert set(train_a) & set(val_a) == set()


def test_deterministic_val_split_different_seed_differs():
    stems = [f"id_{i:05d}" for i in range(500)]
    _, val_a = deterministic_val_split(stems, seed=1, val_fraction=0.02)
    _, val_b = deterministic_val_split(stems, seed=2, val_fraction=0.02)
    assert val_a != val_b


def test_fixed_eval_subset_deterministic_and_bounded():
    val_stems = [f"val_{i:03d}" for i in range(560)]
    a = fixed_eval_subset(val_stems, seed=7, n=24)
    b = fixed_eval_subset(val_stems, seed=7, n=24)
    assert a == b
    assert len(a) == 24
    assert set(a).issubset(set(val_stems))


def test_fixed_eval_subset_capped_by_available_size():
    val_stems = [f"val_{i:03d}" for i in range(10)]
    result = fixed_eval_subset(val_stems, seed=7, n=24)
    assert len(result) == 10


# ============================================================================
# 4) BiRefNet resmi mantık parçaları
# ============================================================================
@pytest.mark.parametrize(
    "epoch,total_epochs,finetune_last_epochs,expected",
    [
        (90, 100, -10, False),
        (91, 100, -10, True),
        (100, 100, -10, True),
        (1, 100, 0, False),  # finetune_last_epochs=0 -> "choose 0 to skip" (config.py yorumu), hep False
        (100, 100, 0, False),
        # Kısa koşu koruması: EPOCHS <= |ft| -> hile TAMAMEN atlanır (pencere epoch 1'in
        # öncesine düşer, decay üssü daha ilk epoch'ta n>1 olurdu — review Critical 1 knock-on).
        (1, 6, -10, False),
        (6, 6, -10, False),
        (10, 10, -10, False),
        # EPOCHS > |ft| -> resmi koşul aynen geçerli; üs otomatik olarak n>=1'den başlar.
        (10, 20, -10, False),
        (11, 20, -10, True),
    ],
)
def test_should_apply_finetune_reweight(epoch, total_epochs, finetune_last_epochs, expected):
    assert should_apply_finetune_reweight(epoch, total_epochs, finetune_last_epochs) is expected


def test_finetune_reweight_exponent_starts_at_one_when_applicable():
    # Hile uygulandığı İLK epoch'ta üs n=1 olmalı (0.9^1) — hiçbir kısayolda n>1 ile başlamamalı.
    total_epochs, ft = 20, -10
    first_applicable = next(
        e for e in range(1, total_epochs + 1) if should_apply_finetune_reweight(e, total_epochs, ft)
    )
    assert first_applicable - (total_epochs + ft) == 1


def test_effective_lr_dis5k_vs_other_task():
    lr_dis5k = effective_lr("DIS5K", batch_size=2, accum_steps=4)
    lr_matting = effective_lr("Matting", batch_size=2, accum_steps=4)
    assert lr_dis5k == pytest.approx(1e-4 * (8 / 4) ** 0.5)
    assert lr_matting == pytest.approx(1e-5 * (8 / 4) ** 0.5)
    assert lr_dis5k == pytest.approx(lr_matting * 10)


def test_effective_lr_override_bypasses_formula():
    assert effective_lr("Matting", batch_size=2, accum_steps=4, base_lr_override=3e-5) == 3e-5


# ============================================================================
# 5) config.py yaması (idempotentlik — review Critical 2)
# ============================================================================
# BiRefNet main dalındaki gerçek satırların birebir kopyası (curl ile doğrulandı).
_CONFIG_SNIPPET = """\
class Config():
    def __init__(self) -> None:
        self.batch_size = 8                                     # Multi-GPU+BF16 training...
        self.sys_home_dir = [os.path.expanduser('~'), '/workspace'][1]   # Default, custom
        self.task = ['DIS5K', 'COD', 'HRSOD', 'General', 'General-2K', 'Matting'][0]
"""


def test_apply_config_patches_basic():
    out = apply_config_patches(_CONFIG_SNIPPET, task="Matting", sys_home_dir="/content/dis_data", batch_size=2)
    assert "self.task = ['DIS5K', 'COD', 'HRSOD', 'General', 'General-2K', 'Matting'][5]" in out
    assert "self.sys_home_dir = [os.path.expanduser('~'), '/content/dis_data'][1]" in out
    assert "self.batch_size = 2 " in out or "self.batch_size = 2\n" in out or "self.batch_size = 2" in out


def test_apply_config_patches_is_idempotent():
    once = apply_config_patches(_CONFIG_SNIPPET, task="Matting", sys_home_dir="/content/dis_data", batch_size=2)
    twice = apply_config_patches(once, task="Matting", sys_home_dir="/content/dis_data", batch_size=2)
    assert once == twice


def test_apply_config_patches_reparameterizable_after_previous_patch():
    # Aynı VM'de kullanıcı BATCH/görev değiştirip yeniden koşarsa da çalışmalı.
    once = apply_config_patches(_CONFIG_SNIPPET, task="Matting", sys_home_dir="/content/dis_data", batch_size=2)
    again = apply_config_patches(once, task="General", sys_home_dir="/content/other", batch_size=4)
    assert "'Matting'][3]" in again
    assert "'/content/other'" in again
    assert "self.batch_size = 4" in again


def test_apply_config_patches_raises_on_unknown_source():
    with pytest.raises(ValueError):
        apply_config_patches("class Config: pass", task="Matting", sys_home_dir="/x", batch_size=2)
    with pytest.raises(ValueError):
        apply_config_patches(_CONFIG_SNIPPET, task="YokBoyleGorev", sys_home_dir="/x", batch_size=2)


# ============================================================================
# 6) copy_pairs (boyut doğrulamalı kopyalama — review Important 3)
# ============================================================================
def _make_pair_tree(tmp_path, stems, im_content=b"IMDATA-123", gt_content=b"GTDATA-456"):
    src_im, src_gt = tmp_path / "src_im", tmp_path / "src_gt"
    dst_im, dst_gt = tmp_path / "dst_im", tmp_path / "dst_gt"
    for d in (src_im, src_gt, dst_im, dst_gt):
        d.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        (src_im / f"{stem}.jpg").write_bytes(im_content)
        (src_gt / f"{stem}.png").write_bytes(gt_content)
    return src_im, src_gt, dst_im, dst_gt


def test_copy_pairs_copies_and_is_idempotent(tmp_path):
    stems = ["a", "b", "c"]
    src_im, src_gt, dst_im, dst_gt = _make_pair_tree(tmp_path, stems)
    assert copy_pairs(stems, src_im, src_gt, dst_im, dst_gt) == 3
    assert copy_pairs(stems, src_im, src_gt, dst_im, dst_gt) == 0  # ikinci koşu no-op
    for stem in stems:
        assert (dst_im / f"{stem}.jpg").read_bytes() == b"IMDATA-123"
        assert (dst_gt / f"{stem}.png").read_bytes() == b"GTDATA-456"


def test_copy_pairs_repairs_truncated_gt(tmp_path):
    # im tam ama gt kesik (yarım kalmış Drive kopyası) -> çift YENİDEN kopyalanmalı.
    stems = ["a"]
    src_im, src_gt, dst_im, dst_gt = _make_pair_tree(tmp_path, stems)
    copy_pairs(stems, src_im, src_gt, dst_im, dst_gt)
    (dst_gt / "a.png").write_bytes(b"GT")  # kesik gt simülasyonu (im boyutu hâlâ doğru)
    assert copy_pairs(stems, src_im, src_gt, dst_im, dst_gt) == 1
    assert (dst_gt / "a.png").read_bytes() == b"GTDATA-456"


def test_copy_pairs_repairs_truncated_im(tmp_path):
    stems = ["a"]
    src_im, src_gt, dst_im, dst_gt = _make_pair_tree(tmp_path, stems)
    copy_pairs(stems, src_im, src_gt, dst_im, dst_gt)
    (dst_im / "a.jpg").write_bytes(b"IM")
    assert copy_pairs(stems, src_im, src_gt, dst_im, dst_gt) == 1
    assert (dst_im / "a.jpg").read_bytes() == b"IMDATA-123"


def test_copy_pairs_parallel_matches_serial(tmp_path):
    # Paralel (varsayılan max_workers=16) sonucun, tek-iş-parçacıklı (max_workers=1)
    # koşumla BİREBİR aynı dizin ağacını ürettiğini kanıtlar — aynı fixture (200 çift,
    # her biri kendine özgü içerik), iki ayrı hedef ağaca kopyalanır, sonra karşılaştırılır.
    stems = [f"stem_{i:04d}" for i in range(200)]
    src_im, src_gt = tmp_path / "src_im", tmp_path / "src_gt"
    src_im.mkdir()
    src_gt.mkdir()
    for stem in stems:
        (src_im / f"{stem}.jpg").write_bytes(f"IMG-DATA-{stem}".encode())
        (src_gt / f"{stem}.png").write_bytes(f"GT-DATA-{stem}".encode())

    dst_im_serial, dst_gt_serial = tmp_path / "dst_im_serial", tmp_path / "dst_gt_serial"
    dst_im_parallel, dst_gt_parallel = tmp_path / "dst_im_parallel", tmp_path / "dst_gt_parallel"
    for d in (dst_im_serial, dst_gt_serial, dst_im_parallel, dst_gt_parallel):
        d.mkdir()

    n_serial = copy_pairs(stems, src_im, src_gt, dst_im_serial, dst_gt_serial, max_workers=1)
    n_parallel = copy_pairs(stems, src_im, src_gt, dst_im_parallel, dst_gt_parallel, max_workers=16)
    assert n_serial == n_parallel == len(stems)

    def _tree(d):
        return {p.name: p.read_bytes() for p in d.iterdir()}

    assert _tree(dst_im_serial) == _tree(dst_im_parallel)
    assert _tree(dst_gt_serial) == _tree(dst_gt_parallel)

    # İkinci koşum (idempotentlik) her iki modda da no-op olmalı.
    assert copy_pairs(stems, src_im, src_gt, dst_im_serial, dst_gt_serial, max_workers=1) == 0
    assert copy_pairs(stems, src_im, src_gt, dst_im_parallel, dst_gt_parallel, max_workers=16) == 0


def test_copy_pairs_collects_errors_and_raises_first_with_count(tmp_path):
    # Kaynakta OLMAYAN bir stem varsa o çiftin kopyalanması hata verir; ama diğer
    # TÜM çiftler yine de işlenmeli (kısmi ilerleme kaybolmamalı) ve sonda İLK hata
    # toplam hata sayısıyla birlikte fırlatılmalı.
    stems = ["a", "missing", "b"]
    src_im, src_gt, dst_im, dst_gt = _make_pair_tree(tmp_path, ["a", "b"])
    with pytest.raises(RuntimeError, match=r"1/3.*missing"):
        copy_pairs(stems, src_im, src_gt, dst_im, dst_gt)
    # "a" ve "b" hatasız çiftler olduğu için yine de kopyalanmış olmalı.
    assert (dst_im / "a.jpg").exists()
    assert (dst_im / "b.jpg").exists()


# ============================================================================
# 7) Kalıcı VAL bölünmesi (review Important 2)
# ============================================================================
def test_load_or_create_val_split_first_run_persists(tmp_path):
    stems = [f"id_{i:05d}" for i in range(1000)]
    persist = tmp_path / "val_stems.json"
    train, val = load_or_create_val_split(stems, seed=42, val_fraction=0.02, persist_path=persist)
    assert persist.exists()
    saved = json.loads(persist.read_text())
    assert saved["val_stems"] == val
    assert len(val) == 20
    assert set(train) | set(val) == set(stems)


def test_load_or_create_val_split_loads_existing_and_keeps_new_stems_in_train(tmp_path):
    stems = [f"id_{i:05d}" for i in range(1000)]
    persist = tmp_path / "val_stems.json"
    _, val_first = load_or_create_val_split(stems, seed=42, val_fraction=0.02, persist_path=persist)

    # Veri seti BÜYÜDÜ (Faz 2 yeniden koştu, 200 yeni çift) — val kümesi DEĞİŞMEMELİ,
    # yeni stem'lerin tamamı train'e gitmeli (belgelenmiş tercih, sızıntı yok).
    grown = stems + [f"new_{i:05d}" for i in range(200)]
    train2, val2 = load_or_create_val_split(grown, seed=42, val_fraction=0.02, persist_path=persist)
    assert val2 == val_first
    assert all(s in train2 for s in (f"new_{i:05d}" for i in range(200)))
    assert set(train2) & set(val2) == set()


def test_load_or_create_val_split_drops_vanished_val_stems(tmp_path):
    stems = [f"id_{i:05d}" for i in range(1000)]
    persist = tmp_path / "val_stems.json"
    _, val_first = load_or_create_val_split(stems, seed=42, val_fraction=0.02, persist_path=persist)
    shrunk = [s for s in stems if s != val_first[0]]  # bir val görseli diskten silindi
    _, val2 = load_or_create_val_split(shrunk, seed=42, val_fraction=0.02, persist_path=persist)
    assert val2 == val_first[1:]
