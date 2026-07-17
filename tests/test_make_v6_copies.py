"""scripts/make_v6_copies.py testleri — küçük (128px) sahte TRAIN çiftleriyle hızlı koşu.

Doğrulanan sözleşmeler (bkz. make_v6_copies modül docstring'i):
- kadraj-kırpma (`_e00`): pencere özne bbox'ını gerçekten keser (kırpılmış gt'de
  kenara değen alpha>0.5 piksel var), alpha değerleri kırpmada DEĞİŞMEZ (saf
  dilimleme), pencere alanı >= orijinalin %50'si,
- mixed (`_m00`/`_m01`): yalnız hem katı (>0.9 oranı >= %8) hem yumuşak
  (0.05-0.95 oranı >= %8) gt'li transparent çiftler seçilir; alpha augment'te
  AYNEN korunur, RGB değişir,
- `_e/_m` türevleri kaynak olarak KULLANILMAZ, `_o00` kaynak OLABİLİR (tercihli),
- determinizm (aynı seed bit-identical) + idempotentlik (var olan dosya yeniden
  üretilmez, manifest'te tekrar yok),
- manifest satırları `{"id", "category"}`, kategori = kaynak kategorisi.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import make_v6_copies as mv  # noqa: E402

SIZE = 128  # kare tuval — hızlı koşu


def _write_pair(im_dir: Path, gt_dir: Path, stem: str, alpha: np.ndarray, color=(0, 180, 0)) -> None:
    im_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    # Düz renk DEĞİL gradyan: düz renkli görüntü, jitter + JPEG kuantizasyonu
    # sonrası tesadüfen bit-identical kalabiliyor — "augment RGB'yi değiştirdi"
    # testinin anlamlı olması için dokulu içerik gerekli.
    grad = np.linspace(0, 75, SIZE, dtype=np.float32)
    rgb = np.zeros((SIZE, SIZE, 3), dtype=np.float32)
    rgb[..., 0] = color[0] + grad[None, :]
    rgb[..., 1] = color[1] + grad[:, None]
    rgb[..., 2] = color[2] + grad[None, :] * 0.5
    rgb = rgb.clip(0, 255).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(im_dir / f"{stem}.jpg", format="JPEG", quality=92)
    Image.fromarray(alpha, mode="L").save(gt_dir / f"{stem}.png")


def _square_alpha() -> np.ndarray:
    """Katı çekirdek (40:88 = 255) + yumuşak çerçeve (32:96 = 120) — bbox
    (alpha>25) 32..96; %20-60 kesikler hep katı çekirdekten geçer (kenara
    değen alpha>0.5 piksel garantisi)."""
    a = np.zeros((SIZE, SIZE), dtype=np.uint8)
    a[32:96, 32:96] = 120
    a[40:88, 40:88] = 255
    return a


def _mixed_alpha() -> np.ndarray:
    """Karma opaklık: katı bölge ~%14 (>0.9), yumuşak bölge ~%25 (0.05-0.95)."""
    a = np.zeros((SIZE, SIZE), dtype=np.uint8)
    a[8:56, 8:56] = 255  # katı: 48x48 / 128^2 ≈ %14
    a[64:120, 8:80] = 128  # yumuşak: 56x72 / 128^2 ≈ %25
    return a


def _soft_alpha() -> np.ndarray:
    """Yalnız yumuşak: katı oranı 0 (< %8) — mixed'e SEÇİLMEMELİ."""
    a = np.zeros((SIZE, SIZE), dtype=np.uint8)
    a[16:112, 16:112] = 128
    return a


def _solid_alpha() -> np.ndarray:
    """Yalnız katı (binary): yumuşak oranı 0 (< %8) — mixed'e SEÇİLMEMELİ."""
    a = np.zeros((SIZE, SIZE), dtype=np.uint8)
    a[16:112, 16:112] = 255
    return a


@pytest.fixture
def env(tmp_path):
    """Sahte TRAIN havuzu: general (_o00 tercihli + _v00), camouflage,
    transparent (karma/yumuşak/katı) ve bir `_e00` türevi (kaynak olamaz)."""
    im_dir = tmp_path / "TRAIN" / "im"
    gt_dir = tmp_path / "TRAIN" / "gt"
    cats: dict[str, str] = {}
    for stem in ("srcA_o00", "srcA_v00", "srcB_o00", "srcB_v00"):
        _write_pair(im_dir, gt_dir, stem, _square_alpha())
        cats[stem] = "general"
    _write_pair(im_dir, gt_dir, "camo0_v00", _square_alpha())
    cats["camo0_v00"] = "camouflage"
    # türev stem — kaynak OLARAK KULLANILMAMALI (türevin türevi olmasın)
    _write_pair(im_dir, gt_dir, "srcA_o00_e00", _square_alpha())
    cats["srcA_o00_e00"] = "general"
    # transparent havuzu
    _write_pair(im_dir, gt_dir, "tr_mix_v00", _mixed_alpha())
    cats["tr_mix_v00"] = "transparent"
    _write_pair(im_dir, gt_dir, "tr_soft_v00", _soft_alpha())
    cats["tr_soft_v00"] = "transparent"
    _write_pair(im_dir, gt_dir, "tr_solid_v00", _solid_alpha())
    cats["tr_solid_v00"] = "transparent"
    return {"im": im_dir, "gt": gt_dir, "cats": cats, "out": tmp_path / "out"}


def _run(env, out_dir=None, seed=42, edge_count=100, mixed_cap=100, **kw):
    return mv.run(
        env["im"], env["gt"], env["cats"],
        out_dir if out_dir is not None else env["out"],
        seed=seed, edge_count=edge_count, mixed_cap=mixed_cap, **kw,
    )


def _manifest_rows(out_dir: Path) -> list[dict]:
    path = out_dir / "manifest.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _gt_arr(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path), dtype=np.uint8)


def _find_window(src: np.ndarray, crop: np.ndarray) -> tuple[int, int] | None:
    """Kırpma penceresi her eksende ya 0'dan başlar ya görüntü kenarında biter
    (pencere kesilmeyen yönde kenara kadar uzanır) — 4 aday konum denenir."""
    sh, sw = src.shape
    ch, cw = crop.shape
    for y0 in {0, sh - ch}:
        for x0 in {0, sw - cw}:
            if np.array_equal(src[y0 : y0 + ch, x0 : x0 + cw], crop):
                return y0, x0
    return None


# ==========================================================================
# Kadraj-kırpma (_e00)
# ==========================================================================
def test_edge_crop_cuts_subject_touches_border_and_preserves_alpha(env):
    _run(env, mixed_cap=0)
    rows = [r for r in _manifest_rows(env["out"]) if r["id"].endswith("_e00")]
    assert rows, "hiç edge-crop üretilmedi"
    for row in rows:
        src_stem = row["id"][: -len("_e00")]
        src = _gt_arr(env["gt"] / f"{src_stem}.png")
        crop = _gt_arr(env["out"] / "gt" / f"{row['id']}.png")
        # pencere gerçekten küçüldü ve alanı orijinalin >= %50'si
        assert crop.shape != src.shape
        assert crop.size >= 0.5 * src.size
        # özne kadraja değiyor: en az bir kenarda alpha > 0.5 (127) piksel
        borders = np.concatenate([crop[0, :], crop[-1, :], crop[:, 0], crop[:, -1]])
        assert (borders > 127).any(), f"{row['id']}: özne kırpma kenarına değmiyor"
        # alpha değerleri kırpmada DEĞİŞMEDİ: crop, kaynağın saf bir dilimi
        assert _find_window(src, crop) is not None, f"{row['id']}: gt saf dilim değil (alpha değişmiş?)"


def test_edge_crop_derived_stems_not_used_but_o00_preferred(env):
    _run(env, mixed_cap=0)  # edge_count=100 > havuz -> tüm uygun kaynaklar kullanılır
    ids = {r["id"] for r in _manifest_rows(env["out"])}
    assert "srcA_o00_e00_e00" not in ids  # türevin türevi YOK
    assert "srcA_o00_e00" in ids  # _o00 kaynak OLABİLİR (tercihli)
    # uygun havuz: 4 general + 1 camo + 3 transparent = 8 (türev hariç)
    assert len([i for i in ids if i.endswith("_e00")]) == 8


def test_edge_crop_proportional_and_preferred_first(env):
    """Kota havuzdan küçükken: kategori başına orantılı dağılım + general
    içinde önce `_o00`'lı (gerçek arka planlı) kaynaklar seçilir."""
    del env["cats"]["tr_mix_v00"], env["cats"]["tr_soft_v00"], env["cats"]["tr_solid_v00"]
    _run(env, edge_count=3, mixed_cap=0)  # havuz: 4 general + 1 camo
    rows = _manifest_rows(env["out"])
    by_cat: dict[str, list[str]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r["id"])
    # largest remainder: general 3*4/5=2.4 -> 2, camo 3*1/5=0.6 -> 1
    assert len(by_cat["general"]) == 2
    assert len(by_cat["camouflage"]) == 1
    # tercih: general'in 2 kotası _o00'lı kaynaklara gitti
    assert set(by_cat["general"]) == {"srcA_o00_e00", "srcB_o00_e00"}


def test_edge_crop_manifest_category_is_source_category(env):
    _run(env, mixed_cap=0)
    for row in _manifest_rows(env["out"]):
        assert set(row) == {"id", "category"}
        src_stem = row["id"].rsplit("_", 1)[0]
        assert row["category"] == env["cats"][src_stem]


# ==========================================================================
# Karma-opaklık (_m00/_m01)
# ==========================================================================
def test_mixed_selection_applies_thresholds(env):
    _run(env, edge_count=0)
    ids = {r["id"] for r in _manifest_rows(env["out"])}
    assert ids == {"tr_mix_v00_m00", "tr_mix_v00_m01"}  # yalnız karma gt seçildi
    for r in _manifest_rows(env["out"]):
        assert r["category"] == "transparent"


def test_mixed_preserves_alpha_but_changes_rgb(env):
    _run(env, edge_count=0)
    src_gt = _gt_arr(env["gt"] / "tr_mix_v00.png")
    src_im = (env["im"] / "tr_mix_v00.jpg").read_bytes()
    for ci in range(2):
        out_gt = _gt_arr(env["out"] / "gt" / f"tr_mix_v00_m{ci:02d}.png")
        assert np.array_equal(out_gt, src_gt), "augment alpha'yı değiştirdi (geometri/flip sızmış?)"
        out_im = (env["out"] / "im" / f"tr_mix_v00_m{ci:02d}.jpg").read_bytes()
        assert out_im != src_im  # renk jitter / JPEG artifact RGB'yi değiştirdi
    # iki kopya birbirinden de farklı (bağımsız rng akışları)
    a = (env["out"] / "im" / "tr_mix_v00_m00.jpg").read_bytes()
    b = (env["out"] / "im" / "tr_mix_v00_m01.jpg").read_bytes()
    assert a != b


def test_mixed_cap_limits_sources_deterministically(env):
    """cap=2 -> tek kaynak (sıralı ilk uygun stem) x 2 kopya."""
    _write_pair(env["im"], env["gt"], "tr_amix_v00", _mixed_alpha())
    env["cats"]["tr_amix_v00"] = "transparent"  # sıralamada tr_mix'ten ÖNCE
    counts = _run(env, edge_count=0, mixed_cap=2)
    assert counts == {"mixed": 2}
    ids = {r["id"] for r in _manifest_rows(env["out"])}
    assert ids == {"tr_amix_v00_m00", "tr_amix_v00_m01"}


# ==========================================================================
# Determinizm + idempotentlik + manifest
# ==========================================================================
def test_deterministic_same_seed_bit_identical(env):
    counts1 = _run(env, out_dir=env["out"] / "a")
    counts2 = _run(env, out_dir=env["out"] / "b")
    assert counts1 == counts2 and sum(counts1.values()) > 0
    ids1 = {r["id"] for r in _manifest_rows(env["out"] / "a")}
    ids2 = {r["id"] for r in _manifest_rows(env["out"] / "b")}
    assert ids1 == ids2
    for stem in ids1:
        for sub, ext in (("im", ".jpg"), ("gt", ".png")):
            fa = (env["out"] / "a" / sub / f"{stem}{ext}").read_bytes()
            fb = (env["out"] / "b" / sub / f"{stem}{ext}").read_bytes()
            assert fa == fb, f"{stem}: aynı seed farklı {sub} üretti"


def test_different_seed_changes_edge_windows(env):
    _run(env, out_dir=env["out"] / "a", seed=42, mixed_cap=0)
    _run(env, out_dir=env["out"] / "b", seed=7, mixed_cap=0)
    diff = 0
    for stem in {r["id"] for r in _manifest_rows(env["out"] / "a")}:
        pa = env["out"] / "a" / "gt" / f"{stem}.png"
        pb = env["out"] / "b" / "gt" / f"{stem}.png"
        if pb.exists() and pa.read_bytes() != pb.read_bytes():
            diff += 1
    assert diff > 0, "farklı seed hiçbir kırpma penceresini değiştirmedi"


def test_idempotent_second_run_produces_nothing(env):
    counts1 = _run(env)
    assert sum(counts1.values()) > 0
    rows1 = _manifest_rows(env["out"])
    sample = env["out"] / "im" / f"{rows1[0]['id']}.jpg"
    mtime = sample.stat().st_mtime_ns
    counts2 = _run(env)
    assert counts2 == {}  # hiçbir yeni öğe üretilmedi
    rows2 = _manifest_rows(env["out"])
    assert rows1 == rows2  # manifest'te tekrar yok
    assert sample.stat().st_mtime_ns == mtime  # dosyaya dokunulmadı


def test_file_exists_but_manifest_row_missing_gets_completed(env):
    _run(env)
    manifest = env["out"] / "manifest.jsonl"
    rows = _manifest_rows(env["out"])
    dropped = rows[0]
    img_path = env["out"] / "im" / f"{dropped['id']}.jpg"
    before = img_path.read_bytes()
    manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows[1:]))
    counts = _run(env)
    assert counts == {}  # dosya zaten vardı -> üretim sayılmadı
    assert {r["id"] for r in _manifest_rows(env["out"])} == {r["id"] for r in rows}
    assert img_path.read_bytes() == before  # dosya yeniden üretilmedi


def test_exclude_stems_removes_sources(env):
    """VAL sızıntı koruması: exclude_stems'teki kaynaklardan türev üretilmez."""
    _run(env, mixed_cap=0, exclude_stems={"srcA_o00", "camo0_v00"})
    ids = {r["id"] for r in _manifest_rows(env["out"])}
    assert "srcA_o00_e00" not in ids
    assert "camo0_v00_e00" not in ids
    assert "srcB_o00_e00" in ids  # kalan havuz çalışmaya devam eder


def test_is_mixed_opacity_thresholds():
    assert mv.is_mixed_opacity(np.asarray(_mixed_alpha(), dtype=np.float32) / 255.0)
    assert not mv.is_mixed_opacity(np.asarray(_soft_alpha(), dtype=np.float32) / 255.0)
    assert not mv.is_mixed_opacity(np.asarray(_solid_alpha(), dtype=np.float32) / 255.0)
