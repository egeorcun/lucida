"""scripts/make_design.py testleri — küçük (<=128px) sahte kaynaklarla hızlı koşu.

Doğrulanan sözleşmeler (bkz. make_design modül docstring'i):
- halftone/posterize/ink filtreleri alpha'yı DEĞİŞTİRMEZ (bit karşılaştırma),
- duman eklemesi GT'ye yansır: nesne bbox'ı DIŞINDA 0.05-0.6 bandında piksel var,
  nesne İÇİNDE duman 0,
- zemin köşeleri GT'de 0 (MARGIN_FRAC kenar bandı garantisi),
- kavisli yazı üretilir ve GT'ye işlenir,
- determinizm (aynı seed bit-identical) + idempotentlik + manifest
  {"id","category"} sözleşmesi ve `design_{i:05d}_c00` stem kalıbı.

make_textfx'in kendi testleri de yeşil kalmalı — make_design yalnız import eder,
make_textfx'i DEĞİŞTİRMEZ.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import make_design as md  # noqa: E402

COUNT = 3
CANVAS = (96, 128)
STEM_RE = re.compile(r"^design_\d{5}_c00$")


def _write_solid(path: Path, size, color, mode="RGB") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, size, color).save(path)


def _write_alpha(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="L").save(path)


@pytest.fixture
def env(tmp_path):
    """Sahte fg çiftleri (katı kare alpha) + ToonOut çiftleri."""
    fg_root = tmp_path / "fg"
    for i in range(2):
        _write_solid(fg_root / "im" / f"obj{i}.jpg", (96, 96), (0, 180, 60))
        a = np.zeros((96, 96), dtype=np.uint8)
        a[24:72, 24:72] = 255
        _write_alpha(fg_root / "gt" / f"obj{i}.png", a)

    toon_dir = tmp_path / "toonout"
    for i in range(2):
        _write_solid(toon_dir / "im" / f"toon{i}.jpg", (96, 96), (200, 60, 30))
        a = np.zeros((96, 96), dtype=np.uint8)
        a[16:80, 32:64] = 255
        _write_alpha(toon_dir / "gt" / f"toon{i}.png", a)

    return {"fg": [fg_root], "toon": toon_dir, "out": tmp_path / "out"}


def _run(env, out_dir=None, seed=42, count=COUNT, **kw):
    return md.run(
        out_dir if out_dir is not None else env["out"],
        bg_dir=None,  # kullanılmaz — zemin sentetik
        fg_dirs=env["fg"],
        toonout_dir=env["toon"],
        font_dir=None,  # PIL varsayılan fontuna düşer
        seed=seed,
        count=count,
        canvas_range=CANVAS,
        **kw,
    )


def _manifest_rows(out_dir: Path) -> list[dict]:
    path = out_dir / "manifest.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _pair_paths(out_dir: Path, stem: str) -> tuple[Path, Path]:
    return out_dir / "im" / f"{stem}.jpg", out_dir / "gt" / f"{stem}.png"


# ==========================================================================
# Baskı filtreleri — alpha bit-birebir korunur, RGB değişir
# ==========================================================================
@pytest.mark.parametrize("kind", ["halftone", "posterize", "ink"])
def test_print_filter_preserves_alpha_bitwise(kind):
    rng = md._item_rng(42, f"filter_{kind}")
    rgb = rng.integers(0, 256, (64, 64, 3)).astype(np.uint8)
    alpha = (rng.uniform(0, 1, (64, 64))).astype(np.float32)  # yumuşak değerli GT
    alpha_bytes = alpha.tobytes()
    rgb2, alpha2 = md.apply_print_filter(rgb, alpha, rng, kind)
    assert alpha2.tobytes() == alpha_bytes  # bit karşılaştırma: alpha AYNEN
    assert rgb2.shape == rgb.shape and rgb2.dtype == np.uint8
    assert not np.array_equal(rgb2, rgb)  # filtre gerçekten uygulanmış


def test_print_filter_none_is_identity():
    rng = md._item_rng(42, "filter_none")
    rgb = rng.integers(0, 256, (32, 32, 3)).astype(np.uint8)
    alpha = np.ones((32, 32), dtype=np.float32)
    rgb2, alpha2 = md.apply_print_filter(rgb, alpha, rng, "none")
    assert np.array_equal(rgb2, rgb) and alpha2.tobytes() == alpha.tobytes()


def test_halftone_darker_means_more_ink():
    """Tramın özü: koyu bölge daha çok mürekkep noktası üretir."""
    rng1 = md._item_rng(1, "ht_dark")
    rng2 = md._item_rng(1, "ht_dark")  # aynı akış -> aynı cell/ink seçimi
    dark = np.full((64, 64, 3), 30, dtype=np.uint8)
    light = np.full((64, 64, 3), 225, dtype=np.uint8)
    out_d = md._filter_halftone(dark, rng1)
    out_l = md._filter_halftone(light, rng2)
    ink_d = (out_d != np.asarray(md._PAPER_RGB)).any(axis=-1).mean()
    ink_l = (out_l != np.asarray(md._PAPER_RGB)).any(axis=-1).mean()
    assert ink_d > ink_l


# ==========================================================================
# Duman — nesne dışına 0.05-0.6 bandı, nesne içinde 0
# ==========================================================================
def test_smoke_alpha_band_outside_object():
    rng = md._item_rng(42, "smoke")
    alpha = np.zeros((128, 128), dtype=np.float32)
    alpha[40:88, 40:88] = 1.0
    smoke = md._smoke_alpha(alpha, rng, reach_frac=0.1)
    assert smoke.shape == alpha.shape
    assert float(smoke[alpha > 0.05].max(initial=0.0)) == 0.0  # nesne içi 0
    outside = smoke.copy()
    outside[40:88, 40:88] = 0.0
    band = (outside > 0.05) & (outside <= 0.6)
    assert int(band.sum()) > 0  # bbox DIŞINDA 0.05-0.6 bandında duman var
    assert float(smoke.max()) <= md.SMOKE_HI + 1e-6


# ==========================================================================
# Kavisli yazı — yay geometrisi + GT'ye işlenme
# ==========================================================================
def _default_font():
    try:
        return ImageFont.load_default(28)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def test_curved_text_arches_upward():
    img = md._curved_text_rgba("OOOOOOOO", _default_font(), (255, 0, 0, 255), theta=1.2)
    a = np.asarray(img)[..., 3]
    assert int((a > 0).sum()) > 0
    cols = np.nonzero(a.any(axis=0))[0]
    tops = np.array([np.nonzero(a[:, c])[0].min() for c in cols])
    n = len(cols)
    mid = tops[n // 3 : 2 * n // 3].min()
    edges = min(tops[: n // 6].max(initial=0), tops[-n // 6 :].max(initial=0))
    assert mid < edges  # orta harfler yayın tepesinde (arch up)


def test_curved_text_reaches_gt(env, monkeypatch):
    """Yalnız yazı kalan bir kompozisyonda (özne/dekor/ışıma kapalı, kavis
    zorunlu) GT sıfırdan farklıysa kavisli yazı GT'ye işlenmiş demektir."""
    monkeypatch.setattr(md, "CURVED_TEXT_PROB", 1.0)
    monkeypatch.setattr(md, "RAY_PROB", 0.0)
    monkeypatch.setattr(md, "DECOR_RANGE", (0, 0))
    rng = md._item_rng(7, "curved_gt")
    rgb, alpha = md._render_design_sample(rng, (128, 128), [], [], [])
    assert int((alpha > 0).sum()) > 0
    # köşe bandı garantisi burada da geçerli
    assert alpha[0, 0] == 0 and alpha[-1, -1] == 0


# ==========================================================================
# Tam koşu — manifest / stem / zemin / GT bandı
# ==========================================================================
def test_run_generates_pairs_and_manifest(env):
    counts = _run(env)
    assert counts == {"design": COUNT}

    rows = _manifest_rows(env["out"])
    assert len(rows) == COUNT
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids))
    for row in rows:
        assert set(row) == {"id", "category"}
        assert row["category"] == "design"
        assert STEM_RE.match(row["id"]), row["id"]

    for stem in ids:
        img_path, gt_path = _pair_paths(env["out"], stem)
        assert img_path.exists() and gt_path.exists()
        img = Image.open(img_path)
        assert img.mode == "RGB"
        gt = Image.open(gt_path)
        assert gt.mode == "L"
        assert img.size == gt.size


def test_gt_corners_zero_and_bg_paperlike(env):
    """Zemin GT'de alpha=0: kenar bandı garantisiyle köşeler her örnekte 0;
    im köşeleri kağıt/pastel zeminin açık tonunda kalır."""
    _run(env)
    for row in _manifest_rows(env["out"]):
        img_path, gt_path = _pair_paths(env["out"], row["id"])
        a = np.asarray(Image.open(gt_path))
        for corner in (a[0, 0], a[0, -1], a[-1, 0], a[-1, -1]):
            assert corner == 0, f"{row['id']}: GT köşesi 0 değil ({corner})"
        rgb = np.asarray(Image.open(img_path))
        for corner in (rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]):
            assert corner.min() >= 160, f"{row['id']}: zemin köşesi açık ton değil {corner}"


def test_gt_has_semi_transparent_band(env):
    """Duman/ışıma/eskitme GT'ye yansır: kaynak alpha'lar tamamen katı (0/255)
    olduğu halde üretilen GT'lerde 0.05-0.6 bandında piksel bulunur."""
    _run(env)
    found = False
    for row in _manifest_rows(env["out"]):
        arr = np.asarray(Image.open(_pair_paths(env["out"], row["id"])[1]), dtype=np.float32) / 255.0
        if int(((arr > 0.05) & (arr <= 0.6)).sum()) > 0:
            found = True
            break
    assert found, "hiçbir GT'de 0.05-0.6 bandında yarı saydam piksel yok"


# ==========================================================================
# Determinizm + idempotentlik + resume
# ==========================================================================
def test_deterministic_same_seed_bit_identical(env):
    counts1 = _run(env, out_dir=env["out"] / "a")
    counts2 = _run(env, out_dir=env["out"] / "b")
    assert counts1 == counts2
    ids1 = {r["id"] for r in _manifest_rows(env["out"] / "a")}
    ids2 = {r["id"] for r in _manifest_rows(env["out"] / "b")}
    assert ids1 == ids2
    for stem in ids1:
        img1, gt1 = _pair_paths(env["out"] / "a", stem)
        img2, gt2 = _pair_paths(env["out"] / "b", stem)
        assert img1.read_bytes() == img2.read_bytes(), f"{stem}: aynı seed farklı image üretti"
        assert gt1.read_bytes() == gt2.read_bytes(), f"{stem}: aynı seed farklı gt üretti"


def test_different_seed_changes_output(env):
    _run(env, out_dir=env["out"] / "a", seed=42)
    _run(env, out_dir=env["out"] / "b", seed=7)
    img1, _ = _pair_paths(env["out"] / "a", "design_00000_c00")
    img2, _ = _pair_paths(env["out"] / "b", "design_00000_c00")
    assert img1.read_bytes() != img2.read_bytes()


def test_idempotent_skips_existing(env):
    counts1 = _run(env)
    assert counts1 == {"design": COUNT}
    counts2 = _run(env)
    assert counts2 == {}  # ikinci koşuda hiçbir yeni öğe üretilmedi
    assert len(_manifest_rows(env["out"])) == COUNT  # manifest'te tekrar yok


def test_file_exists_but_manifest_row_missing_gets_completed(env):
    """Dosya kaydı ile manifest append'i arasında kesinti: dosyalar durur,
    satır eksik — yeniden koşuda dosya ÜRETİLMEZ, yalnız satır tamamlanır."""
    _run(env)
    manifest = env["out"] / "manifest.jsonl"
    rows = _manifest_rows(env["out"])
    dropped = rows[0]
    img_path, _ = _pair_paths(env["out"], dropped["id"])
    before_bytes = img_path.read_bytes()
    before_mtime = img_path.stat().st_mtime_ns
    manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows[1:]))

    counts = _run(env)
    assert counts == {}  # dosya zaten vardı -> üretim sayılmadı
    after = _manifest_rows(env["out"])
    assert {r["id"] for r in after} == {r["id"] for r in rows}
    assert img_path.read_bytes() == before_bytes
    assert img_path.stat().st_mtime_ns == before_mtime  # dosyaya hiç dokunulmadı


def test_partial_then_resume_matches_full_run(env):
    """Kesinti simülasyonu: bazı çiftler silinip yeniden koşulur — devam koşusu
    kesintisiz koşuyla bit-birebir aynı dosyaları üretmeli."""
    dir_full = env["out"] / "full"
    dir_resume = env["out"] / "resume"
    _run(env, out_dir=dir_full)
    _run(env, out_dir=dir_resume)

    rows = _manifest_rows(dir_resume)
    keep, drop = rows[::2], rows[1::2]
    assert drop
    for row in drop:
        img_path, gt_path = _pair_paths(dir_resume, row["id"])
        img_path.unlink()
        gt_path.unlink()
    (dir_resume / "manifest.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep)
    )

    counts = _run(env, out_dir=dir_resume)
    assert counts == {"design": len(drop)}  # yalnız silinenler yeniden üretildi

    for stem in {r["id"] for r in _manifest_rows(dir_full)}:
        img_f, gt_f = _pair_paths(dir_full, stem)
        img_r, gt_r = _pair_paths(dir_resume, stem)
        assert img_f.read_bytes() == img_r.read_bytes(), f"{stem}: devam koşusu image'ı farklı"
        assert gt_f.read_bytes() == gt_r.read_bytes(), f"{stem}: devam koşusu gt'si farklı"


# ==========================================================================
# Kaynak havuzu — zorunluluk + VAL hariç tutma
# ==========================================================================
def test_design_requires_sources(env, tmp_path):
    with pytest.raises(SystemExit, match="design"):
        md.run(tmp_path / "o", fg_dirs=[], toonout_dir=None, seed=42, count=2,
               canvas_range=CANVAS)


def test_exclude_fg_stems_removes_pool(env, tmp_path):
    """Tüm kaynak stem'leri hariç tutulursa havuz boşalır -> SystemExit
    (VAL sızıntı koruması fiilen fg seçiminde uygulanıyor)."""
    with pytest.raises(SystemExit, match="design"):
        md.run(
            tmp_path / "o", fg_dirs=env["fg"], toonout_dir=env["toon"], seed=42,
            count=1, canvas_range=CANVAS,
            exclude_fg_stems={"obj0", "obj1", "toon0", "toon1"},
        )


def test_toonout_only_pool_works(env, tmp_path):
    out = tmp_path / "toon_only"
    counts = md.run(out, fg_dirs=[], toonout_dir=env["toon"], font_dir=None,
                    seed=42, count=1, canvas_range=CANVAS)
    assert counts == {"design": 1}
    assert len(_manifest_rows(out)) == 1
