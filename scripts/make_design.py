"""v7 eğitimi için `design` kategorisi üreticisi — baskı-tasarımı / sticker /
tişört grafiği tarzı sentetik örnekler (GitHub issue #2: halftone, mürekkep
dokusu, dumanlı kenarlar ve beyaza eriyen ışımalı tasarımlarda model özneyi
siliyor ya da hayaletleştiriyor; stil-domain açığı bu kategoriyle kapatılır).

Her örnek bir "baskı tasarımı" kompozisyonudur:

- **Zemin**: kağıt-beyazı/krem düz renk (245-255 bandı) ya da hafif kağıt
  dokusu (düşük genlikli gürültü); `PASTEL_BG_PROB` (%15) olasılıkla açık
  pastel düz renk. Zemin GT'de alpha=0 — ayrıca kanvasın dış kenarında
  `MARGIN_FRAC`'lik bir bant her elemanın alpha'sından SIFIRLANIR (baskı
  "kenar boşluğu"; GT köşeleri her zaman 0 — test sözleşmesi).
- **Stilize özne (1-2 adet)**: `fg_dirs` (trans460/HIM2K, im/+gt/ çiftleri) ve
  `toonout_dir` havuzlarından alınan kesite BASKI-STİLİ filtre uygulanır —
  KRİTİK: filtre YALNIZ RGB'ye dokunur, alpha AYNEN (bit-birebir) kalır
  (`apply_print_filter`). Filtre menüsü: (a) halftone (luminance'ı nokta
  ızgarasına çevirir — nokta yarıçapı KOYULUKLA ölçekli, klasik gazete tramı),
  (b) posterize (3-5 seviye) + doygunluk artışı, (c) yüksek kontrast
  "mürekkep" (eşikleme + kenar vurgusu), (d) filtresiz (%25). ToonOut
  kaynakları zaten illüstrasyon — onlara çoğunlukla filtresiz/posterize düşer.
- **Dumanlı kenar / airbrush** (`_smoke_alpha`): öznenin alpha'sının dışına
  kıvrılan, `SMOKE_LO..SMOKE_HI` (0.1-0.5) bandında bulut/duman lekeleri
  EKLENİR — GT'ye de AYNEN girer (duman tasarımın parçasıdır; Reddit Photo
  1'in silinme nedeni tam bu doku). Organik görünüm iki oktavlı value-noise
  ("Perlin benzeri") maskesiyle sağlanır.
- **Işıma/patlama** (`RAY_PROB`=%50): öznenin ARKASINA radyal ışın demeti veya
  glow — GT'de yarı saydam (`RAY_ALPHA_LO..RAY_ALPHA_HI` = 0.15-0.6).
- **Display yazı (1-2 blok)**: make_textfx'in yazı makinesi YENİDEN KULLANILIR
  (`_get_font`/`_draw_text_rgba`/`_rand_text` import edilir, kopyalanmaz). Ek
  özellikler: KAVİSLİ yazı (harfler tek tek yay üzerine — `_curved_text_rgba`),
  istifli çok satırlı blok (`_stacked_text_rgba`), eskitme/distress (yazı
  alpha'sından value-noise grunge maskesiyle parça eksiltme — GT'ye de aynen).
  Konum: üst ve/veya alt bant.
- **Küçük dekorlar**: yıldız/şimşek/sıçrama lekeleri (2-6 adet, katı veya yarı
  saydam) — basit vektör çizimler.
- **GT = tüm elemanların alpha UNION'ı** (`1-(1-a)(1-b)` zinciri); zemin hiç
  katılmaz.

SÖZLEŞMELER (scripts/make_textfx.py ile BİREBİR AYNI):
- Stem kalıbı `design_{i:05d}_c00`; çıktı `out_dir/im/{stem}.jpg` (JPEG q92) +
  `out_dir/gt/{stem}.png` (L modu) — `_save_pair` make_textfx'ten import.
- Manifest: `{"id": stem, "category": "design"}` satırları JSONL APPEND.
- Determinizm: `_item_rng(seed, stem)` — aynı seed + aynı stem -> bit-identical
  çıktı, işlem sırasından bağımsız (resume güvenliği). DİKKAT: kaynak havuzu
  (fg_dirs içeriği / exclude_fg_stems) değişirse çıktı da değişir — havuz
  seçim indeksleri havuz listesi üzerinden çözülür.
- İdempotentlik: im+gt çifti diskte varsa üretim atlanır; dosya var ama
  manifest satırı eksikse yalnız satır tamamlanır.

`bg_dir` parametresi make_textfx.run() imza kalıbıyla uyum için kabul edilir
ama KULLANILMAZ — zemin tamamen sentetiktir (kağıt/pastel).

Kullanım:
    uv run python scripts/make_design.py --out-dir data/train_design \
        --fg-dirs data/raw_train/trans460_pairs data/raw_train/him2k_merged \
        --toonout-dir /content/downloads/toonout --font-dir /content/fonts \
        --seed 42 --count 6000
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

# make_textfx ile AYNI dizinde (scripts/) — CLI'da script dizini, Colab/test
# tarafında sys.path'e eklenen scripts/ üzerinden import edilir. Yazı/efekt
# makinesi ve ortak sözleşme yardımcıları KOPYALANMAZ, import edilir.
from make_textfx import (  # noqa: F401  (yeniden dışa açılan ortak yardımcılar)
    _append_manifest,
    _bright_color,
    _draw_text_rgba,
    _get_font,
    _item_rng,
    _load_alpha,
    _load_font_paths,
    _load_manifest_ids,
    _load_rgb_capped,
    _pairs_from_dir,
    _rand_text,
    _save_pair,
    _star_points,
)
from make_textfx import _CHARS

# Kaynak havuzlarında 100MP+ görsel olabilir (bkz. make_textfx aynı not).
Image.MAX_IMAGE_PIXELS = None

DEFAULT_COUNT = 6000
DEFAULT_CANVAS = (448, 768)

MARGIN_FRAC = 0.02          # kanvas kenar bandı — GT'de her zaman 0 (köşe garantisi)
PASTEL_BG_PROB = 0.15       # açık pastel düz zemin olasılığı
PAPER_NOISE_PROB = 0.5      # kağıt-beyazı dalında hafif doku olasılığı

FILTER_NONE_PROB = 0.25     # normal kaynakta filtresiz pay (menünün son dalı)
SUBJECT_FRAC_LO, SUBJECT_FRAC_HI = 0.35, 0.7  # özne uzun kenarı / kanvas kısa kenarı
SECOND_SUBJECT_PROB = 0.35
TOON_SUBJECT_PROB = 0.35    # iki havuz da doluysa öznenin ToonOut'tan gelme payı

SMOKE_LO, SMOKE_HI = 0.1, 0.5      # duman alpha bandı (GT'ye aynen girer)
RAY_PROB = 0.5
RAY_ALPHA_LO, RAY_ALPHA_HI = 0.15, 0.6

CURVED_TEXT_PROB = 0.4      # kavisli yazı payı
STACKED_TEXT_PROB = 0.35    # (kavisli seçilmediyse) istifli çok satırlı blok payı
DISTRESS_PROB = 0.5         # eskitme/grunge maskesi olasılığı
SECOND_TEXT_PROB = 0.5      # ikinci yazı bandı olasılığı
DECOR_RANGE = (2, 6)        # dekor adedi (dahil-dahil)

# Halftone/mürekkep "boya" paleti: klasik siyah + tek renk baskı mürekkepleri.
_INK_COLORS: list[tuple[int, int, int]] = [
    (18, 18, 18), (120, 20, 30), (20, 40, 120), (26, 84, 46), (90, 30, 110),
]
_PAPER_RGB = (250, 249, 245)


# ==========================================================================
# Gürültü yardımcıları — duman ve grunge maskeleri için "Perlin benzeri"
# (iki oktavlı value-noise; make_textfx'in gaussian araç kalıbından türedi).
# ==========================================================================
def _value_noise(rng: np.random.Generator, h: int, w: int, cell_px: int) -> np.ndarray:
    """Kaba rastgele ızgaranın bilinear büyütülmesi — [0,1] float32 (H, W)."""
    gh = max(2, round(h / max(1, cell_px)))
    gw = max(2, round(w / max(1, cell_px)))
    grid = (rng.uniform(0.0, 1.0, (gh, gw)) * 255).astype(np.uint8)
    up = Image.fromarray(grid, mode="L").resize((w, h), Image.BILINEAR)
    return np.asarray(up, dtype=np.float32) / 255.0


def _perlin_noise(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    """İki oktavlı value-noise, [0,1]'e normalize — bulut/duman/grunge dokusu."""
    n = 0.65 * _value_noise(rng, h, w, max(8, min(h, w) // 6)) + 0.35 * _value_noise(
        rng, h, w, max(3, min(h, w) // 18)
    )
    n -= n.min()
    mx = float(n.max())
    return (n / mx).astype(np.float32) if mx > 0 else n.astype(np.float32)


# ==========================================================================
# Zemin — kağıt beyazı / krem / açık pastel (GT'de alpha=0)
# ==========================================================================
def _design_bg(rng: np.random.Generator, size: tuple[int, int]) -> np.ndarray:
    """Baskı zemini (H, W, 3 uint8): 245-255 bandı kağıt beyazı/krem (bazen
    hafif dokulu) veya %15 olasılıkla açık pastel düz renk."""
    w, h = size
    if rng.uniform() < PASTEL_BG_PROB:
        col = (255 - rng.integers(12, 60, 3)).astype(np.float32)  # açık pastel
        arr = np.broadcast_to(col, (h, w, 3)).astype(np.float32).copy()
        return arr.round().clip(0, 255).astype(np.uint8)

    base = float(rng.integers(245, 256))
    col = np.array([base, base, base], dtype=np.float32)
    if rng.uniform() < 0.5:  # krem tonu: mavi kanal hafif kısılır
        col[1] -= float(rng.uniform(0.0, 4.0))
        col[2] -= float(rng.uniform(2.0, 10.0))
    arr = np.broadcast_to(col, (h, w, 3)).astype(np.float32).copy()
    if rng.uniform() < PAPER_NOISE_PROB:  # düşük genlikli kağıt dokusu
        amp = float(rng.uniform(1.5, 4.0))
        arr += amp * rng.standard_normal((h, w, 1)).astype(np.float32)
    return arr.round().clip(0, 255).astype(np.uint8)


# ==========================================================================
# Baskı-stili filtreler — YALNIZ RGB'ye; alpha AYNEN döner (bit-birebir)
# ==========================================================================
def _luminance(rgb: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 -> [0,1] float32 luminance."""
    f = rgb.astype(np.float32)
    return (0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]) / 255.0


def _filter_halftone(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Klasik gazete tramı: hücre bazlı ortalama luminance -> nokta yarıçapı
    (koyu bölge = büyük mürekkep noktası). Tam vektörel — hücre döngüsü yok."""
    h, w = rgb.shape[:2]
    cell = int(rng.integers(4, 11))
    lum = _luminance(rgb)
    ch, cw = math.ceil(h / cell), math.ceil(w / cell)
    pad_lum = np.pad(lum, ((0, ch * cell - h), (0, cw * cell - w)), mode="edge")
    lum_c = pad_lum.reshape(ch, cell, cw, cell).mean(axis=(1, 3))
    radius_c = (cell / 2.0) * np.sqrt(np.clip(1.0 - lum_c, 0.0, 1.0)) * float(
        rng.uniform(1.05, 1.35)
    )
    radius = np.repeat(np.repeat(radius_c, cell, axis=0), cell, axis=1)[:h, :w]
    ly = (np.arange(h, dtype=np.float32) % cell) - (cell - 1) / 2.0
    lx = (np.arange(w, dtype=np.float32) % cell) - (cell - 1) / 2.0
    d2 = ly[:, None] ** 2 + lx[None, :] ** 2
    dot = d2 <= radius**2
    ink = np.asarray(_INK_COLORS[int(rng.integers(0, len(_INK_COLORS)))], dtype=np.uint8)
    paper = np.asarray(_PAPER_RGB, dtype=np.uint8)
    return np.where(dot[..., None], ink[None, None, :], paper[None, None, :])


def _filter_posterize(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Posterize (3-5 seviye) + doygunluk artışı."""
    levels = int(rng.integers(3, 6))
    step = 256.0 / levels
    q = (np.floor(rgb.astype(np.float32) / step) * (255.0 / (levels - 1))).clip(0, 255)
    im = Image.fromarray(q.astype(np.uint8), mode="RGB")
    im = ImageEnhance.Color(im).enhance(float(rng.uniform(1.2, 1.8)))
    return np.asarray(im, dtype=np.uint8)


def _filter_ink(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Yüksek kontrast 'mürekkep': luminance eşiklemesi + hafif kenar vurgusu."""
    lum = _luminance(rgb)
    thresh = float(rng.uniform(0.35, 0.6))
    gray8 = (lum * 255).astype(np.uint8)
    edges = np.asarray(
        Image.fromarray(gray8, mode="L").filter(ImageFilter.FIND_EDGES), dtype=np.float32
    ) / 255.0
    ink_mask = (lum <= thresh) | (edges > 0.3)
    ink = np.asarray(_INK_COLORS[int(rng.integers(0, len(_INK_COLORS)))], dtype=np.uint8)
    paper = np.asarray(_PAPER_RGB, dtype=np.uint8)
    return np.where(ink_mask[..., None], ink[None, None, :], paper[None, None, :])


def apply_print_filter(
    rgb: np.ndarray, alpha: np.ndarray, rng: np.random.Generator, kind: str
) -> tuple[np.ndarray, np.ndarray]:
    """Baskı-stili filtreyi YALNIZ RGB'ye uygular; alpha AYNEN (aynı dizi,
    bit-birebir) döner — kategori tasarımının kritik sözleşmesi: filtre stil
    değiştirir, saydamlık ground-truth'u değiştirmez."""
    if kind == "halftone":
        return _filter_halftone(rgb, rng), alpha
    if kind == "posterize":
        return _filter_posterize(rgb, rng), alpha
    if kind == "ink":
        return _filter_ink(rgb, rng), alpha
    return rgb, alpha  # "none"


def _pick_filter(rng: np.random.Generator, is_toon: bool) -> str:
    """Filtre menüsü: normal kaynakta 4 dal eşit (%25 filtresiz); ToonOut zaten
    illüstrasyon olduğundan çoğunlukla filtresiz/posterize."""
    u = float(rng.uniform())
    if is_toon:
        if u < 0.5:
            return "none"
        return "posterize" if u < 0.9 else "ink"
    if u < 0.25:
        return "halftone"
    if u < 0.5:
        return "posterize"
    if u < 1.0 - FILTER_NONE_PROB:
        return "ink"
    return "none"


# ==========================================================================
# Dumanlı kenar / airbrush — alpha'ya dışa kıvrılan yarı saydam duman lekeleri
# ==========================================================================
def _smoke_alpha(
    alpha: np.ndarray, rng: np.random.Generator, reach_frac: float | None = None
) -> np.ndarray:
    """Nesne sınırından DIŞA kıvrılan duman/bulut lekeleri: [SMOKE_LO, SMOKE_HI]
    bandında, nesnenin İÇİNDE (alpha > 0.05) her zaman 0. Zarf = alpha'nın
    gaussian dışa bulanığı; doku = Perlin benzeri value-noise (lekeli kesim +
    hafif blur ile organik kenar)."""
    h, w = alpha.shape
    if reach_frac is None:
        reach_frac = float(rng.uniform(0.05, 0.12))
    reach = max(2.0, min(h, w) * reach_frac)
    soft = np.asarray(
        Image.fromarray((alpha * 255).clip(0, 255).astype(np.uint8), mode="L").filter(
            ImageFilter.GaussianBlur(reach)
        ),
        dtype=np.float32,
    ) / 255.0
    envelope = np.clip(soft * float(rng.uniform(1.6, 2.4)), 0.0, 1.0)
    noise = _perlin_noise(rng, h, w)
    smoke = (SMOKE_LO + (SMOKE_HI - SMOKE_LO) * noise) * envelope
    smoke = smoke * (noise > float(rng.uniform(0.25, 0.45)))  # lekeli/parçalı kesim
    smoke = np.asarray(
        Image.fromarray((smoke * 255).astype(np.uint8), mode="L").filter(
            ImageFilter.GaussianBlur(1.0)
        ),
        dtype=np.float32,
    ) / 255.0
    smoke[alpha > 0.05] = 0.0  # duman yalnız DIŞA — nesne içi GT'si değişmez
    return np.clip(smoke, 0.0, SMOKE_HI).astype(np.float32)


# ==========================================================================
# Stilize özne katmanı — kesit + baskı filtresi + duman (alpha'lar ayrı eleman)
# ==========================================================================
def _resize_pair(
    rgb: np.ndarray, alpha: np.ndarray, size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """size = (w, h); RGB LANCZOS, alpha BILINEAR (make_composites kalıbı)."""
    rgb_r = np.asarray(
        Image.fromarray(rgb, mode="RGB").resize(size, Image.LANCZOS), dtype=np.uint8
    )
    a_r = np.asarray(
        Image.fromarray((alpha * 255).clip(0, 255).astype(np.uint8), mode="L").resize(
            size, Image.BILINEAR
        ),
        dtype=np.float32,
    ) / 255.0
    return rgb_r, a_r


def _subject_layers(
    rng: np.random.Generator,
    pair: tuple[Path, Path],
    is_toon: bool,
    canvas_min: int,
    max_w: int,
    max_h: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], tuple[int, int]] | None:
    """Tek öznenin (duman + gövde) eleman listesi ve katman boyutu (lw, lh).

    Elemanlar: [(duman_rgb, duman_alpha), (özne_rgb, özne_alpha)] — duman önce
    kompozit edilir, özne üstüne biner; GT union'ı ikisini de içerir. Boş
    alpha'lı kaynakta None (özne atlanır)."""
    im_path, gt_path = pair
    rgb = _load_rgb_capped(im_path)
    alpha = _load_alpha(gt_path, (rgb.shape[1], rgb.shape[0]))
    ys, xs = np.nonzero(alpha > 0.05)
    if xs.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    rgb_c, a_c = rgb[y0:y1, x0:x1], alpha[y0:y1, x0:x1]

    # Baskı-stili filtre — YALNIZ RGB'ye (apply_print_filter sözleşmesi).
    rgb_c, a_c = apply_print_filter(rgb_c, a_c, rng, _pick_filter(rng, is_toon))

    # Ölçek: özne uzun kenarı kanvas kısa kenarının %35-70'i.
    target = canvas_min * float(rng.uniform(SUBJECT_FRAC_LO, SUBJECT_FRAC_HI))
    reach_frac = float(rng.uniform(0.05, 0.12))
    ch, cw = a_c.shape
    scale = target / max(ch, cw)
    new_w = max(1, int(round(cw * scale)))
    new_h = max(1, int(round(ch * scale)))
    rgb_s, a_s = _resize_pair(rgb_c, a_c, (new_w, new_h))

    pad = max(2, int(round(2.5 * reach_frac * min(new_w, new_h))))
    lw, lh = new_w + 2 * pad, new_h + 2 * pad
    a_p = np.zeros((lh, lw), dtype=np.float32)
    a_p[pad : pad + new_h, pad : pad + new_w] = a_s
    subj_rgb = np.zeros((lh, lw, 3), dtype=np.float32)
    subj_rgb[pad : pad + new_h, pad : pad + new_w] = rgb_s.astype(np.float32)

    smoke = _smoke_alpha(a_p, rng, reach_frac)
    smoke_col = np.clip(
        float(rng.integers(150, 236)) + rng.uniform(-12.0, 12.0, 3), 0, 255
    ).astype(np.float32)
    smoke_rgb = np.broadcast_to(smoke_col, (lh, lw, 3)).astype(np.float32).copy()

    layers = [(smoke_rgb, smoke), (subj_rgb, a_p)]

    # Kanvasa sığmıyorsa katmanlar oransal küçültülür (duman dahil).
    if lw > max_w or lh > max_h:
        f = min(max_w / lw, max_h / lh)
        nw, nh = max(1, int(lw * f)), max(1, int(lh * f))
        resized = []
        for l_rgb, l_a in layers:
            r2, a2 = _resize_pair(l_rgb.round().clip(0, 255).astype(np.uint8), l_a, (nw, nh))
            resized.append((r2.astype(np.float32), a2))
        layers, (lw, lh) = resized, (nw, nh)
    return layers, (lw, lh)


# ==========================================================================
# Işıma/patlama — öznenin arkasına radyal ışın demeti veya glow (yarı saydam)
# ==========================================================================
def _ray_layer(
    rng: np.random.Generator, size: tuple[int, int], center: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Kanvas boyutunda (rgb, alpha) eleman: radyal ışın demeti (sunburst) veya
    gaussian glow. Alpha [RAY_ALPHA_LO, RAY_ALPHA_HI] bandında yarı saydam —
    GT'ye aynen girer (kenar bandı kompozitte ayrıca sıfırlanır)."""
    w, h = size
    cx, cy = center
    val = float(rng.uniform(RAY_ALPHA_LO, RAY_ALPHA_HI))
    color = np.asarray(
        (255, 255, 255) if rng.uniform() < 0.4 else _bright_color(rng), dtype=np.float32
    )
    if rng.uniform() < 0.5:  # glow: beyaza eriyen yumuşak ışıma
        sigma = min(w, h) * float(rng.uniform(0.08, 0.18))
        yy = (np.arange(h, dtype=np.float32) - cy)[:, None]
        xx = (np.arange(w, dtype=np.float32) - cx)[None, :]
        a = (val * np.exp(-(xx**2 + yy**2) / (2 * sigma**2))).astype(np.float32)
    else:  # sunburst: eşit aralıklı ışın kamaları
        mask = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(mask)
        n = int(rng.integers(8, 17))
        rot0 = float(rng.uniform(0, 2 * math.pi))
        r = 0.5 * min(w, h) * float(rng.uniform(0.7, 1.1))
        half = (math.pi / n) * float(rng.uniform(0.25, 0.45))
        for k in range(n):
            ang = rot0 + k * 2 * math.pi / n
            p1 = (cx + r * math.cos(ang - half), cy + r * math.sin(ang - half))
            p2 = (cx + r * math.cos(ang + half), cy + r * math.sin(ang + half))
            d.polygon([(cx, cy), p1, p2], fill=255)
        a = (np.asarray(mask, dtype=np.float32) / 255.0) * val
    rgb = np.broadcast_to(color, (h, w, 3)).astype(np.float32).copy()
    return rgb, a.astype(np.float32)


# ==========================================================================
# Küçük dekorlar — yıldız / şimşek / sıçrama lekeleri (basit vektör çizimler)
# ==========================================================================
_BOLT_PTS = [(0.45, 0.0), (0.62, 0.0), (0.46, 0.42), (0.68, 0.42),
             (0.30, 1.0), (0.44, 0.55), (0.26, 0.55)]


def _decor_layer(
    rng: np.random.Generator, size: tuple[int, int], margin: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """2-6 dekoru tek RGBA katmanına çizer; hiç dekor yoksa None."""
    w, h = size
    n = int(rng.integers(DECOR_RANGE[0], DECOR_RANGE[1] + 1))
    if n <= 0:
        return None
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for _ in range(n):
        s = min(w, h) * float(rng.uniform(0.025, 0.07))
        lo_x, hi_x = margin + s, w - margin - s
        lo_y, hi_y = margin + s, h - margin - s
        cx = float(rng.uniform(lo_x, hi_x)) if hi_x > lo_x else w / 2
        cy = float(rng.uniform(lo_y, hi_y)) if hi_y > lo_y else h / 2
        a = int(float(rng.uniform(0.4, 1.0)) * 255)  # katı veya yarı saydam
        col = _bright_color(rng) + (a,)
        kind = int(rng.integers(0, 3))
        if kind == 0:  # yıldız
            d.polygon(_star_points(cx, cy, s, s * 0.45, n=int(rng.integers(4, 7))), fill=col)
        elif kind == 1:  # şimşek
            ang = float(rng.uniform(-0.5, 0.5))
            ca, sa = math.cos(ang), math.sin(ang)
            pts = []
            for px, py in _BOLT_PTS:
                dx, dy = (px - 0.45) * 2 * s, (py - 0.5) * 2 * s
                pts.append((cx + dx * ca - dy * sa, cy + dx * sa + dy * ca))
            d.polygon(pts, fill=col)
        else:  # sıçrama lekesi: merkez damla + uydu damlacıklar
            r0 = s * 0.55
            d.ellipse([cx - r0, cy - r0, cx + r0, cy + r0], fill=col)
            for _ in range(int(rng.integers(3, 8))):
                ang = float(rng.uniform(0, 2 * math.pi))
                dist = s * float(rng.uniform(0.7, 1.1))
                rr = r0 * float(rng.uniform(0.15, 0.4))
                px, py = cx + dist * math.cos(ang), cy + dist * math.sin(ang)
                d.ellipse([px - rr, py - rr, px + rr, py + rr], fill=col)
    arr = np.asarray(layer, dtype=np.float32)
    return arr[..., :3], arr[..., 3] / 255.0


# ==========================================================================
# Display yazı — make_textfx yazı makinesi + kavis / istif / eskitme
# ==========================================================================
def _word(rng: np.random.Generator, lo: int = 4, hi: int = 10) -> str:
    n = int(rng.integers(lo, hi))
    return "".join(_CHARS[int(rng.integers(0, len(_CHARS)))] for _ in range(n))


def _curved_text_rgba(
    text: str,
    font,
    fill: tuple[int, int, int, int],
    theta: float,
    stroke_width: int = 0,
    stroke_fill: tuple[int, int, int] | None = None,
) -> Image.Image:
    """KAVİSLİ display yazı: harfler tek tek, tepe noktası üstte olan bir yay
    (arch) üzerine yerleştirilir ve yayın teğetine döndürülür. `theta` toplam
    yay açısı (radyan). Deterministiktir (rng almaz) — testler doğrudan çağırır."""
    text = text.strip() or "A"
    theta = max(0.15, min(float(theta), 2.4))
    pad = max(2, stroke_width + 2)
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    space_w = max(2, int(probe.textlength("i", font=font)))
    glyphs: list[tuple[Image.Image | None, float]] = []
    for ch in text:
        if ch.isspace():
            glyphs.append((None, float(space_w)))
            continue
        img = _draw_text_rgba(ch, font, fill, stroke_width, stroke_fill, pad)
        glyphs.append((img, float(img.width - 2 * pad)))
    tracking = 2.0
    total_w = sum(wch for _, wch in glyphs) + tracking * max(0, len(glyphs) - 1)
    radius = total_w / theta
    gh_max = max((img.height for img, _ in glyphs if img is not None), default=8)
    sag = radius * (1 - math.cos(theta / 2))
    cw = int(total_w + 2 * gh_max)
    chh = int(sag + 2 * gh_max)
    canvas = Image.new("RGBA", (cw, chh), (0, 0, 0, 0))
    cx = cw / 2.0
    cy = gh_max * 0.5 + radius  # çember merkezi; yay tepesi y ~= gh_max*0.5
    cum = 0.0
    for img, wch in glyphs:
        phi = -theta / 2 + (cum + wch / 2) / radius
        cum += wch + tracking
        if img is None:
            continue
        rot = img.rotate(-math.degrees(phi), expand=True, resample=Image.BICUBIC)
        gx = cx + radius * math.sin(phi)
        gy = cy - radius * math.cos(phi)
        canvas.alpha_composite(
            rot, (int(round(gx - rot.width / 2)), int(round(gy - rot.height / 2)))
        )
    bbox = canvas.getbbox()
    return canvas.crop(bbox) if bbox else canvas


def _stacked_text_rgba(
    rng: np.random.Generator,
    font,
    fill: tuple[int, int, int, int],
    stroke_width: int,
    stroke_fill: tuple[int, int, int] | None,
) -> Image.Image:
    """İstifli çok satırlı display blok: 2-3 satır, ortalanmış."""
    pad = max(2, stroke_width + 2)
    lines = []
    for _ in range(int(rng.integers(2, 4))):
        n_words = 1 if rng.uniform() < 0.7 else 2
        text = " ".join(_word(rng, 3, 8) for _ in range(n_words))
        lines.append(_draw_text_rgba(text, font, fill, stroke_width, stroke_fill, pad))
    gap = max(1, int(0.12 * max(im.height for im in lines)))
    bw = max(im.width for im in lines)
    bh = sum(im.height for im in lines) + gap * (len(lines) - 1)
    block = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
    y = 0
    for im in lines:
        block.alpha_composite(im, ((bw - im.width) // 2, y))
        y += im.height + gap
    return block


def _distress(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Eskitme: yazı alpha'sından value-noise grunge maskesiyle parça eksiltir
    (GT'ye de aynen yansır — eksik parça tasarımın kendisidir)."""
    arr = np.array(img)
    noise = _perlin_noise(rng, arr.shape[0], arr.shape[1])
    keep = noise > float(rng.uniform(0.15, 0.35))
    arr[..., 3] = (arr[..., 3] * keep).astype(np.uint8)
    return Image.fromarray(arr, mode="RGBA")


def _ink_or_bright(rng: np.random.Generator) -> tuple[int, int, int]:
    """Baskı yazısı rengi: %50 koyu mürekkep, %50 parlak display rengi."""
    if rng.uniform() < 0.5:
        c = rng.integers(0, 70, 3)
        return (int(c[0]), int(c[1]), int(c[2]))
    return _bright_color(rng)


def _text_block(
    rng: np.random.Generator, canvas_size: tuple[int, int], font_paths: list[Path]
) -> Image.Image:
    """Tek display yazı bloğu: kavisli / istifli / tek satır (+eskitme)."""
    cmin = min(canvas_size)
    font_size = max(10, int(cmin * float(rng.uniform(0.07, 0.16))))
    font = _get_font(font_paths, rng, font_size)
    fill = _ink_or_bright(rng) + (255,)
    stroke_width, stroke_fill = 0, None
    if rng.uniform() < 0.4:
        stroke_width = max(1, font_size // 12)
        stroke_fill = _ink_or_bright(rng)
    u = float(rng.uniform())
    if u < CURVED_TEXT_PROB:
        theta = float(rng.uniform(0.5, 1.6))
        img = _curved_text_rgba(_word(rng).upper(), font, fill, theta, stroke_width, stroke_fill)
    elif u < CURVED_TEXT_PROB + STACKED_TEXT_PROB:
        img = _stacked_text_rgba(rng, font, fill, stroke_width, stroke_fill)
    else:
        img = _draw_text_rgba(
            _rand_text(rng), font, fill, stroke_width, stroke_fill, max(2, stroke_width + 2)
        )
    if rng.uniform() < DISTRESS_PROB:
        img = _distress(img, rng)
    return img


# ==========================================================================
# Örnek kompozisyonu — GT = tüm elemanların alpha union'ı, zemin alpha=0
# ==========================================================================
def _paste_element(
    rgb_small: np.ndarray, a_small: np.ndarray, x0: int, y0: int, size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Küçük katmanı kanvas boyutunda (rgb float, alpha float) elemana gömer."""
    w, h = size
    sh, sw = a_small.shape
    rgb_full = np.zeros((h, w, 3), dtype=np.float32)
    a_full = np.zeros((h, w), dtype=np.float32)
    rgb_full[y0 : y0 + sh, x0 : x0 + sw] = rgb_small
    a_full[y0 : y0 + sh, x0 : x0 + sw] = a_small
    return rgb_full, a_full


def _rgba_to_element(img: Image.Image, x0: int, y0: int, size: tuple[int, int]):
    arr = np.asarray(img, dtype=np.float32)
    return _paste_element(arr[..., :3], arr[..., 3] / 255.0, x0, y0, size)


def _fit_rgba(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    if img.width <= max_w and img.height <= max_h:
        return img
    f = min(max_w / img.width, max_h / img.height)
    return img.resize((max(1, int(img.width * f)), max(1, int(img.height * f))), Image.LANCZOS)


def _render_design_sample(
    rng: np.random.Generator,
    size: tuple[int, int],
    fg_pairs: list[tuple[Path, Path]],
    toon_pairs: list[tuple[Path, Path]],
    font_paths: list[Path],
) -> tuple[np.ndarray, np.ndarray]:
    """Tek design örneği: (kompozit RGB uint8, alpha float32 [0,1]).

    Eleman sırası (alta -> üste): ışıma -> özne(ler; duman + gövde) -> dekorlar
    -> yazı blokları. GT tüm eleman alpha'larının union'ı; kanvasın MARGIN_FRAC
    kenar bandı her elemanda sıfırlanır (zemin köşeleri GT'de daima 0)."""
    w, h = size
    m = max(2, int(MARGIN_FRAC * min(w, h)))
    bg = _design_bg(rng, size)

    elements: list[tuple[np.ndarray, np.ndarray]] = []
    subject_elements: list[tuple[np.ndarray, np.ndarray]] = []
    centers: list[tuple[float, float]] = []

    # 1) Stilize özne(ler) — 1-2 adet (havuz boşsa 0; ör. testlerde yalnız yazı).
    n_sub = (1 + (1 if rng.uniform() < SECOND_SUBJECT_PROB else 0)) if (fg_pairs or toon_pairs) else 0
    for _ in range(n_sub):
        if toon_pairs and fg_pairs:
            use_toon = rng.uniform() < TOON_SUBJECT_PROB
        else:
            use_toon = bool(toon_pairs)
        pool = toon_pairs if use_toon else fg_pairs
        pair = pool[int(rng.integers(0, len(pool)))]
        built = _subject_layers(rng, pair, use_toon, min(w, h), w - 2 * m, h - 2 * m)
        if built is None:
            continue
        layers, (lw, lh) = built
        x0 = int(rng.integers(m, max(m, w - m - lw) + 1))
        y0 = int(rng.integers(m, max(m, h - m - lh) + 1))
        for l_rgb, l_a in layers:
            subject_elements.append(_paste_element(l_rgb, l_a, x0, y0, size))
        centers.append((x0 + lw / 2.0, y0 + lh / 2.0))

    # 2) Işıma/patlama — öznenin ARKASINA (%50).
    if centers and rng.uniform() < RAY_PROB:
        elements.append(_ray_layer(rng, size, centers[0]))
    elements += subject_elements

    # 3) Küçük dekorlar.
    decor = _decor_layer(rng, size, m)
    if decor is not None:
        elements.append(decor)

    # 4) Display yazı blokları — üst ve/veya alt bant.
    n_text = 1 + (1 if rng.uniform() < SECOND_TEXT_PROB else 0)
    bands = ["top", "bottom"] if n_text == 2 else (["top"] if rng.uniform() < 0.5 else ["bottom"])
    for band in bands:
        img = _fit_rgba(_text_block(rng, size, font_paths), max(1, int(0.9 * (w - 2 * m))),
                        max(1, int(0.28 * h)))
        tw, th = img.size
        x0 = int(round((w - tw) / 2 + float(rng.uniform(-0.08, 0.08)) * w))
        x0 = min(max(m, x0), max(m, w - m - tw))
        jitter = int(rng.integers(0, max(1, int(0.08 * h))))
        y0 = m + jitter if band == "top" else max(m, h - m - th - jitter)
        elements.append(_rgba_to_element(img, x0, y0, size))

    # Kompozit + GT union'ı (kenar bandı her elemanda sıfırlanır).
    out_rgb = bg.astype(np.float32)
    total_a = np.zeros((h, w), dtype=np.float32)
    for el_rgb, el_a in elements:
        el_a = el_a.copy()
        el_a[:m, :] = 0.0
        el_a[h - m :, :] = 0.0
        el_a[:, :m] = 0.0
        el_a[:, w - m :] = 0.0
        out_rgb = el_a[..., None] * el_rgb + (1 - el_a[..., None]) * out_rgb
        total_a = 1.0 - (1.0 - total_a) * (1.0 - el_a)
    return out_rgb.round().clip(0, 255).astype(np.uint8), total_a.astype(np.float32)


# ==========================================================================
# Üretim döngüsü + orkestrasyon (make_textfx.gen_text / run kalıbı)
# ==========================================================================
def gen_design(
    count: int,
    out_im_dir: Path,
    out_gt_dir: Path,
    fg_pairs: list[tuple[Path, Path]],
    toon_pairs: list[tuple[Path, Path]],
    font_paths: list[Path],
    seed: int,
    existing_ids: set[str],
    canvas_range: tuple[int, int] = DEFAULT_CANVAS,
) -> tuple[list[dict], int, int]:
    """(manifest satırları, üretilen çift sayısı, atlanan çift sayısı) döndürür."""
    new_rows: list[dict] = []
    generated = skipped = 0
    lo, hi = canvas_range
    for i in range(count):
        stem = f"design_{i:05d}_c00"
        img_path = out_im_dir / f"{stem}.jpg"
        gt_path = out_gt_dir / f"{stem}.png"
        row = {"id": stem, "category": "design"}
        if img_path.exists() and gt_path.exists():
            skipped += 1
            if stem not in existing_ids:
                new_rows.append(row)  # dosya var, manifest satırı eksik -> yalnız satır
            continue
        rng = _item_rng(seed, stem)
        w = int(rng.integers(lo, hi + 1))
        h = int(rng.integers(lo, hi + 1))
        rgb, alpha = _render_design_sample(rng, (w, h), fg_pairs, toon_pairs, font_paths)
        _save_pair(rgb, alpha, img_path, gt_path)
        new_rows.append(row)
        generated += 1
    return new_rows, generated, skipped


def run(
    out_dir: Path,
    bg_dir: Path | None = None,  # imza uyumu (make_textfx kalıbı) — KULLANILMAZ
    fg_dirs: list[Path] | None = None,
    toonout_dir: Path | None = None,
    font_dir: Path | None = None,
    seed: int = 42,
    count: int = DEFAULT_COUNT,
    out_manifest: Path | None = None,
    canvas_range: tuple[int, int] = DEFAULT_CANVAS,
    exclude_fg_stems: set[str] | None = None,
) -> dict[str, int]:
    """design üreticisini koşturur; {"design": yeni üretilen} döndürür (yalnız
    >0 ise — make_textfx.run() kalıbı). `bg_dir` kullanılmaz (zemin sentetik).

    `exclude_fg_stems`: kaynak olarak KULLANILMAYACAK ham fg stem'leri (VAL
    sızıntı koruması — çağıran val_stems.json'dan türetir; bkz.
    training/v7_veri_guncelleme_hucresi.py). DİKKAT: havuz değişirse aynı
    seed'in çıktıları da değişir — koruma kümesi koşular arasında sabit
    tutulmalı (resume aynı kümeyle yapılmalı)."""
    out_dir = Path(out_dir)
    out_im_dir = out_dir / "im"
    out_gt_dir = out_dir / "gt"
    out_im_dir.mkdir(parents=True, exist_ok=True)
    out_gt_dir.mkdir(parents=True, exist_ok=True)
    out_manifest = Path(out_manifest) if out_manifest else out_dir / "manifest.jsonl"
    existing_ids = _load_manifest_ids(out_manifest)

    fg_pairs: list[tuple[Path, Path]] = []
    for d in fg_dirs or []:
        fg_pairs += _pairs_from_dir(Path(d))
    toon_pairs = _pairs_from_dir(Path(toonout_dir)) if toonout_dir else []
    if exclude_fg_stems:
        fg_pairs = [p for p in fg_pairs if p[0].stem not in exclude_fg_stems]
        toon_pairs = [p for p in toon_pairs if p[0].stem not in exclude_fg_stems]
    if count > 0 and not (fg_pairs or toon_pairs):
        raise SystemExit(
            "design için kaynak im/gt çifti bulunamadı (--fg-dirs kökleri im/ + gt/ "
            "içermeli ve/veya --toonout-dir verilmeli)"
        )
    font_paths = _load_font_paths(Path(font_dir) if font_dir else None)

    rows, generated, skipped = gen_design(
        count, out_im_dir, out_gt_dir, fg_pairs, toon_pairs, font_paths, seed,
        existing_ids, canvas_range=canvas_range,
    )

    # manifest'e yalnız yeni id'ler (run içi güvenlik dedup'u dahil — make_textfx kalıbı)
    fresh: list[dict] = []
    seen = set(existing_ids)
    for row in rows:
        if row["id"] not in seen:
            seen.add(row["id"])
            fresh.append(row)
    if fresh:
        _append_manifest(out_manifest, fresh)

    print(f"{generated} yeni çift yazıldı, {skipped} zaten vardı (atlandı)")
    return {"design": generated} if generated else {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out-dir", required=True, help="çıktı kökü (im/ + gt/ + manifest.jsonl)")
    parser.add_argument("--bg-dir", default=None,
                        help="imza uyumu için kabul edilir — KULLANILMAZ (zemin sentetik)")
    parser.add_argument(
        "--fg-dirs", nargs="*", default=[],
        help="özne kaynak kökleri; her kök im/ + gt/ alt dizinleri içermeli (stem eşleşmeli)",
    )
    parser.add_argument("--toonout-dir", default=None, help="ToonOut kökü (im/ + gt/)")
    parser.add_argument("--font-dir", default=None,
                        help=".ttf/.otf/.ttc font havuzu (yoksa PIL varsayılanı)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--out-manifest", default=None, help="varsayılan: <out-dir>/manifest.jsonl")
    parser.add_argument(
        "--exclude-stems-file", default=None,
        help="her satırda bir ham fg stem'i (VAL sızıntı koruması) — kaynak olarak kullanılmaz",
    )
    args = parser.parse_args()
    exclude = None
    if args.exclude_stems_file:
        exclude = {
            line.strip()
            for line in Path(args.exclude_stems_file).read_text().splitlines()
            if line.strip()
        }
    run(
        Path(args.out_dir),
        bg_dir=Path(args.bg_dir) if args.bg_dir else None,
        fg_dirs=[Path(d) for d in args.fg_dirs],
        toonout_dir=Path(args.toonout_dir) if args.toonout_dir else None,
        font_dir=Path(args.font_dir) if args.font_dir else None,
        seed=args.seed,
        count=args.count,
        out_manifest=Path(args.out_manifest) if args.out_manifest else None,
        exclude_fg_stems=exclude,
    )


if __name__ == "__main__":
    main()
