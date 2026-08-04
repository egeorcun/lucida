"""scripts/poster_mode.py — the decisive poster/POD policy.

Contracts (the Ideogram-policy discussion, 2026-07-28):
- nothing survives outside kept components (haze impossible);
- "open" (default): CONFIDENT-zero holes stay transparent and floating
  islands inside them (drips) are removed; HESITANT holes print solid;
- "solid": every small hole prints solid;
- large enclosed windows stay open in every mode;
- the outer edge keeps the model's soft alpha.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    "poster_mode", Path(__file__).parent.parent / "scripts" / "poster_mode.py")
pm = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("poster_mode", pm)
_SPEC.loader.exec_module(pm)


def _scene():
    """200x200: big square subject with a smoky hole, a letter-like counter
    containing a small kept drip, a big window, and outside haze."""
    a = np.zeros((200, 200), dtype=np.float32)
    a[20:180, 20:180] = 1.0
    a[40:60, 40:60] = 0.35            # smoky hole (model hesitated)
    a[80:100, 40:60] = 0.0            # counter: confident zero...
    a[88:92, 48:52] = 1.0             # ...with a floating drip inside
    a[120:170, 100:160] = 0.0         # big window (stays open everywhere)
    a[5:12, 5:12] = 0.2               # outside haze speck
    return a


def test_outside_haze_always_dies():
    for mode in ("open", "solid"):
        out = pm.poster_alpha(_scene(), counters=mode)
        assert out[5:12, 5:12].max() == 0.0


def test_open_mode_clears_counters_and_drips():
    out = pm.poster_alpha(_scene(), counters="open")
    assert out[85:95, 45:55].max() == 0.0, "counter must be fully transparent"
    assert out[89:91, 49:51].max() == 0.0, "floating drip must be removed"
    assert out[45:55, 45:55].min() > 0.99, \
        "hesitant (smoky) hole prints solid even in open mode"


def test_solid_mode_fills_small_holes():
    out = pm.poster_alpha(_scene(), counters="solid")
    assert out[45:55, 45:55].min() > 0.99, "smoky hole prints solid"
    assert out[85:95, 45:55].min() > 0.99, "counter prints solid"


def test_big_window_stays_open_in_all_modes():
    for mode in ("open", "solid"):
        out = pm.poster_alpha(_scene(), counters=mode, max_hole_frac=0.1)
        assert out[130:160, 110:150].max() == 0.0, mode


def test_invalid_mode_raises():
    import pytest
    with pytest.raises(ValueError):
        pm.poster_alpha(_scene(), counters="banana")


def test_counters_auto_resolves_by_page_color():
    import numpy as np
    a = _scene()
    white_page = np.full((200, 200, 3), 250.0, dtype=np.float32)
    cream_page = np.full((200, 200, 3), (246.0, 240.0, 214.0), dtype=np.float32)
    open_like = pm.poster_alpha(a, counters="auto", rgb=white_page, haze_matting=False)
    solid_like = pm.poster_alpha(a, counters="auto", rgb=cream_page, haze_matting=False)
    # counter bolgesi: beyaz sayfada acik, krem sayfada dolu
    assert open_like[85:95, 45:55].max() == 0.0
    assert solid_like[85:95, 45:55].min() > 0.99


def test_haze_matting_thins_draining_page_like_haze_only():
    """The CHEESE airiness lesson: page-like haze that drains to the
    silhouette edge drops to ink density; outline-locked whites survive."""
    import numpy as np
    a = np.zeros((200, 200), dtype=np.float32)
    a[20:180, 20:180] = 1.0
    rgb = np.full((200, 200, 3), 250.0, dtype=np.float32)   # page-white everywhere
    rgb[20:180, 20:180] = 240.0                              # near-page haze block
    rgb[80:120, 80:120] = (200, 30, 30)                      # a red element inside
    out = pm.poster_alpha(a, counters="open", rgb=rgb, haze_matting=True)
    assert out[30, 100] < 0.2, "draining page-like haze must thin out"
    assert out[100, 100] > 0.9, "colored content keeps full alpha"
    out_off = pm.poster_alpha(a, counters="open", rgb=rgb, haze_matting=False)
    assert out_off[30, 100] > 0.9, "matting off -> untouched"


def test_chromatic_rescue_restores_saturated_ink_the_model_erased():
    """The rainbow lesson: on a flat page, saturated color far from the page
    color is ink even when the model zeroed it; page-colored and near-page
    (gray shadow) pixels are never resurrected."""
    import numpy as np
    a = np.zeros((200, 200), dtype=np.float32)
    a[20:100, 20:180] = 1.0                                  # kept art block
    rgb = np.full((200, 200, 3), 250.0, dtype=np.float32)    # white page
    rgb[20:100, 20:180] = (40, 40, 40)                       # the kept ink
    rgb[120:150, 20:180] = (0, 160, 220)                     # rainbow band, model erased
    rgb[160:170, 20:180] = (215, 215, 215)                   # faint gray shadow
    out = pm.poster_alpha(a, counters="open", rgb=rgb, haze_matting=False)
    assert out[130:140, 60:140].min() > 0.9, "saturated band must come back solid"
    assert out[164:168, 60:140].max() == 0.0, "near-page shadow stays removed"
    out_norgb = pm.poster_alpha(a, counters="open", rgb=None, haze_matting=False)
    assert out_norgb[130:140, 60:140].max() == 0.0, "no rgb -> no rescue"


def test_chromatic_rescue_ignores_white_gaps_between_ink():
    """Halftone/small-type lesson: page-colored gaps surrounded by colored
    ink must NOT be resurrected — rescue judges each pixel's own color."""
    import numpy as np
    a = np.zeros((200, 200), dtype=np.float32)
    a[20:100, 20:180] = 1.0
    rgb = np.full((200, 200, 3), 250.0, dtype=np.float32)
    rgb[20:100, 20:180] = (40, 40, 40)
    # yoğun noktalı bant: 2px'lik nokta / 3px'lik beyaz boşluk dokusu
    for y in range(140, 170, 5):
        for x in range(20, 180, 5):
            rgb[y:y+2, x:x+2] = (230, 60, 40)
    out = pm.poster_alpha(a, counters="open", rgb=rgb, haze_matting=False)
    gaps = np.all(rgb == 250.0, axis=-1)
    gaps[:120] = False
    assert out[gaps].max() < 0.3, "noktalar arası sayfa boşlukları şeffaf kalmalı"


def test_page_colored_hesitant_hole_stays_open_on_white_page():
    """O/D/G lesson: a counter whose interior IS the page stays open even
    when the model hesitated; only non-page interiors may print solid."""
    import numpy as np
    a = np.zeros((200, 200), dtype=np.float32)
    a[40:160, 40:160] = 1.0
    a[96:102, 96:102] = 0.4          # kararsız KÜÇÜK iç boşluk (punto boşluğu)
    rgb = np.full((200, 200, 3), 250.0, dtype=np.float32)
    rgb[40:160, 40:160] = (30, 30, 30)
    rgb[96:102, 96:102] = 250.0      # boşluğun içi sayfa rengi
    out = pm.poster_alpha(a, counters="open", rgb=rgb, haze_matting=False)
    assert out[97:101, 97:101].max() < 0.5, "küçük sayfa renkli boşluk açık kalmalı"
    rgb2 = rgb.copy()
    rgb2[96:102, 96:102] = (140, 140, 140)   # gri duman içi
    out2 = pm.poster_alpha(a, counters="open", rgb=rgb2, haze_matting=False)
    assert out2[97:101, 97:101].min() > 0.9, "sayfa-dışı kararsız boşluk dolmalı"
    a3 = a.copy(); rgb3 = rgb.copy()
    a3[96:102, 96:102] = 1.0
    a3[70:130, 70:130] = np.minimum(a3[70:130, 70:130], 0.4)  # BÜYÜK beyaz cep
    rgb3[70:130, 70:130] = 250.0
    out3 = pm.poster_alpha(a3, counters="open", rgb=rgb3, haze_matting=False)
    assert out3[90:110, 90:110].min() > 0.9, "yaprak ölçeğindeki beyaz cep dolu kalmalı"


def test_half_confident_white_edge_element_prints_solid():
    """v18 eldiven dersi: kenara bağlı, sayfa renkli, model ~0.5 veren bölge
    dolu basılır; küçük veya düşük güvenli olanlar etkilenmez."""
    import numpy as np
    a = np.zeros((200, 200), dtype=np.float32)
    a[40:160, 40:120] = 1.0                    # gövde
    a[80:130, 120:170] = 0.55                  # kenara bağlı beyaz uzuv (yarı emin)
    rgb = np.full((200, 200, 3), 250.0, dtype=np.float32)
    rgb[40:160, 40:120] = (30, 30, 30)
    rgb[80:130, 120:170] = 250.0               # uzuv sayfa renkli
    out = pm.poster_alpha(a, counters="open", rgb=rgb, haze_matting=False)
    assert out[95:115, 135:160].min() > 0.95, "yarı emin beyaz uzuv dolu olmalı"
    a2 = a.copy(); a2[80:130, 120:170] = 0.35  # düşük güven -> eski davranış
    out2 = pm.poster_alpha(a2, counters="open", rgb=rgb, haze_matting=False)
    assert out2[95:115, 135:160].max() < 0.95, "düşük güvenli uzuv dolu basılmamalı"


def test_subject_mask_referee_protects_and_fills():
    """SAM3 hakemi (2026-07-30): maske içi kararsız beyaz dolar, maske dışı
    davranış maskesizle birebir aynı kalır."""
    import numpy as np
    a = np.zeros((200, 200), dtype=np.float32)
    a[40:160, 40:120] = 1.0                    # gövde
    a[80:130, 120:170] = 0.2                   # kenara bağlı beyaz uzuv (eldiven, zayıf)
    rgb = np.full((200, 200, 3), 250.0, dtype=np.float32)
    rgb[40:160, 40:120] = (30, 30, 30)
    sm = np.zeros((200, 200), dtype=bool)
    sm[40:160, 40:170] = True                  # özne: gövde + uzuv
    out = pm.poster_alpha(a, counters="open", rgb=rgb, haze_matting=False,
                          subject_mask=sm)
    assert out[95:115, 135:160].min() > 0.9, "özne içi eldiven dolmalı"
    out_none = pm.poster_alpha(a, counters="open", rgb=rgb, haze_matting=False)
    assert out_none[95:115, 135:160].max() < 0.5, "maskesiz eski davranış"
    # maske dışında birebir aynılık: maskeyi tamamen alakasız köşeye koy
    sm2 = np.zeros((200, 200), dtype=bool); sm2[:10, :10] = True
    out2 = pm.poster_alpha(a, counters="open", rgb=rgb, haze_matting=False,
                           subject_mask=sm2)
    assert np.allclose(out2, out_none), "sessiz maske çıktıyı değiştirmemeli"


def test_decontaminate_recovers_foreground_color():
    """Süt dersi: beyaz sayfayla karışmış yarı saydam gri, gerçek koyu
    mürekkep rengine geri çözülür; opak ve silinmiş pikseller değişmez."""
    import numpy as np
    rgb = np.full((50, 50, 3), 250.0, dtype=np.float32)
    alpha = np.zeros((50, 50), dtype=np.float32)
    rgb[10:20, 10:20] = 30.0;  alpha[10:20, 10:20] = 1.0      # opak mürekkep
    # gerçek fg=50 gri, alfa 0.5, sayfa 250 -> görünen piksel 150
    rgb[30:40, 10:20] = 150.0; alpha[30:40, 10:20] = 0.5
    fg = pm.decontaminate(rgb, alpha)
    assert np.allclose(fg[10:20, 10:20], 30.0), "opak değişmez"
    assert np.allclose(fg[0, 0], 250.0), "silinen değişmez"
    assert abs(float(fg[35, 15, 0]) - 50.0) < 2.0, "yarı saydam fg geri çözülür"


def test_defringe_pulls_ink_color_to_the_edge():
    """Parlayan-P dersi: kenar bandındaki gri (sayfa karışmış) pikseller iç
    mürekkep rengini alır; içi olmayan ince çizgiler dokunulmaz kalır."""
    import numpy as np
    rgb = np.full((60, 60, 3), 250.0, dtype=np.float32)
    alpha = np.zeros((60, 60), dtype=np.float32)
    rgb[20:40, 20:40] = 10.0                    # siyah blok
    alpha[20:40, 20:40] = 1.0
    rgb[19, 20:40] = 140.0; alpha[19, 20:40] = 0.98   # kirli AA kenarı (gri, yüksek alfa)
    out = pm.defringe(rgb, alpha)
    assert float(out[19, 30, 0]) < 30, "kenar pikseli iç siyahı almalı"
    assert float(out[30, 30, 0]) == 10.0, "iç değişmez"
    # içi olmayan ince çizgi: 1px'lik uzak çizgi rengini korur
    rgb2 = np.full((60, 60, 3), 250.0, dtype=np.float32)
    a2 = np.zeros((60, 60), dtype=np.float32)
    rgb2[5, 5:15] = 80.0; a2[5, 5:15] = 0.9
    out2 = pm.defringe(rgb2, a2)
    assert float(out2[5, 10, 0]) == 80.0, "ulaşılamaz iç -> renk korunur"
