# Faz 0: Kurulum + Baseline Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Açık kaynak bg-removal modellerini (BiRefNet_HR, RMBG-2.0) lokal MPS'te çalıştırıp, kategorili bir test seti üzerinde Ideogram referansıyla karşılaştıran benchmark harness'ı ve gap analizi raporunu üretmek.

**Architecture:** `bgr/` paketi Segmenter arayüzünü ve model implementasyonlarını taşır; `benchmark/` paketi metrikleri (SAD/MAE/Grad/Conn), manifest tabanlı test setini, koşucuyu ve HTML galeriyi taşır. Her model çıktısı `results/<model>/<id>.png` olarak diske yazılır; metrikler ve galeri bu dosyalardan üretilir.

**Tech Stack:** Python 3.12 + uv, PyTorch ≥2.4 (MPS), transformers (BiRefNet/RMBG-2.0 `trust_remote_code`), Pillow, numpy, scipy, pytest, fal-client (Ideogram referansı), huggingface_hub.

## Global Constraints

- Proje kökü: `/Users/egeo/Documents/Projects/my-bg-remover` (tüm komutlar buradan koşar).
- Python ≥3.12; venv `uv` ile `.venv/` içinde; komutlar `uv run ...` ile.
- PyTorch ≥2.4 zorunlu (MPS'te eski Adam bug'ı nedeniyle); cihaz seçimi her zaman `mps varsa mps, yoksa cpu`.
- Alpha matte sözleşmesi (her modül): `np.ndarray, dtype=float32, shape=(H,W), değerler [0,1]`, giriş görseliyle aynı çözünürlükte.
- `data/` ve `results/` git'e girmez (`.gitignore`).
- RMBG-2.0 gated: HF'de lisans onayı + `huggingface-cli login` gerekir — bu kullanıcı aksiyonudur, kod `GatedRepoError`'ı anlaşılır mesajla raporlar.
- fal.ai çağrıları `FAL_KEY` env değişkeni ister; Ideogram maliyet koruması: bir koşuda en fazla 250 görsel.
- Commit mesajları Türkçe, conventional commits; her commit sonu: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: Proje iskeleti + ortam

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `bgr/__init__.py`, `benchmark/__init__.py`, `tests/__init__.py`, `tests/test_sanity.py`
- Create (boş dizinler, `.gitkeep` ile): `data/.gitkeep`, `results/.gitkeep`, `training/.gitkeep`, `serving/.gitkeep`, `scripts/.gitkeep`

**Interfaces:**
- Produces: çalışan `uv run pytest` ortamı; sonraki tüm task'lar bu venv'i kullanır.

- [ ] **Step 1: pyproject.toml yaz**

```toml
[project]
name = "my-bg-remover"
version = "0.1.0"
description = "Ideogram seviyesinde background remover"
requires-python = ">=3.12"
dependencies = [
    "torch>=2.4",
    "torchvision",
    "transformers>=4.44",
    "timm",
    "einops",
    "kornia",
    "pillow>=10",
    "numpy>=1.26",
    "scipy>=1.12",
    "huggingface-hub>=0.24",
    "fal-client>=0.4",
    "requests",
]

[dependency-groups]
dev = ["pytest>=8"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.uv]
package = false
```

Not: `timm`, `einops`, `kornia` — BiRefNet'in `trust_remote_code` kodunun runtime bağımlılıklarıdır.

- [ ] **Step 2: .gitignore yaz**

```gitignore
.venv/
__pycache__/
*.pyc
data/*
!data/.gitkeep
results/*
!results/.gitkeep
.env
.DS_Store
```

- [ ] **Step 3: Paket iskeletini ve sanity testini oluştur**

`bgr/__init__.py`, `benchmark/__init__.py`, `tests/__init__.py` boş dosyalar. `tests/test_sanity.py`:

```python
import torch


def test_torch_version_and_device():
    major, minor = (int(x) for x in torch.__version__.split(".")[:2])
    assert (major, minor) >= (2, 4)
    assert torch.backends.mps.is_available()
```

- [ ] **Step 4: venv kur ve testi koş**

Run: `cd /Users/egeo/Documents/Projects/my-bg-remover && uv sync && uv run pytest -v`
Expected: `test_torch_version_and_device PASSED` (1 passed)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .gitignore uv.lock bgr benchmark tests data/.gitkeep results/.gitkeep training/.gitkeep serving/.gitkeep scripts/.gitkeep
git commit -m "chore: proje iskeleti, uv ortamı ve sanity testi"
```

---

### Task 2: Matting metrikleri (SAD, MAE, MSE, Grad, Conn)

**Files:**
- Create: `benchmark/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces (hepsi `pred: np.ndarray float32 (H,W) [0,1]`, `gt: aynı` alır, `float` döner):
  - `sad(pred, gt)` — mutlak fark toplamı / 1000 (literatür geleneği)
  - `mae(pred, gt)` — ortalama mutlak fark
  - `mse(pred, gt)` — ortalama kare fark
  - `grad_error(pred, gt, sigma=1.4)` — gaussian gradyan farkı karelerinin toplamı / 1000
  - `conn_error(pred, gt, step=0.1)` — bağlantılılık hatası / 1000
  - `all_metrics(pred, gt) -> dict[str, float]` — `{"sad":..,"mae":..,"mse":..,"grad":..,"conn":..}`

- [ ] **Step 1: Failing testleri yaz**

`tests/test_metrics.py`:

```python
import numpy as np
import pytest

from benchmark.metrics import all_metrics, conn_error, grad_error, mae, mse, sad


@pytest.fixture
def square_alpha():
    gt = np.zeros((100, 100), dtype=np.float32)
    gt[25:75, 25:75] = 1.0
    return gt


def test_identical_alphas_give_zero(square_alpha):
    for fn in (sad, mae, mse, grad_error, conn_error):
        assert fn(square_alpha, square_alpha) == pytest.approx(0.0, abs=1e-6)


def test_sad_counts_absolute_difference(square_alpha):
    pred = square_alpha.copy()
    pred[0, 0:10] = 0.5  # 10 piksel, 0.5 fark -> SAD = 5/1000
    assert sad(pred, square_alpha) == pytest.approx(0.005)


def test_mae_and_mse(square_alpha):
    pred = np.clip(square_alpha + 0.1, 0, 1).astype(np.float32)
    assert mae(pred, square_alpha) == pytest.approx(0.075, abs=0.01)
    assert mse(pred, square_alpha) < mae(pred, square_alpha)


def test_grad_penalizes_blurry_edges(square_alpha):
    from scipy import ndimage
    blurry = ndimage.gaussian_filter(square_alpha, sigma=3).astype(np.float32)
    shifted = np.roll(square_alpha, 1, axis=0)
    assert grad_error(blurry, square_alpha) > 0
    assert grad_error(shifted, square_alpha) > 0


def test_conn_penalizes_disconnected_blobs(square_alpha):
    disconnected = square_alpha.copy()
    disconnected[5:10, 5:10] = 1.0  # ana kareden kopuk küçük blob
    assert conn_error(disconnected, square_alpha) > 0


def test_all_metrics_keys(square_alpha):
    m = all_metrics(square_alpha, square_alpha)
    assert set(m) == {"sad", "mae", "mse", "grad", "conn"}
```

- [ ] **Step 2: Testin FAIL ettiğini doğrula**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'benchmark.metrics'`

- [ ] **Step 3: benchmark/metrics.py implement et**

Standart matting-eval implementasyonları (GCA-Matting/MatteFormer değerlendirme kodlarıyla aynı tanımlar):

```python
"""Matting kalite metrikleri.

Sözleşme: pred ve gt float32, (H, W), [0, 1]. SAD/Grad/Conn literatür
geleneğiyle 1000'e bölünür (küçük okunur sayılar için).
"""
import numpy as np
from scipy import ndimage


def _check(pred: np.ndarray, gt: np.ndarray) -> None:
    if pred.shape != gt.shape:
        raise ValueError(f"shape uyuşmuyor: {pred.shape} vs {gt.shape}")


def sad(pred: np.ndarray, gt: np.ndarray) -> float:
    _check(pred, gt)
    return float(np.abs(pred - gt).sum()) / 1000.0


def mae(pred: np.ndarray, gt: np.ndarray) -> float:
    _check(pred, gt)
    return float(np.abs(pred - gt).mean())


def mse(pred: np.ndarray, gt: np.ndarray) -> float:
    _check(pred, gt)
    return float(((pred - gt) ** 2).mean())


def _gauss_gradient(img: np.ndarray, sigma: float) -> np.ndarray:
    gx = ndimage.gaussian_filter(img, sigma, order=[0, 1])
    gy = ndimage.gaussian_filter(img, sigma, order=[1, 0])
    return np.sqrt(gx**2 + gy**2)


def grad_error(pred: np.ndarray, gt: np.ndarray, sigma: float = 1.4) -> float:
    _check(pred, gt)
    pred_g = _gauss_gradient(pred.astype(np.float64), sigma)
    gt_g = _gauss_gradient(gt.astype(np.float64), sigma)
    return float(((pred_g - gt_g) ** 2).sum()) / 1000.0


def conn_error(pred: np.ndarray, gt: np.ndarray, step: float = 0.1) -> float:
    _check(pred, gt)
    pred = pred.astype(np.float64)
    gt = gt.astype(np.float64)
    thresh_steps = np.arange(0, 1 + step, step)
    round_down_map = -np.ones_like(gt)
    for i in range(1, len(thresh_steps)):
        gt_thresh = gt >= thresh_steps[i]
        pred_thresh = pred >= thresh_steps[i]
        intersection = (gt_thresh & pred_thresh).astype(np.uint8)
        labels, num = ndimage.label(intersection)
        if num == 0:
            omega = np.zeros_like(gt)
        else:
            sizes = ndimage.sum(intersection, labels, range(1, num + 1))
            omega = (labels == (np.argmax(sizes) + 1)).astype(np.float64)
        flag = (round_down_map == -1) & (omega == 0)
        round_down_map[flag] = thresh_steps[i - 1]
    round_down_map[round_down_map == -1] = 1
    gt_diff = gt - round_down_map
    pred_diff = pred - round_down_map
    phi_gt = 1 - gt_diff * (gt_diff >= 0.15)
    phi_pred = 1 - pred_diff * (pred_diff >= 0.15)
    return float(np.abs(phi_pred - phi_gt).sum()) / 1000.0


def all_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    return {
        "sad": sad(pred, gt),
        "mae": mae(pred, gt),
        "mse": mse(pred, gt),
        "grad": grad_error(pred, gt),
        "conn": conn_error(pred, gt),
    }
```

- [ ] **Step 4: Testlerin PASS ettiğini doğrula**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add benchmark/metrics.py tests/test_metrics.py
git commit -m "feat: matting metrikleri (SAD/MAE/MSE/Grad/Conn)"
```

---

### Task 3: Segmenter arayüzü + BiRefNet_HR (MPS inference)

**Files:**
- Create: `bgr/segmenter.py`, `scripts/infer.py`
- Test: `tests/test_segmenter.py`

**Interfaces:**
- Produces:
  - `bgr.segmenter.Segmenter` (ABC): `name: str` özniteliği, `predict_alpha(image: PIL.Image.Image) -> np.ndarray` metodu (float32, (H,W), [0,1], giriş çözünürlüğünde)
  - `bgr.segmenter.BiRefNetSegmenter(model_id: str, input_size: int, name: str)` — transformers `AutoModelForImageSegmentation` tabanlı genel implementasyon (RMBG-2.0 da aynı sınıfı kullanır, Task 4)
  - `bgr.segmenter.get_device() -> torch.device`
- Consumes: Task 1 ortamı.

- [ ] **Step 1: Failing testleri yaz**

`tests/test_segmenter.py` (gerçek model — yavaş test, `slow` marker):

```python
import numpy as np
import pytest
from PIL import Image, ImageDraw

from bgr.segmenter import BiRefNetSegmenter, get_device


@pytest.fixture(scope="module")
def toy_image():
    img = Image.new("RGB", (640, 480), (30, 120, 30))
    d = ImageDraw.Draw(img)
    d.ellipse([200, 100, 440, 380], fill=(220, 60, 60))
    return img


def test_get_device_is_mps():
    assert get_device().type == "mps"


@pytest.mark.slow
def test_birefnet_hr_alpha_contract(toy_image):
    seg = BiRefNetSegmenter(
        model_id="ZhengPeng7/BiRefNet_HR", input_size=2048, name="birefnet-hr"
    )
    alpha = seg.predict_alpha(toy_image)
    assert alpha.dtype == np.float32
    assert alpha.shape == (480, 640)
    assert 0.0 <= alpha.min() and alpha.max() <= 1.0
    # elipsin merkezi özne, köşe arka plan olmalı
    assert alpha[240, 320] > 0.5
    assert alpha[10, 10] < 0.5
```

`pyproject.toml`'a marker ekle (`[tool.pytest.ini_options]` altına):

```toml
markers = ["slow: gerçek model yükleyen yavaş testler"]
```

- [ ] **Step 2: Testin FAIL ettiğini doğrula**

Run: `uv run pytest tests/test_segmenter.py -v -m slow`
Expected: FAIL, `ModuleNotFoundError: No module named 'bgr.segmenter'`

- [ ] **Step 3: bgr/segmenter.py implement et**

```python
"""Segmenter arayüzü ve BiRefNet ailesi implementasyonu.

Sözleşme: predict_alpha(PIL.Image) -> np.float32 (H, W), [0, 1],
giriş görseliyle aynı çözünürlükte.
"""
from abc import ABC, abstractmethod

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def get_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class Segmenter(ABC):
    name: str

    @abstractmethod
    def predict_alpha(self, image: Image.Image) -> np.ndarray: ...


class BiRefNetSegmenter(Segmenter):
    """BiRefNet mimarisi tabanlı tüm HF modelleri (BiRefNet_HR, RMBG-2.0...)."""

    def __init__(self, model_id: str, input_size: int, name: str):
        from transformers import AutoModelForImageSegmentation

        self.name = name
        self.input_size = input_size
        self.device = get_device()
        self.model = AutoModelForImageSegmentation.from_pretrained(
            model_id, trust_remote_code=True
        )
        self.model.to(self.device).eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    @torch.no_grad()
    def predict_alpha(self, image: Image.Image) -> np.ndarray:
        rgb = image.convert("RGB")
        inp = self.transform(rgb).unsqueeze(0).to(self.device)
        preds = self.model(inp)[-1].sigmoid().cpu()
        alpha = transforms.functional.resize(preds[0], rgb.size[::-1])[0]
        return alpha.clamp(0, 1).numpy().astype(np.float32)
```

- [ ] **Step 4: Testlerin PASS ettiğini doğrula**

Run: `uv run pytest tests/test_segmenter.py -v -m "slow or not slow"`
Expected: 2 passed (BiRefNet_HR HF cache'te hazır; ilk yüklemede `trust_remote_code` onayı için `HF_HUB_DISABLE_PROGRESS_BARS=1` gerekmez, ama sandbox dışı ağ erişimi gerekebilir)

- [ ] **Step 5: scripts/infer.py smoke CLI yaz**

```python
"""Tek görselde inference: uv run python scripts/infer.py girdi.jpg cikti.png"""
import sys

import numpy as np
from PIL import Image

from bgr.segmenter import BiRefNetSegmenter


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    seg = BiRefNetSegmenter(
        model_id="ZhengPeng7/BiRefNet_HR", input_size=2048, name="birefnet-hr"
    )
    img = Image.open(src)
    alpha = seg.predict_alpha(img)
    rgba = img.convert("RGB").copy()
    rgba.putalpha(Image.fromarray((alpha * 255).astype(np.uint8)))
    rgba.save(dst)
    print(f"kaydedildi: {dst}")


if __name__ == "__main__":
    main()
```

Run (herhangi bir gerçek fotoğrafla): `uv run python scripts/infer.py <foto.jpg> /tmp/out.png`
Expected: transparan PNG oluşur; görsel olarak makul kesim.

- [ ] **Step 6: Commit**

```bash
git add bgr/segmenter.py scripts/infer.py tests/test_segmenter.py pyproject.toml
git commit -m "feat: Segmenter arayüzü ve BiRefNet_HR MPS inference"
```

---

### Task 4: RMBG-2.0 erişimi + model registry

**Files:**
- Create: `bgr/registry.py`
- Test: `tests/test_registry.py`

**Interfaces:**
- Produces:
  - `bgr.registry.MODEL_SPECS: dict[str, dict]` — anahtarlar: `"birefnet-hr"`, `"rmbg-2.0"`; değerler `{"model_id": str, "input_size": int}`
  - `bgr.registry.get_segmenter(name: str) -> Segmenter` — bilinmeyen ad için `KeyError`; gated repo hatasında kullanıcıya lisans onayı + `huggingface-cli login` talimatı içeren `RuntimeError`
- Consumes: `BiRefNetSegmenter` (Task 3).

- [ ] **Step 1: Kullanıcı aksiyonu — RMBG-2.0 lisansını onaylat**

Kullanıcıya söyle (bloklayıcı):
1. Tarayıcıda https://huggingface.co/briaai/RMBG-2.0 aç → "Agree and access repository".
2. Terminale `! huggingface-cli login` yazıp HF token'ı ile giriş yap (token: https://huggingface.co/settings/tokens).

Doğrulama: `uv run python -c "from huggingface_hub import auth_check; auth_check('briaai/RMBG-2.0'); print('erişim OK')"`
Expected: `erişim OK`

- [ ] **Step 2: Failing testleri yaz**

`tests/test_registry.py`:

```python
import numpy as np
import pytest
from PIL import Image, ImageDraw

from bgr.registry import MODEL_SPECS, get_segmenter


def test_known_model_names():
    assert set(MODEL_SPECS) == {"birefnet-hr", "rmbg-2.0"}


def test_unknown_name_raises():
    with pytest.raises(KeyError):
        get_segmenter("yok-boyle-model")


@pytest.mark.slow
def test_rmbg2_alpha_contract():
    img = Image.new("RGB", (320, 240), (200, 200, 200))
    ImageDraw.Draw(img).rectangle([100, 60, 220, 180], fill=(20, 20, 160))
    seg = get_segmenter("rmbg-2.0")
    alpha = seg.predict_alpha(img)
    assert alpha.dtype == np.float32
    assert alpha.shape == (240, 320)
    assert float(alpha.max()) <= 1.0 and float(alpha.min()) >= 0.0
```

- [ ] **Step 3: Testin FAIL ettiğini doğrula**

Run: `uv run pytest tests/test_registry.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'bgr.registry'`

- [ ] **Step 4: bgr/registry.py implement et**

```python
"""İsimle segmenter üretimi. Yeni model eklemek = MODEL_SPECS'e satır eklemek."""
from bgr.segmenter import BiRefNetSegmenter, Segmenter

MODEL_SPECS: dict[str, dict] = {
    "birefnet-hr": {"model_id": "ZhengPeng7/BiRefNet_HR", "input_size": 2048},
    "rmbg-2.0": {"model_id": "briaai/RMBG-2.0", "input_size": 1024},
}

_GATED_HELP = (
    "{model_id} gated bir model. Şunları yap:\n"
    "1) https://huggingface.co/{model_id} adresinde lisansı onayla\n"
    "2) `huggingface-cli login` ile giriş yap"
)


def get_segmenter(name: str) -> Segmenter:
    spec = MODEL_SPECS[name]  # bilinmeyen ad -> KeyError
    try:
        return BiRefNetSegmenter(
            model_id=spec["model_id"], input_size=spec["input_size"], name=name
        )
    except Exception as e:  # GatedRepoError / 401
        if "gated" in str(e).lower() or "401" in str(e):
            raise RuntimeError(_GATED_HELP.format(model_id=spec["model_id"])) from e
        raise
```

- [ ] **Step 5: Testlerin PASS ettiğini doğrula**

Run: `uv run pytest tests/test_registry.py -v -m "slow or not slow"`
Expected: 3 passed (rmbg-2.0 ağırlıkları ilk koşuda iner, ~1GB)

- [ ] **Step 6: Commit**

```bash
git add bgr/registry.py tests/test_registry.py
git commit -m "feat: model registry ve RMBG-2.0 erişimi"
```

---

### Task 5: Test seti — manifest + GT'li setlerin indirilmesi

**Files:**
- Create: `benchmark/testset.py`, `scripts/build_testset.py`
- Create (indirme sonucu, git dışı): `data/testset/images/*`, `data/testset/gt/*`, `data/testset/manifest.jsonl`
- Test: `tests/test_testset.py`

**Interfaces:**
- Produces:
  - Manifest formatı (JSONL, her satır): `{"id": str, "image": str, "category": str, "gt_alpha": str | null}` — path'ler proje köküne göre göreli; `category ∈ {hair, transparent, thin, product, complex, illustration, general}`
  - `benchmark.testset.load_manifest(path: str) -> list[dict]` — şema doğrulaması yapar (eksik anahtar/bilinmeyen kategori → `ValueError`)
  - `benchmark.testset.append_entries(path: str, entries: list[dict]) -> None`
- Consumes: yok (bağımsız).

- [ ] **Step 1: Failing testleri yaz**

`tests/test_testset.py`:

```python
import json

import pytest

from benchmark.testset import append_entries, load_manifest


def test_roundtrip(tmp_path):
    p = tmp_path / "m.jsonl"
    rows = [
        {"id": "a1", "image": "data/x/a1.jpg", "category": "hair", "gt_alpha": "data/x/a1.png"},
        {"id": "b2", "image": "data/x/b2.jpg", "category": "product", "gt_alpha": None},
    ]
    append_entries(str(p), rows)
    assert load_manifest(str(p)) == rows


def test_invalid_category_raises(tmp_path):
    p = tmp_path / "m.jsonl"
    p.write_text(json.dumps({"id": "x", "image": "i.jpg", "category": "ucan-kus", "gt_alpha": None}) + "\n")
    with pytest.raises(ValueError, match="kategori"):
        load_manifest(str(p))


def test_missing_key_raises(tmp_path):
    p = tmp_path / "m.jsonl"
    p.write_text(json.dumps({"id": "x", "image": "i.jpg"}) + "\n")
    with pytest.raises(ValueError, match="anahtar"):
        load_manifest(str(p))
```

- [ ] **Step 2: FAIL doğrula**

Run: `uv run pytest tests/test_testset.py -v`
Expected: FAIL, `ModuleNotFoundError`

- [ ] **Step 3: benchmark/testset.py implement et**

```python
"""Manifest tabanlı test seti. JSONL: id, image, category, gt_alpha (nullable)."""
import json

CATEGORIES = {"hair", "transparent", "thin", "product", "complex", "illustration", "general"}
_KEYS = {"id", "image", "category", "gt_alpha"}


def _validate(row: dict) -> None:
    missing = _KEYS - set(row)
    if missing:
        raise ValueError(f"eksik anahtar(lar): {sorted(missing)}")
    if row["category"] not in CATEGORIES:
        raise ValueError(f"bilinmeyen kategori: {row['category']}")


def load_manifest(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                _validate(row)
                rows.append(row)
    return rows


def append_entries(path: str, entries: list[dict]) -> None:
    for row in entries:
        _validate(row)
    with open(path, "a") as f:
        for row in entries:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
```

- [ ] **Step 4: PASS doğrula**

Run: `uv run pytest tests/test_testset.py -v`
Expected: 3 passed

- [ ] **Step 5: GT'li kaynak setleri araştır ve indir**

Hedef dağılım (~180 görsel): hair ~40, transparent ~25, thin ~20, product ~30, complex ~30, illustration ~20, general ~15.

Kaynak setler ve elde etme yolu (indirme linkleri zamanla değiştiği için önce HF Hub'da ara, bulamazsan resmi repo'nun README'sindeki Google Drive linkini kullan):

1. HF Hub'da mirror ara (uygulayıcı WebSearch/`huggingface_hub` API kullanır):

```python
from huggingface_hub import HfApi
api = HfApi()
for q in ["AIM-500", "P3M-10k", "DIS5K", "Transparent-460", "AM-2k"]:
    hits = api.list_datasets(search=q, limit=5)
    print(q, "->", [d.id for d in hits])
```

2. Bulunan mirror'ları `huggingface_hub.snapshot_download(repo_id, repo_type="dataset", local_dir="data/raw/<set>")` ile indir; mirror yoksa resmi kaynak: AIM-500/AM-2k → https://github.com/ViTAE-Transformer/ViTAE-Transformer-Matting, P3M → https://github.com/JizhiziLi/P3M, DIS5K → https://github.com/xuebinqin/DIS (Drive linkleri README'lerde; `gdown` ile indir, `uv add gdown --group dev` gerekirse).
3. Her setten deterministik örnekle (seed=42), görsel + GT alpha'yı `data/testset/images|gt/` altına `{set}_{orig_ad}` id'siyle kopyala ve manifest'e yaz. Kategori eşlemesi: P3M → hair; Transparent-460 → transparent; AIM-500 → tip etiketine göre hair/thin/general; DIS-VD → thin/complex; AM-2k → complex; ürün ve illüstrasyon görselleri GT'siz kategori olarak sonraki adımda gelir.

Bu mantığı `scripts/build_testset.py` içine yaz (indirilen dizin yapısına göre glob'lar uyarlanır — dizin yapısı indirme sonrası `ls` ile doğrulanıp scripte gömülür):

```python
"""GT'li kaynak setlerden kategorili test seti örnekle: uv run python scripts/build_testset.py"""
import random
import shutil
from pathlib import Path

from benchmark.testset import append_entries

random.seed(42)
ROOT = Path(__file__).resolve().parent.parent
OUT_IMG = ROOT / "data/testset/images"
OUT_GT = ROOT / "data/testset/gt"
MANIFEST = ROOT / "data/testset/manifest.jsonl"

# (kaynak_ad, images_glob, gt_glob, kategori, adet) — glob'lar indirme sonrası doğrulanır
SOURCES: list[tuple[str, str, str, str, int]] = [
    ("p3m", "data/raw/p3m/**/original_image/*.jpg", "data/raw/p3m/**/mask/*.png", "hair", 40),
    ("trans460", "data/raw/trans460/**/image/*", "data/raw/trans460/**/alpha/*", "transparent", 25),
    ("disvd", "data/raw/dis5k/DIS-VD/im/*", "data/raw/dis5k/DIS-VD/gt/*", "thin", 20),
    ("aim500", "data/raw/aim500/**/original/*", "data/raw/aim500/**/mask/*", "general", 15),
    ("am2k", "data/raw/am2k/**/original/*", "data/raw/am2k/**/mask/*", "complex", 30),
]


def sample_source(name: str, img_glob: str, gt_glob: str, category: str, n: int) -> list[dict]:
    imgs = sorted(ROOT.glob(img_glob))
    gts = {p.stem: p for p in ROOT.glob(gt_glob)}
    paired = [(i, gts[i.stem]) for i in imgs if i.stem in gts]
    rows = []
    for img, gt in random.sample(paired, min(n, len(paired))):
        rid = f"{name}_{img.stem}"
        dst_i = OUT_IMG / f"{rid}{img.suffix}"
        dst_g = OUT_GT / f"{rid}.png"
        shutil.copy(img, dst_i)
        shutil.copy(gt, dst_g)
        rows.append({"id": rid, "image": str(dst_i.relative_to(ROOT)),
                     "category": category, "gt_alpha": str(dst_g.relative_to(ROOT))})
    return rows


def main() -> None:
    OUT_IMG.mkdir(parents=True, exist_ok=True)
    OUT_GT.mkdir(parents=True, exist_ok=True)
    for src in SOURCES:
        rows = sample_source(*src)
        append_entries(str(MANIFEST), rows)
        print(f"{src[0]}: {len(rows)} örnek")


if __name__ == "__main__":
    main()
```

Run: `uv run python scripts/build_testset.py`
Expected: her kaynaktan beklenen adet; `wc -l data/testset/manifest.jsonl` ≈ 130

- [ ] **Step 6: GT'siz kategorileri ekle (product, illustration, complex gerçek dünya)**

Ürün (~30), illüstrasyon (~20) ve gerçek dünya (~15) görseli GT'siz eklenir (`gt_alpha: null`) — bunlar Ideogram'la subjektif galeri karşılaştırması için. Kaynak: kullanıcının kendi fotoğrafları `data/testset/incoming/` klasörüne atılır + telifsiz kaynaklardan (Unsplash/Pexels) indirilen ürün/illüstrasyon görselleri. Ekleme komutu `scripts/build_testset.py`'ye alt komut olarak eklenir:

```python
def add_unlabeled(folder: str, category: str) -> None:
    rows = []
    for img in sorted((ROOT / folder).glob("*")):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        rid = f"user_{category}_{img.stem}"
        dst = OUT_IMG / f"{rid}{img.suffix}"
        shutil.copy(img, dst)
        rows.append({"id": rid, "image": str(dst.relative_to(ROOT)),
                     "category": category, "gt_alpha": None})
    append_entries(str(MANIFEST), rows)
    print(f"{category}: {len(rows)} GT'siz görsel eklendi")
```

`main()`'e CLI argümanı ekle: `uv run python scripts/build_testset.py add data/testset/incoming product`

Run: (görseller toplandıktan sonra) `uv run python scripts/build_testset.py add data/testset/incoming product`
Expected: manifest toplamı ~180'e ulaşır; `load_manifest` hatasız okur.

- [ ] **Step 7: Commit**

```bash
git add benchmark/testset.py scripts/build_testset.py tests/test_testset.py
git commit -m "feat: manifest tabanlı test seti ve kaynak set örnekleme"
```

---

### Task 6: Ideogram referans çıktıları (fal.ai)

**Files:**
- Create: `benchmark/ideogram.py`, `scripts/fetch_ideogram.py`
- Test: `tests/test_ideogram.py` (API çağrısı mock'lu)

**Interfaces:**
- Produces:
  - `benchmark.ideogram.fetch_reference(image_path: str, out_path: str) -> None` — fal.ai `fal-ai/ideogram/remove-background` çağırır, çıktı RGBA PNG'yi `out_path`'e yazar; `out_path` varsa atlar (idempotent, tekrar ücret ödenmez)
  - Çıktı düzeni: `results/ideogram/<id>.png`
- Consumes: `load_manifest` (Task 5).

- [ ] **Step 1: Failing testi yaz (mock'lu — gerçek API'ye para ödemeden)**

`tests/test_ideogram.py`:

```python
from unittest.mock import patch

from PIL import Image

from benchmark.ideogram import fetch_reference


def test_skips_existing_output(tmp_path):
    out = tmp_path / "x.png"
    Image.new("RGBA", (4, 4)).save(out)
    with patch("benchmark.ideogram.fal_client") as m:
        fetch_reference("gercek-degil.jpg", str(out))
        m.subscribe.assert_not_called()


def test_calls_fal_and_saves(tmp_path):
    src = tmp_path / "in.jpg"
    Image.new("RGB", (4, 4), (255, 0, 0)).save(src)
    out = tmp_path / "out.png"
    fake_png = tmp_path / "fake_result.png"
    Image.new("RGBA", (4, 4), (0, 255, 0, 128)).save(fake_png)
    with (
        patch("benchmark.ideogram.fal_client") as m,
        patch("benchmark.ideogram._download") as dl,
    ):
        m.upload_file.return_value = "https://fal.example/in.jpg"
        m.subscribe.return_value = {"image": {"url": "https://fal.example/out.png"}}
        dl.side_effect = lambda url, path: fake_png.rename(path)
        fetch_reference(str(src), str(out))
    assert out.exists()
```

- [ ] **Step 2: FAIL doğrula**

Run: `uv run pytest tests/test_ideogram.py -v`
Expected: FAIL, `ModuleNotFoundError`

- [ ] **Step 3: benchmark/ideogram.py implement et**

```python
"""fal.ai Ideogram remove-background referans çıktıları ($0.01/görsel).

FAL_KEY env değişkeni gerekir. Idempotent: çıktı varsa API çağrılmaz.
"""
import os
from pathlib import Path

import fal_client
import requests

ENDPOINT = "fal-ai/ideogram/remove-background"


def _download(url: str, path: str) -> None:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    Path(path).write_bytes(r.content)


def fetch_reference(image_path: str, out_path: str) -> None:
    if Path(out_path).exists():
        return
    if not os.environ.get("FAL_KEY"):
        raise RuntimeError("FAL_KEY tanımlı değil: https://fal.ai/dashboard/keys")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    url = fal_client.upload_file(image_path)
    result = fal_client.subscribe(ENDPOINT, arguments={"image_url": url})
    _download(result["image"]["url"], out_path)
```

- [ ] **Step 4: PASS doğrula**

Run: `uv run pytest tests/test_ideogram.py -v`
Expected: 2 passed

- [ ] **Step 5: Toplu çekme scripti yaz ve KÜÇÜK denemeyle doğrula**

`scripts/fetch_ideogram.py`:

```python
"""Manifest'teki tüm görseller için Ideogram referansı çek.
Kullanım: uv run python scripts/fetch_ideogram.py [--limit N]
"""
import argparse
from pathlib import Path

from benchmark.ideogram import fetch_reference
from benchmark.testset import load_manifest

ROOT = Path(__file__).resolve().parent.parent
MAX_PER_RUN = 250  # maliyet koruması (~$2.50 tavan)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=MAX_PER_RUN)
    n = min(ap.parse_args().limit, MAX_PER_RUN)
    rows = load_manifest(str(ROOT / "data/testset/manifest.jsonl"))[:n]
    for i, row in enumerate(rows, 1):
        out = ROOT / "results/ideogram" / f"{row['id']}.png"
        fetch_reference(str(ROOT / row["image"]), str(out))
        print(f"[{i}/{len(rows)}] {row['id']}")


if __name__ == "__main__":
    main()
```

Run (önce 3 görsellik deneme — kullanıcıdan FAL_KEY alınır, ~$0.03): `FAL_KEY=... uv run python scripts/fetch_ideogram.py --limit 3`
Expected: `results/ideogram/` altında 3 RGBA PNG. Sonra kullanıcı onayıyla tam koşu (~$1.80).

- [ ] **Step 6: Commit**

```bash
git add benchmark/ideogram.py scripts/fetch_ideogram.py tests/test_ideogram.py
git commit -m "feat: Ideogram referans çıktıları (fal.ai, idempotent)"
```

---

### Task 7: Benchmark koşucusu

**Files:**
- Create: `benchmark/run.py`
- Test: `tests/test_run.py`

**Interfaces:**
- Produces:
  - `benchmark.run.run_benchmark(models: list[str], manifest_path: str, out_dir: str) -> dict` — her model için her manifest görselinde alpha üretir → `<out_dir>/<model>/<id>.png` (gri tonlama, 8-bit); GT'li satırlarda metrik hesaplar; döndürdüğü ve `<out_dir>/metrics.json`'a yazdığı yapı:
    `{"per_image": {model: {id: {metrik: değer}}}, "per_category": {model: {kategori: {metrik: ortalama}}}, "overall": {model: {metrik: ortalama}}}`
  - CLI: `uv run python -m benchmark.run --models birefnet-hr,rmbg-2.0 --manifest data/testset/manifest.jsonl --out results/baseline`
- Consumes: `get_segmenter` (Task 4), `load_manifest` (Task 5), `all_metrics` (Task 2).

- [ ] **Step 1: Failing testi yaz (sahte segmenter'la — model yüklemeden)**

`tests/test_run.py`:

```python
import json
from unittest.mock import patch

import numpy as np
from PIL import Image

from benchmark.run import run_benchmark


class FakeSeg:
    name = "fake"

    def predict_alpha(self, image):
        w, h = image.size
        return np.ones((h, w), dtype=np.float32)


def _make_testset(tmp_path):
    img = tmp_path / "a.jpg"
    Image.new("RGB", (8, 8), (10, 10, 10)).save(img)
    gt = tmp_path / "a.png"
    Image.fromarray(np.full((8, 8), 255, np.uint8)).save(gt)
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(json.dumps({
        "id": "a", "image": str(img), "category": "general", "gt_alpha": str(gt),
    }) + "\n")
    return manifest


def test_run_benchmark_outputs_and_metrics(tmp_path):
    manifest = _make_testset(tmp_path)
    with patch("benchmark.run.get_segmenter", return_value=FakeSeg()):
        result = run_benchmark(["fake"], str(manifest), str(tmp_path / "out"))
    assert (tmp_path / "out/fake/a.png").exists()
    assert result["per_image"]["fake"]["a"]["sad"] == 0.0  # tam isabet
    assert result["overall"]["fake"]["mae"] == 0.0
    assert (tmp_path / "out/metrics.json").exists()
```

- [ ] **Step 2: FAIL doğrula**

Run: `uv run pytest tests/test_run.py -v`
Expected: FAIL, `ModuleNotFoundError`

- [ ] **Step 3: benchmark/run.py implement et**

```python
"""Benchmark koşucusu: modeller x manifest -> alpha PNG'ler + metrics.json."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.metrics import all_metrics
from benchmark.testset import load_manifest
from bgr.registry import get_segmenter


def _load_alpha(path: str) -> np.ndarray:
    img = Image.open(path)
    if img.mode == "RGBA":
        return np.asarray(img.split()[-1], dtype=np.float32) / 255.0
    return np.asarray(img.convert("L"), dtype=np.float32) / 255.0


def run_benchmark(models: list[str], manifest_path: str, out_dir: str) -> dict:
    rows = load_manifest(manifest_path)
    out = Path(out_dir)
    per_image: dict = {}
    for name in models:
        seg = get_segmenter(name)
        model_dir = out / name
        model_dir.mkdir(parents=True, exist_ok=True)
        per_image[name] = {}
        for row in rows:
            dst = model_dir / f"{row['id']}.png"
            if not dst.exists():
                alpha = seg.predict_alpha(Image.open(row["image"]))
                Image.fromarray((alpha * 255).astype(np.uint8)).save(dst)
            if row["gt_alpha"]:
                pred = _load_alpha(str(dst))
                gt = _load_alpha(row["gt_alpha"])
                per_image[name][row["id"]] = all_metrics(pred, gt)

    categories = {r["id"]: r["category"] for r in rows}
    per_category: dict = {}
    overall: dict = {}
    for name, images in per_image.items():
        cat_acc: dict = defaultdict(lambda: defaultdict(list))
        for rid, m in images.items():
            for k, v in m.items():
                cat_acc[categories[rid]][k].append(v)
        per_category[name] = {
            c: {k: float(np.mean(v)) for k, v in ms.items()} for c, ms in cat_acc.items()
        }
        keys = {k for m in images.values() for k in m}
        overall[name] = {
            k: float(np.mean([m[k] for m in images.values()])) for k in keys
        }

    result = {"per_image": per_image, "per_category": per_category, "overall": overall}
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    result = run_benchmark(a.models.split(","), a.manifest, a.out)
    print(json.dumps(result["overall"], indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: PASS doğrula**

Run: `uv run pytest tests/test_run.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add benchmark/run.py tests/test_run.py
git commit -m "feat: benchmark koşucusu (alpha çıktıları + kategori metrikleri)"
```

---

### Task 8: HTML karşılaştırma galerisi

**Files:**
- Create: `benchmark/gallery.py`
- Test: `tests/test_gallery.py`

**Interfaces:**
- Produces:
  - `benchmark.gallery.build_gallery(manifest_path: str, results_dir: str, models: list[str], out_html: str) -> None` — satır başına: orijinal görsel + her modelin RGBA kompoziti (dama tahtası arka planda) + varsa `ideogram/` çıktısı; kategoriye göre gruplu, `<img>` path'leri göreli
  - CLI: `uv run python -m benchmark.gallery --manifest ... --results results/baseline --models birefnet-hr,rmbg-2.0 --out results/baseline/gallery.html`
- Consumes: `load_manifest` (Task 5); `results/<model>/<id>.png` düzeni (Task 7); `results/ideogram/<id>.png` (Task 6).

- [ ] **Step 1: Failing testi yaz**

`tests/test_gallery.py`:

```python
import json

import numpy as np
from PIL import Image

from benchmark.gallery import build_gallery


def test_gallery_contains_rows_and_images(tmp_path):
    img = tmp_path / "a.jpg"
    Image.new("RGB", (8, 8)).save(img)
    (tmp_path / "results/m1").mkdir(parents=True)
    Image.fromarray(np.full((8, 8), 200, np.uint8)).save(tmp_path / "results/m1/a.png")
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(json.dumps({
        "id": "a", "image": str(img), "category": "hair", "gt_alpha": None,
    }) + "\n")
    out = tmp_path / "results/gallery.html"
    build_gallery(str(manifest), str(tmp_path / "results"), ["m1"], str(out))
    html = out.read_text()
    assert "hair" in html and 'id="a"' in html and "m1/a" in html
```

- [ ] **Step 2: FAIL doğrula**

Run: `uv run pytest tests/test_gallery.py -v`
Expected: FAIL, `ModuleNotFoundError`

- [ ] **Step 3: benchmark/gallery.py implement et**

```python
"""Yan yana HTML galeri: orijinal | model alpha kompozitleri | Ideogram."""
import argparse
import html
import os
from collections import defaultdict
from pathlib import Path

from benchmark.testset import load_manifest

_CSS = """
body{font-family:sans-serif;background:#111;color:#eee;margin:16px}
.row{display:flex;gap:8px;margin:8px 0;align-items:flex-start}
.cell{text-align:center;font-size:11px}
img{max-width:260px;max-height:260px;
 background:repeating-conic-gradient(#666 0% 25%,#999 0% 50%) 0/16px 16px}
h2{border-bottom:1px solid #444;padding-bottom:4px}
"""


def _img_cell(out_dir: Path, src: Path, label: str) -> str:
    rel = os.path.relpath(src, out_dir)
    return f'<div class="cell"><img src="{html.escape(rel)}"><br>{html.escape(label)}</div>'


def build_gallery(manifest_path: str, results_dir: str, models: list[str], out_html: str) -> None:
    rows = load_manifest(manifest_path)
    results = Path(results_dir)
    out = Path(out_html)
    out_dir = out.parent
    by_cat = defaultdict(list)
    for row in rows:
        by_cat[row["category"]].append(row)

    parts = [f"<style>{_CSS}</style><h1>bg-remover benchmark</h1>"]
    for cat in sorted(by_cat):
        parts.append(f"<h2>{html.escape(cat)}</h2>")
        for row in by_cat[cat]:
            cells = [_img_cell(out_dir, Path(row["image"]).resolve(), "orijinal")]
            if row["gt_alpha"]:
                cells.append(_img_cell(out_dir, Path(row["gt_alpha"]).resolve(), "GT"))
            for m in models:
                p = results / m / f"{row['id']}.png"
                if p.exists():
                    cells.append(_img_cell(out_dir, p.resolve(), m))
            ideo = results.parent / "ideogram" / f"{row['id']}.png"
            if ideo.exists():
                cells.append(_img_cell(out_dir, ideo.resolve(), "ideogram"))
            parts.append(f'<div class="row" id="{html.escape(row["id"])}">{"".join(cells)}</div>')
    out.write_text("\n".join(parts))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    build_gallery(a.manifest, a.results, a.models.split(","), a.out)
    print(f"galeri: {a.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: PASS doğrula**

Run: `uv run pytest tests/test_gallery.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add benchmark/gallery.py tests/test_gallery.py
git commit -m "feat: yan yana HTML karşılaştırma galerisi"
```

---

### Task 9: Baseline koşusu + gap analizi raporu

**Files:**
- Create: `docs/reports/2026-07-baseline-gap-analysis.md`
- Output (git dışı): `results/baseline/*`

**Interfaces:**
- Consumes: Task 5 manifest'i, Task 6 Ideogram çıktıları, Task 7 koşucu, Task 8 galeri.
- Produces: kategorize gap analizi raporu — Faz 2'nin (veri) girdi belgesi.

- [ ] **Step 1: Tam test suite'ini koş**

Run: `uv run pytest -v -m "not slow" && uv run pytest -v -m slow`
Expected: tamamı PASS

- [ ] **Step 2: Baseline benchmark'ı koş**

Run: `uv run python -m benchmark.run --models birefnet-hr,rmbg-2.0 --manifest data/testset/manifest.jsonl --out results/baseline`
Expected: `results/baseline/{birefnet-hr,rmbg-2.0}/` dolu; `metrics.json` overall + per_category değerleri yazdırır. (~180 görsel x 2 model, MPS'te tahmini 15-45 dk.)

- [ ] **Step 3: Galeriyi üret ve aç**

Run: `uv run python -m benchmark.gallery --manifest data/testset/manifest.jsonl --results results/baseline --models birefnet-hr,rmbg-2.0 --out results/baseline/gallery.html && open results/baseline/gallery.html`
Expected: tarayıcıda kategori bazlı yan yana karşılaştırma (orijinal | GT | modeller | Ideogram).

- [ ] **Step 4: Gap analizi raporunu yaz**

`docs/reports/2026-07-baseline-gap-analysis.md` şablonu (galeri incelemesi ve metrics.json ile doldurulur — kullanıcıyla BİRLİKTE galeri gezilerek):

```markdown
# Baseline Gap Analizi (Ideogram vs açık modeller)

## Sayısal sonuçlar (GT'li kategoriler)
| Kategori | Model | SAD | MAE | Grad | Conn |
|---|---|---|---|---|---|
(metrics.json'dan)

## Kategori bazlı gözlemler (galeri incelemesi)
### hair — ...
### transparent — ...
### thin — ...
### product — ...
### complex — ...
### illustration — ...

## Sonuç: Faz 2 veri öncelikleri
1. (en kötü kategori -> hangi dataset/sentetik veri)
2. ...

## Sonuç: Faz 1 post-processing beklentileri
- (kenar halosu görülen örnekler -> decontamination kazanç tahmini)
```

- [ ] **Step 5: Commit**

```bash
git add docs/reports/2026-07-baseline-gap-analysis.md
git commit -m "docs: baseline gap analizi raporu"
```

---

## Self-Review Notları

- **Spec kapsaması:** Faz 0'ın 6 maddesi ↔ Task 1 (kurulum), Task 3-4 (modeller+MPS), Task 5 (test seti), Task 6 (Ideogram), Task 2+7+8 (harness), Task 9 (gap raporu). BEN2 ve BiRefNet_dynamic spec'te "referans" olarak geçiyor — bilinçli olarak V1 registry dışında bırakıldı (YAGNI; registry'ye satır eklemek tek satır, gap analizi gerektirirse eklenir).
- **Bilinen belirsizlik:** Task 5'teki dataset glob'ları indirme sonrası dizin yapısına göre uyarlanacak — bu, plandaki tek "koşarken doğrula" noktası; task içinde açıkça işaretlendi.
- **Tip tutarlılığı:** alpha sözleşmesi (float32, (H,W), [0,1]) tüm task'larda aynı; `get_segmenter`/`load_manifest`/`all_metrics` imzaları tüketildikleri yerlerle eşleşiyor.
