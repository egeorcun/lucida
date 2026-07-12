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
            model_id, trust_remote_code=True, dtype=torch.float32
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


class LocalBiRefNetSegmenter(BiRefNetSegmenter):
    """Kendi fine-tune checkpoint'imizle yüklenen BiRefNet.

    Mimari `arch_id`'den (HF, `trust_remote_code=True`) kurulur — bu yalnızca
    doğru sınıfı/kodu getirmek için bir başlangıç noktası; ağırlıklar hemen
    ardından `ckpt_path`'teki kendi checkpoint'imizle TAMAMEN override edilir.

    Checkpoint formatı: `training/train_colab.ipynb`'nin
    `save_and_sync_checkpoint`'i — `torch.save({"model": state_dict,
    "optimizer": ..., "lr_scheduler": ..., "epoch": int}, path)`. `state_dict`
    `torch.compile` altında eğitildiyse `_orig_mod.` önekli olabilir (resmi
    BiRefNet `train.py` ile aynı davranış: önek KALDIRILMADAN kaydedilir,
    yükleme sırasında temizlenir — bkz. `utils.check_state_dict`).
    """

    def __init__(
        self,
        ckpt_path: str,
        input_size: int,
        name: str,
        arch_id: str = "ZhengPeng7/BiRefNet_HR",
    ):
        super().__init__(model_id=arch_id, input_size=input_size, name=name)
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "model" not in payload:
            raise KeyError(
                f"checkpoint'te 'model' anahtarı yok ({ckpt_path}); "
                f"bulunan anahtarlar: {sorted(payload.keys())}"
            )
        state_dict = payload["model"]
        if any(k.startswith("_orig_mod.") for k in state_dict):
            state_dict = {
                k.removeprefix("_orig_mod."): v for k, v in state_dict.items()
            }
        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            diff = self.model.load_state_dict(state_dict, strict=False)
            raise RuntimeError(
                "strict load_state_dict başarısız: checkpoint mimariyle TAM "
                "eşleşmiyor (sessiz kısmi yükleme YAPILMADI).\n"
                f"  eksik anahtarlar ({len(diff.missing_keys)}): {diff.missing_keys}\n"
                f"  fazla anahtarlar ({len(diff.unexpected_keys)}): {diff.unexpected_keys}\n"
                f"  checkpoint: {ckpt_path}, arch: {arch_id}"
            ) from e
        self.model.to(self.device).eval()


class InSPyReNetSegmenter(Segmenter):
    """InSPyReNet (ACCV 2022) — `transparent-background` paketi üzerinden.

    Paket kendi ön/son işlemesini yapar; `process(..., type="map")` girdiyle
    aynı boyutta gri tonlamalı alpha haritası döner. Alpha sözleşmesi diğer
    segmenter'larla aynı: float32, (H, W), [0, 1], giriş çözünürlüğünde.
    Cihaz: paket MPS'i resmi desteklemediği için CPU'ya sabitlenir (yavaş ama
    deterministik; benchmark tek seferlik koşulduğundan kabul edilebilir)."""

    def __init__(self, name: str = "inspyrenet"):
        from transparent_background import Remover

        self.name = name
        self.remover = Remover(mode="base", device="cpu")

    def predict_alpha(self, image: Image.Image) -> np.ndarray:
        out = self.remover.process(image.convert("RGB"), type="map")
        alpha = np.asarray(out.convert("L"), dtype=np.float32) / 255.0
        return alpha
