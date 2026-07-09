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
