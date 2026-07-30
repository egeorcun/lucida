"""SAM3 semantic referee for the poster policy (spec 2026-07-30 study).

WHY: 13-duel Ideogram study, official verdict 12-1 — their entire edge is
SEMANTIC knowledge of what whites belong to the subject (fur, gloves, badge
interiors). SAM3's promptable concept segmentation rents that knowledge:
a small prompt battery yields subject-instance masks, and `poster_alpha`
consumes their union as `subject_mask` — protective evidence only, so a
missed concept can never regress the output (see the poster_alpha
docstring).

False-positive guard: instances below SCORE_MIN are dropped; the battery is
deliberately conservative (subject-like concepts only, never page/text —
text is already Lucida's strongest category).
"""
import numpy as np
import torch
from PIL import Image

PROMPT_BATTERY = (
    "cartoon character",
    "person",
    "animal",
    "glove",
    "hand",
    "badge emblem",
)
SCORE_MIN = 0.60

_model = None
_processor = None


def _load(device: str):
    global _model, _processor
    if _model is None:
        from transformers import Sam3Model, Sam3Processor
        _model = Sam3Model.from_pretrained("facebook/sam3", dtype=torch.float32).to(device)
        _processor = Sam3Processor.from_pretrained("facebook/sam3")
    return _model, _processor


def subject_mask(image: Image.Image, prompts: tuple[str, ...] = PROMPT_BATTERY,
                 score_min: float = SCORE_MIN, device: str | None = None) -> np.ndarray:
    """Union of high-confidence subject-instance masks (bool HxW)."""
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu")
    model, processor = _load(device)
    W, H = image.size
    union = np.zeros((H, W), dtype=bool)
    for prompt in prompts:
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_instance_segmentation(
            outputs, threshold=0.4, mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist())[0]
        scores = results.get("scores")
        for i, m in enumerate(results["masks"]):
            if scores is not None and float(scores[i]) < score_min:
                continue
            union |= m.cpu().numpy().astype(bool)
    return union
