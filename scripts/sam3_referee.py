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

# auto_prompts vocabulary: CLIP zero-shot picks which concepts are present,
# SAM3 then segments only those. ABSTAIN classes absorb subject-less designs
# (pure lettering, landscapes) so softmax cannot hallucinate a subject.
AUTO_SUBJECTS = (
    "cartoon character", "person", "woman", "man", "animal", "dog", "cat",
    "tiger", "bird", "bear", "glove", "hand", "shoe", "flower",
    "badge emblem", "mascot", "robot", "skull", "car", "food", "cheese",
    "pumpkin", "ghost", "razor", "astronaut", "deer", "monster", "baby",
    "dinosaur", "fish",
)
AUTO_ABSTAIN = (
    "text lettering only", "typography quote design", "landscape scenery",
    "abstract pattern", "logo wordmark",
)

_clip = None
_clip_proc = None


def auto_prompts(image: Image.Image, top_k: int = 4, prob_min: float = 0.10,
                 device: str | None = None) -> tuple[str, ...]:
    """Pick SAM3 concept prompts automatically: CLIP zero-shot ranks a
    design-domain vocabulary; abstain classes win on subject-less designs
    and the fixed PROMPT_BATTERY is returned as the safe fallback. SAM3
    self-filters absent concepts (returns zero instances), so a borderline
    pick costs inference time, never correctness."""
    global _clip, _clip_proc
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu")
    if _clip is None:
        from transformers import CLIPModel, CLIPProcessor
        _clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
        _clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    vocab = list(AUTO_SUBJECTS) + list(AUTO_ABSTAIN)
    inputs = _clip_proc(text=[f"a design featuring {c}" for c in vocab],
                        images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        probs = _clip(**inputs).logits_per_image.softmax(dim=-1)[0]
    ranked = sorted(zip(vocab, probs.tolist()), key=lambda kv: -kv[1])
    top = [(c, p) for c, p in ranked[:5] if p > prob_min]
    subjects = [c for c, _ in top if c in AUTO_SUBJECTS][:top_k]
    if not subjects or (top and top[0][0] in AUTO_ABSTAIN):
        return PROMPT_BATTERY
    # the Pumpkin lesson (2026-08-05): the multiplex SAM3 never found the
    # right glove under "glove"/"hand" at any threshold, but "cartoon
    # character" caught both hands — the whole-figure prompt always rides
    # along as protective evidence.
    return tuple(dict.fromkeys(subjects + ["cartoon character", "glove", "hand"]))

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
