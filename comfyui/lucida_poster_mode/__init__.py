"""Lucida Poster Mode — decisive POD/poster policy on a Lucida alpha.

Wire: Remove Background (Lucida) mask + the ORIGINAL image -> this node.
Outputs the processed MASK and a ready RGBA IMAGE. `counters` and `haze`
default to auto (resolved from the page color); override per image if
needed. Model to pair with: lucida-mix95-comfy.safetensors.
"""
import numpy as np
import torch

from .poster_mode import decontaminate, defringe, poster_alpha


class LucidaPosterMode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "mask": ("MASK",),
            "counters": (["auto", "open", "solid"],),
            "haze": (["auto", "on", "off"],),
            # auto: CLIP fotograf/tasarim ayrimi yapar; fotografta politika
            # kendini kapatir ve ham alfa doner (Medal dersi, 2026-08-05).
            "domain": (["auto", "design"],),
        }, "optional": {
            "subject_mask": ("MASK",),   # SAM3 hakemi: özne örnekleri (koruyucu kanıt)
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("rgba", "mask", "preview_black")
    FUNCTION = "apply"
    CATEGORY = "image/background removal"

    # rgba: dogrudan Save Image'a baglanacak nihai RGBA cikti.
    # mask: on plan=1 konvansiyonu — JoinImageWithAlpha kullanacaksan once
    # InvertMask gerekir (Join, maskeyi 1-alpha olarak yorumlar).
    def apply(self, image, mask, counters="auto", haze="auto", domain="auto",
              subject_mask=None):
        out_rgba, out_masks, out_prev = [], [], []
        n = mask.shape[0]
        for i in range(n):
            a = mask[i].detach().cpu().float().numpy()
            img_t = image[min(i, image.shape[0] - 1)].detach().cpu().float()
            img_t = img_t[..., :3]  # RGBA gelirse (ör. subgraph'in birleşik çıkışı) alfayı at
            if img_t.shape[:2] != a.shape:
                a = torch.nn.functional.interpolate(
                    torch.from_numpy(a)[None, None], size=img_t.shape[:2],
                    mode="bilinear", align_corners=False)[0, 0].numpy()
            rgb = img_t.numpy() * 255.0
            if domain == "auto":
                from PIL import Image as PILImage
                from .sam3_referee import design_domain
                if not design_domain(PILImage.fromarray(rgb.astype(np.uint8))):
                    # fotograf: politika ve finish atlanir, ham alfa doner
                    print("[LucidaPosterMode] foto alan -> politika atlandi")
                    ap = np.clip(a, 0.0, 1.0)
                    out_masks.append(torch.from_numpy(ap))
                    rgba = np.concatenate([rgb / 255.0, ap[..., None]], axis=-1)
                    out_rgba.append(torch.from_numpy(rgba.astype(np.float32)))
                    comp = rgb * ap[..., None] / 255.0
                    out_prev.append(torch.from_numpy(comp.astype(np.float32)))
                    continue
            hm = {"auto": "auto", "on": True, "off": False}[haze]
            sm = None
            if subject_mask is not None:
                smi = subject_mask[min(i, subject_mask.shape[0]-1)].detach().cpu().float()
                if tuple(smi.shape) != a.shape:
                    smi = torch.nn.functional.interpolate(
                        smi[None, None], size=a.shape, mode="nearest")[0, 0]
                sm = smi.numpy() > 0.5
            ap = poster_alpha(a, counters=counters, rgb=rgb, haze_matting=hm,
                              subject_mask=sm)
            out_masks.append(torch.from_numpy(ap))
            fg = decontaminate(rgb, ap)  # sut dersi: yarı saydamlar gerçek renkte
            fg = defringe(fg, ap)        # parlayan-P dersi: kenar rengi içeriden
            rgba = np.concatenate([fg / 255.0, ap[..., None]], axis=-1)
            out_rgba.append(torch.from_numpy(rgba.astype(np.float32)))
            comp = fg * ap[..., None] / 255.0  # siyah üstüne premultiplied önizleme
            out_prev.append(torch.from_numpy(comp.astype(np.float32)))
        return (torch.stack(out_rgba), torch.stack(out_masks), torch.stack(out_prev))


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
_clip_state = {}


class LucidaAutoPrompts:
    """CLIP zero-shot ile SAM3 kavram istemlerini otomatik secer.
    prompt1..3 -> her biri ayri CLIPTextEncode + SAM3 Detect dalina gider;
    ozne bulunamazsa guvenli varsayilanlar doner."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt1", "prompt2", "prompt3")
    FUNCTION = "pick"
    CATEGORY = "image/background removal"

    def pick(self, image):
        from PIL import Image as PILImage
        if not _clip_state:
            from transformers import CLIPModel, CLIPProcessor
            dev = "mps" if torch.backends.mps.is_available() else (
                "cuda" if torch.cuda.is_available() else "cpu")
            _clip_state["m"] = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32").to(dev).eval()
            _clip_state["p"] = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32")
            _clip_state["dev"] = dev
        arr = (image[0].detach().cpu().float().numpy()[..., :3] * 255.0).astype(np.uint8)
        img = PILImage.fromarray(arr)
        vocab = list(AUTO_SUBJECTS) + list(AUTO_ABSTAIN)
        inputs = _clip_state["p"](text=[f"a design featuring {c}" for c in vocab],
                                  images=img, return_tensors="pt",
                                  padding=True).to(_clip_state["dev"])
        with torch.no_grad():
            probs = _clip_state["m"](**inputs).logits_per_image.softmax(dim=-1)[0]
        ranked = sorted(zip(vocab, probs.tolist()), key=lambda kv: -kv[1])
        top = [(c, pr) for c, pr in ranked[:5] if pr > 0.10]
        subjects = [c for c, _ in top if c in AUTO_SUBJECTS][:2]
        if top and top[0][0] in AUTO_ABSTAIN:
            subjects = []
        # Balkabagi dersi (2026-08-05): multiplex SAM3 "glove"/"hand" ile sag
        # eli hicbir esikte bulamadi, "cartoon character" iki eli de buldu —
        # butun-figur istemi her zaman bataryada kalir (koruyucu guvence).
        if "cartoon character" not in subjects:
            subjects.append("cartoon character")
        while len(subjects) < 3:
            subjects.append("glove" if "glove" not in subjects else "hand")
        print(f"[LucidaAutoPrompts] secilen istemler: {subjects}")
        return (subjects[0], subjects[1], subjects[2])


class LucidaReferee:
    """Tam hakem: CLIP oto-istem + transformers SAM3 (facebook/sam3, fp32)
    tek node'da. Comfy'nin multiplex fp16 SAM3'u uc ayri hatada (eldiven,
    sag el, dagilmis dekor nesneleri) instance kacirdi; tam model hepsini
    buluyor. Cikti MASK dogrudan LucidaPosterMode.subject_mask'e baglanir.
    Ilk calistirmada ~3.4GB model iner/yuklenir; istem basina birkac sn."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("subject_mask", "prompts")
    FUNCTION = "run"
    CATEGORY = "image/background removal"

    def run(self, image):
        from PIL import Image as PILImage
        from .sam3_referee import auto_prompts, subject_mask
        masks = []
        used = []
        for i in range(image.shape[0]):
            arr = (image[i].detach().cpu().float().numpy()[..., :3] * 255.0).astype(np.uint8)
            img = PILImage.fromarray(arr)
            prompts = auto_prompts(img)
            used.append(", ".join(prompts))
            print(f"[LucidaReferee] istemler: {prompts}")
            sm = subject_mask(img, prompts=prompts)
            masks.append(torch.from_numpy(sm.astype(np.float32)))
        return (torch.stack(masks), " | ".join(used))


NODE_CLASS_MAPPINGS = {"LucidaPosterMode": LucidaPosterMode,
                       "LucidaAutoPrompts": LucidaAutoPrompts,
                       "LucidaReferee": LucidaReferee}
NODE_DISPLAY_NAME_MAPPINGS = {"LucidaPosterMode": "Lucida Poster Mode",
                              "LucidaAutoPrompts": "Lucida Auto Prompts",
                              "LucidaReferee": "Lucida Referee (SAM3)"}
