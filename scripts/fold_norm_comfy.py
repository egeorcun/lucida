"""Fold ImageNet normalization into BiRefNet weights for ComfyUI.

WHY (2026-07-28, the ComfyUI discrepancy): ComfyUI's background-removal
path (comfy/background_removal/birefnet.py + birefnet.json with
mean=[0,0,0], std=[1,1,1]) feeds the model RAW [0,1] pixels, while BiRefNet
weights are trained on ImageNet-normalized input. Measured on the YOU'RE
HAPPY artwork with lucida-mix95: white-element alpha collapses from 0.39
(normalized) to 0.14 (raw) — the entire fill==background fix silently
vanishes inside ComfyUI. Comfy-Org's own bundled weights are NOT folded
either (byte-identical to the HF exports), so every BiRefNet model runs
degraded there; robust photo cases hide it, ambiguous design cases don't.

THE FOLD: for every conv that consumes the (possibly patched) image,
    y = W·((x-mu)/sigma) + b  ==  (W/sigma)·x + (b - sum(W·mu/sigma))
so W' = W/sigma (per input channel) and b' = b - sum over in/kh/kw of
W·mu/sigma. The image enters BiRefNet through SIX gates (BiRefNet_HR,
dec_ipt + dec_ipt_split): bb.patch_embed.proj (3ch) and
decoder.ipt_blk1..5.conv1 (3/48/192/768/3072 ch). The ipt blocks take
image2patches output with 'b c (hg h) (wg w) -> b (c hg wg) h w' layout —
CHANNEL-MAJOR, so mu/sigma expand with repeat_interleave(in_ch // 3).

Validated end to end through ComfyUI's own loader (comfy.bg_removal_model):
folded weights reproduce the normalized-pipeline behavior (white-element
alpha 0.41 vs reference 0.39; background unchanged at 0.0005).

Usage:
    uv run python scripts/fold_norm_comfy.py \
        --src lucida-mix95.safetensors --dst lucida-mix95-comfy.safetensors
"""
import argparse

import torch
from safetensors.torch import load_file, save_file

MEAN = torch.tensor([0.485, 0.456, 0.406])
STD = torch.tensor([0.229, 0.224, 0.225])
IMAGE_GATE_IN_CH = (3, 48, 192, 768, 3072)


def fold_norm(sd: dict) -> list[str]:
    """Mutates `sd` in place; returns the folded weight keys."""
    targets = ["bb.patch_embed.proj.weight"]
    for k, t in sd.items():
        if ("ipt_blk" in k and k.endswith(".conv1.weight")
                and t.ndim == 4 and t.shape[1] in IMAGE_GATE_IN_CH):
            targets.append(k)
    folded = []
    for wk in sorted(set(targets)):
        bk = wk[: -len("weight")] + "bias"
        if bk not in sd:
            raise KeyError(f"{wk}: bias missing — cannot fold")
        W = sd[wk].float()
        r = W.shape[1] // 3
        mu = MEAN.repeat_interleave(r)
        sig = STD.repeat_interleave(r)
        sd[wk] = (W / sig.view(1, -1, 1, 1)).to(sd[wk].dtype)
        delta = (W * (mu / sig).view(1, -1, 1, 1)).sum(dim=(1, 2, 3))
        sd[bk] = (sd[bk].float() - delta).to(sd[bk].dtype)
        folded.append(wk)
    return folded


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    a = ap.parse_args()
    sd = {k: v.clone() for k, v in load_file(a.src).items()}
    folded = fold_norm(sd)
    save_file(sd, a.dst)
    print(f"{len(folded)} convs folded -> {a.dst}")
    for k in folded:
        print("  ", k)


if __name__ == "__main__":
    main()
