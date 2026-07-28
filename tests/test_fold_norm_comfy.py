"""scripts/fold_norm_comfy.py — the ComfyUI normalization fold.

The contract: for a conv consuming the (channel-major patched) image,
folded(W, b) applied to RAW pixels must equal original(W, b) applied to
ImageNet-NORMALIZED pixels, to float32 precision.
"""
import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_SPEC = importlib.util.spec_from_file_location(
    "fold_norm_comfy", Path(__file__).parent.parent / "scripts" / "fold_norm_comfy.py")
fnc = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("fold_norm_comfy", fnc)
_SPEC.loader.exec_module(fnc)


def _conv_equiv(in_ch: int):
    torch.manual_seed(7)
    W = torch.randn(8, in_ch, 3, 3)
    b = torch.randn(8)
    r = in_ch // 3
    mu = fnc.MEAN.repeat_interleave(r).view(1, -1, 1, 1)
    sig = fnc.STD.repeat_interleave(r).view(1, -1, 1, 1)
    x = torch.rand(2, in_ch, 16, 16)
    ref = F.conv2d((x - mu) / sig, W, b, padding=1)
    key = "decoder.ipt_blk2.conv1" if in_ch != 3 else "decoder.ipt_blk1.conv1"
    sd = {f"{key}.weight": W.clone(), f"{key}.bias": b.clone(),
          "bb.patch_embed.proj.weight": torch.randn(4, 3, 4, 4),
          "bb.patch_embed.proj.bias": torch.randn(4)}
    folded = fnc.fold_norm(sd)
    assert f"{key}.weight" in folded
    out = F.conv2d(x, sd[f"{key}.weight"], sd[f"{key}.bias"], padding=1)
    assert torch.allclose(ref, out, atol=1e-4), f"max diff {(ref - out).abs().max()}"


def test_fold_equivalence_plain_3ch():
    _conv_equiv(3)


def test_fold_equivalence_patched_48ch():
    _conv_equiv(48)


def test_fold_targets_all_six_gates():
    sd = {}
    for k, c in [("bb.patch_embed.proj", 3), ("decoder.ipt_blk1.conv1", 3),
                 ("decoder.ipt_blk2.conv1", 48), ("decoder.ipt_blk3.conv1", 192),
                 ("decoder.ipt_blk4.conv1", 768), ("decoder.ipt_blk5.conv1", 3072),
                 ("decoder.other.conv1", 64)]:
        sd[f"{k}.weight"] = torch.randn(4, c, 3, 3)
        sd[f"{k}.bias"] = torch.randn(4)
    folded = fnc.fold_norm(sd)
    assert len(folded) == 6
    assert "decoder.other.conv1.weight" not in folded
