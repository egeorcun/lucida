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
    from bgr.pipeline import PipelineSegmenter

    base_name, _, suffix = name.partition("+")
    spec = MODEL_SPECS[base_name]  # bilinmeyen ad -> KeyError
    try:
        base = BiRefNetSegmenter(
            model_id=spec["model_id"], input_size=spec["input_size"], name=base_name
        )
    except Exception as e:
        if "gated" in str(e).lower() or "401" in str(e):
            raise RuntimeError(_GATED_HELP.format(model_id=spec["model_id"])) from e
        raise
    if suffix == "refine":
        return PipelineSegmenter(base, refine=True)
    if suffix:
        raise KeyError(f"bilinmeyen varyant: +{suffix}")
    return base
