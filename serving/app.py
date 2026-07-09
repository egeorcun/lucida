"""Lokal bg-remove servisi: uv run uvicorn serving.app:app --port 8756"""
import io
import threading

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

from bgr.pipeline import PipelineSegmenter
from bgr.registry import MODEL_SPECS, get_segmenter

app = FastAPI(title="my-bg-remover")
_SEGMENTERS: dict[str, object] = {}
_SEGMENTERS_LOCK = threading.Lock()


def _load_segmenter(name: str):
    if name not in _SEGMENTERS:
        with _SEGMENTERS_LOCK:
            if name not in _SEGMENTERS:
                _SEGMENTERS[name] = get_segmenter(name)
    return _SEGMENTERS[name]


@app.get("/health")
def health():
    return {"status": "ok", "models": sorted(MODEL_SPECS)}


@app.post("/remove")
def remove(
    file: UploadFile,
    model: str = "rmbg-2.0",
    refine: bool = False,
    decontaminate: bool = True,
):
    try:
        seg = _load_segmenter(model + ("+refine" if refine else ""))
    except KeyError:
        raise HTTPException(400, f"bilinmeyen model: {model}")
    try:
        img = Image.open(io.BytesIO(file.file.read()))
        img.load()
    except Exception:
        raise HTTPException(400, "geçersiz görsel dosyası")
    pipe = seg if isinstance(seg, PipelineSegmenter) else PipelineSegmenter(seg)
    out = pipe.process(img, decontaminate=decontaminate)
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")
