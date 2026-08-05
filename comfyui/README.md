# Lucida Design Pipeline for ComfyUI

The `design-expert` pipeline as ComfyUI custom nodes: **Lucida m35** weights +
poster policy + SAM3 semantic referee + finish package (decontaminate, defringe).

No-install version: the same pipeline runs in the browser at
[spaces/egeorcun/lucida-design](https://huggingface.co/spaces/egeorcun/lucida-design).

## Install

1. **Nodes** — copy the node package into ComfyUI:

   ```bash
   cp -r comfyui/lucida_poster_mode <ComfyUI>/custom_nodes/
   ```

   Python deps inside ComfyUI's environment: `transformers>=5`, `scipy`, `pillow`
   (SAM3 and CLIP load through `transformers`).

2. **Weights** — download `lucida-m35-comfy.safetensors` from the
   [Hugging Face repo](https://huggingface.co/egeorcun/lucida) and place it in:

   ```
   <ComfyUI>/models/background_removal/lucida-m35-comfy.safetensors
   ```

   This is the folded (Normalize baked into the first conv) export for ComfyUI's
   native `RemoveBackground` node — do not use it with the `transformers` snippet
   from the main README.

3. **SAM3 (referee)** — the referee uses [facebook/sam3](https://huggingface.co/facebook/sam3)
   (gated). Accept the license with your HF account and log in once inside the
   environment that runs ComfyUI (`hf auth login`). The model (~3.4 GB) and
   CLIP ViT-B/32 download automatically on first run.

4. **Workflow** — drag [`docs/comfyui/lucida_design_expert.json`](../docs/comfyui/lucida_design_expert.json)
   onto the ComfyUI canvas:

   ```
   Load Image ──► Remove Background (lucida-m35-comfy) ──► Lucida Poster Mode ──► RGBA
        └───────► Lucida Referee (SAM3) ── subject_mask ──────┘
   ```

## Nodes

| Node | What it does |
|---|---|
| `Lucida Referee (SAM3)` | CLIP zero-shot picks concept prompts (with a decor-ensemble sweep), full SAM3 segments them; the union is protective-only subject evidence. First run is slow (model load); a few seconds per prompt afterwards. |
| `Lucida Poster Mode` | The poster policy + finish package on the raw alpha. `counters`/`haze` default to auto. Feed the referee mask into `subject_mask` — without it the policy still runs, with weaker protection of page-colored subject whites. |
| `Lucida Auto Prompts` | Legacy: prompt strings for ComfyUI's native SAM3 nodes. The bundled multiplex fp16 SAM3 misses instances the full model catches — prefer `Lucida Referee`. |

## Notes

- The node package is self-contained: `poster_mode.py` and `sam3_referee.py` here are
  copies of `scripts/` at the matching commit. When updating, copy all three files.
- Outputs: `rgba` (connect straight to Save Image), `mask` (foreground=1 — invert
  before `JoinImageWithAlpha`, which reads it as 1−alpha), `preview_black`
  (premultiplied preview on black).
