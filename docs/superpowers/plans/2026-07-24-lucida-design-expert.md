# Lucida Design Expert Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the design-expert model campaign: two new data generators (typography with exact counter GT, OpenClipart collages), an adaptive-erosion hinge loss, the v16 preset/notebook wiring, the Colab data cell, and the staged-resolution training runbook — per `docs/superpowers/specs/2026-07-24-lucida-design-expert.md`.

**Architecture:** New generators follow the established `scripts/make_*.py` contracts (deterministic `_item_rng`, resume-safe, `im/ gt/ manifest.jsonl` layout, progress prints every 250). The loss change is a per-sample erosion-band selection inside the existing `bg_hinge_loss`. Training reuses `train_colab.ipynb` with a new preset and a `TRAIN_SIZE` parameter for the 1536 polish epoch.

**Tech Stack:** Python 3.12, uv, pytest, PIL/numpy/scipy, torch (losses only), cairosvg (clipart rendering), HF `datasets` (Crello/OpenClipart), Colab A100/H100.

## Global Constraints

- Base checkpoint: **epoch_14**; recipe (shares, lambda, gates) NEVER changes mid-campaign.
- Epoch shares (exact): design_real .22, typography .16, clipart .10, transparent .14, complex .12, illustration .08, design .06, text .06, fx .03, photo residue .03.
- Hinge: lambda=3, max_soft_ratio gate 0.12 unchanged; adaptive band 3px (hard-edged GT) / 11px (soft GT).
- Release bar: design_real ≤ 0.08; user's visual panel passes BY EYE; text ≤ 0.0091 and illustration ≤ 0.0092; photo regressions allowed+documented; nothing ships without the user's visual approval.
- Budget: campaign ≤ 190 units of the 215 available; GATE after epoch 16 before the 1536 epoch.
- All public text English; NO Claude attribution anywhere (standing rule).
- All work on branch `v14` (rename to `design-expert` at first commit).

---

### Task 1: Typography generator

**Files:**
- Create: `scripts/make_typography.py`
- Test: `tests/test_make_typography.py`

**Interfaces:**
- Consumes: `_get_font`, `_draw_text_rgba`, `_rand_text`, `_renders_latin` from `scripts/make_textfx.py` (import, don't copy); `_item_rng`, `_save_pair`, `_append_manifest`, `_load_manifest_ids` patterns from `scripts/make_bokeh_copies.py` (copy with attribution comments, per repo convention).
- Produces: `run(out_dir: Path, font_dir: Path, bg_pool_dirs: list[Path], count: int = 6000, seed: int = 21) -> int`; stems `typo_{i:05d}`; manifest rows `{"id", "category": "typography"}`.

- [ ] **Step 1: Write the failing tests** — `tests/test_make_typography.py` with: (a) `test_counters_are_zero_in_gt`: render the word "BOOB" in a bold font at 300px over a flat background; assert the GT has ≥2 connected zero-regions fully enclosed by α=1 pixels (letter counters), using `scipy.ndimage.label` on `(gt==0)` restricted to the text bbox interior; (b) `test_white_text_on_white_bg_gt_exact`: white fill over white background — image nearly uniform but GT still carries the glyphs (`gt.max()==1`, glyph-area fraction > 1%); (c) `test_distress_never_touches_gt`: distress variant changes RGB inside glyphs but GT is bit-identical to the undistressed render of the same stem seed; (d) determinism + resume tests (same-seed bit-identical; second run skips all).
- [ ] **Step 2: Run tests, verify failure** — `uv run pytest tests/test_make_typography.py -q` → import error.
- [ ] **Step 3: Implement `scripts/make_typography.py`** — per sample: pick 1-2 fonts; render 1-3 display lines large (glyph height 25-60% of canvas short side; canvas 640-1024); GT = union of glyph alphas (counters are zero by font geometry — that IS the point); background menu: flat color 30%, WHITE 20%, gradient 20%, texture (value-noise) 15%, image from `bg_pool_dirs` 15%; optional distress (value-noise mask multiplied into BOTH rgb-visibility and NOT into gt — distress erodes ink visually via rgb blending toward bg while gt stays glyph-exact… implement as: rgb = mix(bg, ink, glyph_alpha*distress_mask) but gt = glyph_alpha, documented as "print wear look, alpha contract preserved"); crops: 40% of samples are tight crops around the text block so glyphs are LARGE (the resolution lesson).
- [ ] **Step 4: Tests pass** — `uv run pytest tests/test_make_typography.py -q`.
- [ ] **Step 5: Commit** — `git add scripts/make_typography.py tests/test_make_typography.py && git commit -m "feat: typography generator with exact counter ground truth"`.

### Task 2: Clipart collage generator

**Files:**
- Create: `scripts/make_clipart.py`
- Test: `tests/test_make_clipart.py`

**Interfaces:**
- Consumes: `cairosvg.svg2png` (new dep: `uv pip install cairosvg`, add to Colab cell installs); the make_design compositing pattern.
- Produces: `run(out_dir: Path, svg_dir: Path, count: int = 5000, seed: int = 33, white_bg_share: float = 0.35) -> int`; stems `clip_{i:05d}`; category `"clipart"`.

- [ ] **Step 1: Failing tests** — (a) `test_svg_renders_to_exact_alpha`: an inline SVG circle renders to alpha with interior 1 and exterior 0; (b) `test_white_fill_clipart_on_white_bg_kept_in_gt`: white-filled SVG flower over white background — GT keeps the flower (Stay Fresh lesson); (c) collage composition: 3-6 elements, GT = union, background NOT in GT; (d) determinism/resume. Test SVGs are inline strings written to tmp files.
- [ ] **Step 2: Verify failure.**
- [ ] **Step 3: Implement** — render SVG → RGBA at random scale (cairosvg, `output_width`), compose 3-6 elements with rotation/overlap onto background (white `white_bg_share` of the time — over-representing the white-on-white lesson; else flat pastel/gradient); GT = union of element alphas.
- [ ] **Step 4: Tests pass.** — full suite `uv run pytest tests/ -q -m "not slow"`.
- [ ] **Step 5: Commit** — `"feat: OpenClipart collage generator (pixel-exact SVG alpha)"`.

### Task 3: Adaptive erosion band in the hinge loss

**Files:**
- Modify: `training/torch_losses.py` (bg_hinge_loss)
- Test: `tests/test_torch_losses.py` (extend)

**Interfaces:**
- Produces: `bg_hinge_loss(..., erosion_px=11, hard_erosion_px=3, hard_soft_ratio=0.01)` — samples whose GT soft-ratio ≤ `hard_soft_ratio` use the tight band; others keep `erosion_px`. Backward compatible defaults.

- [ ] **Step 1: Failing tests** — (a) `test_hard_edged_sample_gets_tight_band`: vector-like GT (binary), residue placed 5px from the subject edge (inside the old 11px band, outside the new 3px band) MUST be penalized; (b) `test_soft_sample_keeps_wide_band`: photo-like GT (has soft edge ring), same 5px residue must NOT be penalized; (c) mixed batch: gradients land accordingly per sample.
- [ ] **Step 2: Verify failure** (the 5px residue currently escapes in (a)).
- [ ] **Step 3: Implement** — compute per-sample soft ratio (existing code); build two eroded-bg masks (`_eroded_bg_mask(gt, erosion_px)` and `_eroded_bg_mask(gt, hard_erosion_px)`); select per sample with `torch.where(hard_mask_flag.view(-1,1,1,1), bg_tight, bg_wide)`.
- [ ] **Step 4: Tests pass** — all 15+ loss tests green.
- [ ] **Step 5: Commit** — `"feat: adaptive erosion band — edge-hugging bloom on hard-edged art is now penalized"`.

### Task 4: Preset V16, testset categories, notebook wiring

**Files:**
- Modify: `training/train_colab_lib.py` (SAMPLER_PRESET_V16 + docstring with the spec's share table verbatim), `benchmark/testset.py` (CATEGORIES += `"typography"`, `"clipart"`), `training/train_colab.ipynb` (cell 2: `SAMPLER_PRESET="v16"`, `EPOCHS=16`, new `TRAIN_SIZE = 1024` param; cell (d): after `apply_config_patches`, regex-patch BiRefNet `config.py`: `re.sub(r"self\.size\s*=\s*\([^)]+\)", f"self.size = ({TRAIN_SIZE}, {TRAIN_SIZE})", s)`; cell 13 guards: v16 requires ≥500 `typo_*` AND ≥500 `clip_*` stems)
- Test: `tests/test_train_colab_lib.py`, `tests/test_train_colab_notebook.py`

- [ ] Steps: failing preset test (shares sum 1.0, design_real .22, typography .16) → implement → notebook JSON edits via the established python-patch pattern → full suite green → commit `"feat: v16 preset + TRAIN_SIZE parameter + typography/clipart guards"`.

### Task 5: Benchmark additions — typography holdout + visual panel tool

**Files:**
- Create: `data/testset_typography/` via `make_typography.run(count=12, seed=999)` (disjoint seed), manifest category `typography`.
- Create: `scripts/make_visual_panel.py` — takes a dir of ~15 real-world images (user's six failure cases + POD mockups, collected in `data/visual_panel/`), runs N registry models, emits one dark-checkerboard side-by-side JPEG per image into `results/visual_panel/<model>/…` plus a combined contact sheet.
- Test: `tests/test_make_visual_panel.py` (runs with a stub segmenter registered in the test, not a real model).

- [ ] Steps: failing test (panel file exists, has 1+N columns, checkerboard applied under RGBA) → implement → generate the 12 typography holdout pairs + commit them (small, deterministic) → commit `"feat: typography holdout + visual panel tooling"`.

### Task 6: v16 Colab data cell

**Files:**
- Create: `training/v16_data_update_cell.py` — v14 cell lineage (report/log, Errno 5 retry, module-cache purge, copy_pairs merge, flush). Stages: env (pip installs: datasets, cairosvg) → fonts (expand GOOGLE_FONT_PATHS to ~60 OFL fonts — the list is written out in the cell) → typography 6k (`make_typography.run`, bg pool = local tar-extracted TRAIN images) → openclipart download (`nyuuzyou/openclipart` via datasets, sample ~20k SVGs) → clipart 5k → design_real expansion (train split count=10500 resume-safe into the SAME `data/train_design_real` dir + validation split count=1000 into `data/train_design_real_val`) → export all → Drive merge (fresh stems only) → flush.
- Test: syntax parse + `tests/test_train_colab_lib.py` unchanged-contract run.

- [ ] Steps: write cell (complete code, no placeholders — copy the v14 cell and replace the generate/export/merge stages) → `python3 -c "import ast; ast.parse(...)"` → full local suite → commit `"feat: v16 data cell — typography, clipart, design_real expansion"` → push.

### Task 7: Campaign runbook (operational, no code)

- [ ] Rename branch: `git branch -m v14 design-expert && git push -u origin design-expert && git push origin --delete v14`; update the Colab driver cells' `REPO_BRANCH`.
- [ ] CPU session: run v16 data cell (~3-4h: Crello cached? NO — new VM, re-downloads ~1h; typography+clipart ~1.5h; merge ~1h). Confirm "Drive flush COMPLETE".
- [ ] GPU session: driver with `EPOCHS=16` (trains epochs 15+16 back-to-back, ~7.4h, ~96 units; resume-safe).
- [ ] Local: benchmark epochs 15 & 16 on design_real + typography holdout + classic design/text/illustration/transparent/complex + photo categories (informational) + visual panel contact sheets.
- [ ] **GATE (user decision):** trend right → GPU session with `EPOCHS=17`, `TRAIN_SIZE=1536` (~7h, ~90 units; OOM fallback 1280). Else stop with ~119 units intact.
- [ ] Final: three candidates × (numbers + panels) → user picks by eye or rejects all. Release flow (separate `egeorcun/lucida-design` HF repo, Space picker, ComfyUI file+template, README two-product rewrite) begins ONLY after the user's visual approval.

## Self-Review

- Spec coverage: data mix (T1/T2/T6), adaptive band (T3), preset/resolution (T4), evaluation+panel (T5), staged training+gate+budget (T7), distribution (T7 final, gated). ✓
- No placeholders: generator/loss tasks carry concrete test definitions and implementation outlines with exact contracts; the two long file bodies (T1/T2 implementations, T6 cell) follow named, existing in-repo patterns line-for-line — the implementer copies `make_bokeh_copies.py`/`v14_data_update_cell.py` scaffolding as instructed. ✓
- Type consistency: `run()` signatures, stem prefixes (`typo_`, `clip_`, `crello_`), category names (`typography`, `clipart`, `design_real`) used identically across T1-T6. ✓
