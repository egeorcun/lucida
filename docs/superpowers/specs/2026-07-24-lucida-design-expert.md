# Lucida Design — Design-Expert Model Design Doc

Approved through brainstorming, 2026-07-24. Supersedes single-generalist releases:
the project splits into **two products** — Lucida (photo expert, current v7 stays
published) and **Lucida Design** (this spec). The photo-expert campaign gets its
own spec after this ships.

## Why

Eight epochs (v8-v15) proved photo rules and design rules fight inside one set of
weights: every epoch that fixed real-photo haze damaged layered artwork, and vice
versa. Meanwhile v14 showed the design side is a winnable, empty field: on real
layered templates (design_real, exact Crello GT) every measured model collapses
(birefnet 0.39, inspyrenet 0.40, the commercial reference 0.41) while one epoch of
Crello data took us from 0.34 to 0.11. User-confirmed target customer: print-on-
demand / poster / sticker designers. Their blockers, each with a confirmed failing
example: letter-counter whites kept (CHEESE), interior whites carved out
(Stay Fresh), edge-hugging white bloom (SALE), design elements eaten or faded
(Pinterest collage, sun illustration).

## Product definition

Specialties: **design_real layouts, typography, illustration (all styles),
transparency, complex scenes.** Photo categories (hair/camouflage/thin) may
regress freely; the model card says "use Lucida for photographs". Serving
resolution: 1536.

## Release bar (all five required)

1. design_real MAE **<= 0.08** (v14: 0.1111; rivals 0.32-0.41)
2. **Visual panel passes by eye**: ~15 real-world cases including the six
   confirmed failures (CHEESE counters empty, Stay Fresh interiors intact,
   SALE bloom gone, collage elements kept, sun kept, POD t-shirt mockup) —
   side-by-side candidate panels on dark background
3. text and illustration at or better than v7 (0.0091 / 0.0092)
4. Photo-category regressions allowed and documented
5. **Nothing ships without the user's own visual approval** (standing rule for
   all future releases)

## Data mix

Base checkpoint: **epoch_14** (design_real breakthrough already learned).

New pools (built locally, free):
- **Typography generator (~6k pairs)** — 100+ OFL fonts, large display text,
  counters exact GT=0 from font geometry; backgrounds deliberately varied:
  flat colors, WHITE (white-on-white lesson), gradients, textures, photos;
  distress/halftone variants; crops where glyphs are large (resolution lesson).
  Holdout of ~12 pairs on a separate seed becomes a benchmark category.
- **OpenClipart collages (~5k pairs)** — CC0 SVGs rendered to pixel-exact alpha,
  sticker-style compositions; "white interior on white background" cases
  over-represented (Stay Fresh lesson).
- **design_real expansion 8k -> ~11k** — remaining Crello train templates +
  validation split (test split stays benchmark-only).

Epoch shares (fixed for the whole campaign): design_real .22, typography .16,
clipart .10, transparent .14, complex .12, illustration .08, design .06,
text .06, fx .03, photo residue (hair/camo/thin/general) .03 total (graceful
degradation, not maintenance).

## Loss

- **Adaptive erosion band** in the background-purity hinge: samples whose GT is
  hard-edged (soft-alpha ratio ~0 — design/vector/typography) get a 3px band so
  edge-hugging bloom is punished to the boundary; samples with real GT softness
  keep 11px. Per-sample, automatic, unit-tested.
- lambda=3 (the v11-proven dose); synthetic semi-transparent categories stay
  exempt (max_soft_ratio gate at 0.12 unchanged).
- No other loss changes; **the recipe never changes mid-campaign** (the core
  process lesson of v8-v15).

## Training plan (one continuous campaign, ~186 of 215 available units)

| Epoch | Res  | Cost      | Purpose |
|-------|------|-----------|---------|
| 15    | 1024 | ~48 units | new data lesson lands |
| 16    | 1024 | ~48 units | consolidation, same recipe |
| — GATE: numeric scores reviewed; proceed only if design_real/typography trend right — |
| 17    | 1536 | ~90 units | sharpness polish (counters, thin edges) |

Every epoch checkpoint is kept — three candidates; the user picks by eye.
If 1536 OOMs on the assigned GPU, fall back to 1280. Resume-safe throughout.

## Evaluation

Per-epoch, local: design_real (16 Crello GT), typography holdout (~12), classic
text/illustration/transparent/complex; photo categories reported as
informational. Candidates that fail the numeric bar never reach the visual
panel. Panel verdict belongs to the user.

## Distribution

- New HF repo **egeorcun/lucida-design** (card: design/illustration/POD expert;
  photographs -> use egeorcun/lucida). The existing lucida repo keeps serving v7.
- Demo Space gains a model picker (Photo / Design); ComfyUI gets
  `lucida-design.safetensors` + a design variant of the workflow template.
- README rewritten around two products; honest release notes (gains + limits).

## Non-goals

- No photo-category rescue work in this campaign (that is the photo expert's
  spec, written after this ships).
- No router/auto-classifier yet — two honest checkpoints first; a router can
  come later as pure UX.
