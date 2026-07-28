# v18 — Edge-Limb and Atmosphere-Density Lessons

## Why (eye-test marathon, 2026-07-28)

Two model-level gaps survived the entire trainless poster-policy campaign, both
proven locally undecidable by measurement:

1. **Edge limbs** (YOU'RE HAPPY gloves): page-colored, stroke-outlined limbs
   attached to a figure's silhouette edge get raw alpha 0.03-0.17. Their pixel
   statistics are inseparable from letter counters (both white, both enclosed,
   alpha ranges interleaved). Only the model can learn "this white is a hand".
2. **Atmosphere density** (I HEARD CHEESE smoke): the model prints halftone
   smoke/glow fields at alpha ~1.0 while the Ideogram reference renders them as
   ink density (dark smoke 0.5-0.65, whites near 0, warm tint by chroma).
   Neutral dark smoke and dark body shading cannot be separated by color rules
   — two attempts damaged solid art and were reverted.

Recipe = v17 (fg_hinge 3.0, LR_SCALE 1.0, resume from epoch_17, one epoch @1024)
with SAMPLER_PRESET_V18 injecting two new synthetic categories.

## Data

### `limb` — scripts/make_clipart.py, render_limb_sample (6,000 pairs)

- White page (the ambiguity is white-on-white; ~0.85 white, rest pale flat).
- One BODY: random SVG, target 0.45-0.70 of canvas short side, center-biased.
- 2-4 LIMBS attached to the body silhouette: procedural blobs (ellipse /
  two-lobe mitten / rounded polygon), fill = page color +/- small jitter,
  dark stroke 2-6 px, placed so 30-70% of the limb lies OUTSIDE the body
  alpha. 1-2 interior "knuckle" strokes attached to the contour (the YH
  pattern: strokes touch the outline, never floating islands).
- GT = body ∪ limb-fill ∪ strokes = 1.0. Stems `limb_{i:05d}`, category `limb`.

### `atmosphere` — scripts/make_atmosphere.py (4,000 pairs)

- White page; 1-2 dark SVG anchor elements (the smoke belongs to something).
- Around/behind anchors: low-frequency density field D in [0,1] (value noise
  masked to 1-3 blobs). Two render modes per field, mixed:
  - **halftone**: ink dots on a jittered grid, dot radius ∝ D; dot color
    near-black to mid-gray;
  - **wash**: page mixed toward gray/warm smoke color by D.
- Optional glow burst (0.35 prob): radial warm rays from an anchor edge +
  white-hot core.
- **GT is the density, not the dot mask**: gt = max(a_dots, 0.85·D) inside
  the field (halftone), gt = wash-mix fraction (wash), rays at their drawn
  alpha. This is the whole lesson — smoke prints as translucency.
- Stems `atmo_{i:05d}`, category `atmosphere`.

### Probe sets (eval, HF egeorcun/lucida-eval probe/)

- `testset_limb`: 24 held-out pairs (seed 993). Gate metric: fill_alpha over
  eroded GT==1 (limbs count — they are GT 1.0).
- `testset_atmosphere`: 24 held-out pairs (seed 994). Gate metric: mae
  (falls only if the model outputs density instead of 1.0).

## SAMPLER_PRESET_V18

limb .09, atmosphere .07 (16% combined — the 2026-07-26 underdose lesson:
clipart2 at 3k/1 epoch never taught the gloves); shaved from: design_real
.16→.12, typography_outline .14→.11, transparent .12→.09, complex .10→.09,
typography .08→.07, design_real_ambig .08→.07, clipart .06→.05,
illustration .06→.05, fx .03→.02. Sum stays 1.0.

## Gates (v17 discipline — no blind spends)

0. **Data eye gate (free)**: dark-checker preview grids of both generators go
   to the user BEFORE any Colab spend.
1. **Smoke gate (~2-3 units)**: EPOCH_NUM_SAMPLES=1200 from epoch_17.
   PASS = probe_limb fill_alpha UP by >0.01 AND probe_atmosphere mae DOWN
   AND design_real/typography bg_mae within +0.002.
2. **Full epoch (~48 units)** only on PASS: EPOCHS=18 @1024.
3. Final: benchmark + the four eye-test artworks re-rendered; release bar
   stays "user approves by eye".

## Verification

- Unit tests per generator (determinism, GT contracts: limb GT solid across
  the limb; atmosphere GT mid-alpha in dot gaps, ≤ D ceiling; knuckle strokes
  attached to contour).
- Local 24-pair preview at seed 993/994 doubles as probe-set dry run.
