# v19 Data Recipe — Subject-Whites Semantics (2026-08-05)

Single theme, per the post-v18 decision: **the model must stop scoring
subject-owned whites as page**. No new rule work rides on this; the poster
policy and the SAM3 referee stay frozen so the training effect is
measurable in isolation.

## Why now: the referee-era case log

Four eye-test failures between 2026-08-04 and 2026-08-05, all one class.
Today every one of them is patched by the SAM3 referee — which means the
model itself still fails them, and the referee is a single point of
failure (three of the four regressions were triggered by the weaker
multiplex SAM3 in ComfyUI missing an instance the full model catches).

| Case | Design | Raw model verdict | What saved it |
|---|---|---|---|
| 1 | YOU'RE HAPPY gloves | fills at 0.03–0.17 (left), 0.6 (right, post-v18) | referee lift |
| 2 | Pumpkin peace-sign hands | cream fills at median **0.001** — confident zero | referee hole-fill (via "cartoon character" prompt) |
| 3 | Summer Vibes shells / sand dollar / pearls | 46–66% of light pixels < 0.05 — confident-zero carving | decor-sweep battery |
| 4 | Summer Vibes cream water drop | **0.87+ (model was right!)** — the atmosphere law melted it | "water drop" melt exemption |

Case 4 is the mirror image of the others: the policy cannot tell a glossy
neutral element from milk residue, so the model must learn to make smoke
*graded* (density alpha) while elements stay solid — the same lesson as
v18's atmosphere cell, but with real-form data this time.

## The v18 post-mortem constraints (binding)

1. **No synthetic blob shapes.** e17 aced the synthetic limb probe (0.987)
   and scored 0.06 on real gloves. Every foreground element in v19 data
   must be a real-form asset (actual clipart/watercolor drawings), not a
   generated blob.
2. **Probes must be held-out REAL images.** The synthetic probe ceiling
   masked the failure until the eye test. v19's gate is the model alone —
   referee OFF — on the four cases above plus new specimens.
3. **Dose matters.** v18's 3k×1-epoch limb cell moved the right glove
   0.37→0.65 and the left not at all. The v19 cell is larger and trains
   longer (see quantities).

## Data cells

### Cell A — outlined page-colored fills (cases 1–3)

The signature: an element whose fill is within Δ<20 of the page color,
sealed (or half-sealed) by a drawn contour, attached to or floating near
a larger figure.

- **Assets:** real POD-style clipart with four-finger cartoon gloves/hands,
  white-filled limbs; marine/decor packs (shells with ribbed white
  interiors, sand dollars, pearls, bows); white fur animals (tiger/cat
  chest+muzzle class from the duel).
- **Composites:** paste on white and warm-white pages (the Pumpkin lesson:
  [254,251,244]-class pages are in distribution), plus 10% tinted-cream
  pages labeled *solid-counters* style so Petersburg behavior survives.
- **GT protocol:** alpha = final desired matte — fills SOLID 1.0, page
  pockets between elements 0.0. GT is authored from the asset's own
  transparency, never from a model.
- **Scatter variant (Summer Vibes class):** 8–20 small objects per canvas,
  no dominant figure, so the model learns whites belong to objects even
  without a big anchor silhouette.
- **Quantity:** 5,000 composites (vs v18's 3,000 single-cell), sampling
  weight ≥ 2× base.

### Cell B — glossy neutrals vs graded atmosphere (case 4 + CHEESE)

Same canvas carries BOTH classes so the contrast is learnable:

- glossy neutral elements (water drops, pearls, glass highlights) with
  contour + specular structure → GT solid;
- real halftone/airbrush smoke and glow fields → GT = ink density (the
  Ideogram-fitted plane: 0.845·dist + 0.205·coverage − 0.01), authored by
  running the fitted curve on the source, then hand-spot-checked.
- **Quantity:** 2,000 composites. This is the only cell where GT is
  derived rather than asset-native; cap its weight at 1× so a bad
  derivation cannot dominate.

### Cell C — SAM3 data engine (the ZIM-style loop, first live use)

The referee already produces correct masks on the exact images the model
fails. Turn today's patch into tomorrow's labels:

- Collect unlabeled real POD designs (Etsy-style corpus, target 1,500).
- Run the FULL pipeline (m35 + policy + full-SAM3 referee) → output alpha.
- Keep only images where referee-ON vs referee-OFF outputs differ by >2%
  of pixels — those differences are precisely the semantic class the
  model lacks.
- Eye-skim at contact sheet level, drop bad sheets, use survivors as
  training pairs (input, referee-ON alpha).
- **Quantity target:** 1,000 survivors.

## Gate (before any Colab spend)

- Assemble Cells A–C, then run the CURRENT m35 on a 100-image held-out
  slice to record the baseline carving rate (fraction of GT-solid
  page-colored pixels scored < 0.3).
- Train gate: v19 passes only if, **referee OFF**, (a) carving rate drops
  by ≥ 60% on held-out, (b) the four case images pass the eye test, and
  (c) the 203-image classic benchmark shows no category regression
  (text bar 0.0095 must not worsen).
- The 13-duel catalog rerun (referee ON) is the final eye gate — the
  referee must become insurance, not scaffolding.

## Explicit non-goals

- No policy/rule changes bundled into the v19 cycle.
- No new synthetic texture generators.
- Dark-page pocket semantics (overthink class) stays filed for v20 — it
  needs its own cell and must not dilute this one.
