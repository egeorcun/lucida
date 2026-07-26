# Fill==Background: the Definitive Fix (Lucida Design)

**User directive (2026-07-26):** "kesin çözülmesi için uğraş, deneme yapma
artık" — no more one-epoch experiments; engineer the fix, prove it cheaply,
then spend GPU once.

## The failure

A design element whose color equals the page color (cream letter fills on a
cream page, white petals/hands on a white page) is deleted or made
semi-transparent. Measured on the user's real artwork (Stay Fresh original,
YOU'RE HAPPY): every stage-1/2 candidate hollows the fills; v7 is the least
bad. The outline data patch (3000 pairs, 1 epoch, end-of-schedule LR)
improved soup1617 only marginally — right direction, hopeless dose.

## Root cause

Every background-quality intervention since v8 pushed in ONE direction:
bg_purity BCE, bg hinge, tighter erosion bands — all price "alpha left on
background". Nothing ever priced "hole carved inside an opaque element".
Under that asymmetric objective, the rational policy for an ambiguous pixel
is DELETE. Fill==bg is the purest ambiguity, so it exposes the learned
erase-bias most clearly. More data alone cannot beat an objective that
keeps rewarding deletion; the objective itself must become symmetric.

## The fix (three legs, one verification gate)

### 1. fg_hinge_loss — the mirror force (training/torch_losses.py)

Hinge over the ERODED true-foreground region (GT == 1 eroded by
`erosion_px`): `mean(relu(logit(1 - tau_p) - logit))`. Constant gradient for
every interior pixel predicted below the tolerated opacity — the only shape
that can grind holes shut, exactly as the bg hinge was the only shape that
could grind haze down (v10 lesson, mirrored). Same contracts as
bg_hinge_loss:
- `max_soft_ratio` gate: semi-transparent categories (fx/transparent glow)
  are exempt per-sample — pressure applies only where the GT says OPAQUE.
- adaptive band: hard-edged samples (soft ratio <= 0.01) use
  `hard_erosion_px` so the pressure reaches close to the outline; soft
  samples keep the wide band (fur interiors are not flattened).
- Weight: `FG_HINGE_LAMBDA`, same order as the bg hinge's lambda. Both
  hinges active together = symmetric objective.

### 2. Ambiguity data at full dose (v17 data update)

- **typography_outline pool (new, 6000):** outline menu from index 0
  (OUTLINE_PROB .75, FILL_EQ_BG .50 — raised), stems `typo2_*`, category
  `typography`.
- **clipart outline mode:** enclosing dark stroke around SVG elements over
  a page colored EXACTLY like the element's dominant color (share 0.30 of
  a new 3000-pair `clip2_*` pool).
- **design_real_ambig pool (new, 4000):** REAL Crello templates
  re-composited over a page filled with the dominant color of a randomly
  chosen foreground element (exact GT from layers, real-world layout, the
  ambiguity injected deliberately). Closest generator to the failing cases.
- Sampler preset v17: design_real .18, design_real_ambig .08,
  typography .10, typography_outline .12, clipart .08, clipart2 .04,
  transparent .14, complex .12, illustration .08, design .04, text .04,
  fx .03, hair .01, camouflage .01, thin .005, general .005 (= 1.0).

### 3. Training shape

Resume from **epoch_16** (not 17 — the 1536 epoch bought nothing and its
LR is fully decayed), `LR_SCALE = 1.0` (fresh pressure for relearning a
bias), 2 epochs @1024 (~92 units). No mid-campaign recipe changes.

### Verification gate (BEFORE the full spend)

- **fill_eq_bg probe set** (data/testset_fill_eq_bg): 24 synthetic
  fill==bg pairs (typography + clipart + design_real_ambig generators,
  held-out seeds) + the two real cases (Stay Fresh original, YOU'RE HAPPY
  original when provided). New metric: `fill_alpha` = mean alpha over
  eroded GT==1 (the hole metric; target mean >= 0.95).
- **Smoke probe (~3 units):** 600 steps @1024 from epoch_16 with the v17
  recipe. PASS = probe fill_alpha clearly rising vs epoch_16 baseline AND
  bg_mae flat on a 30-image classic slice. Only a PASS unlocks the 2-epoch
  spend. FAIL = back to the drawing board at zero additional cost.

## Success criteria (release re-gate)

- fill_eq_bg probe: fill_alpha >= 0.95 mean, >= 0.90 min per real case.
- design_real <= 0.08 preserved; typography <= 0.009; bg metrics within
  +0.002 of e16 (no smear regression).
- User's eye on the panel (Stay Fresh full + YOU'RE HAPPY + POD cases) —
  final authority, unchanged.

## Cost

Data prep: free (CPU). Smoke probe ~3 units. Full run ~92 units.
Total ~95 units, spent only after the smoke gate passes.
