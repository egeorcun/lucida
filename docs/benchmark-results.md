# Lucida v13 comparison (203 images, MAE — lower is better)

Source: `results/baseline/metrics.json` (`per_category` section), 2026-07 v13 run.
The released v13 weights are the uniform checkpoint soup of the lineage's epoch-8 (v9)
and epoch-13 checkpoints — see the training story in the README.

| category | lucida-v13 | lucida-v7 | inspyrenet | ideogram | rmbg-2.0 | birefnet-hr |
|---|---|---|---|---|---|---|
| camouflage | 0.0227 | 0.0270 | 0.0582 | 0.1179 | 0.1405 | 0.0752 |
| transparent | 0.0338 | 0.0358 | 0.0725 | 0.0343 | 0.0741 | 0.0687 |
| complex | 0.0465 | 0.0484 | 0.0110 | 0.1046 | 0.0241 | 0.0385 |
| thin | 0.0321 | 0.0322 | 0.0166 | 0.0521 | 0.0180 | 0.0196 |
| hair | 0.0093 | 0.0093 | 0.0069 | 0.0112 | 0.0045 | 0.0048 |
| text | 0.0103 | 0.0091 | 0.0181 | 0.0123 | 0.0173 | 0.0207 |
| fx | 0.0211 | 0.0180 | 0.0269 | 0.0165 | 0.0268 | 0.0272 |
| illustration | 0.0082 | 0.0092 | 0.0242 | 0.0215 | 0.0125 | 0.0157 |
| design | 0.0254 | 0.0235 | 0.0587 | 0.0518 | 0.0478 | 0.0544 |
| **OVERALL** | **0.0250** | **0.0257** | **0.0295** | **0.0507** | **0.0401** | **0.0346** |

## Background purity (new in v13: residue over the eroded GT==0 region)

| metric | lucida-v13 | lucida-v7 | inspyrenet | ideogram | rmbg-2.0 | birefnet-hr |
|---|---|---|---|---|---|---|
| bg_mae | 0.00910 | 0.00958 | 0.00798 | 0.02627 | 0.02193 | 0.01545 |
| bg_smear (share of bg pixels with alpha > 0.05) | 1.46% | 1.60% | 1.29% | 3.20% | 2.28% | 1.92% |
