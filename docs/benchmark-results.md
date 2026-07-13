# Lucida v5 karşılaştırma (191 görsel, MAE — düşük iyi)

> Snapshot of `results/baseline/lucida-v5-comparison.md` (that directory is git-ignored).
> Methodology and reproduction commands: [benchmark.md](benchmark.md).
> `bgr-v4` = the epoch-4 checkpoint of the same fine-tune; `+refine` = optional edge-refinement pass.

| kategori | lucida-v5 | lucida-v5+refine | bgr-v4 | inspyrenet | ideogram | rmbg-2.0 | birefnet-hr |
|---|---|---|---|---|---|---|---|
| camouflage | 0.0273 | 0.0272 | 0.0271 | 0.0582 | 0.1179 | 0.1405 | 0.0752 |
| transparent | 0.0376 | 0.0376 | 0.0405 | 0.0725 | 0.0343 | 0.0741 | 0.0687 |
| complex | 0.0666 | 0.0665 | 0.0664 | 0.0110 | 0.1046 | 0.0241 | 0.0385 |
| thin | 0.0350 | 0.0350 | 0.0375 | 0.0166 | 0.0521 | 0.0180 | 0.0196 |
| hair | 0.0087 | 0.0088 | 0.0106 | 0.0069 | 0.0112 | 0.0045 | 0.0048 |
| text | 0.0126 | 0.0136 | 0.0119 | 0.0181 | 0.0123 | 0.0173 | 0.0207 |
| fx | 0.0321 | 0.0300 | 0.0288 | 0.0269 | 0.0165 | 0.0268 | 0.0272 |
| illustration | 0.0095 | 0.0093 | 0.0129 | 0.0242 | 0.0215 | 0.0125 | 0.0157 |
| **OVERALL** | **0.0304** | **0.0303** | **0.0316** | **0.0277** | **0.0506** | **0.0396** | **0.0334** |
