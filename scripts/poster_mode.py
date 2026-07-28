"""Poster/POD mode: decisive, Ideogram-style silhouette policy on top of a
Lucida alpha (spec discussion 2026-07-28 — "the eye rewards decisiveness").

The base model separates page from design well (corner bg ~0.0005) but takes
element-wise risks (translucent gloves, carved fills). This trainless policy
trades that ambition for certainty:

1. binarize at `keep_thresh`, drop stray components smaller than
   `min_area_frac` of the canvas (keeps sparkles, kills specks);
2. zero the alpha OUTSIDE kept components (no residual haze, ever);
3. fill enclosed holes smaller than `max_hole_frac` of their component's
   bbox when EITHER (a) the model hesitated there (mean raw alpha >=
   `hole_alpha_gate`; a smoky glove is indecision -> solid), OR (b) the
   hole surrounds KEPT design content (the E/A/P lesson: decorative white
   ink inside letter counters — the model keeps the drip, deletes the rest,
   and the ragged mix looks broken; merging drip + hole reproduces the
   original solid counter). A counter the model zeroes ENTIRELY (the O)
   stays cleanly transparent. Large gaps (arm-to-body windows) stay open;
4. the outer edge keeps the model's soft alpha (no jaggies).
"""
import numpy as np
from scipy import ndimage


def poster_alpha(alpha: np.ndarray, keep_thresh: float = 0.3,
                 min_area_frac: float = 0.00005,
                 max_hole_frac: float = 0.25,
                 hole_alpha_gate: float = 0.12,
                 counters: str = "open") -> np.ndarray:
    """`counters` (user pick 2026-07-28: OPEN is the default):
    - "open" (apparel, DEFAULT): CONFIDENT-zero holes (letter counters) stay
      transparent and kept islands floating inside them (counter drips) are
      removed; holes where the model HESITATED (smoky gloves) still print
      solid — decisiveness in both directions;
    - "solid" (sticker): every small hole prints solid — the on-page look.
    Kept content is flattened to full opacity (posters are flat art); only
    the outer edge keeps the model's soft alpha."""
    if counters not in ("open", "solid"):
        raise ValueError(f"counters must be 'open' or 'solid', got {counters!r}")
    a = alpha.astype(np.float32)
    h, w = a.shape
    solid = a > keep_thresh
    lab, n = ndimage.label(solid)
    if n == 0:
        return a
    keep = np.zeros_like(solid)
    min_area = min_area_frac * h * w
    for i in range(1, n + 1):
        comp = lab == i
        if comp.sum() >= min_area:
            keep |= comp

    # fill enclosed holes per kept component (bounded by max_hole_frac)
    filled = keep.copy()
    comp_lab, cn = ndimage.label(keep)
    for i in range(1, cn + 1):
        comp = comp_lab == i
        ys, xs = np.nonzero(comp)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        box = comp[y0:y1, x0:x1]
        box_filled = ndimage.binary_fill_holes(box)
        holes = box_filled & ~box
        hlab, hn = ndimage.label(holes)
        bbox_area = box.shape[0] * box.shape[1]
        a_box = a[y0:y1, x0:x1]
        comp_box = comp_lab[y0:y1, x0:x1]
        for j in range(1, hn + 1):
            hole = hlab == j
            if hole.sum() > max_hole_frac * bbox_area:
                continue
            if counters == "solid":
                filled[y0:y1, x0:x1] |= hole
            elif float(a_box[hole].mean()) >= hole_alpha_gate:
                # the model HESITATED here (smoky glove) -> print solid,
                # in EVERY mode; only confident zeros stay open
                filled[y0:y1, x0:x1] |= hole

    if counters == "open":
        # delete kept islands floating inside holes: components fully
        # enclosed by a filled-minus-kept region (counter drips)
        whole = ndimage.binary_fill_holes(keep)
        comp_lab2, cn2 = ndimage.label(keep)
        sizes = ndimage.sum(keep, comp_lab2, range(1, cn2 + 1))
        main_ids = {int(i) + 1 for i, sz in enumerate(sizes)
                    if sz >= 0.005 * h * w}
        for i in range(1, cn2 + 1):
            if i in main_ids:
                continue
            comp = comp_lab2 == i
            grown = ndimage.binary_dilation(comp, iterations=3)
            # island whose neighborhood is inside another component's filled body
            surroundings = whole & ~keep
            if (grown & ~comp & surroundings).sum() >= 0.5 * (grown & ~comp).sum():
                keep &= ~comp
                filled &= ~comp

    out = a * keep                     # decisiveness: nothing outside survives
    out = np.maximum(out, filled.astype(np.float32))  # holes print solid
    # restore the soft outer edge: original alpha wins in the edge band
    edge = keep & ~ndimage.binary_erosion(keep, iterations=2)
    out[edge] = a[edge]
    return np.clip(out, 0.0, 1.0)
