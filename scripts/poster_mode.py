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
                 counters: str = "auto",
                 rgb: np.ndarray | None = None,
                 haze_matting: bool | str = "auto",
                 haze_scale: float = 130.0) -> np.ndarray:
    """`counters` (user pick 2026-07-28: OPEN is the default):
    - "open" (apparel, DEFAULT): CONFIDENT-zero holes (letter counters) stay
      transparent and kept islands floating inside them (counter drips) are
      removed; holes where the model HESITATED (smoky gloves) still print
      solid — decisiveness in both directions;
    - "solid" (sticker): every small hole prints solid — the on-page look.
    Kept content is flattened to full opacity (posters are flat art); only
    the outer edge keeps the model's soft alpha."""
    if counters not in ("open", "solid", "auto"):
        raise ValueError(f"counters must be 'open', 'solid' or 'auto', got {counters!r}")
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

    if counters == "auto":
        # one discriminator, zero knobs (2026-07-28): a near-pure-white page
        # means counters read as holes (open); a tinted/cream page means the
        # page color doubles as ink and holes must print solid.
        counters = "solid"
        if rgb is not None:
            probe = ~keep
            if probe.sum() > 500:
                pg = np.median(rgb[probe].reshape(-1, 3), axis=0)
                if pg.min() > 243 and (pg.max() - pg.min()) < 8:
                    counters = "open"

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
                # the model HESITATED here -> print solid (smoky glove) —
                # UNLESS the hole is TINY relative to its component (a letter
                # counter, the Ideogram-parity lesson 2026-07-28): tiny
                # hesitant holes open cleanly, size separates them from smoke
                comp_area = float((comp_box == comp_box[ndimage.binary_dilation(hole) & ~hole][0]
                                   if (ndimage.binary_dilation(hole) & ~hole).any() else 0).sum()) or 1.0
                if hole.sum() >= 0.02 * comp_area:
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

    # decisiveness with GLOW RESPECT (the CHEESE lesson, 2026-07-28):
    # confident foreground flattens to solid; the model's mid-alpha keeps
    # its softness ONLY where the region drains to the silhouette edge (a
    # glow/airbrush field fading into the page — flattening those turned
    # halos into opaque white slabs while Ideogram kept them soft). Interior
    # -locked mid-alpha pockets (smoky gloves) still flatten solid.
    out = a * keep
    confident = keep & (a >= 0.7)
    out[confident] = 1.0
    mid = keep & ~confident
    if mid.any():
        boundary = keep & ~ndimage.binary_erosion(keep, iterations=3)
        mlab, mn = ndimage.label(mid)
        for i in range(1, mn + 1):
            region = mlab == i
            if not (region & boundary).any():   # interior-locked pocket
                out[region] = 1.0
    fill_new = filled & ~keep
    out[fill_new] = 1.0
    # restore the soft outer edge: original alpha wins in the edge band
    edge = keep & ~ndimage.binary_erosion(keep, iterations=2)
    out[edge] = a[edge]

    if haze_matting == "auto" and rgb is not None:
        # page-color heuristic (Stay Fresh regression, 2026-07-28): density
        # matting is gold on WHITE-page glow posters (CHEESE) and poison on
        # cream-ink designs whose elements share the page color. Auto turns
        # it on only for a near-pure-white, low-chroma page.
        probe = out < 0.05
        if probe.sum() > 500:
            pg = np.median(rgb[probe].reshape(-1, 3), axis=0)
            haze_matting = bool(pg.min() > 243 and (pg.max() - pg.min()) < 8)
        else:
            haze_matting = False
    if haze_matting and rgb is not None:
        # the Ideogram-airiness lesson (CHEESE): the model keeps glow/smoke
        # near-opaque (fx training bias) while the reference renders it as
        # INK DENSITY — distance from the page color. Applied ONLY to
        # boundary-DRAINING page-like regions: petal/glove whites are
        # outline-locked and never drain, so they are untouched.
        removed = out < 0.05
        if removed.sum() > 500:
            page = np.median(rgb[removed].reshape(-1, 3), axis=0)
            # halftone suppression: dotted textures inflate raw color
            # distance and smuggled white residue past the gate — measure
            # density on a median-smoothed image instead
            smooth = np.stack([ndimage.median_filter(rgb[..., c], size=5)
                               for c in range(3)], axis=-1)
            density = np.clip(np.linalg.norm(smooth - page, axis=-1) / haze_scale, 0.0, 1.0)
            pagey = (density < 0.8) & (out > 0.05)
            plab, pn = ndimage.label(pagey)
            boundary = keep & ~ndimage.binary_erosion(keep, iterations=3)
            for i in range(1, pn + 1):
                region = plab == i
                if (region & boundary).any():        # drains to the silhouette
                    out[region] = np.minimum(out[region], density[region].astype(np.float32))
    return np.clip(out, 0.0, 1.0)
