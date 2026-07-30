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
                 haze_scale: float = 130.0,
                 haze_ink_gate: float = 0.35,
                 haze_cov_radius: float = 0.08,
                 haze_cov_lo: float = 0.20,
                 haze_cov_hi: float = 0.60,
                 subject_mask: np.ndarray | None = None) -> np.ndarray:
    """`counters` (user pick 2026-07-28: OPEN is the default):
    - "open" (apparel, DEFAULT): CONFIDENT-zero holes (letter counters) stay
      transparent and kept islands floating inside them (counter drips) are
      removed; holes where the model HESITATED (smoky gloves) still print
      solid — decisiveness in both directions;
    - "solid" (sticker): every small hole prints solid — the on-page look.
    Kept content is flattened to full opacity (posters are flat art); only
    the outer edge keeps the model's soft alpha.

    `subject_mask` (the SAM3 referee, 2026-07-30): an optional boolean mask
    of SUBJECT instances (characters, animals, gloves, badges) from a
    semantic model. Used asymmetrically — only as PROTECTIVE evidence:
    pixels inside the mask are never melted by the haze stage, page-colored
    pockets inside it never open, and hesitant page-colored regions inside
    it print solid (the gloves). Where the mask is silent, behavior is
    identical to subject_mask=None, so a missed concept can never regress
    the output."""
    if counters not in ("open", "solid", "auto"):
        raise ValueError(f"counters must be 'open', 'solid' or 'auto', got {counters!r}")
    a = alpha.astype(np.float32)
    h, w = a.shape

    if rgb is not None:
        # CHROMATIC RESCUE (the rainbow lesson, 2026-07-28): on a flat-page
        # design a pixel whose color sits FAR from the page color is ink by
        # definition — yet the model erases wide soft-gradient elements
        # (rainbow swoosh, airbrushed glow, a pink razor) as page haze.
        # Restore alpha from color evidence BEFORE any silhouette decision.
        # Page-colored elements (white gloves) are untouched: color cannot
        # decide those. The pixel's OWN color decides — a neighborhood-
        # smoothed distance resurrects page-white gaps between halftone dots
        # and between small-type letters (their median neighborhood is ink).
        # Lone saturated noise pixels still die in the speck filter.
        low = a < 0.05
        if low.sum() > 500:
            page_color = np.median(rgb[low].reshape(-1, 3), axis=0)
            page_dist = np.linalg.norm(rgb - page_color, axis=-1)
            rescue = np.clip((page_dist / haze_scale - 0.5) / 0.5, 0.0, 1.0)
            a = np.maximum(a, rescue.astype(np.float32))
        else:
            page_dist = None
    else:
        page_dist = None
    if subject_mask is not None:
        sm = subject_mask.astype(bool)
        assert sm.shape == a.shape, "subject_mask shape mismatch"
        # THE GLOVES RULE: hesitant page-colored pixels INSIDE a subject
        # instance are the subject's own whites — lift them to solid before
        # any silhouette decision.
        lift = sm & (a > 0.05) & (a < keep_thresh + 0.4)
        a = np.where(lift, np.maximum(a, 0.95), a)
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
                # (2026-07-30) a dark-flat-page variant of this rule was
                # tried (the overthink duel) and REVERTED the same day: it
                # gutted full-bleed dark designs (the AImpala badge interior
                # and its I/l counters) — dark pocket vs dark design is a
                # semantic call, filed for v19 training data instead.

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
        pd_box = page_dist[y0:y1, x0:x1] if page_dist is not None else None
        for j in range(1, hn + 1):
            hole = hlab == j
            if hole.sum() > max_hole_frac * bbox_area:
                continue
            if counters == "solid":
                filled[y0:y1, x0:x1] |= hole
                continue
            if subject_mask is not None and \
                    subject_mask[y0:y1, x0:x1][hole].mean() > 0.6:
                # referee: hole inside a subject instance -> print solid
                filled[y0:y1, x0:x1] |= hole
                continue
            if pd_box is not None and hole.sum() < 0.0012 * h * w and \
                    float(np.median(pd_box[hole])) < 0.3 * haze_scale:
                # the hole IS the page showing through (O/D/G counters in
                # small type, 2026-07-28): a SMALL page-colored interior
                # stays open no matter how much the model hesitated. The
                # size cap keeps big page-colored pockets (a daisy petal
                # bay) printing solid — at petal scale white is ink.
                continue
            if float(a_box[hole].mean()) >= hole_alpha_gate:
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
            if (region & boundary).any():
                # HALF-CONFIDENT WHITE ELEMENT (the v18 gloves, 2026-07-30):
                # after the limb lesson the model answers "probably element"
                # (raw ~0.5) on page-colored edge-attached limbs it used to
                # erase (~0.1-0.3). A draining page-colored region the model
                # scores >=0.45 is an element — print it SOLID instead of
                # leaving a half-ghost for the haze stage to melt. Chromatic
                # glows are untouched (page_dist gate).
                if page_dist is not None and region.sum() >= 0.0012 * h * w and \
                        float(np.median(page_dist[region])) < 0.3 * haze_scale and \
                        float(a[region].mean()) >= 0.45:
                    out[region] = 1.0
                continue
            # interior-locked pocket
            if subject_mask is not None and float(subject_mask[region].mean()) > 0.6:
                out[region] = 1.0    # referee: pocket inside a subject -> solid
                continue
            if counters == "open" and page_dist is not None and \
                    region.sum() < 0.0012 * h * w and \
                    float(np.median(page_dist[region])) < 0.3 * haze_scale:
                # O/D/G lesson (2026-07-28): a SMALL page-colored pocket
                # the model hesitated on is the page showing through a
                # counter — open it. The size cap protects petal-scale
                # white pockets (the daisy-bite regression): at element
                # scale, page-colored means white ink, not page.
                out[region] = 0.0
            else:
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
            # halftone suppression, both directions (2026-07-28): dotted
            # textures inflate RAW distance (residue smuggled past the gate),
            # while dense ink neighborhoods inflate SMOOTHED distance (white
            # gaps between small-type letters escape cleaning). Membership
            # takes the MIN of both so page-colored pixels always qualify;
            # the assigned value uses the pixel's OWN distance so real ink
            # dots inside a hazy region keep their opacity.
            # Ideogram alignment, round 3 (2026-07-28, fitted pixel-by-pixel
            # on the CHEESE duel): in ATMOSPHERE (halftone texture fields —
            # smoke, bursts, speed lines) the reference alpha is a plane in
            # (ink distance, ink coverage):
            #     alpha = 0.845*(dist/2*scale) + 0.205*coverage - 0.01
            # (mean |err| 0.065). Whites get ~0 REGARDLESS of coverage — the
            # bright-burst look comes from dense ray INK, never from opaque
            # page white. Atmosphere = neighborhoods containing page-white
            # gaps (page_frac > 0.10); solid art (dog body, cheese, letters)
            # has no gaps and is untouched.
            pdist = np.linalg.norm(rgb - page, axis=-1)
            ink_frac = ndimage.uniform_filter(
                (pdist > haze_ink_gate * haze_scale).astype(np.float32),
                size=max(9, int(min(h, w) * haze_cov_radius)))
            atm_curve = np.clip(0.845 * np.clip(pdist / (2.0 * haze_scale), 0.0, 1.0)
                                + 0.205 * ink_frac - 0.01, 0.0, 1.0)
            # atmosphere = HALFTONE TEXTURE: ink and page gaps interleaved at
            # dot scale (fine window catches the dot spacing). A solid white
            # ELEMENT (daisy petal) is a large kept page-colored blob with no
            # ink inside — protected, the daisy-ghost regression of round 3.
            fine = max(7, int(min(h, w) * 0.008))
            ink_fine = ndimage.uniform_filter(
                (pdist > haze_ink_gate * haze_scale).astype(np.float32), size=fine)
            page_fine = ndimage.uniform_filter(
                (pdist < 0.3 * haze_scale).astype(np.float32), size=fine)
            # three atmosphere signatures: halftone texture (ink dots and
            # page gaps interleaved), smooth near-page veils (the original
            # white-residue case), and smooth NEUTRAL gray washes out to
            # the full ink scale (the flank smoke at distance ~100 that
            # matches neither — 2026-07-28). Solid white ELEMENTS match
            # the veil signature too and are carved out by the blob rule.
            chroma = rgb.max(-1) - rgb.min(-1)
            atmosphere = ((ink_fine > 0.10) & (page_fine > 0.10)) \
                | (pdist < 0.5 * haze_scale) \
                | ((pdist < haze_scale) & (chroma < 25.0))
            wlab, wn = ndimage.label(pdist < 0.3 * haze_scale)
            if wn:
                wsizes = ndimage.sum(np.ones_like(pdist), wlab, range(1, wn + 1))
                kept_frac = ndimage.mean((out > 0.5).astype(np.float32),
                                         wlab, range(1, wn + 1))
                inkiness = ndimage.mean(ink_fine, wlab, range(1, wn + 1))
                # kept_frac ~1.0 separates elements from veils: an ink
                # contour SEALS a petal blob away from the page, so every
                # pixel of it is kept; a veil blob is continuous with the
                # removed page and dilutes far below that. SOLIDITY seals
                # the remaining leak (2026-07-28 "nasıl farketmiyorsun"):
                # a petal is a hole-free slab, while the milky white mesh
                # of a sparse halftone field is riddled with dot holes —
                # that mesh is atmosphere and must melt.
                for wi in np.nonzero((wsizes > 0.0012 * h * w)
                                     & (kept_frac > 0.9) & (inkiness < 0.25))[0]:
                    blob = wlab == (wi + 1)
                    filled = ndimage.binary_fill_holes(blob)
                    if blob.sum() >= 0.92 * filled.sum():
                        atmosphere &= ~blob
            # interior-locked exemption, DARK-RING edition (2026-07-28
            # "sadece onlar değil"): stipple shading INSIDE solid dark art
            # (dog-nose highlights, ring-dark 0.61-0.66, Ideogram keeps at
            # 0.96+) is exempt; an enclosed smoke pocket sits in mixed
            # texture (ring-dark 0.39, Ideogram melts to 0.39) and melts
            # like any draining region. Measured separation on the CHEESE
            # duel — the only local signal out of four tried that matches
            # Ideogram's keep/melt decision.
            # CHROMA is the through-line of the duel data (2026-07-28):
            # everything Ideogram melts is NEUTRAL (gray smoke chroma ~2,
            # white pockets ~12); everything it keeps is chromatic (burst
            # core 122, pink glow 63) or sealed in dark art (nose stipple,
            # dark-ring 0.61+). Melt neutral pixels only; exempt locked
            # regions ringed by dark OR chromatic ink.
            # the WHITES law of the duel data (2026-07-28, binned table):
            # for near-page pixels Ideogram's alpha is a smooth ramp in
            # PIXEL CHROMA (0-10 -> 0.05, 25-45 -> 0.55, 70+ -> 0.91) —
            # tinted paper IS an ink wash. But the ramp holds only INSIDE
            # luminous elements: a faint warm white in the outskirts melts
            # like neutral haze, so the ramp is gated on a neighborhood of
            # genuinely saturated ink (burst rays, pink glow, rainbow).
            sat_ink = ndimage.uniform_filter(
                ((chroma >= 55.0) & (pdist > 0.5 * haze_scale))
                .astype(np.float32), size=max(9, int(min(h, w) * 0.015)))
            ramp = np.clip(chroma / 70.0, 0.0, 1.0) ** 0.85
            atm_curve = np.maximum(atm_curve, np.where(sat_ink > 0.15, ramp, 0.0))
            pagey = atmosphere & (out > 0.05)
            plab, pn = ndimage.label(pagey)
            boundary = keep & ~ndimage.binary_erosion(keep, iterations=3)
            rw = max(9, int(min(h, w) * 0.03))
            for i in range(1, pn + 1):
                region = plab == i
                if not (region & boundary).any() and \
                        region.sum() >= 0.0008 * h * w:
                    # only SIZABLE sealed pockets earn the exemption: a
                    # nose-highlight is 0.1-0.2% of the canvas; the small
                    # smoke curls hugging the dark silhouette are haze
                    ring = ndimage.binary_dilation(region, iterations=rw) & ~region
                    if ring.any() and float(
                            (pdist[ring] > 1.9 * haze_scale).mean()) >= 0.5:
                        continue    # sealed inside solid dark art (nose stipple)
                # PAGE-CONTACT ANCHOR (the carved-face lesson, 2026-07-30):
                # "drains to the boundary band" is topology-fragile — one
                # flipped edge pixel connected face-interior highlights to
                # the boundary and the melt carved a forehead. Atmosphere by
                # definition bleeds into the REMOVED page, so melt only
                # regions whose ring actually touches removed pixels. Sealed
                # whites (eye whites, skin highlights, stroke-sealed gloves)
                # never do, whatever the region graph says.
                ring = ndimage.binary_dilation(region, iterations=3) & ~region
                if not ring.any() or float((out[ring] < 0.05).mean()) < 0.15:
                    continue
                if subject_mask is not None:
                    sel = region & ~subject_mask.astype(bool)   # referee: subject pixels never melt
                else:
                    sel = region
                out[sel] = np.minimum(out[sel], atm_curve[sel].astype(np.float32))
    return np.clip(out, 0.0, 1.0)


def decontaminate(rgb: np.ndarray, alpha: np.ndarray,
                  page: np.ndarray | None = None,
                  min_alpha: float = 0.2) -> np.ndarray:
    """Foreground color recovery (the milky-whites lesson, 2026-07-30):
    a semi-transparent pixel still carries the PAGE color mixed in — on a
    dark garment it renders as milk. Invert pixel = a*fg + (1-a)*page to
    recover fg; Ideogram ships decontaminated color, which is half of why
    its smoke reads dark instead of milky. `page` defaults to the median
    color of removed pixels. Returns a new RGB float array."""
    rgb = rgb.astype(np.float32)
    a = alpha.astype(np.float32)
    if page is None:
        removed = a < 0.05
        page = (np.median(rgb[removed].reshape(-1, 3), axis=0)
                if removed.sum() > 500 else np.float32([255.0, 255.0, 255.0]))
    fg = rgb.copy()
    sel = (a > 0.05) & (a < 0.95)
    aa = a[sel][:, None]
    fg[sel] = np.clip((rgb[sel] - (1.0 - aa) * page) / np.maximum(aa, min_alpha),
                      0.0, 255.0)
    return fg
