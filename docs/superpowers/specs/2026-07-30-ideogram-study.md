# Ideogram Study — 13-Artwork Duel Catalog (2026-07-30)

Method: 13 real design artworks (10 new + CHEESE/YH/rainbow) through
lucida-m95v18-35 + poster policy vs `fal-ai/ideogram/remove-background`,
side-by-side on dark checker, eye + pixel verdicts.

## Verdict table

| Class (artwork) | Winner | Why |
|---|---|---|
| Cream-on-cream stamp (Petersburg) | **US, decisive** | Ideogram ghosts the whole white stamp art — the Stay Fresh weakness confirmed twice |
| Distressed retro (Sunset) | **US, decisive** | Ideogram renders grunge texture as semi-density: muddy, holes in stripes |
| Vintage badge (AImpala) | **US** | Ideogram carves the engraved sky texture inside the badge |
| Cream engraving (astronaut) | **US** | Ideogram deletes the suit's cream whites as page (v17 fill lesson pays) |
| Colored page (HAPPY TIMES) | US, slight | Ideogram deletes the red arc fill inside the rainbow |
| Dark-page cartoon (Spooky) | parity | both correct incl. black cat on black page |
| Graffiti (BORN) | parity | |
| Photo-hybrid (dog) | parity | |
| White script on black (overthink) | **IDE** | it knows black IS the page between letters; our fix (dark flat pages → open pockets) closes most of the gap, giant enclosed surrounds remain |
| Watercolor animal (tiger) | **IDE, decisive** | white FUR belongs to the subject; our whites-melt eats the muzzle/chest |
| Gloves (YOU'RE HAPPY) | **IDE** | semantic white-limb knowledge; v18 got the right glove to raw 0.6 but poster can't finish |
| Halftone smoke (CHEESE) | IDE, slight | their dark smoke sits at 0.5-0.65; our fitted curve leaves milkier whites |
| Airbrush glow (rainbow) | IDE, slight | their glow stays full and soft; our color-proximity melt clips the fade |

Bottom line: on flat POD graphics (the im2lazy core) we now WIN or tie most
classes; Ideogram's remaining edge is one thing only — SEMANTIC knowledge of
what whites belong to the subject (fur, gloves, glow, smoke).

## Lessons to steal (v19)

1. **Subject-whites semantics** — the single theme behind every IDE win:
   training pairs with white-furred animals, white clothing/gloves/limbs,
   white props on white pages; GT keeps them solid. Real shapes, not blobs
   (v18's mistake: ellipse limbs ≠ four-finger gloves; probe was too easy —
   e17 aced synthetic 0.987 while scoring 0.06 on the real glove).
2. **Real-style halftone smoke** — dot fields sampled to match POD art
   (dense, figure-attached), density GT; v18's synthetic fields did not
   transfer (real smoke still raw 0.96-0.99).
3. **Airbrush glow fullness** — glow fade must survive; policy currently
   clips it by color proximity.
4. **Page-pocket removal generalizes to any flat page color** — DONE in
   policy (dark flat pages → open pockets, commit 87cf454); huge enclosed
   surrounds still capped by max_hole_frac.

## Ideogram weaknesses to keep beating (marketing angles)

- cream/tinted page + page-colored ink (Petersburg, Stay Fresh)
- distressed/grunge texture (Sunset)
- badge/engraving interiors (AImpala, astronaut)
