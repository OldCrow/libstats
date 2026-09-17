# libstats docs/UML style — "corvid cartography" (ratified 2026-09-17)

## Palette (from docs/branding/*.svg)
- ink backdrop      #161b23   (full-canvas background)
- band / panel      #232a36   (family bands, grouped regions, table panels)
- parchment card    #e7e0cf   (every class / node / step box)
- gold              #c8a24e   (band strokes, headings on ink, arrows, accents)
- ink text          #161b23   (text on parchment)
- muted on parchment #6b6455  (secondary text on cards)
- muted on ink      #9a9382   (secondary text on bands/ink)
- danger / error    #c0563f   (only for error paths, e.g. Result::error, contention)
- ok / fast path    #6f9a5a   (only for success / lock-free paths)
Never introduce other hues. White text is NOT used; gold or parchment carries emphasis on ink.

## Canvas
- viewBox width 860 (matches existing four), height as needed; rx=0 canvas rect fill #161b23.
- font-family="system-ui,-apple-system,Helvetica,Arial,sans-serif" on the root <g>.
- Title (top-left, gold, 15px bold, letter-spacing 0.5) + one-line subtitle (#9a9382, 10.5px).
- Footer bottom-right: "libstats v2.4.0  ·  docs/UML/<file>.svg" in #9a9382 9.5px.

## Boxes
- Card: rect fill #e7e0cf, stroke #c8a24e 1.2, rx 7. Title 11px bold ink; detail lines 9.5px ink; tag line 8.5px #6b6455.
- Band/region: rect fill #232a36, stroke #c8a24e 1.5, rx 12. Heading 11.5px bold gold UPPERCASE letter-spacing 0.5, top-left inside the band, 14px from top.
- Abstract/interface: same card, title in italic, stereotype line «interface» / «mixin» 8.5px #6b6455 above title.
- Minimum gap between any two boxes: 12px. Minimum gap between a box and a band edge: 12px.

## Arrows — the part that must be readable
Two relationship kinds, never ambiguous:
- INHERITANCE / "is-a": SOLID gold line 1.6px, HOLLOW TRIANGLE head (marker: polygon points "0 0,10 5,0 10" fill #161b23 stroke #c8a24e 1.4, markerWidth 12 markerHeight 12 refX 10 refY 5).
- DELEGATION / "calls into": DASHED gold line 1.6px, dasharray "5 3", OPEN V head (marker: path "M0 0 L10 5 L0 10" fill none stroke #c8a24e 1.6).
- DATA / control FLOW (pipeline diagrams): solid gold 1.6px, FILLED small triangle head (polygon "0 0,8 4,0 8" fill #c8a24e). Ok-path variant in #6f9a5a, error-path variant in #c0563f, same head shape.
Routing rules:
1. Orthogonal segments only (horizontal/vertical), 90° corners; never a diagonal.
2. An arrow never passes over or through any box it does not connect. Route around: leave the source box on a free side, run in a gutter (band margin or inter-box aisle), enter the target on a free side.
3. Every arrow has at least 28px of visible shaft before the head, so line style (solid vs dashed) is legible.
4. Arrow heads never overlap text. Enter a box at an edge midpoint or a clearly free point.
5. When several arrows share a corridor, offset them 6px apart; do not overlap shafts.
6. Label an arrow only when the kind is not obvious from the legend; labels 8.5px #9a9382 with a #161b23 halo rect behind.
Legend box (parchment card, bottom-left) on every diagram that uses ≥2 arrow kinds: one sample of each kind with its name.

## Text rules
- ABC: accurate, brief, clear. No marketing words.
- Every number in a diagram must come from the current tree (v2.4.0 / main): dispatch_thresholds.h for thresholds, HEADER_ARCHITECTURE_GUIDE.md for the roster, CHANGELOG for #143 wording.
- Use ≥ 8.5px text; nothing smaller.

## Decision outputs (added 2026-09-17 after review)
Every decision diamond's two outputs are coloured the same way in every
diagram, regardless of what the branches mean downstream: the AFFIRMATIVE
branch (yes / ok / fast path) is #6f9a5a, the NEGATIVE branch (no / error /
contention / fallback) is #c0563f, with the branch label in the same colour.
Plain gold is for unconditional flow only. The legend names both.
