# Image analysis — agent-vision perception stage

ProcAgen3D's perception stages (depth/normal/edge priors, ViT features, material
verification) are replaced in this skill by *your* structured reading of the
reference image. The outputs are unchanged copies of the used references at
`<out>/reference_NN.<ext>` plus `<out>/priors.md`, all written **before any
code**. Unstructured glancing produces fused parts and wrong proportions —
the checklist below is the stage; do it in order, against the saved copies.

## 0. Preserve the inputs

Create `<out>` and copy every image used to inform the asset byte-for-byte to
the output root. Name the copies `reference_01.<ext>`, `reference_02.<ext>`,
etc. in the order supplied, preserving each image's format and extension.
Record the original filename, URL, or attachment label for every saved copy in
the `references` table in `priors.md`. Analyze these saved copies from this
point onward so the asset remains reproducible when the original prompt or
attachment is unavailable.

If the runtime exposes only a decoded or resized representation rather than
the original bytes, save the highest-resolution representation available with
the correct extension and disclose the substitution in the table's notes.

## 1. Identity and scale anchor

Images carry no absolute scale. Name the object class and anchor real-world
dimensions from class knowledge (a city bike wheel ≈ 0.66 m; a dining chair
seat ≈ 0.45 m high; a mug ≈ 0.1 m). State the anchor explicitly:
`scale anchor: wheel diameter 0.66 m (assumed 700C city wheel)`. Every other
dimension derives from ratios measured against this anchor.

## 2. Part inventory (part-coverage ground truth)

List every visually distinct component with counts — this becomes the design
part table and the standard your own inspection will judge against later.
Count repeated instances exactly (spokes, bolts, drawers, slats: count them
in the image, don't guess "many"). Note parts that are implied but occluded
(the far pedal, the fourth chair leg) and mark them `inferred`.

## 3. Proportions (depth/edge prior)

Measure ratios off the image, not from memory: total height : width : depth;
each major part's extent relative to the anchor. Describe the silhouette per
visible view in words ("side view: two equal circles, centers ~1.6 wheel
diameters apart; seat above rear wheel center"). Note characteristic angles
(fork rake, roof slope, leg splay) in degrees.

## 4. Symmetry and structure

Declare the symmetry the build should exploit: bilateral (vehicles,
furniture, characters), radial with count (stool legs, bolt circles, wheel
spokes), or none. Unseen sides: state the symmetry assumption used to
complete them (`back face: mirrored from front, inferred`).

## 5. Material palette (verified colors)

Per part family, sample the color from a *mid-tone* region — not from
highlights or shadow — and cross-check the same material at two spots in the
image; if they disagree wildly, the surface is reflective/textured: record
the base tone and say `glossy` / `textured`. Output one family per distinct
real-world finish with RGB estimates and roughness/metallic guesses — 3–8
for standard tier, 12+ for showcase (split two-tone paint, lens colors,
chrome vs satin trim; see detail.md §Materials).

## 6. Articulation inference

Which parts plausibly move, where the pivot is, what the travel is: doors
(hinge edge), wheels (axle), lids (back edge), drawers (prismatic, depth of
travel). Only joints visible or strongly implied by the object class —
declare each with type, pivot location, axis, and a plausible limit range.

## 7. Detail inventory (fine-grained ground truth — showcase tier)

The part inventory (§2) captures what the object *is*; this section captures
what makes it *fine-grained*. Sweep the image as a 3×3 grid so no region is
skipped, and in each region enumerate every sub-feature with an approximate
size: seams and panel gaps, trim strips, badges/lettering (transcribe the
text), grille slats (count), tread pattern, rivets/bolts/handles, stripes
and graphics (with edge positions), vents, wipers, hinges. Decompose to the
ladder in detail.md: a "mirror" is housing + stem + glass; a "wheel" is
casing + tread + rim + lip + holes + hub + lugs.

Then name 3–5 **identity features** — the details a viewer would use to
recognize *this specific object* (the KC roof lamps, the tailgate lettering,
the tri-color stripe). A build that misses an identity feature fails
inspection even when every global verdict passes.

## priors.md template

```markdown
# Priors — <object>
scale anchor: <part> = <value> m (<reason>)
overall: H x W x D ≈ <..> m (front faces -Y)

## references
| saved copy | original source | notes |
|------------|-----------------|-------|
| reference_01.jpg | <filename, URL, or attachment label> | exact copy |

## parts
| part | count | key dims (m) | notes |
|------|-------|--------------|-------|
| Wheel_Front | 1 | d 0.66, w 0.035 | 32 spokes visible |
| ... | | | inferred: far pedal |

## proportions & silhouette
- side: <...>  front: <...>
- symmetry: bilateral (x=0); bolt circle radial x8

## materials
| family | rgb | rough/metal | applies to |
|--------|-----|-------------|-----------|

## joints
| joint | type | pivot | axis | limits |

## detail inventory (per 3x3 region; showcase tier)
| region | sub-feature | count | approx size | notes |
|--------|------------|-------|-------------|-------|

identity features: <3-5 bullets>

## uncertainties
- <anything the image does not settle — state the assumption chosen>
```

Keep priors honest: an `uncertainties` entry beats a confident invention.
When the image contradicts the text request, surface the conflict instead of
silently picking one.
