# Image analysis — agent-vision perception stage

Use structured agent vision for semantic judgment and registered image-fit
gates for numeric projection evidence. Write unchanged reference copies,
`<out>/priors.md`, `<out>/reconstruction_plan.json`, and version-2
`<out>/fit_spec.json` **before any code**. Read `reconstruction-planning.md`
and `image-fit.md` completely and measure against the saved copies. Optional
learned depth/normal/feature priors may supplement this stage later; they never
replace the registered shape, pose, mask, landmark, ratio, and layout contract.
Unstructured glancing produces fused parts, wrong primitive families, and
foreshortening baked into dimensions.

## Contents

0. Preserve inputs
1. Identity, scale, frame, and pose
2. Parts and complexity
3. Proportions, primitive priors, and curved-form blueprint
4. Symmetry and structure
5. Materials
6. Articulation
7. Detail inventory and priors template

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

## 1b. Canonical frame, camera, and pose

Before measuring proportions, declare canonical front/up/right and
longitudinal axes, symmetry plane, and contact convention. Then separate:

- registered camera azimuth/elevation/roll and projection;
- any evidence-backed rigid lean of the whole object;
- every visible articulated chain as ordered joint centers.

Use long edges, visible planar faces, circles/ellipses, symmetry, and contact
evidence—not only the overall bbox. In a floating single-object image, keep
geometry canonical and explain rigid view tilt with the camera unless support
evidence proves a physical lean. Build articulated pose with joint/assembly
transforms; do not change dimensions to imitate a bend. Put frame axes and
chains in version-2 `fit_spec.json`.

## 2. Part inventory (part-coverage ground truth)

List every visually distinct component with counts — this becomes the design
part table and the standard your own inspection will judge against later.
Count repeated instances exactly (spokes, bolts, drawers, slats: count them
in the image, don't guess "many"). Note parts that are implied but occluded
(the far pedal, the fourth chair leg) and mark them `inferred`.

Also count visible **feature groups** for complexity, treating a repeated array
as one group with an instance count. Classify `simple | moderate | complex |
extreme` from `reconstruction-planning.md`; object category and triangle count
are not complexity evidence. Mark the occupied cells of an object-centric 3×3
grid in `complexity.occupied_regions`; this is the coverage contract for later
detail work.

## 3. Proportions (depth/edge prior)

Only after camera/pose decomposition, measure ratios off the image, not from
memory: total height : width : depth;
each major part's extent relative to the anchor. Describe the silhouette per
visible view in words ("side view: two equal circles, centers ~1.6 wheel
diameters apart; seat above rear wheel center"). Note characteristic angles
(fork rake, roof slope, leg splay) in degrees. Put 6–12 identity-bearing
landmarks and the ratios between them in `fit_spec.json`; prose alone is not a
measured prior.

For a multi-object reference, add an instance table with normalized image
bboxes/centroids and pairwise layout/depth relations. Do not collapse a scene
to one overall bounding box or give every independent object the same facing
direction. Details and schema: `image-fit.md`.

## 3a. Primitive-family priors

For every silhouette-bearing macro/meso mass, compare at least two candidate
families. Record straight/parallel edges, planar fields, section
constancy/change, curvature across faces, and edge treatment. Choose the
simplest supported family and state the rejected alternative.

A rounded corner does not make a body an ellipsoid. A camera body, receiver,
housing, or faceted armor panel with broad planar fields remains a box/prism
plus bevel/chamfer. Use ellipsoid/capsule only when curvature continues across
the face; use loft only when 3+ measured sections change. Write decisions to
`reconstruction_plan.json` and tag meshes with `procagen3d_shape_family`.

## 3b. Form blueprint (curved or mixed targets)

Classify the overall form profile as `rectilinear | curved | mixed`. If it is
`curved` or `mixed`, read `complex-forms.md` now and make primary geometry
measurable before choosing helpers:

- Reuse the already-solved reference camera and pose; distinguish perspective
  convergence or joint rotation from real taper. Mark every camera/pose value
  approximate unless calibrated.
- Name 3–5 **macro identity forms** (continuous roof arc, rear-haunch swell,
  thigh-to-knee pinch) separately from badges/colors in §7.
- Trace 6–12 normalized `(u,v)` landmarks along each visible identity contour
  and state the image-to-world mapping used.
- For each primary mass, declare `topology`, `method`, longitudinal axis, and
  5–12 cross-section/spine stations. Each station records center, width,
  bottom/top or depth, twist, evidence view, and confidence.
- List negative spaces and surface relationships: openings, undercuts, limb
  gaps, parent surface, attachment band, smooth transition versus hard seam.
- Mark hidden sections `inferred`; constrain them by symmetry/class priors
  without claiming reference verification.

A varying-section target built from a deformed box or overlapping ellipsoids is
a representation error. The converse is equally important: a constant-section
planar/prismatic target rebuilt as a loft or ellipsoid is also a representation
error. The primitive prior and form blueprint jointly ground the probe.

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
real-world finish with RGB estimates and roughness/metallic guesses. Meet the
complexity-adaptive floor in `detail.md`; split two-tone paint, lens colors,
chrome, and satin trim into their real finishes.

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

Convert the inventory to `reconstruction_plan.json`: one entry per visible
feature group with a semantic name pattern, `min_count`, priority, and
object-centric 3×3 region. Arrays remain one group. Make every non-inferred
group required, cover every declared occupied region, and schedule hidden
inferred work last.

## priors.md template

```markdown
# Priors — <object>
scale anchor: <part> = <value> m (<reason>)
overall: H x W x D ≈ <..> m (front faces -Y)
form profile: <rectilinear | curved | mixed>
complexity: <simple | moderate | complex | extreme> (<N> visible feature groups)
canonical frame: front <axis>, up <axis>, right <axis>; symmetry <...>
reference camera: <projection; azimuth, elevation; FOV or ortho-scale estimate>
reference pose: <rigid root pose + articulated chains; inferred items marked>
fit contract: fit_spec.json v2 (camera, local silhouette, pose, landmarks/ratios)
reconstruction contract: reconstruction_plan.json (shape priors + detail groups)

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

## primitive-family priors
| mass | chosen family / edge treatment | evidence | rejected alternative | confidence |
|------|--------------------------------|----------|----------------------|------------|
| Body | box + 2 mm bevel | planar face, parallel rails | ellipsoid: no face curvature | high |

## pose
| frame axis / chain | ordered landmarks | reference angles/bends | confidence |
|--------------------|-------------------|------------------------|------------|

## form blueprint (curved/mixed)
macro identity forms: <3-5 bullets>

| primary mass | topology / method | axis | stations or outline | evidence / confidence |
|--------------|-------------------|------|---------------------|-----------------------|
| BodyShell | continuous / loft | Y | t=0: center,width,bottom,crown; ... | ref 01 side+top / medium |

negative spaces & transitions: <openings, undercuts, attachment bands, seams>

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
