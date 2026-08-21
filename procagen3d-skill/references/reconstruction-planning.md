# Image reconstruction planning

Read this for every image-conditioned asset before estimating dimensions or
choosing geometry. Produce `reconstruction_plan.json` before any Blender code.
It is the machine-readable contract for primitive family, reference pose, and
complexity-adaptive detail.

## Contents

1. Solve frame and pose before shape
2. Select the simplest supported shape family
3. Classify perceptual complexity
4. Author the reconstruction plan
5. Pass the reconstruction probe

## 1. Solve frame and pose before shape

A 2D tilt is not a 3D taper. Perspective foreshortening is not a short part.
Separate four things before measuring proportions:

1. **Canonical object frame:** state front, up, right, longitudinal axes, the
   symmetry plane, and the ground/contact convention. Keep load-bearing
   dimensions in this frame.
2. **Registered camera:** estimate projection, azimuth, elevation, roll, FOV
   or orthographic scale, target, and image shift. Use long parallel edges,
   planar face visibility, circles/ellipses, and symmetry—not object bbox alone.
3. **Rigid reference pose:** record any physical lean/rotation of the complete
   object. In a single unsupported product image, camera rotation and rigid
   object rotation are gauge-equivalent; keep the asset canonical and assign
   the relative view rotation to the camera unless contact/support evidence
   proves a real lean.
4. **Articulated pose:** trace every visible kinematic chain as ordered joint
   centers: shoulder→elbow→wrist, hip→knee→ankle→toe, boom root→hinge→tip.
   Record left and right chains separately. Build the reference pose through
   assembly/joint transforms; never change limb dimensions to imitate a bend.

Estimate dimensions only after this decomposition. Measure along object axes
or with pose-corrected ratios. Mark depth and occluded links inferred when one
view cannot determine them.

### Scenes: separate objects share a floor, not a volume

When the reference holds several distinct objects, declare each as an
`instances` entry in `fit_spec.json`. That declaration is what turns on
`SCENE_INTERPENETRATION`, which measures how deeply one declared instance is
buried inside another by containment rather than by bounding box. A lamp
standing on a side table measures about 0%; a lamp sunk into a sofa back
measures its true depth. Over 3% warns, over 10% fails.

The scoping matters both ways: without instance declarations the check stays
off, because the sub-assemblies of one articulated body are *supposed* to
overlap — a shoulder inside its socket is correct. A deliberate overlap between
two real instances goes in `"allowed_intersections": [{"a": ..., "b": ...}]`.

Place scene objects by their contact and facing, not by nudging each until its
projected box lands in the right place. A chair rotated 65° away from the table
it should face keeps almost the same footprint from one camera, so the bounding
box tells you nothing — its silhouette region and the camera solve are what
notice.

### Everything must be joined to something

`DETACHED_PARTS` treats the asset as one connected solid: any mesh island
separated from the rest by more than 1% of the object's size is a floating
part. This catches what a single registered view structurally cannot — a head
with no neck under it projects exactly like a head with one, and the gap only
appears when you look from somewhere else.

The fix is almost always a missing connector rather than a wrong position:
build the neck, the stem, the mount, the axle. Pieces that really are separate
in the reference go in `"detached_groups": ["FloatingBit_*"]`.

### Symmetry is where depth comes from

One view cannot tell you how far forward a shoulder sits. It can tell you that
the left one sits exactly as far forward as the right. For a bilateral subject
that single fact removes about half the free depth parameters, and those free
parameters are the whole reason a model can look correct from the reference
camera and be wrong in 3D.

So build both sides from one builder and one set of constants, with the side as
a sign:

```python
for side, sign in (("L", 1.0), ("R", -1.0)):
    hip = Vector((sign * HIP_HALF_SPAN, HIP_DEPTH, HIP_HEIGHT))
    build_leg(f"Thigh_{side}", hip, ...)
```

Never author `Thigh_L` and `Thigh_R` as two independent sets of coordinates.
The moment their depths are typed separately, one of them is wrong and the
front view will not show you which.

Declare the contract in the plan:

```json
"symmetry": {
  "plane": "x",
  "origin_m": 0.0,
  "tolerance": 0.015,
  "asymmetric": ["Shield_*", "Rifle_*"]
}
```

`SYMMETRY` pairs meshes by name (`Foo_L`/`Foo_R`, `Foo_Left`/`Foo_Right`,
`LeftFoo`/`RightFoo`) and compares each pair's mirrored centroid and size
against the object's own scale. It reports the mismatch *along* the mirror
plane separately, because that component is exactly the depth or height error
the reference camera cannot see. Genuinely one-sided equipment goes in
`asymmetric`; a subject that is really not bilateral declares `"plane": null`
with a reason. Once three or more left/right pairs exist, the contract is
required — silence is not an answer.

### Rigid equipment is authored from one axis, not from endpoints

A rifle, spear, mast, axle, or pipe run is a single straight object. Author it
as one origin plus one direction and derive every station along that line:

```python
RIFLE_AXIS = (RIFLE_MUZZLE - RIFLE_BUTT).normalized()
RIFLE_LENGTH = (RIFLE_MUZZLE - RIFLE_BUTT).length
def station(t):                      # t in 0..1 from butt to muzzle
    return RIFLE_BUTT + RIFLE_AXIS * (RIFLE_LENGTH * t)
receiver = box_between("Rifle_Receiver", station(0.00), station(0.26), ...)
barrel   = box_between("Rifle_Barrel",   station(0.26), station(1.00), ...)
```

Placing the butt, the receiver end, and the muzzle as three independent points
is how a straight weapon comes out bent. Each point gets nudged until it sits
right in the reference, and because one view cannot see depth, the depth
component of each is chosen freely — so the parts end up pointing in different
directions while the projection still looks correct. A real run produced a
receiver and barrel 22° apart, with the muzzle sitting 3.2 units off the
receiver's own axis on a 10.7-unit weapon, and it looked fine from the
reference camera.

`RIGID_AXIS` finds every long assembly — grouped by transform parent, measured
from the parts' own axes so a diagonal rod is not mistaken for a compact one —
whose parts point more than 8° apart, and requires a stated decision for each.
A radial array such as wheel spokes reads as a disc and is never asked.

```json
"rigid_axes": [
  {"pattern": "Rifle_*", "rigid": true, "max_deviation_deg": 5},
  {"pattern": "Leg_L_*", "rigid": false, "reason": "knee joint bends in the reference"}
]
```

`"rigid": true` holds the parts to one axis and fails past the tolerance.
`"rigid": false` exempts a genuine articulated chain and requires a reason
naming the joint. Leaving a bent assembly undeclared is itself a failure: the
question is not whether you want to answer it, it is which answer is true. A
weapon, mast, or axle that bends has no true "articulated" answer.

Use `fit_spec.json` version 2. Its `pose.frame_axes` gates projected object-axis
directions; `pose.chains` gates segment directions, bend angles, and normalized
segment lengths. Add at least three tight `silhouette_regions` around
primitive-family decisions and pose-sensitive limbs. A whole-object mask can
pass while every local part is wrong.

## 2. Select the simplest supported shape family

For every silhouette-bearing macro/meso mass, compare at least two candidate
families and choose the lowest-degree family consistent with visible evidence.
Do not choose by object category, material, or the word “rounded.”

| evidence | preferred family | reject unless separately evidenced |
|---|---|---|
| broad planar fields; two or more long straight/parallel edge pairs; nearly constant section | `box` or `prism` + bevel/chamfer | ellipsoid, capsule, free loft |
| constant circular/oval section along a straight axis | `cylinder` or `revolve` | free loft |
| explicit axial radius profile | `revolve` | stacked spheres/cylinders |
| measured section width/depth changes at 3+ stations | `loft` | box stretched to fake taper |
| curved spine with changing section | `sweep` | straight extrusion |
| thin surface following a parent | `shell` / `surface-grid` | solid blob or floating flat card |
| curvature across the complete visible face with no planar field | `sphere`, `ellipsoid`, or `capsule` | rounded box used as a proxy |

Edge treatment and volume family are independent. A camera body, receiver,
housing, battery, or armor plate can have softened edges and remain a box or
prism. Bevel only the boundary. An ellipsoid is justified only when curvature
continues across the face, not because corners are round or highlights are soft.

Record for each mass:

- supporting lines/faces/sections and evidence view;
- family, edge treatment, and confidence;
- the strongest **different** rejected alternative and why it loses (the chosen
  family may never also appear as rejected);
- hidden-axis assumptions separately from visible-family evidence.

Mixed form means evidence-backed coexistence, not a percentage quota. A camera
may be mostly boxes/prisms plus revolved lens parts; a manufactured rifle may be
mostly prisms plus a truly curved grip; a mecha may use faceted armor shells and
small capsule-like joint cores. Never convert correct blocks into lofts merely
to make a `mixed` check pass.

No gate asks for curved mass. The checks run the other way: `FORM_PROMISED_CURVATURE`
fails masses you declared `continuous`/`shell` that were built as flat-sided
blocks, and `FORM_PROFILE_EVIDENCE` fires only when a `curved` profile has
almost no measurable curvature anywhere — and its remedy is to declare
`rectilinear` or `mixed`, not to inflate correct blocks. If your object is a
faceted, assembled thing, say so; boxes are a valid answer and the tool will
never push you off them.

Tag every planned structural mesh so the checker can compare program to plan:

```python
def mark_form(obj, role, topology, method, family, section_count=None):
    obj["procagen3d_form_role"] = role
    obj["procagen3d_topology"] = topology
    obj["procagen3d_form_method"] = method
    obj["procagen3d_shape_family"] = family
    if section_count is not None:
        obj["procagen3d_section_count"] = int(section_count)
    return obj
```

Use `method="analytic-primitive"` for a genuine sphere/ellipsoid/capsule.
Boxes/prisms use `primitive-csg`, `profile-extrude`, or `boolean`; a plan that
says `box` cannot be implemented with `loft`.

### The tag is checked against the geometry, not just against the plan

Every mesh is measured at build time and the signature is stored in
`scene_graph.json`. `check` fails on `SHAPE_FAMILY_MEASURED` when the declared
family contradicts what was actually built, so tagging a UV sphere `box` buys
nothing. Two numbers do most of the work:

- **`fill_ratio`** — solid volume over local bounding-box volume;
- **`planar_area_fraction`** — surface area in the six largest coplanar
  clusters, i.e. "does this thing have broad flat sides?"

Measured on clean primitives:

| built shape | `fill_ratio` | `planar_area_fraction` | `section_variation` |
|---|---:|---:|---:|
| plain box | 1.00 | 1.00 | 0.00 |
| bevelled box (6% radius) | 0.99 | 0.82 | 0.00 |
| heavily rounded box (22%) | 0.89 | 0.51 | 0.05 |
| hexagonal prism | 0.75 | 0.76 | 0.00 |
| cylinder | 0.78 | 0.35 | 0.00 |
| cone | 0.26 | 0.37 | 1.50 |
| sphere | 0.52 | 0.03 | 0.41 |
| ellipsoid | 0.52 | 0.08 | 0.41 |
| tapered loft | 0.41 | 0.32 | 0.64 |

Read the middle column again: **bevelling a box barely moves it, and rounding
it hard still leaves it at 0.51 against an ellipsoid's 0.08.** Softened corners
are not curvature. If a mass measures above 0.40 there, it has flat sides and
belongs to `box` or `prism`, whatever the highlights suggest.

`section_variation` is the prism/loft discriminator: it is zero for anything
with a constant section, so a straight extrusion of a flat profile cannot be
passed off as a `loft`.

## 3. Classify perceptual complexity

Count **visible feature groups**, not repeated instances, source-code lines, or
triangles. Thirty-two spokes are one group with `min_count: 32`; distinct
helmet vents, cheek plates, knee insets, and calf shells are separate groups.
Classify from the reference inventory:

| class | visible groups | typical evidence |
|---|---:|---|
| `simple` | 1–7 | few clean masses, sparse surface language, repeated arrays dominate |
| `moderate` | 8–15 | several assemblies/material breaks, limited panel language |
| `complex` | 16–27 | dense vehicle/mecha panels, many identity-bearing regions or chains |
| `extreme` | 28+ | layered armor/bodywork, equipment, occlusion, many independent identity clusters |

Complexity is reference-specific. A clean quadcopter or city bicycle can be
simple even when spokes/propellers create many mesh instances. A car or mecha
with layered panels, vents, joints, and equipment is usually complex/extreme.

For showcase work, inventory object-centric 3×3 regions before synthesis.
Each non-inferred feature group gets a semantic mesh-name pattern and minimum
count. Use exactly `top-left | top-center | top-right`, `middle-left |
middle-center | middle-right`, and `bottom-left | bottom-center |
bottom-right`. Patterns must be distinct and contain a semantic literal; `*`
is not a feature contract. Build in this order:

1. silhouette and reference pose;
2. identity and structural feature groups in every occupied region;
3. meso panel breaks, joints, and material boundaries;
4. repeated microdetail;
5. hidden inferred detail last.

Never spend the budget on dense knurling, spokes, or bolts while a complex
torso, car flank, head, limb, or cabin remains a single undifferentiated mass.

## 4. Author the reconstruction plan

Write `<out>/reconstruction_plan.json` before code:

```json
{
  "version": 1,
  "camera_solve": {
    "locked": true,
    "candidates_tested": [
      "az 28 el 12 fov 34: chosen — lens ellipse axis ratio 0.62 matches, top plate barely visible",
      "az 40 el 20 fov 34: rejected — shows far too much of the top plate and over-widens the grip"
    ],
    "camera": {
      "projection": "perspective",
      "azimuth_deg": 28.0, "elevation_deg": 12.0, "roll_deg": -3.0,
      "fov_y_deg": 34.0, "distance_m": 2.8, "target_m": [0.0, 0.0, 0.45]
    }
  },
  "complexity": {
    "class": "complex",
    "occupied_regions": [
      "top-center", "middle-left", "middle-center",
      "middle-right", "bottom-center"
    ],
    "drivers": [
      "18 visible feature groups across five occupied regions",
      "four articulated limb chains",
      "layered faceted armor over exposed joint cores"
    ]
  },
  "shape_priors": [
    {
      "id": "receiver_core",
      "pattern": "ReceiverCore",
      "family": "prism",
      "edge_treatment": "2 mm bevel plus explicit chamfers",
      "confidence": "high",
      "evidence": ["two long parallel rails and a planar broadside"],
      "rejected_alternatives": ["ellipsoid: no curvature across the side face"]
    },
    {
      "id": "grip_shells",
      "pattern": "GripShell_*",
      "family": "loft",
      "edge_treatment": "hard seam at receiver attachment",
      "confidence": "medium",
      "evidence": ["width swells at palm and pinches at toe"],
      "rejected_alternatives": ["box: measured section changes at four stations"]
    }
  ],
  "detail_features": [
    {
      "id": "receiver_panel_breaks",
      "pattern": "ReceiverPanel_*",
      "min_count": 6,
      "priority": "identity",
      "region": "middle-center"
    },
    {
      "id": "hidden_back_fasteners",
      "pattern": "BackFastener_*",
      "min_count": 2,
      "priority": "inferred",
      "region": "hidden-back",
      "required": false
    }
  ]
}
```

The example is abbreviated: a `complex` plan must actually list all 16–27
visible groups, not merely claim 18 in `drivers`.

`camera_solve` is mandatory: name at least two viewpoints you actually compared
and why the loser lost, then lock the winner. `check` compares `fit_spec.camera`
against this lock and fails on `CAMERA_LOCK` if they diverge. Solve the camera
once; do not tune it later to rescue a fit.

Priorities are `identity | structural | secondary | micro | inferred`.
Every non-inferred group is required by default. The checker verifies family
tags and construction methods, primary/macro prior coverage, feature patterns
and counts, complexity-band consistency, identity-group minimums, occupied
region coverage, and adaptive detail floors. Use only the object-centric 3×3
region names from §3. A moderate/complex/extreme plan must cover at least
3/5/6 occupied regions respectively, and every declared occupied region needs
a required visible feature group.

Name-pattern coverage proves only that meshes were named after the plan, so it
is backed by a measurement. `REGION_DENSITY` bins every mesh by where its
centre actually sits in the object's 3×3 grid and requires 2/4/8/12 meshes in
each declared occupied region for simple/moderate/complex/extreme. This is what
makes "the torso is one undifferentiated blob" a machine-detectable failure:
two hundred panel slivers on the legs cannot pay for an empty chest.

Repeated parts have a congruence contract. `INSTANCE_CONGRUENCE` groups meshes
by `Name_NN` and compares volume and surface area — both rotation-invariant, so
a builder that bakes each instance's rotation into its vertices still reads as
congruent. Eighteen swords fanned across a back are eighteen calls to one
`build_sword()` differing only in transform, not eighteen shapes eyeballed into
a silhouette.

Arrays whose members genuinely differ — seams tracing panel edges of different
lengths, per-glyph letter blocks, a stack of trim rings — declare it once:

```json
"instance_arrays": [
  {"pattern": "WingBlade_*", "congruent": true, "tolerance": 0.05},
  {"pattern": "BodySeam_*", "congruent": false,
   "reason": "each seam traces a different panel edge"}
]
```

Undeclared arrays warn above 8% spread, and fail above 25% once they reach six
members: at that size an unexplained inconsistency is drift, and the fix is
either one shared builder or one line saying why not.

These gates prove plan compliance and geometric honesty, not that the visual
evidence was interpreted correctly; inspection must still compare the saved
reference.

## 5. Pass the reconstruction probe

For every image-conditioned asset, build a neutral `form_probe.py` containing
every structural mass listed in `shape_priors` (primary plus necessary
secondary structure), important negative spaces, contact parts, and pose/joint
markers. For rectilinear/mixed targets this is also the primitive-family probe;
for curved targets it remains the section/continuity probe.

Correction budget: **2** for simple/moderate, **3** for complex, **4** for
extreme. On exhaustion with two or more reference views, stop — the evidence
was sufficient, so the reconstruction is what is wrong. On exhaustion with a
single view, continue on the best probe and deliver it as approximate, with
`limitations.md` naming every failing gate and the views that would resolve it.

Run `build`, registered `fit`, and `check` on the probe. Read the canonical and
registered views. Pass only when:

- every planned mass looks like its family in all useful views;
- the camera/object frame explains tilt without deforming dimensions;
- all visible chains reproduce segment direction and bend;
- local silhouette regions pass at corners, tapers, limbs, and attachments;
- negative spaces and overlaps match.

If a family is wrong, revise the prior and rewrite that builder. If pose is
wrong, repair camera/root/joint transforms. Do not adjust dimensions or add
detail until those two layers pass.

## 6. Complex and extreme targets: probe one assembly at a time

A stool or a drone fits in one mental chunk, which is why single-pass
generation works for them and collapses on a car, a mecha, or a character. For
`complex` and `extreme` plans, do not write the whole object and hope the
aggregate score tells you what is wrong — it cannot. Work assembly by assembly:

1. Split the object into named assemblies (torso, head, each limb, equipment,
   bodywork zone). Five to nine is typical.
2. Give each one at least one `silhouette_regions` crop in `fit_spec.json`.
   `FIT_REGION_COVERAGE` requires roughly one region per two primary masses, so
   this falls out naturally.
3. Probe and register the assemblies in identity order, most recognisable
   first. Read the per-region rows in `fit_report.json` after every build; a
   failing region names the assembly that is wrong, which whole-frame IoU never
   does.
4. Only add detail to an assembly whose region already passes. Detail on a
   wrong mass is wasted work that the repair budget then has to protect.

Repair budgets scale with the same class: 3 iterations for `simple`/`moderate`,
6 for `complex`, 8 for `extreme`. A complex object legitimately needs more
passes; what it does not get is a lower bar.
