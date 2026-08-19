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

Priorities are `identity | structural | secondary | micro | inferred`.
Every non-inferred group is required by default. The checker verifies family
tags and construction methods, primary/macro prior coverage, feature patterns
and counts, complexity-band consistency, identity-group minimums, occupied
region coverage, and adaptive detail floors. Use only the object-centric 3×3
region names from §3. A moderate/complex/extreme plan must cover at least
3/5/6 occupied regions respectively, and every declared occupied region needs
a required visible feature group. These gates prove plan compliance, not that
the visual evidence was interpreted correctly; inspection must still compare
the saved reference.

## 5. Pass the reconstruction probe

For every image-conditioned asset, build a neutral `form_probe.py` containing
every structural mass listed in `shape_priors` (primary plus necessary
secondary structure), important negative spaces, contact parts, and pose/joint
markers. For rectilinear/mixed targets this is also the primitive-family probe;
for curved targets it remains the section/continuity probe.

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
