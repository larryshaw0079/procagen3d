# Complex forms — curved, streamlined, and irregular surfaces

Use this reference for any target whose identity depends on changing
cross-sections, compound curvature, taper/twist, a continuous highlight, or
non-rectangular armor/panels. A sports-car body and biomechanical mecha are
form-dominant even though both are "hard surface." More parts, bevels, and
triangles cannot repair the wrong representation family.

## Contents

1. Route topology before geometry
2. Build the form blueprint
3. Pass the shape-first probe
4. Use constructive curved builders
5. Apply subject recipes
6. Inspect and repair form

## 1. Route topology before geometry

Classify every silhouette-bearing macro/meso part before choosing a helper:

| topology | visual meaning | use | do not use |
|----------|----------------|-----|------------|
| `continuous` | one volume with section changes and uninterrupted surface flow | `loft`, varying-profile `sweep`, `revolve`, authored `subdivision`, `surface-grid`, `nurbs` | box, deformed box, straight extrusion, overlapping ellipsoids |
| `shell` | thin panel following a parent volume | surface grid + Solidify, lofted skin, subdiv patch | floating flat plate across a curved parent |
| `assembled` | manufactured block/plate with intentional hard junctions | primitive CSG, polygon profile extrusion, boolean, revolved solid, bevel | pretending it is a blended body mass |
| `strand` | cable, horn, pipe, trim rail | curve or profile sweep | chain of capsules unless joints are visible |
| `relief` | seam, ridge, recess, badge, inset | shallow profile extrusion, curve, boolean, decal | counting it as macro form |

Declare the overall form profile as `rectilinear`, `curved`, or `mixed`.
Choose `curved` when continuous/shell forms dominate; choose `mixed` for
mecha and machinery that deliberately combine sculpted masses with assembled
solids. Run `check --form <profile>` later.

Set the same profile on the program root so plain `check` auto-detects it:

```python
root["procagen3d_form_profile"] = "curved"  # or "mixed" / "rectilinear"
```

For each structural macro/meso mass, record this row in the design header and
mark the built object with the same contract. Use `primary` for silhouette,
volume, and identity forms; use `secondary` for supports, contacts, and
deliberately assembled structure:

```python
from procagen3d_runtime import mark_form

mark_form(body, "primary", "continuous", "loft", section_count=8)
```

Use these exact method values: `loft`, `sweep`, `revolve`, `subdivision`,
`surface-grid`, `nurbs`, `curve`, `solidify`, `profile-extrude`,
`primitive-csg`, `boolean`, or `decal`. Tag every structural mass that enters
form review, but do not inflate coverage by tagging seams, fasteners, decals,
or trim. Those are detail, not structural form.

The checker audits valid contracts on the largest structural masses and uses
their bounding-envelope volume as a macro-form proxy. A small token loft does
not compensate for a body assembled from rounded boxes; legitimately blocky
mechanisms remain valid `assembled` structure in a mixed target.

## 2. Build the form blueprint

Do not infer a curve from adjectives. Measure a compact guide from the saved
reference:

1. Estimate and register the full reference camera in `fit_spec.json` as
   required by `image-fit.md`: projection, position/target (or
   azimuth/elevation/distance), roll, FOV or orthographic scale, and image
   shift. Optionally put the coarse three-value preview contract on the root:

   ```python
   root["procagen3d_reference_projection"] = "perspective"
   root["procagen3d_reference_camera"] = [azimuth_deg, elevation_deg, fov_deg]
   # Or: "orthographic" with [azimuth_deg, elevation_deg, ortho_scale_m].
   ```

   Treat that root property as a backward-compatible preview only: it
   auto-centers the asset and cannot prove registration. Run `procagen3d fit`
   for the scored `renders/reference_match.png`. Mark estimates approximate
   until the registered mask/landmark gates pass.
2. Trace 6–12 normalized `(u, v)` landmarks along each identity-bearing
   contour. Convert them to meters with one explicit image-to-world mapping.
3. For a loft, choose a longitudinal axis and 5–12 stations. At each station
   record center, half-width, bottom, shoulder/belt height, crown/top, local
   depth, twist, and the evidence view. Add stations at every curvature
   extremum, hard crease, opening boundary, and attachment.
4. For a polygon panel, record its ordered outline and thickness. For a sweep,
   record spine points plus width/height/twist per station. Small semantic
   control arrays are constructive parameters, not forbidden vertex soup.
5. Record negative spaces separately: wheel arches, limb-to-torso gaps,
   undercuts, vents, and joint clearances. A correct outer AABB can still hide
   a solid-filled opening.
6. Name 3–5 **macro identity forms** separately from detail identity features:
   e.g. a continuous nose-to-canopy arc, paired rear haunch shoulders, or an
   EVA thigh swell that pinches sharply into the knee.

Single-view evidence constrains the visible contour, not the hidden side.
Use symmetry/class priors for hidden sections, lower their confidence, and
keep them smooth and simple. With calibrated front/side/top references, fit
all silhouettes together; a visual hull may guide blockout but must not be
the final editable surface because it cannot recover concavities.

## 3. Pass the shape-first probe

For `curved` or `mixed`, create `<out>/form_probe.py` before the full program.
Build only primary masses, joint-center markers, ground-contact parts, and
negative-space openings. Use one neutral material; omit seams, labels,
fasteners, tread, and material variation.

```sh
procagen3d build <out>/form_probe.py --out <out>/form_probe --form-diagnostics
procagen3d check <out>/form_probe --tier quick --form <profile>
```

Read both `renders/sheet.png` and `renders/form_sheet.png`. Pass only when:

- `check` has no `FORM_*` failure, and each `FORM_*` warning is fixed or
  justified by named, deliberately assembled structure;
- front/right/top/iso silhouettes agree with the form blueprint;
- cross-sections swell, pinch, taper, and twist at the named stations;
- negative spaces remain open and attachment patches meet cleanly;
- primary transitions read as one intended surface, not overlapping lumps;
- the reference-matched view agrees without collapsing in unseen views.

Allow at most two probe corrections. If the body family is wrong, rewrite the
single primary builder (box → loft/sweep/surface), not its dimensions or
decoration. If the probe still fails, stop or request better views; do not
decorate a rejected form. Once passed, transfer those builders and station
constants to the full program before adding secondary structure and detail.
For repeated complex modules, validate one master (one fender, thigh shell,
leaf, blade)
from a close and an orbit view before mirroring/instancing it.

## 4. Use constructive curved builders

Prefer compact parametric rings, profiles, and grids. Keep ring point counts
consistent and winding consistent. Inspect cap winding, outward normals,
non-manifold edges, and end seams in the first probe.

Import `loft_rings`, `sweep_profile`, and `revolve_profile` from
`procagen3d_runtime`; do not copy or improvise their mesh-construction bodies.
The build freezes the tested implementation into the delivered `program.py`.
The asset source should contain only the semantic control points and calls.

### Variable-section loft

Use for bodies, hulls, fairings, fenders, torsos, thighs, shins, forearms, and
asymmetric housings. Supply equal-count closed rings in world/local XYZ,
ordered consistently around the section.

```python
from procagen3d_runtime import loft_rings

body = loft_rings(
    "Body",
    BODY_RINGS,
    body_material,
    subdivision=1,
    role="primary",
    topology="continuous",
)
```

Start with 5–12 rings and 8–24 points per ring. Add support rings near a hard
break instead of raising subdivision globally. Catmull-Clark shrinks a cage;
place the authored cage to compensate, or use denser interpolated rings with
`subdivision=0` when measured silhouettes must be interpolated exactly.

### Varying-profile sweep

Use for a horn, intake lip, curved rail, tapered cable, organic limb core, or
other form organized around a spine. The runtime constructs a
parallel-transport frame and reuses its tested loft implementation; do not use
a constant circular bevel for a section that visibly changes.

```python
from procagen3d_runtime import sweep_profile

horn = sweep_profile(
    "Horn",
    HORN_SPINE,
    HORN_PROFILE,
    HORN_SCALES,
    horn_material,
    role="primary",
)
```

Order `profile` consistently; use 8–16 points for visible oval/polygon
sections. Add an attachment collar as a separate assembled part when the
sweep meets a housing—do not leave a pinched point contact.

### Other representation families

- **Revolve** an explicit radial profile for bottles, hubs, bowls, domes, and
  ogives. A Screw-modifier lathe should keep at least five authored profile
  samples; spend 48–72 evaluated segments on a showcase macroform.
- **Polygon profile extrusion** for deliberately planar/faceted armor, blades,
  brackets, fins, and aero planes. Use a 5–12 point outline and real thickness;
  classify it `assembled`, never `continuous`.
- **Surface grid + Solidify** for a chair back, canopy, fairing, leaf, or panel
  defined by analytic `P(u,v)`. Vary width/depth/cup across the grid and add
  side/end walls; tag it `shell` + `surface-grid` or `solidify`.
- **Authored subdivision cage** for a compound shell when section lofts cannot
  express branching transitions. Use enough control loops to locate shoulders,
  creases, openings, and attachment boundaries. Eight cube vertices plus
  Bevel/Subdivision is still a rounded box and fails the form contract.
- **NURBS/Bezier curve** for tubes and trim with a circular section. Convert to
  mesh before delivery when it must pass mesh-only part checks.
- Apply booleans **after** the primary envelope passes. Cut wheel arches,
  vents, and cavities from a coherent body; then add a conforming flare/lip.

## 5. Apply subject recipes

### Streamlined vehicle

- Loft the center tub/hood/roof/deck along the longitudinal axis with stations
  at nose tip, mouth rear, front axle, cowl, cabin peak, rear shoulder, rear
  axle, and tail.
- Loft each fender/haunch from shared attachment stations. Blend or overlap
  only inside hidden attachment bands; independent ellipsoids read as pontoons.
- Build glazing as conforming shells derived from the cabin surface. Cut real
  wheel openings before adding flares. Keep splitter, wing, canards, and thin
  fences as assembled profile extrusions.

### Biomechanical or irregular mecha

- Build anatomical/biomechanical cores as tapered capsules, lofts, or sweeps.
- Build thigh, shin, forearm, shoulder, and torso armor as separate asymmetric
  4–8 ring loft shells so joints and gaps remain articulated.
- Use polygon profile extrusions for intentionally faceted chest, knee, ankle,
  and pylon plates; primitives remain valid for joint hubs and hard mechanisms.
- Use sweeps for horns, cables, and curved spines. Preserve collars and visible
  gaps at every joint; do not let armor shells fuse across pivots.

## 6. Inspect and repair form

Judge form before materials and microdetail:

- **silhouette:** compare traced landmarks in reference, front, right, top, and
  iso views; no view may collapse into a slab;
- **volume/cross-section:** confirm the named swells, pinches, crowns, undercuts,
  and thickness changes—broadside similarity alone is insufficient;
- **continuity/highlight flow:** read `form_sheet.png`; unwanted flat runs,
  segmented ellipsoid joins, and box corners are failures;
- **negative space:** verify every wheel arch, limb gap, undercut, and cavity;
- **attachment:** inspect collars, root blends, seams, and load/contact patches.

If an identity-bearing compound transition remains segmented or faceted, keep
`shape — FAIL` even when every badge, material, and repeated detail is present.
Repair the highest-impact primary surface first. Preserve already-passing
dimensions, joints, openings, and views explicitly. When the representation
family is wrong, rewrite that one builder; additive decoration is not a repair.
