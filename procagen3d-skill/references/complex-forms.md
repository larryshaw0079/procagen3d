# Complex forms — curved, streamlined, and irregular surfaces

Use this reference for any target whose identity depends on changing
cross-sections, compound curvature, taper/twist, a continuous highlight, or
non-rectangular armor/panels. A sports-car body and biomechanical mecha are
form-dominant even though both are "hard surface." More parts, bevels, and
triangles cannot repair the wrong representation family.

Read `reconstruction-planning.md` first for image-conditioned work. This file
describes genuine complex forms; it does not authorize replacing evidence-backed
boxes/prisms with lofts or ellipsoids.

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
solids. `mixed` means both families are present because the evidence says so;
it is not a minimum continuous-volume percentage. Run `check --form <profile>`
later.

Set the same profile on the program root so plain `check` auto-detects it:

```python
root["procagen3d_form_profile"] = "curved"  # or "mixed" / "rectilinear"
```

For each structural macro/meso mass, record this row in the design header and
mark the built object with the same contract. Use `primary` for silhouette,
volume, and identity forms; use `secondary` for supports, contacts, and
deliberately assembled structure:

```python
def mark_form(obj, role, topology, method, family, section_count=None):
    obj["procagen3d_form_role"] = role          # primary | secondary
    obj["procagen3d_topology"] = topology      # table above
    obj["procagen3d_form_method"] = method     # checker vocabulary below
    obj["procagen3d_shape_family"] = family     # reconstruction plan value
    if section_count is not None:
        obj["procagen3d_section_count"] = int(section_count)
    return obj
```

Use these exact method values: `loft`, `sweep`, `revolve`, `subdivision`,
`surface-grid`, `nurbs`, `curve`, `solidify`, `profile-extrude`,
`primitive-csg`, `boolean`, `analytic-primitive`, or `decal`. Use
`analytic-primitive` only for a genuine sphere/ellipsoid/capsule. Tag every
structural mass that enters form review, but do not inflate coverage by tagging
seams, fasteners, decals, or trim. Those are detail, not structural form.

The checker audits contract coverage on the largest structural masses. A
curved target still needs real continuous macro coverage. A mixed target only
needs genuine continuous/shell and assembled structural representatives, then
checks every version-2 mass against `reconstruction_plan.json`; neither family
must be among the largest masses and no volume percentage is rewarded.
Legitimately blocky mechanisms remain assembled.

### Rectilinear/prismatic guard

Long straight/parallel edges, broad planar fields, and constant sections are
positive evidence for `box`/`prism`, even when corners are beveled. Bevel is
edge treatment, not topology. If those cues dominate, route the part to
primitive CSG or polygon profile extrusion. Do not make a six-ring loft whose
stations differ only enough to satisfy a form warning; that is a rounded-box
proxy and a prior failure.

## 2. Build the form blueprint

Do not infer a curve from adjectives. Measure a compact guide from the saved
reference:

1. Reuse the camera and pose already solved in version-2 `fit_spec.json` as
   required by `reconstruction-planning.md` and `image-fit.md`. Do not tune
   section dimensions until frame-axis and pose-chain evidence is credible.
   Optionally put the coarse three-value preview contract on the root:

   ```python
   root["procagen3d_reference_projection"] = "perspective"
   root["procagen3d_reference_camera"] = [azimuth_deg, elevation_deg, fov_deg]
   # Or: "orthographic" with [azimuth_deg, elevation_deg, ortho_scale_m].
   ```

   Treat that root property as a backward-compatible preview only: it
   auto-centers the asset and cannot prove registration. Run `procagen3d fit`
   for the scored `renders/reference_match.png`. Mark estimates approximate
   until the registered local-silhouette/pose gates pass.
2. Copy each chosen structural family from `reconstruction_plan.json`; the
   form blueprint only expands masses planned as loft/sweep/shell. Do not
   promote a box/prism because this reference discusses curves.
3. Trace 6–12 normalized `(u, v)` landmarks along each identity-bearing
   contour. Convert them to meters with one explicit image-to-world mapping.
4. For a loft, choose a longitudinal axis and 5–12 stations. At each station
   record center, half-width, bottom, shoulder/belt height, crown/top, local
   depth, twist, and the evidence view. Add stations at every curvature
   extremum, hard crease, opening boundary, and attachment.
5. For a polygon panel, record its ordered outline and thickness. For a sweep,
   record spine points plus width/height/twist per station. Small semantic
   control arrays are constructive parameters, not forbidden vertex soup.
6. Record negative spaces separately: wheel arches, limb-to-torso gaps,
   undercuts, vents, and joint clearances. A correct outer AABB can still hide
   a solid-filled opening.
7. Name 3–5 **macro identity forms** separately from detail identity features:
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
procagen3d fit <out>/form_probe --spec <out>/fit_spec.json
procagen3d check <out>/form_probe --tier quick --form <profile>
```

Read both `renders/sheet.png` and `renders/form_sheet.png`. Pass only when:

- `check` has no `FORM_*` failure, and each `FORM_*` warning is fixed or
  justified by named, deliberately assembled structure;
- front/right/top/iso silhouettes agree with the primitive plan/form blueprint;
- local silhouette, frame-axis, and articulated pose-chain gates pass;
- cross-sections swell, pinch, taper, and twist at the named stations;
- negative spaces remain open and attachment patches meet cleanly;
- primary transitions read as one intended surface, not overlapping lumps;
- the reference-matched view agrees without collapsing in unseen views.

Allow at most two probe corrections. If a family is wrong, revisit evidence in
the reconstruction plan, then rewrite that single builder in either direction
(ellipsoid/loft → box/prism is as valid as box → loft). Do not tune dimensions
or decoration around the wrong family. If the probe still fails, stop or
request better views; do not
decorate a rejected form. Once passed, transfer those builders and station
constants to the full program before adding secondary structure and detail.
For repeated complex modules, validate one master (one fender, thigh shell,
leaf, blade)
from a close and an orbit view before mirroring/instancing it.

## 4. Use constructive curved builders

Prefer compact parametric rings, profiles, and grids. Keep ring point counts
consistent and winding consistent. Inspect cap winding, outward normals,
non-manifold edges, and end seams in the first probe.

### Variable-section loft

Use for bodies, hulls, fairings, fenders, torsos, thighs, shins, forearms, and
asymmetric housings. Supply equal-count closed rings in world/local XYZ,
ordered consistently around the section.

```python
def loft_rings(name, rings, mat, *, cap=True, subdivision=0,
               role="primary", topology="continuous"):
    rings = [[Vector(point) for point in ring] for ring in rings]
    if len(rings) < 2 or len(rings[0]) < 3:
        raise ValueError("loft needs at least 2 rings with 3 points each")
    ring_size = len(rings[0])
    if any(len(ring) != ring_size for ring in rings):
        raise ValueError("all loft rings must have equal point counts")

    verts = [tuple(point) for ring in rings for point in ring]
    faces = []
    for row in range(len(rings) - 1):
        a0, b0 = row * ring_size, (row + 1) * ring_size
        for col in range(ring_size):
            nxt = (col + 1) % ring_size
            faces.append((a0 + col, a0 + nxt, b0 + nxt, b0 + col))
    if cap:
        faces.append(tuple(reversed(range(ring_size))))
        last = (len(rings) - 1) * ring_size
        faces.append(tuple(last + col for col in range(ring_size)))

    mesh = bpy.data.meshes.new(name + "_Mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.validate(verbose=True)
    mesh.update(calc_edges=True)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(mat)
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    if subdivision:
        mod = obj.modifiers.new("Form_Subdivision", "SUBSURF")
        mod.subdivision_type = "CATMULL_CLARK"
        mod.levels = subdivision
        mod.render_levels = subdivision
    mark_form(obj, role, topology, "loft", "loft", len(rings))
    obj["procagen3d_ring_points"] = ring_size
    return obj
```

Start with 5–12 rings and 8–24 points per ring. Add support rings near a hard
break instead of raising subdivision globally. Catmull-Clark shrinks a cage;
place the authored cage to compensate, or use denser interpolated rings with
`subdivision=0` when measured silhouettes must be interpolated exactly.

### Varying-profile sweep

Use for a horn, intake lip, curved rail, tapered cable, organic limb core, or
other form organized around a spine. Reuse `loft_rings` after constructing a
parallel-transport frame; do not use a constant circular bevel for a section
that visibly changes.

```python
def sweep_profile(name, spine, profile, scales, mat, *, role="primary"):
    spine = [Vector(point) for point in spine]
    if len(spine) < 2 or len(scales) != len(spine):
        raise ValueError("sweep needs matching spine and scale stations")
    tangents = []
    for i in range(len(spine)):
        before = spine[max(0, i - 1)]
        after = spine[min(len(spine) - 1, i + 1)]
        tangents.append((after - before).normalized())

    reference = Vector((0, 0, 1))
    if abs(tangents[0].dot(reference)) > 0.95:
        reference = Vector((1, 0, 0))
    normal = (reference - tangents[0] * reference.dot(tangents[0])).normalized()
    rings = []
    previous = tangents[0]
    for center, tangent, scale in zip(spine, tangents, scales):
        normal = previous.rotation_difference(tangent) @ normal
        normal = (normal - tangent * normal.dot(tangent)).normalized()
        binormal = tangent.cross(normal).normalized()
        sx, sy = ((scale, scale) if isinstance(scale, (int, float)) else scale)
        rings.append([center + normal * (u * sx) + binormal * (v * sy)
                      for u, v in profile])
        previous = tangent
    obj = loft_rings(name, rings, mat, role=role, topology="continuous")
    mark_form(obj, role, "continuous", "sweep", "sweep", len(rings))
    return obj
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

- Build genuinely anatomical/biomechanical cores as tapered capsules, lofts,
  or sweeps; compact joint hubs may use analytic primitives.
- Route each armor mass independently from visible evidence. Broad planar
  faces, hard crease networks, and constant thickness use polygon prisms or
  faceted shells. Use 4–8 ring loft shells only where the armor itself visibly
  swells/pinches across the face.
- Do not wrap an ellipsoid around every upper arm, thigh, shin, or shoulder.
  Many mecha derive identity from layered faceted plates over a much smaller
  curved inner core; preserve those plate boundaries and gaps.
- Use polygon profile extrusions for faceted chest, knee, ankle, pylon, forearm,
  and calf plates; primitives remain valid for joint hubs and hard mechanisms.
- Use sweeps for horns, cables, and curved spines. Preserve collars and visible
  gaps at every joint; do not let armor shells fuse across pivots.

## 6. Inspect and repair form

Judge form before materials and microdetail:

- **silhouette:** compare traced landmarks in reference, front, right, top, and
  iso views; no view may collapse into a slab;
- **volume/cross-section:** confirm the named swells, pinches, crowns, undercuts,
  and thickness changes—broadside similarity alone is insufficient;
- **family/highlight flow:** read `form_sheet.png`; unwanted flat runs or box
  corners fail planned continuous forms, while unwanted bulges/face curvature
  fail planned boxes and prisms; segmented ellipsoid joins always fail;
- **negative space:** verify every wheel arch, limb gap, undercut, and cavity;
- **attachment:** inspect collars, root blends, seams, and load/contact patches.

If an identity-bearing compound transition remains segmented or faceted, keep
`shape — FAIL` even when every badge, material, and repeated detail is present.
Repair the highest-impact primary surface first. Preserve already-passing
dimensions, joints, openings, and views explicitly. When the representation
family is wrong, rewrite that one builder; additive decoration is not a repair.
