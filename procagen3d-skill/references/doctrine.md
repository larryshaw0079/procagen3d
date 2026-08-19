# ProcAgen3D representation doctrine

What makes a program a *ProcAgen3D* program, as opposed to a script that happens
to emit a mesh. The build harness and `check` enforce most of this; the rest
is on you. Violations found by gates are never acceptable residue.

## Contents

1. Program, units, dimensions, names, pivots, and hierarchy
2. Canonical helpers and program skeleton
3. Geometry and reconstruction strategy
4. Repair doctrine and pre-build checklist

## Program contract

- Defines `def build()` that constructs the whole scene into the (empty)
  current scene and returns the root object. Ends with
  `if __name__ == "__main__": build()` so it also runs in bare Blender.
- Pure `bpy` / `bmesh` / `mathutils` / `math`. No rendering, no export, no
  file/network I/O, no `subprocess` — the harness owns execution (these are
  flagged as `[PROCAGEN3D:WARN:PROGRAM]`).
- Deterministic: no unseeded randomness. If organic variation is wanted,
  seed explicitly (`random.seed(7)`) so every rebuild is identical.
- Units are meters at real-world scale, Z up, ground plane at z = 0 (feet,
  wheels, bases touch z = 0). The object faces **-Y** — the canonical
  "front" render looks from -Y toward +Y, so a car's windshield, a face, a
  drawer front must point toward -Y.
- Keep load-bearing dimensions in this canonical object frame. For a reference
  image, solve camera and rigid/articulated pose separately; never bake image
  tilt or foreshortening into part size.

## Named constants for dimensions

Every load-bearing dimension is a named constant at the top, in meters, with
physical joint locations as vectors — this is what makes the asset
parametric and locally editable:

```python
BASE_FLANGE_R   = 0.170
BASE_BOLT_COUNT = 8
SHOULDER = Vector((0.0,  0.000, 0.405))   # physical joint locations
ELBOW    = Vector((0.0, -0.105, 0.865))
```

Derive dependent sizes from constants (`SEAT_TOP = SEAT_HEIGHT`, leg length
computed from height and splay) instead of repeating literals.

## Part naming

- Each component is a separately named mesh object, PascalCase, semantic:
  `Seat`, `Fork_Crown`, `Left_Pedal_Crank`.
- Repeated instances are named individually: `Spoke_17`, never one merged
  `Spokes`. Loops build them: `for i in range(SPOKE_COUNT): ... name =
  f"Spoke_{i+1}"`.
- Never leave a default name (`Cube`, `Sphere.001`) — `check` fails these.
  Rename immediately after every `primitive_*_add` via
  `bpy.context.object.name = ...`.

## Pivots and origins

Every part that should move owns its origin at its physical pivot: a door's
hinge edge, a wheel's axle, a limb's proximal joint. Set it with the cursor:

```python
bpy.context.scene.cursor.location = HINGE     # a Vector constant
bpy.ops.object.origin_set(type="ORIGIN_CURSOR")
```

Static parts may keep their geometric-center origin.

## Hierarchy

One root empty named after the object (`Bicycle`), semantic group empties
where the assembly has real sub-assemblies (`Frame`, `Front_Wheel_Assembly`,
`Left_Arm`), meshes as leaves. Parent with the keep-world helper below —
naive `obj.parent = x` shifts the child by the parent's transform.

## Materials

A purposeful set of physically based families assigned by part meaning (Wood,
PaintedSteel, Rubber, Glass, Brass...), not per-mesh one-offs. Reuse via the
helper and meet the complexity-adaptive floor in `detail.md`.

## Canonical helpers (paste into every program)

These are tested against Blender 4.5 headless. Take what the program needs;
`add_joint` and `reparent_keep_world` are required verbatim whenever joints
or parenting are used — the validators rely on their conventions.

```python
import math

import bpy
from mathutils import Vector


def make_material(name, color, roughness=0.6, metallic=0.0):
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (*color, 1.0)
        bsdf.inputs["Roughness"].default_value = roughness
        bsdf.inputs["Metallic"].default_value = metallic
        mat.diffuse_color = (*color, 1.0)   # workbench/viewport color
    return mat


def box(name, size, location, mat):
    """Axis-aligned box with applied scale (avoids the unapplied-scale trap)."""
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = size
    bpy.ops.object.transform_apply(scale=True)
    obj.data.materials.append(mat)
    return obj


def mark_form(obj, role, topology, method, family, section_count=None):
    """Bind a structural mesh to the form + reconstruction-plan contract."""
    obj["procagen3d_form_role"] = role
    obj["procagen3d_topology"] = topology
    obj["procagen3d_form_method"] = method
    obj["procagen3d_shape_family"] = family
    if section_count is not None:
        obj["procagen3d_section_count"] = int(section_count)
    return obj


def reparent_keep_world(obj, new_parent):
    """Parent without moving the child (bpy default parenting shifts it)."""
    bpy.context.view_layer.update()
    mw = obj.matrix_world.copy()
    obj.parent = new_parent
    bpy.context.view_layer.update()
    obj.matrix_world = mw


def add_joint(name, parent, child, jtype, axis, limits=None, origin=None):
    """ProcAgen3D joint: an empty at the pivot; re-parents parent <- joint <- child
    preserving world transforms. jtype: 'revolute' | 'prismatic' | 'fixed'.
    axis: world-space at rest pose. limits: [lo, hi] in degrees (revolute)
    or meters (prismatic)."""
    j = bpy.data.objects.new(name, None)
    j.empty_display_type = "ARROWS"
    j.empty_display_size = 0.05
    bpy.context.scene.collection.objects.link(j)
    j.location = origin if origin is not None else child.matrix_world.translation
    j["procagen3d_joint_type"] = jtype
    j["procagen3d_joint_axis"] = list(axis)
    if limits is not None:
        j["procagen3d_joint_limits"] = list(limits)
    j["procagen3d_joint_child"] = child.name
    j["procagen3d_joint_parent"] = parent.name if parent else ""
    reparent_keep_world(j, parent)
    reparent_keep_world(child, j)
    return j
```

## Program skeleton

```python
"""<Object name> — ProcAgen3D program.

Parts: <root> > <group> > <part table with counts>
Joints: <name> (<type>, axis, limits) ...
"""
import math

import bpy
from mathutils import Vector

# --- constants (meters) ---
SEAT_HEIGHT = 0.45
LEG_COUNT = 3
# ... every load-bearing dimension ...

# --- canonical helpers here ---

def build_seat(mat):
    ...
    return obj

def build_leg(i, mat):
    ...
    return obj

def build():
    root = bpy.data.objects.new("Stool", None)
    bpy.context.scene.collection.objects.link(root)
    wood = make_material("Wood", (0.44, 0.29, 0.17))
    reparent_keep_world(build_seat(wood), root)
    for i in range(LEG_COUNT):
        reparent_keep_world(build_leg(i, wood), root)
    return root

if __name__ == "__main__":
    build()
```

## Geometry strategy, briefly

Choose the simplest evidence-backed family from
`references/reconstruction-planning.md`, then use constructive solid modeling
with primitives, arrays of instances, booleans (bmesh or modifier), bevels,
and simple bmesh extrusions. A shape
you can name (flange, strut, rim, shell) should come from a parameterized
builder function, not from magic vertex dumps. Compact, semantic control
arrays—profile points, section rings, sweep spines, and surface-grid
parameters—are explicitly constructive and preferred when primitives cannot
represent the form. Hollow containers: build wall
panels or use a boolean cavity cut — then verify wall presence in the top
view of the sheet. Curved masses (body panels, fenders, revolved forms) are
buildable with these tools — route compound/changing-section forms through
`references/complex-forms.md`, and use the segment/detail rules in
`references/detail.md`; reserve the "stylized" disclaimer for genuinely
organic subjects (faces, animals, cloth).

A bevel is edge treatment, not a license to change the volume family. Preserve
broad planar fields and constant sections as boxes/prisms. Use
`analytic-primitive` only for a genuinely spherical/ellipsoidal/capsule mass.

## Repair doctrine

Repairs are minimal source edits. The guard (`procagen3d guard old new`)
deterministically rejects a correction that shrinks the source by more than
15%, drops the `build()` entry point, or drops `build_*` part functions.
Renames/merges that are genuinely intended: pass `--allow-drop <name>` /
`--allow-shrink` and say why in the final report.

## Checklist before first build

- [ ] constants block, meters, z=0 ground, front faces -Y
- [ ] image-conditioned: canonical frame + camera + rigid/articulated pose
      solved before dimensions; fit v2 and reconstruction plan authored
- [ ] detail tier declared; standard+: part table decomposed to the
      detail.md ladder; complexity class and feature groups declared
- [ ] form profile declared; curved/mixed: form blueprint + probe passed,
      structural macro/meso meshes tagged with compatible role/topology/method
- [ ] every image-conditioned primary/macro mass has an evidence-backed
      `procagen3d_shape_family` matching `reconstruction_plan.json`
- [ ] every part has a semantic PascalCase name, instances numbered
- [ ] one root empty; groups for real sub-assemblies; keep-world parenting
- [ ] movable parts: origin at pivot, joint declared via `add_joint`
- [ ] materials by meaning via `make_material`
- [ ] `build()` defined, `__main__` guard present, no I/O or render calls
