# ProcAgen3D representation doctrine

What makes a program a *ProcAgen3D* program, as opposed to a script that happens
to emit a mesh. The build harness and `check` enforce most of this; the rest
is on you. Violations found by gates are never acceptable residue.

## Program contract

- Defines `def build()` that constructs the whole scene into the (empty)
  current scene and returns the root object. Ends with
  `if __name__ == "__main__": build()` so it also runs in bare Blender.
- Pure `bpy` / `bmesh` / `mathutils` / `math`, plus the authoring-only
  `procagen3d_runtime` import described below. No rendering, no export, no
  file/network I/O, no `subprocess` — the harness owns execution (these are
  flagged as `[PROCAGEN3D:WARN:PROGRAM]`).
- Deterministic: no unseeded randomness. If organic variation is wanted,
  seed explicitly (`random.seed(7)`) so every rebuild is identical.
- Units are meters at real-world scale, Z up, ground plane at z = 0 (feet,
  wheels, bases touch z = 0). The object faces **-Y** — the canonical
  "front" render looks from -Y toward +Y, so a car's windshield, a face, a
  drawer front must point toward -Y.

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

A small set of physically based families assigned by part meaning (Wood,
PaintedSteel, Rubber, Glass, Brass...), not per-mesh one-offs. Reuse via the
helper; keep 3–8 materials per asset.

## Canonical modeling runtime

Do not rewrite basic modeling helpers in every asset. Import the tested,
versioned authoring API explicitly:

```python
import bpy
from mathutils import Vector
from procagen3d_runtime import (
    add_joint,
    apply_transform,
    box,
    cylinder_between,
    ellipsoid,
    loft_rings,
    make_material,
    mark_form,
    new_group,
    reparent_keep_world,
    revolve_profile,
    sweep_profile,
)
```

`procagen3d build` replaces that module-scope import with the canonical runtime
source and retains the frozen result as `<out>/program.py`. The delivered
program is therefore self-contained and runnable in bare Blender; the shorter
authoring source remains easy to generate and review. Existing self-contained
programs remain supported.

Run `procagen3d lint <source.py>` before an expensive build. A direct
`bpy.ops.object.transform_apply` call is accepted only when `location=`,
`rotation=`, and `scale=` are all present. Prefer `apply_transform(obj,
location=False, rotation=False, scale=True)`, which also makes `obj` the sole
active selection. This gate prevents positioned mesh data from being baked
around the world origin by Blender's omitted `location=True` default.

## Program skeleton

```python
"""<Object name> — ProcAgen3D program.

Parts: <root> > <group> > <part table with counts>
Joints: <name> (<type>, axis, limits) ...
"""
import math

import bpy
from mathutils import Vector
from procagen3d_runtime import box, make_material, reparent_keep_world

# --- constants (meters) ---
SEAT_HEIGHT = 0.45
LEG_COUNT = 3
# ... every load-bearing dimension ...

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

Prefer constructive solid modeling with primitives, arrays of instances,
booleans (bmesh or modifier), bevels, and simple bmesh extrusions. A shape
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

## Repair doctrine

Repairs are minimal source edits. The guard (`procagen3d guard old new`)
deterministically rejects a correction that shrinks the source by more than
15%, drops the `build()` entry point, or drops `build_*` part functions.
Renames/merges that are genuinely intended: pass `--allow-drop <name>` /
`--allow-shrink` and say why in the final report.

## Checklist before first build

- [ ] constants block, meters, z=0 ground, front faces -Y
- [ ] detail tier declared; standard+: part table decomposed to the
      detail.md ladder, floors met by design not by hope
- [ ] form profile declared; curved/mixed: form blueprint + probe passed,
      structural macro/meso meshes tagged with compatible role/topology/method
- [ ] every part has a semantic PascalCase name, instances numbered
- [ ] one root empty; groups for real sub-assemblies; keep-world parenting
- [ ] movable parts: origin at pivot, joint declared via `add_joint`
- [ ] materials by meaning via `make_material`
- [ ] `build()` defined, `__main__` guard present, no I/O or render calls
