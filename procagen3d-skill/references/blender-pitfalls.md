# Blender pitfalls — headless bpy traps (Blender 4.x)

Every trap below is from a real failure mode. Read before writing bpy code;
revisit when a build error or a weird render doesn't make sense.

## Trap 1 — default names survive

`primitive_cube_add` names the object `Cube`; the next one `Cube.001`.
`check` fails both. Rename **immediately** after every add:

```python
bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
obj = bpy.context.object
obj.name = "Seat"          # object name; harness syncs mesh-data name at export
```

Assigning a name that already exists silently renames yours to `.001` —
if `check` reports `DUPLICATE_NAMES`, some loop reused a name.

## Trap 2 — unapplied scale

`obj.scale = (2, 1, 0.1)` makes dimensions lie downstream (booleans, bevel
widths, exports). Size via op parameters (`radius=`, `depth=`) or set
`obj.dimensions` then apply:

```python
obj.dimensions = (0.4, 0.3, 0.25)
bpy.ops.object.transform_apply(scale=True)   # acts on selected+active: do it right after add
```

`check` warns `UNAPPLIED_SCALE` on |scale−1| > 1e-3.

## Trap 3 — naive parenting moves the child

`child.parent = p` keeps the child's *local* matrix, so it jumps by p's
transform. Always use the doctrine helper:

```python
def reparent_keep_world(obj, new_parent):
    bpy.context.view_layer.update()
    mw = obj.matrix_world.copy()
    obj.parent = new_parent
    bpy.context.view_layer.update()
    obj.matrix_world = mw
```

## Trap 4 — stale matrix_world

`matrix_world` is only current after a depsgraph update. After any parenting
or transform change, call `bpy.context.view_layer.update()` before reading
it. Symptom: parts assembled at positions that were true one operation ago.

## Trap 5 — ops act on selection/active state

`transform_apply`, `origin_set`, `join`, `modifier_apply` operate on the
*selected/active* objects — which is whatever the last add left behind. Do
selection-dependent ops immediately after creating the object, or set state
explicitly:

```python
bpy.ops.object.select_all(action="DESELECT")
obj.select_set(True)
bpy.context.view_layer.objects.active = obj
```

## Trap 6 — origin_set goes through the 3D cursor

```python
bpy.context.scene.cursor.location = HINGE
bpy.ops.object.origin_set(type="ORIGIN_CURSOR")   # affects selected+active!
```

Combine Trap 5 + 6: set selection first or run it right after the add.

## Trap 7 — booleans need the modifier applied on the right object

```python
mod = target.modifiers.new("Cut", "BOOLEAN")
mod.operation = "DIFFERENCE"
mod.object = cutter
bpy.context.view_layer.objects.active = target        # modifier_apply uses active
bpy.ops.object.modifier_apply(modifier=mod.name)
bpy.data.objects.remove(cutter, do_unlink=True)       # remove the cutter after
```

Leaving the cutter in the scene adds a phantom part that fails part-coverage
inspection. Alternatively leave modifiers unapplied — the harness exports
with modifiers applied — but then vertex counts in `scene_graph.json` are
evaluated counts. Remove the cutter, or set `cutter.hide_render = True`; the
harness excludes non-renderable helpers from graphs, proof framing, joint
checks, and GLB export.

## Trap 8 — orientation conventions

Meters, Z up, ground at z = 0, **front faces -Y** (the canonical front
render looks from -Y). A model built facing +Y shows its back on the sheet's
front tile and every left/right judgment flips. Align direction vectors
with `Vector.to_track_quat`: `obj.rotation_euler =
direction.to_track_quat("Z", "Y").to_euler()` points the object's +Z along
`direction`.

## Trap 9 — cylinders/spheres have axis Z

`primitive_cylinder_add` is Z-axis aligned. A wheel lies in the XZ plane
(rolling along Y) only after rotating it; an axle along X needs
`rotation_euler=(0, math.pi/2, 0)`. Sanity-check every rotated primitive in
the side view of the sheet.

## Trap 10 — programs must not own I/O

No `render()`, `export_scene`, `wm.save*`, `open()`, network, `subprocess`.
The harness executes, exports, renders. These patterns trigger
`[PROCAGEN3D:WARN:PROGRAM]` and in a delivered asset are doctrine violations.

## Trap 11 — background mode has no UI context

Ops relying on view3d context (`view3d.*`, snapping helpers) fail headless.
Everything in this skill's doctrine (mesh adds, transform_apply, origin_set,
modifier_apply, gltf export) is context-safe. When an op errors with
"context is incorrect", switch to the data API (`bpy.data`, `bmesh`) rather
than fighting the context.

## Trap 12 — mirrored parts are not negative-scaled copies

`obj.scale.x = -1` inverts normals and trips the unapplied-scale gate.
Build mirrored instances by constructing at the mirrored position (negate
the position/angle constants), or apply the mirror into mesh data.
