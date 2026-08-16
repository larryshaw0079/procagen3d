"""Scene-graph dump: bounds, joint records, object/mesh totals."""

import math

import bpy
from mathutils import Vector

from .runtime import depsgraph, mesh_objects


def world_bbox(obj, dg):
    ev = obj.evaluated_get(dg)
    corners = [ev.matrix_world @ Vector(c) for c in ev.bound_box]
    lo = Vector((min(c[i] for c in corners) for i in range(3)))
    hi = Vector((max(c[i] for c in corners) for i in range(3)))
    return lo, hi


def union_bbox(objs, dg):
    lo = Vector((math.inf,) * 3)
    hi = Vector((-math.inf,) * 3)
    for o in objs:
        blo, bhi = world_bbox(o, dg)
        lo = Vector((min(lo[i], blo[i]) for i in range(3)))
        hi = Vector((max(hi[i], bhi[i]) for i in range(3)))
    return lo, hi


def base_axis_plane_counts(mesh):
    """Count normalized coordinate planes in the authored (pre-modifier) cage.

    Normalizing each axis to 0..1 makes the diagnostic scale-independent.  A
    cube or eight-vertex tapered box stays at only a few planes even when a
    Bevel modifier inflates its evaluated triangle count; a real loft, sweep,
    lathe, or surface grid normally has many authored planes on two or more
    axes.
    """
    if not mesh.vertices:
        return [0, 0, 0]
    counts = []
    for axis in range(3):
        values = [v.co[axis] for v in mesh.vertices]
        lo, hi = min(values), max(values)
        span = hi - lo
        if span <= 1e-9:
            counts.append(1)
            continue
        counts.append(len({round((value - lo) / span, 3) for value in values}))
    return counts


def descendants(obj):
    out = []
    stack = list(obj.children)
    while stack:
        c = stack.pop()
        out.append(c)
        stack.extend(c.children)
    return out


def joint_empties():
    return [o for o in bpy.context.scene.objects if "procagen3d_joint_type" in o]


def joint_record(j, dg):
    child_name = str(j.get("procagen3d_joint_child", ""))
    child = bpy.context.scene.objects.get(child_name)
    origin = list(j.matrix_world.translation)
    rec = {
        "name": j.name,
        "type": str(j.get("procagen3d_joint_type", "")),
        "axis": list(j.get("procagen3d_joint_axis", [])),
        "limits": list(j.get("procagen3d_joint_limits", [])),
        "child": child_name,
        "parent": str(j.get("procagen3d_joint_parent", "")),
        "origin_world": origin,
    }
    if child is not None:
        gap = (child.matrix_world.translation - Vector(origin)).length
        rec["child_origin_gap"] = round(gap, 6)
    return rec


def collect_scene_graph():
    dg = depsgraph()
    objects = []
    total_tris = 0
    graph_objects = [
        obj for obj in bpy.context.scene.objects
        if obj.type not in ("CAMERA", "LIGHT")
        and not (obj.type == "MESH" and obj.hide_render)
    ]
    graph_names = {obj.name for obj in graph_objects}
    for obj in graph_objects:
        entry = {
            "name": obj.name,
            "type": obj.type,
            "parent": (obj.parent.name
                       if obj.parent and obj.parent.name in graph_names else None),
            "location": [round(v, 6) for v in obj.location],
            "scale": [round(v, 6) for v in obj.scale],
            "origin_world": [round(v, 6) for v in obj.matrix_world.translation],
        }
        if obj.type == "MESH":
            lo, hi = world_bbox(obj, dg)
            ev = obj.evaluated_get(dg)
            me = ev.to_mesh()
            me.calc_loop_triangles()
            tris = len(me.loop_triangles)
            entry.update(
                {
                    "bbox_world_min": [round(v, 6) for v in lo],
                    "bbox_world_max": [round(v, 6) for v in hi],
                    "dimensions": [round(hi[i] - lo[i], 6) for i in range(3)],
                    "vertex_count": len(me.vertices),
                    "poly_count": len(me.polygons),
                    "triangle_count": tris,
                    "base_vertex_count": len(obj.data.vertices),
                    "base_poly_count": len(obj.data.polygons),
                    "base_axis_plane_counts": base_axis_plane_counts(obj.data),
                    "modifiers": [modifier.type for modifier in obj.modifiers],
                    "materials": [m.name for m in obj.data.materials if m],
                }
            )
            total_tris += tris
            ev.to_mesh_clear()
        props = {
            k: (list(v) if hasattr(v, "__len__") and not isinstance(v, str) else v)
            for k, v in obj.items()
            if isinstance(k, str) and k.startswith("procagen3d_")
        }
        if props:
            entry["custom_props"] = props
        objects.append(entry)

    meshes = mesh_objects()
    graph = {
        "procagen3d_version": "0.2",
        "blender_version": bpy.app.version_string,
        "objects": objects,
        "roots": [o.name for o in graph_objects
                  if o.parent is None or o.parent.name not in graph_names],
        "joints": [joint_record(j, dg) for j in joint_empties()],
        "totals": {
            "objects": len(objects),
            "meshes": len(meshes),
            "triangles": total_tris,
        },
    }
    if meshes:
        lo, hi = union_bbox(meshes, dg)
        graph["world_bbox"] = {
            "min": [round(v, 6) for v in lo],
            "max": [round(v, 6) for v in hi],
            "size": [round(hi[i] - lo[i], 6) for i in range(3)],
        }
    return graph
