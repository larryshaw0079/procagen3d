"""Validate articulation: type, child, pivot, limits, sweep collisions, rest pose."""

import json
import math
from pathlib import Path

import bpy
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

from .runtime import FAIL, OK, WARN, depsgraph, finish, mesh_objects
from .scene import descendants, joint_empties, union_bbox


JOINT_TYPES = ("revolute", "prismatic", "fixed")


def bvh_from_objects(objs, dg):
    verts, polys = [], []
    for o in objs:
        ev = o.evaluated_get(dg)
        me = ev.to_mesh()
        mw = ev.matrix_world
        base = len(verts)
        verts.extend(tuple(mw @ v.co) for v in me.vertices)
        polys.extend(tuple(i + base for i in p.vertices) for p in me.polygons)
        ev.to_mesh_clear()
    if not polys:
        return None
    return BVHTree.FromPolygons(verts, polys)


def stage_joints(args):
    out_dir = Path(args.out)
    blend = out_dir / "scene.blend"
    if not blend.exists():
        print(f"{FAIL}:NO_SCENE] {blend} not found (run build first)")
        finish(1)
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    dg = depsgraph()
    joints = joint_empties()
    if not joints:
        print(f"{OK} no joints declared — nothing to validate")
        finish(0)

    scene_objs = bpy.context.scene.objects
    report = []
    failures = 0
    rest_pose = {j.name: j.matrix_world.copy() for j in joints}
    child_rest = {}

    for j in joints:
        entry = {"joint": j.name, "checks": {}, "collisions": []}
        checks = entry["checks"]
        jtype = str(j.get("procagen3d_joint_type", ""))
        axis = Vector(tuple(j.get("procagen3d_joint_axis", (0, 0, 0))))
        limits = list(j.get("procagen3d_joint_limits", []))
        child = scene_objs.get(str(j.get("procagen3d_joint_child", "")))

        checks["type_valid"] = jtype in JOINT_TYPES
        if not checks["type_valid"]:
            print(f"{FAIL}:JOINT_TYPE] {j.name}: '{jtype}' not in {JOINT_TYPES}")
            failures += 1
        checks["child_exists"] = child is not None
        if child is None:
            print(f"{FAIL}:JOINT_CHILD] {j.name}: child "
                  f"'{j.get('procagen3d_joint_child', '')}' not found")
            failures += 1
            report.append(entry)
            continue
        child_rest[child.name] = child.matrix_world.copy()

        moving = [o for o in [child] + descendants(child)
                  if o.type == "MESH" and not o.hide_render]
        checks["child_has_geometry"] = bool(moving)

        if jtype == "fixed":
            report.append(entry)
            continue

        # axis sanity (axis is in world space at rest pose)
        checks["axis_nonzero"] = axis.length > 1e-6
        if not checks["axis_nonzero"]:
            print(f"{FAIL}:JOINT_AXIS] {j.name}: zero axis")
            failures += 1
            report.append(entry)
            continue
        axis = axis.normalized()

        # pivot must sit on the moving part ("axis on moving part")
        pivot = j.matrix_world.translation
        if moving:
            lo, hi = union_bbox(moving, dg)
            pad = max(0.05, 0.10 * (hi - lo).length)
            on_part = all(lo[i] - pad <= pivot[i] <= hi[i] + pad for i in range(3))
            checks["pivot_on_moving_part"] = on_part
            if not on_part:
                print(f"{FAIL}:JOINT_PIVOT] {j.name}: pivot {tuple(round(v,3) for v in pivot)} "
                      f"lies off the moving part bbox [{tuple(round(v,3) for v in lo)}, "
                      f"{tuple(round(v,3) for v in hi)}]")
                failures += 1

        # limits sanity
        if len(limits) != 2 or limits[0] >= limits[1]:
            checks["limits_declared"] = False
            print(f"{WARN}:JOINT_LIMITS] {j.name}: no usable limits declared "
                  f"({limits}); declare [lo, hi] (deg for revolute, m for prismatic)")
        else:
            checks["limits_declared"] = True
            if jtype == "revolute" and limits[1] - limits[0] >= 300:
                print(f"{WARN}:JOINT_LIMITS] {j.name}: range "
                      f"{limits[1]-limits[0]:.0f} deg >= 300 — generic default? "
                      "Declare a physically plausible range.")
            if jtype == "prismatic" and limits[1] - limits[0] > 5.0:
                print(f"{WARN}:JOINT_LIMITS] {j.name}: prismatic range "
                      f"{limits[1]-limits[0]:.2f} m looks implausible")

        # sweep collision test
        if checks["limits_declared"] and moving:
            moving_names = {o.name for o in moving}
            parent_name = str(j.get("procagen3d_joint_parent", ""))
            excluded = set(moving_names)
            if not args.strict and parent_name:
                parent_obj = scene_objs.get(parent_name)
                if parent_obj is not None:
                    excluded.add(parent_obj.name)
                    # direct mesh children of a group-empty parent share the pivot
                    if parent_obj.type == "EMPTY":
                        excluded.update(c.name for c in parent_obj.children
                                        if c.type == "MESH")
            static = [o for o in mesh_objects() if o.name not in excluded]
            static_bvh = bvh_from_objects(static, dg)
            base_mw = j.matrix_world.copy()
            lo_l, hi_l = limits
            for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
                val = lo_l + frac * (hi_l - lo_l)
                if abs(val) < 1e-9 or static_bvh is None:
                    continue
                if jtype == "revolute":
                    rot = (Matrix.Translation(pivot)
                           @ Matrix.Rotation(math.radians(val), 4, axis)
                           @ Matrix.Translation(-pivot))
                    j.matrix_world = rot @ base_mw
                else:  # prismatic
                    j.matrix_world = Matrix.Translation(axis * val) @ base_mw
                dg = depsgraph()
                moving_bvh = bvh_from_objects(moving, dg)
                if moving_bvh is not None:
                    hits = moving_bvh.overlap(static_bvh)
                    if hits:
                        unit = "deg" if jtype == "revolute" else "m"
                        entry["collisions"].append({"at": val, "pairs": len(hits)})
                        print(f"{WARN}:JOINT_SWEEP] {j.name}: collision with "
                              f"static geometry at {val:.1f} {unit} "
                              f"({len(hits)} face pairs)")
            j.matrix_world = base_mw
            dg = depsgraph()
            checks["sweep_collision_free"] = not entry["collisions"]
        report.append(entry)

    # rest-pose restore check (validator must leave the scene untouched)
    dg = depsgraph()
    rest_ok = True
    for j in joints:
        delta = max(abs(a - b) for ra, rb in zip(j.matrix_world, rest_pose[j.name])
                    for a, b in zip(ra, rb))
        if delta > 1e-6:
            rest_ok = False
            print(f"{FAIL}:REST_POSE] {j.name}: rest pose drifted by {delta:.2e}")
            failures += 1
    for name, mw in child_rest.items():
        obj = scene_objs.get(name)
        if obj is None:
            continue
        delta = max(abs(a - b) for ra, rb in zip(obj.matrix_world, mw)
                    for a, b in zip(ra, rb))
        if delta > 1e-6:
            rest_ok = False
            print(f"{FAIL}:REST_POSE] {name}: rest pose drifted by {delta:.2e}")
            failures += 1

    result = {
        "joints_checked": len(joints),
        "failures": failures,
        "rest_pose_ok": rest_ok,
        "report": report,
    }
    (out_dir / "joints_report.json").write_text(json.dumps(result, indent=2))
    if failures:
        print(f"{FAIL}:JOINTS] {failures} failure(s) across {len(joints)} joint(s)")
        finish(1)
    print(f"{OK} {len(joints)} joint(s) validated "
          f"(warnings above, if any, need judgment — read them)")
    finish(0)
