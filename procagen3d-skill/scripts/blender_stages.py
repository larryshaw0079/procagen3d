"""ProcAgen3D Blender-side stages.

Not meant to be run directly — invoked by scripts/procagen3d.py as:

    blender --background --factory-startup --python-exit-code 1 \
        --python blender_stages.py -- <stage> [args...]

Stages:
    build   Execute a ProcAgen3D program, dump scene_graph.json, export GLB,
            save scene.blend, render canonical views + contact sheet.
    render  Re-render canonical views from an existing scene.blend.
    joints  Validate articulation (pivot placement, axis, limits, sweep
            collisions, rest-pose restore) against an existing scene.blend.

Exit codes are reported both as process exit code and as a stdout sentinel
line ``PROCAGEN3D_EXIT:<code>`` (0 = ok, 1 = failure) so the driver is robust to
Blender's own exit-code quirks.
"""

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

import bpy
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

OK = "[PROCAGEN3D:OK]"
WARN = "[PROCAGEN3D:WARN"
FAIL = "[PROCAGEN3D:FAIL"

VIEW_ORDER = ["front", "right", "iso", "left", "back", "top"]
# Camera direction (unit vector from center toward camera), Blender convention:
# front looks along +Y (camera on -Y side).
VIEW_DIRS = {
    "front": Vector((0.0, -1.0, 0.0)),
    "back": Vector((0.0, 1.0, 0.0)),
    "left": Vector((-1.0, 0.0, 0.0)),
    "right": Vector((1.0, 0.0, 0.0)),
    "top": Vector((0.0, 0.0, 1.0)),
}

JOINT_TYPES = ("revolute", "prismatic", "fixed")


def finish(code):
    print(f"PROCAGEN3D_EXIT:{code}")
    sys.stdout.flush()
    sys.exit(code)


def script_args():
    argv = sys.argv
    return argv[argv.index("--") + 1 :] if "--" in argv else []


def depsgraph():
    bpy.context.view_layer.update()
    return bpy.context.evaluated_depsgraph_get()


def mesh_objects():
    return [o for o in bpy.context.scene.objects if o.type == "MESH"]


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
    for obj in bpy.context.scene.objects:
        if obj.type in ("CAMERA", "LIGHT"):
            continue
        entry = {
            "name": obj.name,
            "type": obj.type,
            "parent": obj.parent.name if obj.parent else None,
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
        "procagen3d_version": "0.1",
        "blender_version": bpy.app.version_string,
        "objects": objects,
        "roots": [o.name for o in bpy.context.scene.objects
                  if o.parent is None and o.type not in ("CAMERA", "LIGHT")],
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


# ---------------------------------------------------------------- rendering

def setup_engine(scene, engine):
    if engine == "workbench":
        scene.render.engine = "BLENDER_WORKBENCH"
        sh = scene.display.shading
        sh.light = "STUDIO"
        sh.color_type = "MATERIAL"
        sh.show_object_outline = True
        sh.show_shadows = False
        sh.background_type = "VIEWPORT"
        sh.background_color = (0.92, 0.92, 0.92)
        scene.display.render_aa = "8"
        return
    # eevee / cycles need a world and a light
    world = bpy.data.worlds.new("ProcAgen3D_World")
    world.color = (0.85, 0.85, 0.85)
    scene.world = world
    sun_data = bpy.data.lights.new("ProcAgen3D_Sun", type="SUN")
    sun_data.energy = 3.0
    sun = bpy.data.objects.new("ProcAgen3D_Sun", sun_data)
    scene.collection.objects.link(sun)
    sun.rotation_euler = (math.radians(50), 0.0, math.radians(30))
    if engine == "eevee":
        scene.render.engine = "BLENDER_EEVEE_NEXT"
        scene.eevee.taa_render_samples = 16
    else:
        scene.render.engine = "CYCLES"
        scene.cycles.samples = 32
        scene.cycles.device = "CPU"


def render_views(out_dir, size, engine):
    scene = bpy.context.scene
    meshes = mesh_objects()
    if not meshes:
        print(f"{FAIL}:NO_MESHES] nothing to render")
        return []
    dg = depsgraph()
    lo, hi = union_bbox(meshes, dg)
    center = (lo + hi) / 2
    extent = max(hi[i] - lo[i] for i in range(3))
    radius = (hi - lo).length / 2 or 1.0

    setup_engine(scene, engine)
    scene.render.resolution_x = size
    scene.render.resolution_y = size
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = True  # composited over gray in the sheet

    cam_data = bpy.data.cameras.new("ProcAgen3D_Cam")
    cam = bpy.data.objects.new("ProcAgen3D_Cam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam

    renders_dir = Path(out_dir) / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for view in VIEW_ORDER:
        if view == "iso":
            cam_data.type = "PERSP"
            cam_data.angle = math.radians(40)
            direction = Vector((1.0, -1.0, 0.75)).normalized()
            dist = radius / math.tan(cam_data.angle / 2) * 1.25
        else:
            cam_data.type = "ORTHO"
            # shared ortho scale = scale normalization across canonical views
            cam_data.ortho_scale = extent * 1.15
            direction = VIEW_DIRS[view]
            dist = radius * 3 + 1.0
        cam.location = center + direction * dist
        look = (center - cam.location).normalized()
        cam.rotation_euler = look.to_track_quat("-Z", "Y").to_euler()
        cam_data.clip_end = max(100.0, dist * 4)
        path = renders_dir / f"{view}.png"
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        written.append(path)
        print(f"{OK} rendered {view} -> {path}")

    sheet = make_contact_sheet(renders_dir, size)
    if sheet:
        written.append(sheet)
    return written


def make_contact_sheet(renders_dir, size):
    try:
        import numpy as np
    except ImportError:
        print(f"{WARN}:NO_NUMPY] contact sheet skipped")
        return None
    bg = np.array([0.92, 0.92, 0.92], dtype=np.float32)
    tiles = []
    for view in VIEW_ORDER:
        path = renders_dir / f"{view}.png"
        img = bpy.data.images.load(str(path))
        px = np.array(img.pixels[:], dtype=np.float32).reshape(size, size, 4)
        alpha = px[..., 3:4]
        flat = np.empty_like(px)
        flat[..., :3] = px[..., :3] * alpha + bg * (1.0 - alpha)
        flat[..., 3] = 1.0
        tiles.append(flat)
        bpy.data.images.remove(img)
    # 2 rows x 3 cols; bpy pixel rows are bottom-up, so row 0 of the sheet is
    # the bottom row of the image -> put the second triple at the bottom.
    top = np.concatenate(tiles[0:3], axis=1)
    bottom = np.concatenate(tiles[3:6], axis=1)
    sheet_px = np.concatenate([bottom, top], axis=0)
    h, w = sheet_px.shape[0], sheet_px.shape[1]
    sheet = bpy.data.images.new("ProcAgen3D_Sheet", width=w, height=h, alpha=True)
    sheet.pixels = sheet_px.ravel().tolist()
    sheet_path = renders_dir / "sheet.png"
    sheet.filepath_raw = str(sheet_path)
    sheet.file_format = "PNG"
    sheet.save()
    print(f"{OK} contact sheet -> {sheet_path} (rows: front|right|iso / left|back|top)")
    return sheet_path


# ---------------------------------------------------------------- build stage

BANNED_PATTERNS = [
    "bpy.ops.render",
    "export_scene",
    "wm.save",
    "wm.open_mainfile",
    "urllib",
    "requests",
    "subprocess",
    "os.system",
]


def stage_build(args):
    program_path = Path(args.program)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    src = program_path.read_text()

    warnings = []
    for pat in BANNED_PATTERNS:
        if pat in src:
            warnings.append(f"program contains banned call '{pat}' "
                            "(programs must only build geometry)")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    diag = {"build_ok": False, "program": program_path.name, "warnings": warnings}
    try:
        ns = {"__name__": "procagen3d_program", "__file__": str(program_path)}
        exec(compile(src, program_path.name, "exec"), ns)
        if "build" not in ns or not callable(ns["build"]):
            raise RuntimeError(
                "program defines no build() entry point "
                "(doctrine: def build() must exist and construct the scene)")
        ns["build"]()
        bpy.context.view_layer.update()
    except Exception:
        tb = traceback.format_exc()
        diag["error"] = tb
        (out_dir / "diagnostics.json").write_text(json.dumps(diag, indent=2))
        print("PROCAGEN3D_BUILD_ERROR")
        print(tb)
        finish(1)

    graph = collect_scene_graph()
    if graph["totals"]["meshes"] == 0:
        diag["error"] = "build() produced no mesh objects"
        (out_dir / "diagnostics.json").write_text(json.dumps(diag, indent=2))
        print("PROCAGEN3D_BUILD_ERROR")
        print(diag["error"])
        finish(1)

    (out_dir / "scene_graph.json").write_text(json.dumps(graph, indent=2))

    # Semantic names must survive into the GLB: mesh datablocks inherit the
    # object's name (otherwise the GLB carries 'Cube.001' mesh names).
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            obj.data.name = obj.name

    # Export GLB before cameras/lights are added for rendering.
    bpy.ops.export_scene.gltf(
        filepath=str(out_dir / "model.glb"),
        export_format="GLB",
        export_extras=True,
        export_apply=True,
    )
    bpy.ops.wm.save_as_mainfile(filepath=str(out_dir / "scene.blend"))

    if not args.no_render:
        render_views(out_dir, args.size, args.engine)

    diag["build_ok"] = True
    diag["stats"] = graph["totals"]
    diag["elapsed_s"] = round(time.time() - t0, 1)
    (out_dir / "diagnostics.json").write_text(json.dumps(diag, indent=2))
    for w in warnings:
        print(f"{WARN}:PROGRAM] {w}")
    print(f"{OK} build complete: {graph['totals']['meshes']} meshes, "
          f"{graph['totals']['triangles']} tris, "
          f"{len(graph['joints'])} joints, {diag['elapsed_s']}s")
    finish(0)


def stage_render(args):
    out_dir = Path(args.out)
    blend = out_dir / "scene.blend"
    if not blend.exists():
        print(f"{FAIL}:NO_SCENE] {blend} not found (run build first)")
        finish(1)
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    render_views(out_dir, args.size, args.engine)
    finish(0)


# ---------------------------------------------------------------- joints stage

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

        moving = [o for o in [child] + descendants(child) if o.type == "MESH"]
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


# ---------------------------------------------------------------- entrypoint

def main():
    parser = argparse.ArgumentParser(prog="blender_stages")
    sub = parser.add_subparsers(dest="stage", required=True)

    p_build = sub.add_parser("build")
    p_build.add_argument("--program", required=True)
    p_build.add_argument("--out", required=True)
    p_build.add_argument("--size", type=int, default=512)
    p_build.add_argument("--engine", default="workbench",
                         choices=["workbench", "eevee", "cycles"])
    p_build.add_argument("--no-render", action="store_true")

    p_render = sub.add_parser("render")
    p_render.add_argument("--out", required=True)
    p_render.add_argument("--size", type=int, default=512)
    p_render.add_argument("--engine", default="workbench",
                          choices=["workbench", "eevee", "cycles"])

    p_joints = sub.add_parser("joints")
    p_joints.add_argument("--out", required=True)
    p_joints.add_argument("--strict", action="store_true")

    args = parser.parse_args(script_args())
    {"build": stage_build, "render": stage_render, "joints": stage_joints}[args.stage](args)


if __name__ == "__main__":
    main()
