"""ProcAgen3D Blender-side stages.

Not meant to be run directly — invoked by scripts/procagen3d.py as:

    blender --background --factory-startup --python-exit-code 1 \
        --python blender_stages.py -- <stage> [args...]

Stages:
    build   Execute a ProcAgen3D program, dump scene_graph.json, export GLB,
            save scene.blend, render canonical views + contact sheet.
    render  Re-render canonical views from an existing scene.blend.
    fit     Render a registered reference view and score image/pose-fit gates.
    joints  Validate articulation (pivot placement, axis, limits, sweep
            collisions, rest-pose restore) against an existing scene.blend.

Exit codes are reported both as process exit code and as a stdout sentinel
line ``PROCAGEN3D_EXIT:<code>`` (0 = ok, 1 = failure) so the driver is robust to
Blender's own exit-code quirks.
"""

import argparse
import fnmatch
import hashlib
import json
import math
import sys
import time
import traceback
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Euler, Matrix, Quaternion, Vector
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
    # Hidden Boolean cutters and construction helpers are not part of the
    # judged/export-facing asset. They must not inflate totals, form envelopes,
    # collision sets, or the camera framing used for proof renders.
    return [o for o in bpy.context.scene.objects
            if o.type == "MESH" and not o.hide_render]


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


def normal_clusters(mesh, merge_deg=6.0):
    """Group face area by surface normal, coarse-bucketed then merged.

    Bucketing is O(faces) and merging is O(buckets^2) on a set that stays small
    for anything that is not already a smooth surface, so this is affordable on
    every mesh at build time.
    """
    buckets = {}
    for poly in mesh.polygons:
        normal = poly.normal
        if normal.length_squared < 1e-12:
            continue
        normal = normal.normalized()
        key = tuple(round(component * 48.0) for component in normal)
        entry = buckets.get(key)
        if entry is None:
            buckets[key] = [normal.copy(), poly.area]
        else:
            entry[1] += poly.area
    clusters = []
    tolerance = math.cos(math.radians(merge_deg))
    for normal, area in sorted(buckets.values(), key=lambda item: -item[1]):
        for cluster in clusters:
            if normal.dot(cluster[0]) >= tolerance:
                cluster[1] += area
                break
        else:
            clusters.append([normal, area])
        if len(clusters) > 4096:
            break
    return sorted((area for _, area in clusters), reverse=True)


def section_variation(mesh, axis, stations=7):
    """Relative spread of cross-section size along ``axis``.

    Sections are cut by intersecting edges with station planes rather than by
    binning vertices, because an extrusion or a lathe carries vertices only at
    its end rings and would otherwise report no section at all.

    Near zero for a box, prism, or cylinder (constant section); large for a
    cone, taper, or loft.  This is what separates a stretched block from a form
    whose section genuinely changes.
    """
    coords = [v.co for v in mesh.vertices]
    lo = min(c[axis] for c in coords)
    hi = max(c[axis] for c in coords)
    span = hi - lo
    if span <= 1e-9:
        return 0.0
    others = [i for i in range(3) if i != axis]
    edges = {key for poly in mesh.polygons for key in poly.edge_keys}
    extents = []
    for index in range(1, stations + 1):
        level = lo + span * index / (stations + 1)
        points = []
        for start_index, end_index in edges:
            a = coords[start_index]
            b = coords[end_index]
            delta = b[axis] - a[axis]
            if abs(delta) <= 1e-12:
                continue
            t = (level - a[axis]) / delta
            if not 0.0 <= t <= 1.0:
                continue
            points.append([a[other] + (b[other] - a[other]) * t
                           for other in others])
        if len(points) < 3:
            continue
        area = 1.0
        for column in range(2):
            values = [p[column] for p in points]
            area *= max(values) - min(values)
        extents.append(math.sqrt(max(area, 0.0)))
    if len(extents) < 3:
        return 0.0
    mean = sum(extents) / len(extents)
    if mean <= 1e-9:
        return 0.0
    return (max(extents) - min(extents)) / mean


def principal_axis(points):
    """Dominant direction of a point cloud, plus how elongated it is.

    Power iteration on the covariance matrix, then one deflation for the second
    eigenvalue.  ``elongation`` near 1 means the cloud has no meaningful long
    axis, so callers can ignore compact parts.
    """
    count = len(points)
    if count < 3:
        return None, 1.0
    centre = [sum(p[i] for p in points) / count for i in range(3)]
    cov = [[0.0] * 3 for _ in range(3)]
    for point in points:
        d = [point[i] - centre[i] for i in range(3)]
        for i in range(3):
            for j in range(3):
                cov[i][j] += d[i] * d[j]

    def multiply(vector):
        return [sum(cov[i][j] * vector[j] for j in range(3)) for i in range(3)]

    def iterate(seed):
        vector = seed
        value = 0.0
        for _ in range(48):
            nxt = multiply(vector)
            length = math.sqrt(sum(c * c for c in nxt))
            if length <= 1e-18:
                return None, 0.0
            vector = [c / length for c in nxt]
            value = length
        return vector, value

    first, lambda1 = iterate([0.5773, 0.5774, 0.5775])
    if first is None:
        return None, 1.0
    for i in range(3):
        for j in range(3):
            cov[i][j] -= lambda1 * first[i] * first[j]
    seed = [1.0, 0.0, 0.0]
    if abs(first[0]) > 0.9:
        seed = [0.0, 1.0, 0.0]
    _, lambda2 = iterate(seed)
    elongation = math.sqrt(lambda1 / lambda2) if lambda2 > 1e-18 else 999.0
    return first, elongation


def shape_signature(mesh):
    """Measure the geometry that actually distinguishes one shape family from another.

    Two numbers carry most of the discrimination:

    ``fill_ratio`` — solid volume over local bounding-box volume.  A box is 1.0,
    a cylinder 0.79, an ellipsoid 0.52, a cone 0.26.

    ``planar_area_fraction`` — surface area lying in the six largest coplanar
    normal clusters.  A box is ~1.0 even after bevelling, a cylinder ~0.4, a
    sphere or loft below 0.1.

    Together they catch an ellipsoid wearing a ``box`` label, which no amount of
    metadata cross-checking can.  Returns None when the mesh is too coarse to
    classify.
    """
    polys = mesh.polygons
    if len(polys) < 4 or not mesh.vertices:
        return None
    coords = [v.co for v in mesh.vertices]
    dims = [max(c[i] for c in coords) - min(c[i] for c in coords)
            for i in range(3)]
    bbox_volume = dims[0] * dims[1] * dims[2]

    mesh.calc_loop_triangles()
    volume = 0.0
    for tri in mesh.loop_triangles:
        a, b, c = (coords[i] for i in tri.vertices)
        volume += a.dot(b.cross(c))
    volume = abs(volume) / 6.0

    boundary = 0
    edge_use = {}
    for poly in polys:
        for key in poly.edge_keys:
            edge_use[key] = edge_use.get(key, 0) + 1
    for count in edge_use.values():
        if count == 1:
            boundary += 1

    areas = normal_clusters(mesh)
    total_area = sum(areas)
    if total_area <= 1e-12:
        return None
    planar = sum(areas[:6]) / total_area
    longest = max(range(3), key=lambda i: dims[i])

    ordered = sorted(dims)
    return {
        "fill_ratio": round(volume / bbox_volume, 4) if bbox_volume > 1e-12 else 0.0,
        "planar_area_fraction": round(planar, 4),
        "largest_planar_fraction": round(areas[0] / total_area, 4),
        "normal_cluster_count": len(areas),
        "section_variation": round(section_variation(mesh, longest), 4),
        "thin_ratio": round(ordered[0] / ordered[2], 4) if ordered[2] > 1e-12 else 0.0,
        "local_dims": [round(d, 6) for d in dims],
        # Rotation-invariant size, so repeated parts stay comparable even when
        # a builder bakes each instance's rotation into its vertices.
        "volume": round(volume, 12),
        "surface_area": round(total_area, 12),
        "closed": boundary == 0,
    }


def assembly_intersections(dg, limit=12):
    """Measure real interpenetration between top-level assemblies.

    Bounding boxes are useless here — a side table tucked beside a sofa arm
    overlaps boxes without the solids touching, while a lamp sunk into a sofa
    back is a genuine fault.  So this intersects actual triangles and then
    reports the *thickness* of the intersecting region: resting contact yields
    a thin sheet whose smallest dimension is near zero, whereas one object
    buried in another yields a region fat in all three directions.
    """
    scene = bpy.context.scene
    roots = [obj for obj in scene.objects
             if obj.parent is None and obj.type in ("EMPTY", "MESH")]
    assemblies = []
    for root in roots:
        for child in root.children:
            members = [child] + descendants(child)
            meshes = [o for o in members if o.type == "MESH" and not o.hide_render]
            if meshes:
                assemblies.append((child.name, meshes))
    if len(assemblies) < 2:
        return []

    trees, extents = {}, {}
    for name, meshes in assemblies:
        tree = bvh_from_objects(meshes, dg)
        if tree is None:
            continue
        trees[name] = tree
        lo, hi = union_bbox(meshes, dg)
        extents[name] = min(hi[i] - lo[i] for i in range(3))

    samples = {}
    for name, meshes in assemblies:
        if name not in trees:
            continue
        points = []
        for obj in meshes:
            evaluated = obj.evaluated_get(dg)
            mesh = evaluated.to_mesh()
            matrix = evaluated.matrix_world
            points.extend(matrix @ v.co for v in mesh.vertices)
            evaluated.to_mesh_clear()
        stride = max(1, len(points) // 4000)
        samples[name] = points[::stride]

    def deepest_inside(points, tree):
        """How far the deepest point sits inside a closed surface.

        Sign comes from the surface normal at the nearest point, so a vertex
        resting exactly on another surface scores ~0 while one buried in a
        solid scores its true depth.
        """
        depth = 0.0
        inside = 0
        for point in points:
            location, normal, _, distance = tree.find_nearest(point)
            if location is None or distance is None:
                continue
            if (point - location).dot(normal) < 0.0:
                inside += 1
                depth = max(depth, distance)
        return depth, inside

    out = []
    names = sorted(trees)
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            if not trees[first].overlap(trees[second]):
                continue
            depth_a, count_a = deepest_inside(samples[second], trees[first])
            depth_b, count_b = deepest_inside(samples[first], trees[second])
            depth = max(depth_a, depth_b)
            reference = min(extents.get(first, 0.0), extents.get(second, 0.0))
            out.append({
                "a": first,
                "b": second,
                "points_inside": count_a + count_b,
                "penetration_m": round(depth, 6),
                "penetration_fraction": (round(depth / reference, 4)
                                         if reference > 1e-9 else 0.0),
            })
    out.sort(key=lambda item: -item["penetration_fraction"])
    return out[:limit]


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
            signature = shape_signature(me)
            if signature is not None:
                world = [ev.matrix_world @ v.co for v in me.vertices]
                axis, elongation = principal_axis(world)
                if axis is not None:
                    signature["world_axis"] = [round(c, 6) for c in axis]
                    signature["elongation"] = round(min(elongation, 999.0), 4)
                entry["shape_signature"] = signature
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
    try:
        overlaps = assembly_intersections(dg)
    except Exception as exc:  # diagnostics must never break a valid build
        print(f"{WARN}:INTERSECTION_SCAN] skipped: {exc}")
        overlaps = []
    if overlaps:
        graph["assembly_intersections"] = overlaps
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
        # 4.2–4.5 shipped EEVEE Next as BLENDER_EEVEE_NEXT; 5.0 restored
        # BLENDER_EEVEE as the only identifier.
        scene.render.engine = (
            "BLENDER_EEVEE" if bpy.app.version >= (5, 0, 0)
            else "BLENDER_EEVEE_NEXT")
        scene.eevee.taa_render_samples = 16
    else:
        scene.render.engine = "CYCLES"
        scene.cycles.samples = 32
        scene.cycles.device = "CPU"


def setup_form_engine(scene):
    """Neutral clay diagnostic: expose surface flow without material camouflage."""
    scene.render.engine = "BLENDER_WORKBENCH"
    sh = scene.display.shading
    sh.light = "STUDIO"
    sh.color_type = "SINGLE"
    sh.single_color = (0.46, 0.49, 0.53)
    sh.show_object_outline = False
    sh.show_shadows = True
    sh.show_cavity = True
    sh.cavity_type = "WORLD"
    sh.show_specular_highlight = True
    sh.background_type = "VIEWPORT"
    sh.background_color = (0.92, 0.92, 0.92)
    scene.display.render_aa = "8"


def reference_camera_contract(scene, discard_invalid=False):
    """Read the first valid root reference-camera contract."""
    key = "procagen3d_reference_camera"
    projection_key = "procagen3d_reference_projection"
    selected = None
    for obj in scene.objects:
        if obj.parent is not None or key not in obj:
            continue
        candidate = None
        try:
            values = [float(v) for v in obj[key]]
            if len(values) == 3:
                azimuth, elevation, framing = values
                raw_projection = obj.get(projection_key, "perspective")
                projection = (raw_projection.lower()
                              if isinstance(raw_projection, str) else None)
                framing_valid = (
                    5.0 <= framing <= 120.0 if projection == "perspective"
                    else 1e-5 <= framing <= 1e6
                    if projection == "orthographic" else False
                )
                if (all(math.isfinite(value) for value in values)
                        and -360.0 <= azimuth <= 360.0
                        and -89.0 < elevation < 89.0 and framing_valid):
                    candidate = (projection, azimuth, elevation, framing)
        except (TypeError, ValueError, OverflowError):
            pass
        if candidate is not None:
            if selected is None:
                selected = candidate
            if not discard_invalid:
                return selected
            continue
        print(f"{WARN}:REFERENCE_CAMERA] invalid reference projection/camera "
              f"contract on {obj.name}: projection="
              f"{obj.get(projection_key, 'perspective')!r}, "
              f"camera={obj.get(key)!r}")
        if discard_invalid:
            del obj[key]
            if projection_key in obj:
                del obj[projection_key]
    return selected


def render_views(out_dir, size, engine, form_diagnostics=False):
    scene = bpy.context.scene
    meshes = mesh_objects()
    if not meshes:
        print(f"{FAIL}:NO_MESHES] nothing to render")
        return [], None
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
    # These products are conditional. Clear exact known paths before each
    # render so a removed/invalid camera contract or disabled form pass cannot
    # leave evidence from an older scene in the inspection directory.
    optional_paths = [renders_dir / "reference_match.png",
                      renders_dir / "form_sheet.png"]
    optional_paths.extend(renders_dir / f"form_{view}.png"
                          for view in VIEW_ORDER)
    for stale_path in optional_paths:
        stale_path.unlink(missing_ok=True)
    written = []

    def position_canonical(view):
        if view == "iso":
            cam_data.type = "PERSP"
            cam_data.angle = math.radians(40)
            direction = Vector((1.0, -1.0, 0.75)).normalized()
            dist = radius / math.sin(cam_data.angle / 2) * 1.15
        else:
            cam_data.type = "ORTHO"
            # shared ortho scale = scale normalization across canonical views
            cam_data.ortho_scale = extent * 1.15
            direction = VIEW_DIRS[view]
            dist = radius * 3 + 1.0
        cam.location = center + direction * dist
        look = (center - cam.location).normalized()
        cam.rotation_euler = look.to_track_quat("-Z", "Y").to_euler()
        cam_data.clip_start = max(
            1e-5, min(0.1, max(dist - radius, 1e-4) * 0.25))
        cam_data.clip_end = max(100.0, dist * 4)

    def render_canonical_set(prefix=""):
        paths = []
        for view in VIEW_ORDER:
            position_canonical(view)
            path = renders_dir / f"{prefix}{view}.png"
            scene.render.filepath = str(path)
            bpy.ops.render.render(write_still=True)
            paths.append(path)
            print(f"{OK} rendered {prefix}{view} -> {path}")
        return paths

    written.extend(render_canonical_set())
    silhouettes = measure_view_silhouettes(renders_dir, size)
    sheet = make_contact_sheet(renders_dir, size)
    if sheet:
        written.append(sheet)

    # Optional reference-camera contract on the root object. The third value
    # is vertical FOV degrees for perspective (default), or vertical world
    # scale when procagen3d_reference_projection == "orthographic". Azimuth 0
    # is canonical front (-Y); positive azimuth turns toward +X.
    camera_contract = reference_camera_contract(scene)
    if camera_contract:
        projection, azimuth, elevation, framing = camera_contract
        az = math.radians(azimuth)
        el = math.radians(elevation)
        direction = Vector((math.sin(az) * math.cos(el),
                            -math.cos(az) * math.cos(el),
                            math.sin(el))).normalized()
        if projection == "orthographic":
            cam_data.type = "ORTHO"
            cam_data.ortho_scale = framing
            dist = radius * 3 + 1.0
        else:
            cam_data.type = "PERSP"
            cam_data.angle = math.radians(framing)
            # Fit the entire bounding sphere. tan() assumes a flat subject and
            # can put wide-FOV cameras inside long/round geometry; angular
            # radius is asin(radius / distance), hence the sin() denominator.
            dist = radius / math.sin(cam_data.angle / 2) * 1.15
        cam.location = center + direction * dist
        look = (center - cam.location).normalized()
        cam.rotation_euler = look.to_track_quat("-Z", "Y").to_euler()
        cam_data.clip_start = max(
            1e-5, min(0.1, max(dist - radius, 1e-4) * 0.25))
        cam_data.clip_end = max(100.0, dist * 4)
        path = renders_dir / "reference_match.png"
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        written.append(path)
        print(f"{OK} rendered reference camera -> {path}")

    if form_diagnostics:
        setup_form_engine(scene)
        written.extend(render_canonical_set("form_"))
        form_sheet = make_contact_sheet(
            renders_dir, size, prefix="form_", output_name="form_sheet.png")
        if form_sheet:
            written.append(form_sheet)
    return written, silhouettes


def measure_view_silhouettes(renders_dir, size):
    """Silhouette coverage of the orthographic canonical views.

    All five ortho views share one ortho_scale and resolution, so their
    foreground areas are directly comparable.  A model that fits a reference
    from one camera by flattening into a bas-relief keeps a large front area
    and loses almost all of its side and top area, which is invisible to any
    single-view image metric but obvious here.
    """
    try:
        import numpy as np
    except ImportError:
        print(f"{WARN}:NO_NUMPY] view silhouette measurement skipped")
        return None
    out = {}
    for view in VIEW_ORDER:
        if view == "iso":  # perspective; not area-comparable with the ortho set
            continue
        path = renders_dir / f"{view}.png"
        if not path.is_file():
            continue
        img = bpy.data.images.load(str(path))
        px = np.array(img.pixels[:], dtype=np.float32).reshape(size, size, 4)
        bpy.data.images.remove(img)
        mask = px[..., 3] > 0.5
        area = float(mask.mean())
        entry = {"area_fraction": round(area, 6)}
        if mask.any():
            rows = np.flatnonzero(mask.any(axis=1))
            cols = np.flatnonzero(mask.any(axis=0))
            entry["extent_fraction"] = [
                round(float(cols[-1] - cols[0] + 1) / size, 6),
                round(float(rows[-1] - rows[0] + 1) / size, 6),
            ]
        out[view] = entry
    return out or None


def make_contact_sheet(renders_dir, size, prefix="", output_name="sheet.png"):
    try:
        import numpy as np
    except ImportError:
        print(f"{WARN}:NO_NUMPY] contact sheet skipped")
        return None
    bg = np.array([0.92, 0.92, 0.92], dtype=np.float32)
    tiles = []
    for view in VIEW_ORDER:
        path = renders_dir / f"{prefix}{view}.png"
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
    image_name = "ProcAgen3D_" + output_name.replace(".png", "")
    sheet = bpy.data.images.new(image_name, width=w, height=h, alpha=True)
    sheet.pixels = sheet_px.ravel().tolist()
    sheet_path = renders_dir / output_name
    sheet.filepath_raw = str(sheet_path)
    sheet.file_format = "PNG"
    sheet.save()
    print(f"{OK} contact sheet -> {sheet_path} (rows: front|right|iso / left|back|top)")
    return sheet_path


# ---------------------------------------------------------------- image-fit stage

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fit_path(out_dir, value, label):
    """Resolve a fit artifact inside the asset output directory."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty path string")
    root = Path(out_dir).resolve()
    candidate = (root / value).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"{label} must stay inside {root}: {value!r}")
    if not candidate.is_file():
        raise ValueError(f"{label} not found: {candidate}")
    return candidate


def finite_values(value, count, label):
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise ValueError(f"{label} must contain {count} numbers")
    try:
        values = [float(item) for item in value]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must contain {count} numbers") from exc
    if not all(math.isfinite(item) for item in values):
        raise ValueError(f"{label} must contain finite numbers")
    return values


def load_rgba(path):
    """Load an image as a top-down float RGBA numpy array."""
    import numpy as np

    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        width, height = (int(value) for value in image.size)
        if width <= 0 or height <= 0:
            raise ValueError(f"image has invalid dimensions: {path}")
        pixels = np.array(image.pixels[:], dtype=np.float32)
        rgba = pixels.reshape(height, width, 4)
        return np.flipud(rgba).copy()
    finally:
        bpy.data.images.remove(image)


def save_rgba(path, rgba):
    """Save a top-down float RGBA numpy array without external imaging deps."""
    import numpy as np

    height, width, channels = rgba.shape
    if channels != 4:
        raise ValueError("save_rgba expects HxWx4 pixels")
    image = bpy.data.images.new(
        "ProcAgen3D_" + Path(path).stem, width=width, height=height, alpha=True)
    try:
        stored = np.flipud(np.clip(rgba, 0.0, 1.0)).astype(np.float32)
        image.pixels = stored.ravel().tolist()
        image.filepath_raw = str(path)
        image.file_format = "PNG"
        image.save()
    finally:
        bpy.data.images.remove(image)


def save_mask(path, mask):
    import numpy as np

    value = mask.astype(np.float32)
    rgba = np.empty((*value.shape, 4), dtype=np.float32)
    rgba[..., :3] = value[..., None]
    rgba[..., 3] = 1.0
    save_rgba(path, rgba)


def reference_mask(reference_rgba, config, out_dir):
    """Return a foreground mask from alpha, a supplied mask, or border color."""
    import numpy as np

    source = str(config.get("source", "auto")).lower()
    alpha_threshold = float(config.get("alpha_threshold", 0.5))
    if not 0.0 <= alpha_threshold <= 1.0:
        raise ValueError("mask.alpha_threshold must be within [0, 1]")
    alpha = reference_rgba[..., 3]
    has_transparency = bool(np.any(alpha < 0.98) and np.any(alpha > alpha_threshold))
    if source == "auto":
        source = "alpha" if has_transparency else "border"

    if source == "alpha":
        if not has_transparency:
            raise ValueError(
                "mask.source='alpha' requested but the reference has no useful alpha")
        mask = alpha > alpha_threshold
    elif source == "file":
        mask_path = fit_path(out_dir, config.get("path"), "mask.path")
        supplied = load_rgba(mask_path)
        if supplied.shape[:2] != reference_rgba.shape[:2]:
            raise ValueError(
                f"mask dimensions {supplied.shape[1]}x{supplied.shape[0]} do not "
                f"match reference {reference_rgba.shape[1]}x{reference_rgba.shape[0]}")
        supplied_alpha = supplied[..., 3]
        if np.any(supplied_alpha < 0.98):
            mask = supplied_alpha > alpha_threshold
        else:
            luminance = supplied[..., :3].mean(axis=2)
            mask = luminance > float(config.get("value_threshold", 0.5))
    elif source == "border":
        rgb = reference_rgba[..., :3]
        height, width = rgb.shape[:2]
        band = max(1, int(round(min(height, width) * 0.02)))
        border = np.concatenate((
            rgb[:band].reshape(-1, 3),
            rgb[-band:].reshape(-1, 3),
            rgb[:, :band].reshape(-1, 3),
            rgb[:, -band:].reshape(-1, 3),
        ), axis=0)
        background = np.median(border, axis=0)
        threshold = float(config.get("color_threshold", 0.08))
        if not 0.0 < threshold <= math.sqrt(3.0):
            raise ValueError("mask.color_threshold must be within (0, sqrt(3)]")
        mask = np.linalg.norm(rgb - background, axis=2) > threshold
    else:
        raise ValueError("mask.source must be auto, alpha, border, or file")

    if bool(config.get("invert", False)):
        mask = ~mask
    if not np.any(mask):
        raise ValueError("reference foreground mask is empty")
    if np.all(mask):
        raise ValueError("reference foreground mask covers the entire image")
    return mask, source


def mask_observation(mask):
    import numpy as np

    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("foreground mask is empty")
    height, width = mask.shape
    bbox = [
        float(xs.min()) / width,
        float(ys.min()) / height,
        float(xs.max() + 1) / width,
        float(ys.max() + 1) / height,
    ]
    centroid = [
        float((xs.astype(np.float64) + 0.5).mean()) / width,
        float((ys.astype(np.float64) + 0.5).mean()) / height,
    ]
    return {
        "bbox_uv": bbox,
        "centroid_uv": centroid,
        "area_fraction": float(mask.mean()),
    }


def bbox_iou(a, b):
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    inter = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 1e-12 else 0.0


def fit_camera(scene, camera_config, width, height):
    projection = str(camera_config.get("projection", "perspective")).lower()
    if projection not in ("perspective", "orthographic"):
        raise ValueError("camera.projection must be perspective or orthographic")
    target = Vector(finite_values(
        camera_config.get("target_m", [0, 0, 0]), 3, "camera.target_m"))
    roll = float(camera_config.get("roll_deg", 0.0))
    if not math.isfinite(roll):
        raise ValueError("camera.roll_deg must be finite")

    if "location_m" in camera_config:
        location = Vector(finite_values(
            camera_config["location_m"], 3, "camera.location_m"))
        direction = location - target
        distance = direction.length
        if distance <= 1e-6:
            raise ValueError("camera.location_m must differ from camera.target_m")
        direction.normalize()
    else:
        azimuth = float(camera_config.get("azimuth_deg", 0.0))
        elevation = float(camera_config.get("elevation_deg", 0.0))
        distance = float(camera_config.get("distance_m", 0.0))
        if not all(math.isfinite(value) for value in (azimuth, elevation, distance)):
            raise ValueError("camera azimuth/elevation/distance must be finite")
        if not -89.0 < elevation < 89.0:
            raise ValueError("camera.elevation_deg must be within (-89, 89)")
        if distance <= 0.0:
            raise ValueError("camera.distance_m must be positive")
        azimuth = math.radians(azimuth)
        elevation = math.radians(elevation)
        direction = Vector((
            math.sin(azimuth) * math.cos(elevation),
            -math.cos(azimuth) * math.cos(elevation),
            math.sin(elevation),
        )).normalized()
        location = target + direction * distance

    data = bpy.data.cameras.new("ProcAgen3D_FitCam")
    camera = bpy.data.objects.new("ProcAgen3D_FitCam", data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    data.sensor_fit = "VERTICAL"
    if projection == "perspective":
        fov = float(camera_config.get("fov_y_deg", 0.0))
        if not math.isfinite(fov) or not 5.0 <= fov <= 120.0:
            raise ValueError("camera.fov_y_deg must be within [5, 120]")
        data.type = "PERSP"
        data.angle = math.radians(fov)
    else:
        scale = float(camera_config.get("ortho_scale_m", 0.0))
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("camera.ortho_scale_m must be positive")
        data.type = "ORTHO"
        data.ortho_scale = scale

    data.shift_x = float(camera_config.get("shift_x", 0.0))
    data.shift_y = float(camera_config.get("shift_y", 0.0))
    if not all(math.isfinite(value) for value in (data.shift_x, data.shift_y)):
        raise ValueError("camera shift values must be finite")
    camera.location = location
    look = (target - location).normalized()
    base_rotation = look.to_track_quat("-Z", "Y")
    roll_rotation = Quaternion(look, math.radians(roll))
    camera.rotation_euler = (roll_rotation @ base_rotation).to_euler()
    data.clip_start = max(1e-5, min(0.1, distance * 0.01))
    data.clip_end = max(100.0, distance * 10.0)

    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = True
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = 1.0
    return camera, {
        "projection": projection,
        "location_m": [float(value) for value in location],
        "target_m": [float(value) for value in target],
        "roll_deg": roll,
        "shift_x": float(data.shift_x),
        "shift_y": float(data.shift_y),
        "resolution_px": [width, height],
        **({"fov_y_deg": float(camera_config["fov_y_deg"])}
           if projection == "perspective" else
           {"ortho_scale_m": float(camera_config["ortho_scale_m"])}),
    }


def place_camera(camera, params):
    """Point an existing camera using the fit-spec parameter convention.

    Same maths as fit_camera, but reusing one camera object so the solver can
    evaluate thousands of candidate viewpoints without leaking datablocks.
    """
    azimuth = math.radians(params["azimuth_deg"])
    elevation = math.radians(params["elevation_deg"])
    direction = Vector((
        math.sin(azimuth) * math.cos(elevation),
        -math.cos(azimuth) * math.cos(elevation),
        math.sin(elevation),
    )).normalized()
    target = Vector((params["target_x"], params["target_y"], params["target_z"]))
    location = target + direction * params["distance_m"]
    camera.location = location
    look = (target - location).normalized()
    base_rotation = look.to_track_quat("-Z", "Y")
    roll_rotation = Quaternion(look, math.radians(params["roll_deg"]))
    camera.rotation_euler = (roll_rotation @ base_rotation).to_euler()
    camera.data.angle = math.radians(params["fov_y_deg"])
    camera.data.shift_x = params["shift_x"]
    camera.data.shift_y = params["shift_y"]
    camera.data.clip_start = max(1e-5, min(0.1, params["distance_m"] * 0.01))
    camera.data.clip_end = max(100.0, params["distance_m"] * 10.0)


CAMERA_SOLVE_BOUNDS = {
    "azimuth_deg": (-180.0, 180.0),
    "elevation_deg": (-88.0, 88.0),
    "roll_deg": (-45.0, 45.0),
    "root_pitch_deg": (-40.0, 40.0),
    "root_yaw_deg": (-60.0, 60.0),
    "root_roll_deg": (-40.0, 40.0),
    "fov_y_deg": (5.0, 120.0),
    "distance_m": (1e-3, 1e6),
    "target_x": (-1e6, 1e6),
    "target_y": (-1e6, 1e6),
    "target_z": (-1e6, 1e6),
    "shift_x": (-2.0, 2.0),
    "shift_y": (-2.0, 2.0),
}


def solve_symmetric(matrix, rhs):
    """Gaussian elimination with partial pivoting on a small dense system."""
    n = len(rhs)
    a = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda r: abs(a[r][column]))
        if abs(a[pivot][column]) < 1e-18:
            return None
        a[column], a[pivot] = a[pivot], a[column]
        inverse = 1.0 / a[column][column]
        for row in range(column + 1, n):
            factor = a[row][column] * inverse
            if factor:
                for k in range(column, n + 1):
                    a[row][k] -= factor * a[column][k]
    out = [0.0] * n
    for row in range(n - 1, -1, -1):
        total = a[row][n] - sum(a[row][k] * out[k] for k in range(row + 1, n))
        out[row] = total / a[row][row]
    return out


def levenberg_marquardt(residual_vector, start, scales, iterations=120):
    """Damped least squares over a residual vector, with numerical Jacobians.

    Camera resection has a long curved valley — distance trades against field
    of view, elevation against target height — which a simplex method crawls
    along and stalls in.  Gauss-Newton with Marquardt damping follows it.
    """
    params = list(start)
    residuals = residual_vector(params)
    cost = sum(r * r for r in residuals)
    n = len(params)
    damping = 1e-3
    for _ in range(iterations):
        jacobian = []
        for index in range(n):
            step = scales[index]
            forward = list(params)
            forward[index] += step
            shifted = residual_vector(forward)
            jacobian.append([(shifted[k] - residuals[k]) / step
                             for k in range(len(residuals))])
        jtj = [[sum(jacobian[i][k] * jacobian[j][k]
                    for k in range(len(residuals)))
                for j in range(n)] for i in range(n)]
        jtr = [sum(jacobian[i][k] * residuals[k] for k in range(len(residuals)))
               for i in range(n)]
        improved = False
        for _ in range(12):
            damped = [row[:] for row in jtj]
            for i in range(n):
                damped[i][i] += damping * max(jtj[i][i], 1e-12)
            delta = solve_symmetric(damped, [-value for value in jtr])
            if delta is None:
                damping *= 10.0
                continue
            candidate = [params[i] + delta[i] for i in range(n)]
            trial = residual_vector(candidate)
            trial_cost = sum(r * r for r in trial)
            if trial_cost < cost:
                params, residuals, cost = candidate, trial, trial_cost
                damping = max(damping * 0.3, 1e-12)
                improved = True
                break
            damping *= 10.0
            if damping > 1e12:
                break
        if not improved:
            break
    return params, cost


def nelder_mead(objective, start, steps, iterations=4000, tolerance=1e-12):
    """Derivative-free simplex minimisation over a fixed parameter ordering."""
    n = len(start)
    simplex = [list(start)]
    for index in range(n):
        point = list(start)
        point[index] += steps[index]
        simplex.append(point)
    scores = [objective(point) for point in simplex]
    for _ in range(iterations):
        order = sorted(range(n + 1), key=lambda i: scores[i])
        simplex = [simplex[i] for i in order]
        scores = [scores[i] for i in order]
        if abs(scores[-1] - scores[0]) <= tolerance * (abs(scores[0]) + tolerance):
            break
        centroid = [sum(point[i] for point in simplex[:-1]) / n for i in range(n)]
        worst = simplex[-1]
        reflected = [centroid[i] + (centroid[i] - worst[i]) for i in range(n)]
        score = objective(reflected)
        if score < scores[0]:
            expanded = [centroid[i] + 2.0 * (centroid[i] - worst[i])
                        for i in range(n)]
            expanded_score = objective(expanded)
            if expanded_score < score:
                simplex[-1], scores[-1] = expanded, expanded_score
            else:
                simplex[-1], scores[-1] = reflected, score
        elif score < scores[-2]:
            simplex[-1], scores[-1] = reflected, score
        else:
            contracted = [centroid[i] + 0.5 * (worst[i] - centroid[i])
                          for i in range(n)]
            contracted_score = objective(contracted)
            if contracted_score < scores[-1]:
                simplex[-1], scores[-1] = contracted, contracted_score
            else:
                best = simplex[0]
                for i in range(1, n + 1):
                    simplex[i] = [best[j] + 0.5 * (simplex[i][j] - best[j])
                                  for j in range(n)]
                    scores[i] = objective(simplex[i])
    order = sorted(range(n + 1), key=lambda i: scores[i])
    return simplex[order[0]], scores[order[0]]


def resect_dlt(correspondences, width, height):
    """Linear camera resection: the globally-consistent starting point.

    Solves the 3x4 projection matrix by SVD (no local minima), then splits it
    into intrinsics, rotation, and centre.  Iterative refinement afterwards
    only has to polish, instead of hunting for the right basin among the
    distance/field-of-view and elevation/height ambiguities.
    """
    import numpy as np

    if len(correspondences) < 6:
        return None
    world = np.array([[p[0], p[1], p[2]] for _, p, _ in correspondences],
                     dtype=np.float64)
    pixels = np.array([[uv[0] * width, uv[1] * height]
                       for _, _, uv in correspondences], dtype=np.float64)

    # Hartley normalisation: raw metres and pixels differ by orders of
    # magnitude and make the design matrix badly conditioned.
    world_centre = world.mean(axis=0)
    world_scale = np.sqrt(3.0) / max(
        1e-12, np.linalg.norm(world - world_centre, axis=1).mean())
    pixel_centre = pixels.mean(axis=0)
    pixel_scale = np.sqrt(2.0) / max(
        1e-12, np.linalg.norm(pixels - pixel_centre, axis=1).mean())
    normalized_world = (world - world_centre) * world_scale
    normalized_pixels = (pixels - pixel_centre) * pixel_scale

    rows = []
    for (x, y, z), (u, v) in zip(normalized_world, normalized_pixels):
        rows.append([x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u])
        rows.append([0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z, -v])
    _, _, vt = np.linalg.svd(np.array(rows, dtype=np.float64))
    projection = vt[-1].reshape(3, 4)

    unnormalize = np.array([[1.0 / pixel_scale, 0, pixel_centre[0]],
                            [0, 1.0 / pixel_scale, pixel_centre[1]],
                            [0, 0, 1.0]])
    scale_world = np.eye(4) * world_scale
    scale_world[3, 3] = 1.0
    scale_world[:3, 3] = -world_centre * world_scale
    projection = unnormalize @ projection @ scale_world

    m = projection[:, :3]
    if abs(np.linalg.det(m)) < 1e-18:
        return None
    centre = -np.linalg.inv(m) @ projection[:, 3]

    # RQ decomposition of M into upper-triangular K and rotation R.
    reverse = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.float64)
    q, r = np.linalg.qr((reverse @ m).T)
    intrinsics = reverse @ r.T @ reverse
    rotation = reverse @ q.T
    signs = np.diag(np.sign(np.diag(intrinsics)))
    intrinsics = intrinsics @ signs
    rotation = signs @ rotation
    if np.linalg.det(rotation) < 0:
        rotation = -rotation
    if abs(intrinsics[2, 2]) < 1e-18:
        return None
    intrinsics = intrinsics / intrinsics[2, 2]

    focal_y = abs(intrinsics[1, 1])
    if focal_y < 1e-9:
        return None
    fov_y = math.degrees(2.0 * math.atan(height / (2.0 * focal_y)))
    if not 1.0 < fov_y < 170.0:
        return None

    # Image-convention rotation rows: x right, y down, z along the view ray.
    forward = Vector(rotation[2, :].tolist()).normalized()
    up = -Vector(rotation[1, :].tolist()).normalized()
    centre_vec = Vector(centre.tolist())
    # Keep the subject in front of the camera; DLT is sign-ambiguous.
    ahead = sum((Vector(p) - centre_vec).dot(forward)
                for _, p, _ in correspondences)
    if ahead < 0:
        forward, up = -forward, -up
    mean_point = Vector((0.0, 0.0, 0.0))
    for _, point, _ in correspondences:
        mean_point += Vector(point)
    mean_point /= len(correspondences)
    distance = max(1e-3, (mean_point - centre_vec).dot(forward))
    target = centre_vec + forward * distance

    direction = (centre_vec - target).normalized()
    elevation = math.degrees(math.asin(max(-1.0, min(1.0, direction.z))))
    azimuth = math.degrees(math.atan2(direction.x, -direction.y))
    look = -direction
    base_up = look.to_track_quat("-Z", "Y") @ Vector((0.0, 1.0, 0.0))
    roll = math.degrees(math.atan2(base_up.cross(up).dot(look),
                                   base_up.dot(up)))
    return {
        "azimuth_deg": azimuth,
        "elevation_deg": max(-88.0, min(88.0, elevation)),
        "roll_deg": ((roll + 180.0) % 360.0) - 180.0,
        "fov_y_deg": max(5.0, min(120.0, fov_y)),
        "distance_m": distance,
        "target_x": target.x, "target_y": target.y, "target_z": target.z,
    }


def solve_camera(scene, camera, correspondences, initial, free_names,
                 width, height):
    """Least-squares camera resection against observed image landmarks.

    ``correspondences`` are (id, world point, observed uv) triples read off the
    reference.  The camera that best explains them is computed rather than
    guessed, so a tilted or rotated subject no longer has to be imitated by
    deforming geometry.
    """
    order = [name for name in free_names]
    params = dict(initial)
    report_seed = []

    # A standing subject leans about its ground contact, so that is the pivot
    # for any root rotation the caller asks us to estimate.
    pivot = Vector((
        sum(p[0] for _, p, _ in correspondences) / len(correspondences),
        sum(p[1] for _, p, _ in correspondences) / len(correspondences),
        min(p[2] for _, p, _ in correspondences),
    ))
    root_names = ("root_pitch_deg", "root_yaw_deg", "root_roll_deg")

    def posed(point):
        if not any(abs(params.get(name, 0.0)) > 1e-12 for name in root_names):
            return Vector(point)
        rotation = Euler((
            math.radians(params.get("root_pitch_deg", 0.0)),
            math.radians(params.get("root_roll_deg", 0.0)),
            math.radians(params.get("root_yaw_deg", 0.0)),
        ), "XYZ").to_matrix()
        return pivot + rotation @ (Vector(point) - pivot)

    def residuals(values=None):
        if values is not None:
            for name, value in zip(order, values):
                low, high = CAMERA_SOLVE_BOUNDS[name]
                params[name] = min(high, max(low, value))
        place_camera(camera, params)
        scene.view_layers[0].update()
        out = []
        for name, point, observed in correspondences:
            projected = world_to_camera_view(scene, camera, posed(point))
            if projected.z <= 0.0:
                out.append((name, 10.0))
                continue
            uv = (float(projected.x), float(1.0 - projected.y))
            out.append((name, math.hypot(uv[0] - observed[0],
                                         uv[1] - observed[1])))
        return out

    def objective(values):
        return sum(error * error for _, error in residuals(values))

    def residual_vector(values):
        for name, value in zip(order, values):
            low, high = CAMERA_SOLVE_BOUNDS[name]
            params[name] = min(high, max(low, value))
        place_camera(camera, params)
        scene.view_layers[0].update()
        out = []
        for _, point, observed in correspondences:
            projected = world_to_camera_view(scene, camera, posed(point))
            if projected.z <= 0.0:
                out.extend((10.0, 10.0))
                continue
            out.append(float(projected.x) - observed[0])
            out.append(float(1.0 - projected.y) - observed[1])
        return out

    jacobian_steps = {
        "azimuth_deg": 0.02, "elevation_deg": 0.02, "roll_deg": 0.02,
        "fov_y_deg": 0.02, "shift_x": 1e-4, "shift_y": 1e-4,
        "root_pitch_deg": 0.02, "root_yaw_deg": 0.02, "root_roll_deg": 0.02,
        "distance_m": max(1e-4, initial["distance_m"] * 1e-4),
        "target_x": max(1e-4, initial["distance_m"] * 1e-4),
        "target_y": max(1e-4, initial["distance_m"] * 1e-4),
        "target_z": max(1e-4, initial["distance_m"] * 1e-4),
    }

    steps = {
        "azimuth_deg": 6.0, "elevation_deg": 4.0, "roll_deg": 3.0,
        "fov_y_deg": 4.0, "shift_x": 0.05, "shift_y": 0.05,
        "root_pitch_deg": 4.0, "root_yaw_deg": 4.0, "root_roll_deg": 3.0,
        "distance_m": max(1e-3, initial["distance_m"] * 0.12),
        "target_x": max(1e-3, initial["distance_m"] * 0.03),
        "target_y": max(1e-3, initial["distance_m"] * 0.03),
        "target_z": max(1e-3, initial["distance_m"] * 0.03),
    }
    best_values, best_score = None, math.inf
    seeds = [dict(initial)]
    linear = resect_dlt(correspondences, width, height)
    if linear is not None:
        seed = dict(initial)
        seed.update({name: value for name, value in linear.items()
                     if name in order or name in initial})
        seeds.insert(0, seed)
        report_seed.append(linear)
    # The linear solution is the reliable start.  A few offsets around the
    # declared guess are kept as a fallback for the case where too few or too
    # coplanar correspondences make resection degenerate.
    for azimuth_offset in (-20.0, 0.0, 20.0):
        for elevation_offset in (-10.0, 0.0, 10.0):
            seed = dict(initial)
            seed["azimuth_deg"] = initial["azimuth_deg"] + azimuth_offset
            seed["elevation_deg"] = max(
                -88.0, min(88.0, initial["elevation_deg"] + elevation_offset))
            seeds.append(seed)
    for index, seed in enumerate(seeds):
        start = [seed[name] for name in order]
        if index > 0:
            # Simplex first to land in a basin, then damped least squares.
            start, _ = nelder_mead(
                objective, start, [steps[name] for name in order],
                iterations=500)
        values, score = levenberg_marquardt(
            residual_vector, start, [jacobian_steps[n] for n in order])
        if score < best_score:
            best_values, best_score = values, score
    for name, value in zip(order, best_values):
        low, high = CAMERA_SOLVE_BOUNDS[name]
        params[name] = min(high, max(low, value))
    final = residuals()
    errors = [error for _, error in final]
    rms = math.sqrt(sum(e * e for e in errors) / len(errors))
    return params, final, rms, (report_seed[0] if report_seed else None)


def matched_geometry(scene, pattern):
    matched = [obj for obj in scene.objects
               if fnmatch.fnmatchcase(obj.name, pattern)]
    if not matched:
        raise ValueError(f"no object matches {pattern!r}")
    meshes = []
    seen = set()
    stack = list(matched)
    while stack:
        obj = stack.pop()
        if obj.name in seen:
            continue
        seen.add(obj.name)
        if obj.type == "MESH" and not obj.hide_render:
            meshes.append(obj)
        stack.extend(obj.children)
    return matched, meshes


def project_world(scene, camera, point):
    projected = world_to_camera_view(scene, camera, Vector(point))
    return [float(projected.x), float(1.0 - projected.y)], float(projected.z)


def projected_instance(scene, camera, dg, pattern):
    matched, meshes = matched_geometry(scene, pattern)
    if not meshes:
        raise ValueError(f"{pattern!r} matches no renderable mesh geometry")
    uv_points = []
    world_lo, world_hi = union_bbox(meshes, dg)
    for obj in meshes:
        evaluated = obj.evaluated_get(dg)
        mesh = evaluated.to_mesh()
        try:
            matrix = evaluated.matrix_world
            for vertex in mesh.vertices:
                uv, depth = project_world(scene, camera, matrix @ vertex.co)
                if depth > 0.0:
                    uv_points.append(uv)
        finally:
            evaluated.to_mesh_clear()
    if not uv_points:
        raise ValueError(f"{pattern!r} projects entirely behind the camera")
    left = min(point[0] for point in uv_points)
    top = min(point[1] for point in uv_points)
    right = max(point[0] for point in uv_points)
    bottom = max(point[1] for point in uv_points)
    center_world = (world_lo + world_hi) / 2.0
    center_uv, center_depth = project_world(scene, camera, center_world)
    return {
        "pattern": pattern,
        "matches": [obj.name for obj in matched],
        "bbox_uv": [left, top, right, bottom],
        "centroid_uv": center_uv,
        "camera_depth_m": center_depth,
    }


def bbox_anchor(bbox, anchor):
    left, top, right, bottom = bbox
    center_x, center_y = (left + right) / 2.0, (top + bottom) / 2.0
    anchors = {
        "bbox_center": [center_x, center_y],
        "bbox_left": [left, center_y],
        "bbox_right": [right, center_y],
        "bbox_top": [center_x, top],
        "bbox_bottom": [center_x, bottom],
        "bbox_top_left": [left, top],
        "bbox_top_right": [right, top],
        "bbox_bottom_left": [left, bottom],
        "bbox_bottom_right": [right, bottom],
    }
    if anchor not in anchors:
        raise ValueError(
            "landmark.anchor must be origin or bbox_center/left/right/top/bottom/corner")
    return anchors[anchor]


def landmark_uv(scene, camera, dg, entry):
    if "world_point_m" in entry:
        uv, depth = project_world(scene, camera, finite_values(
            entry["world_point_m"], 3, "landmark.world_point_m"))
        if depth <= 0.0:
            raise ValueError("landmark.world_point_m projects behind the camera")
        return uv
    pattern = entry.get("pattern")
    if not isinstance(pattern, str) or not pattern:
        raise ValueError("landmark requires pattern or world_point_m")
    anchor = str(entry.get("anchor", "origin"))
    if anchor == "origin":
        matched, _ = matched_geometry(scene, pattern)
        if len(matched) != 1:
            raise ValueError(
                f"origin landmark pattern {pattern!r} matches {len(matched)} objects")
        uv, depth = project_world(scene, camera, matched[0].matrix_world.translation)
        if depth <= 0.0:
            raise ValueError(f"landmark {pattern!r} projects behind the camera")
        return uv
    observation = projected_instance(scene, camera, dg, pattern)
    return bbox_anchor(observation["bbox_uv"], anchor)


def uv_distance(a, b, axis):
    if axis == "x":
        return abs(a[0] - b[0])
    if axis == "y":
        return abs(a[1] - b[1])
    if axis != "distance":
        raise ValueError("ratio.axis must be distance, x, or y")
    return math.hypot(a[0] - b[0], a[1] - b[1])


def directed_angle_deg(a, b):
    """Directed image-plane angle from a to b, in degrees."""
    if math.hypot(b[0] - a[0], b[1] - a[1]) <= 1e-9:
        raise ValueError("pose segment has zero image length")
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def angle_error_deg(a, b):
    """Smallest absolute error between two directed angles."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def joint_angle_deg(a, pivot, b):
    """Unsigned image-plane bend angle at pivot, in degrees."""
    va = (a[0] - pivot[0], a[1] - pivot[1])
    vb = (b[0] - pivot[0], b[1] - pivot[1])
    la = math.hypot(*va)
    lb = math.hypot(*vb)
    if la <= 1e-9 or lb <= 1e-9:
        raise ValueError("pose joint contains a zero-length segment")
    cosine = max(-1.0, min(1.0, (va[0] * vb[0] + va[1] * vb[1]) / (la * lb)))
    return math.degrees(math.acos(cosine))


def fit_gate(gates, gate_id, kind, target, measured, passed, note=""):
    gates.append({
        "id": str(gate_id),
        "kind": kind,
        "target": target,
        "measured": measured,
        "pass": bool(passed),
        "note": note,
    })


def draw_cross(image, uv, color, radius=4):
    height, width = image.shape[:2]
    x = int(round(uv[0] * (width - 1)))
    y = int(round(uv[1] * (height - 1)))
    if not (0 <= x < width and 0 <= y < height):
        return
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    image[y, x0:x1, :3] = color
    image[y0:y1, x, :3] = color


def draw_box(image, bbox, color):
    height, width = image.shape[:2]
    left = max(0, min(width - 1, int(round(bbox[0] * (width - 1)))))
    top = max(0, min(height - 1, int(round(bbox[1] * (height - 1)))))
    right = max(0, min(width - 1, int(round(bbox[2] * (width - 1)))))
    bottom = max(0, min(height - 1, int(round(bbox[3] * (height - 1)))))
    if left >= right or top >= bottom:
        return
    image[top:top + 2, left:right + 1, :3] = color
    image[max(top, bottom - 1):bottom + 1, left:right + 1, :3] = color
    image[top:bottom + 1, left:left + 2, :3] = color
    image[top:bottom + 1, max(left, right - 1):right + 1, :3] = color


# Registered-fit strictness lives here, in code, not in the agent-authored spec.
# A spec that asks for a looser number is clamped back to these values and the
# run is failed, because a self-chosen pass mark measures nothing.  Floors are
# indexed by the reconstruction plan's complexity class: a mecha genuinely
# cannot register as tightly as a camera body, but it may not pick its own bar.
FIT_MASK_IOU_FLOOR = {
    "simple": 0.88, "moderate": 0.84, "complex": 0.80, "extreme": 0.76,
}
FIT_REGION_IOU_FLOOR = {
    "simple": 0.85, "moderate": 0.82, "complex": 0.78, "extreme": 0.74,
}
FIT_DEFAULT_COMPLEXITY = "complex"
FIT_ERROR_CEILINGS = {
    "bbox_max_error": 0.05,
    "centroid_max_error": 0.04,
    "area_ratio_max_error": 0.20,
    "region_area_ratio_max_error": 0.18,
    "landmark_max_error": 0.04,
    "ratio_max_relative_error": 0.10,
    "pose_axis_max_angle_error_deg": 4.0,
    "pose_segment_max_angle_error_deg": 5.0,
    "pose_joint_max_angle_error_deg": 7.0,
    "pose_length_fraction_max_error": 0.04,
    "instance_bbox_max_error": 0.05,
    "instance_centroid_max_error": 0.04,
    "relation_max_error": 0.05,
}
# Wiry subjects (bicycle, drone, antenna arrays) lose IoU to one-pixel
# registration slop in a way a solid body does not.  The allowance is derived
# from the reference mask itself, so it cannot be claimed by declaration.
FIT_THINNESS_RELIEF = 0.20
FIT_THINNESS_RELIEF_CAP = 0.15
# How much evidence the reconstruction actually had to work from.  From one
# view the depth of every part is unobservable, so silhouette error accumulates
# out of depth choices that are each individually defensible; demanding the
# same registration as a multi-view reconstruction asks for something the input
# does not contain.  More views make the problem better posed, so the bar rises.
FIT_VIEW_ADJUSTMENT = {1: -0.08, 2: 0.0, 3: 0.03}
FIT_VIEW_ADJUSTMENT_MAX = 0.03


def count_reference_views(out_dir):
    """Distinct reference images saved for this asset, by content."""
    digests = set()
    for path in sorted(Path(out_dir).glob("reference_[0-9][0-9].*")):
        if path.is_file():
            digests.add(sha256_file(path))
    return max(1, len(digests))
# Reading a point off an image is worth roughly a pixel at best.  Below this,
# the target was computed from the model rather than observed.
LANDMARK_PROVENANCE_FLOOR = 0.001
LANDMARK_PROVENANCE_MIN_SAMPLES = 5


def erode_mask(mask, radius):
    """Plus-shaped binary erosion, ``radius`` iterations."""
    import numpy as np

    out = mask
    for _ in range(max(0, radius)):
        shrunk = out.copy()
        shrunk[1:, :] &= out[:-1, :]
        shrunk[:-1, :] &= out[1:, :]
        shrunk[:, 1:] &= out[:, :-1]
        shrunk[:, :-1] &= out[:, 1:]
        shrunk[0, :] = False
        shrunk[-1, :] = False
        shrunk[:, 0] = False
        shrunk[:, -1] = False
        out = shrunk
    return out


def mask_thinness(mask):
    """Fraction of foreground that a small erosion removes: 0 solid, ~1 wiry."""
    total = int(mask.sum())
    if not total:
        return 0.0
    height, width = mask.shape[:2]
    radius = max(1, int(round(0.004 * min(height, width))))
    return 1.0 - float(erode_mask(mask, radius).sum()) / total


class FitThresholdPolicy:
    """Resolve every fit threshold as the stricter of policy and spec."""

    def __init__(self, complexity_class, views=1):
        self.complexity = (complexity_class
                           if complexity_class in FIT_MASK_IOU_FLOOR
                           else FIT_DEFAULT_COMPLEXITY)
        self.views = max(1, int(views))
        self.thinness = 0.0
        self.violations = []
        self.applied = {}

    def observe_reference(self, reference_fg):
        self.thinness = round(mask_thinness(reference_fg), 4)

    def _relief(self):
        return min(FIT_THINNESS_RELIEF_CAP, FIT_THINNESS_RELIEF * self.thinness)

    def view_adjustment(self):
        return FIT_VIEW_ADJUSTMENT.get(self.views, FIT_VIEW_ADJUSTMENT_MAX)

    def mask_iou_floor(self):
        return round(FIT_MASK_IOU_FLOOR[self.complexity]
                     + self.view_adjustment() - self._relief(), 4)

    def region_iou_floor(self):
        return round(FIT_REGION_IOU_FLOOR[self.complexity]
                     + self.view_adjustment() - self._relief(), 4)

    def floor(self, label, requested, limit):
        """Minimum-style threshold: the spec may raise it, never lower it."""
        if requested is None:
            return limit
        value = float(requested)
        if value < limit - 1e-9:
            self.violations.append(
                f"{label}: spec asks {value:.4g}, policy floor is {limit:.4g}")
            return limit
        return value

    def ceiling(self, label, requested, name):
        """Maximum-style threshold: the spec may tighten it, never loosen it."""
        limit = FIT_ERROR_CEILINGS[name]
        if requested is None:
            return limit
        value = float(requested)
        if value > limit + 1e-9:
            self.violations.append(
                f"{label}: spec asks {value:.4g}, policy ceiling is {limit:.4g}")
            return limit
        return value


def load_complexity_class(out_dir):
    path = Path(out_dir) / "reconstruction_plan.json"
    if not path.is_file():
        return None
    try:
        plan = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    complexity = plan.get("complexity") if isinstance(plan, dict) else None
    if isinstance(complexity, dict):
        return complexity.get("class")
    return None


def stage_fit(args):
    import numpy as np

    out_dir = Path(args.out).resolve()
    blend = out_dir / "scene.blend"
    spec_path = Path(args.spec).resolve()
    renders_dir = out_dir / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "fit_report.json"
    artifact_names = (
        "reference_match.png", "reference_overlay.png",
        "reference_mask.png", "render_mask.png",
    )
    report_path.unlink(missing_ok=True)
    for name in artifact_names:
        (renders_dir / name).unlink(missing_ok=True)

    base_report = {
        "procagen3d_fit_version": 2,
        "fit_spec": spec_path.name,
        "passed": False,
        "gates": [],
    }
    try:
        if not blend.is_file():
            raise ValueError(f"{blend} not found (run build first)")
        if not spec_path.is_file():
            raise ValueError(f"fit spec not found: {spec_path}")
        spec = json.loads(spec_path.read_text())
        spec_version = spec.get("version") if isinstance(spec, dict) else None
        if spec_version not in (1, 2):
            raise ValueError("fit spec must be a JSON object with version: 1 or 2")
        base_report["fit_spec_version"] = spec_version
        pose_config = spec.get("pose")
        if spec_version == 2:
            if not isinstance(pose_config, dict):
                raise ValueError("fit spec version 2 requires a pose object")
            pose_mode = str(pose_config.get("mode", ""))
            if pose_mode not in ("rigid", "articulated", "unobservable"):
                raise ValueError(
                    "pose.mode must be rigid, articulated, or unobservable")
            frame_axes = pose_config.get("frame_axes", [])
            pose_chains = pose_config.get("chains", [])
            if not isinstance(frame_axes, list) or not isinstance(pose_chains, list):
                raise ValueError("pose.frame_axes and pose.chains must be lists")
            if pose_mode == "rigid" and not frame_axes:
                raise ValueError("rigid pose requires at least one frame axis")
            if pose_mode == "articulated" and not pose_chains:
                raise ValueError("articulated pose requires at least one pose chain")
            if pose_mode == "unobservable" and not str(pose_config.get("reason", "")).strip():
                raise ValueError("unobservable pose requires a non-empty reason")
        reference_path = fit_path(
            out_dir, spec.get("reference_image"), "reference_image")
        reference_rgba = load_rgba(reference_path)
        height, width = reference_rgba.shape[:2]
        input_hashes = {
            "fit_spec_sha256": sha256_file(spec_path),
            "reference_sha256": sha256_file(reference_path),
            "scene_graph_sha256": sha256_file(out_dir / "scene_graph.json"),
            "scene_blend_sha256": sha256_file(blend),
        }

        bpy.ops.wm.open_mainfile(filepath=str(blend))
        setup_engine(bpy.context.scene, args.engine)
        camera, normalized_camera = fit_camera(
            bpy.context.scene, spec.get("camera", {}), width, height)
        render_path = renders_dir / "reference_match.png"
        bpy.context.scene.render.filepath = str(render_path)
        bpy.ops.render.render(write_still=True)
        rendered_rgba = load_rgba(render_path)
        if rendered_rgba.shape[:2] != reference_rgba.shape[:2]:
            raise ValueError("registered render dimensions do not match reference")

        thresholds = spec.get("thresholds", {})
        if not isinstance(thresholds, dict):
            raise ValueError("thresholds must be an object")
        policy = FitThresholdPolicy(load_complexity_class(out_dir),
                                    count_reference_views(out_dir))
        gates = []
        mask_config = spec.get("mask")
        reference_fg = None
        render_fg = rendered_rgba[..., 3] > 0.5
        overlay = reference_rgba.copy()
        overlay[..., 3] = 1.0
        if mask_config is not None:
            if not isinstance(mask_config, dict):
                raise ValueError("mask must be an object")
            reference_fg, mask_source = reference_mask(
                reference_rgba, mask_config, out_dir)
            if not np.any(render_fg):
                raise ValueError("registered render mask is empty")
            ref_obs = mask_observation(reference_fg)
            render_obs = mask_observation(render_fg)
            intersection = np.logical_and(reference_fg, render_fg).sum()
            union = np.logical_or(reference_fg, render_fg).sum()
            iou = float(intersection / union) if union else 0.0
            bbox_error = max(abs(a - b) for a, b in zip(
                ref_obs["bbox_uv"], render_obs["bbox_uv"]))
            centroid_error = math.hypot(*(
                ref_obs["centroid_uv"][index] - render_obs["centroid_uv"][index]
                for index in range(2)))
            area_ratio_error = abs(
                render_obs["area_fraction"] / ref_obs["area_fraction"] - 1.0)
            policy.observe_reference(reference_fg)
            min_iou = policy.floor(
                "mask.min_iou",
                mask_config.get("min_iou", thresholds.get("mask_min_iou")),
                policy.mask_iou_floor())
            max_bbox = policy.ceiling(
                "mask.max_bbox_error",
                mask_config.get("max_bbox_error",
                                thresholds.get("bbox_max_error")),
                "bbox_max_error")
            max_centroid = policy.ceiling(
                "mask.max_centroid_error",
                mask_config.get("max_centroid_error",
                                thresholds.get("centroid_max_error")),
                "centroid_max_error")
            max_area = policy.ceiling(
                "mask.max_area_ratio_error",
                mask_config.get("max_area_ratio_error",
                                thresholds.get("area_ratio_max_error")),
                "area_ratio_max_error")
            fit_gate(gates, "mask_iou", "mask", f">= {min_iou:.4f}",
                     round(iou, 6), iou >= min_iou)
            fit_gate(gates, "mask_bbox", "mask", f"<= {max_bbox:.4f}",
                     round(bbox_error, 6), bbox_error <= max_bbox)
            fit_gate(gates, "mask_centroid", "mask", f"<= {max_centroid:.4f}",
                     round(centroid_error, 6), centroid_error <= max_centroid)
            fit_gate(gates, "mask_area_ratio", "mask", f"<= {max_area:.4f}",
                     round(area_ratio_error, 6), area_ratio_error <= max_area)
            save_mask(renders_dir / "reference_mask.png", reference_fg)
            save_mask(renders_dir / "render_mask.png", render_fg)
            overlap = np.logical_and(reference_fg, render_fg)
            reference_only = np.logical_and(reference_fg, ~render_fg)
            render_only = np.logical_and(render_fg, ~reference_fg)
            overlay[overlap, :3] = 0.45 * overlay[overlap, :3] + np.array(
                [0.1, 0.9, 0.2], dtype=np.float32) * 0.55
            overlay[reference_only, :3] = 0.35 * overlay[reference_only, :3] + np.array(
                [1.0, 0.1, 0.1], dtype=np.float32) * 0.65
            overlay[render_only, :3] = 0.35 * overlay[render_only, :3] + np.array(
                [0.1, 0.35, 1.0], dtype=np.float32) * 0.65
            base_report["mask"] = {
                "source": mask_source,
                "reference": ref_obs,
                "render": render_obs,
            }
            if mask_source == "file":
                supplied_mask_path = fit_path(
                    out_dir, mask_config.get("path"), "mask.path")
                input_hashes["mask_sha256"] = sha256_file(supplied_mask_path)
                base_report["mask"]["path"] = supplied_mask_path.name

        silhouette_regions = spec.get("silhouette_regions", [])
        if not isinstance(silhouette_regions, list):
            raise ValueError("silhouette_regions must be a list")
        if spec_version == 2 and len(silhouette_regions) < 3:
            raise ValueError(
                "fit spec version 2 requires at least three silhouette_regions")
        if silhouette_regions and reference_fg is None:
            raise ValueError("silhouette_regions require a foreground mask")
        region_records = []
        region_ids = set()
        region_boxes = set()
        for index, entry in enumerate(silhouette_regions):
            if not isinstance(entry, dict):
                raise ValueError(f"silhouette_regions[{index}] must be an object")
            region_id = str(entry.get("id", f"silhouette_region_{index + 1}"))
            if region_id in region_ids:
                raise ValueError(f"duplicate silhouette region id: {region_id}")
            region_ids.add(region_id)
            bbox = finite_values(
                entry.get("bbox_uv"), 4, f"silhouette region {region_id}.bbox_uv")
            if (not all(0.0 <= value <= 1.0 for value in bbox)
                    or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]):
                raise ValueError(
                    f"silhouette region {region_id} has an invalid normalized bbox")
            bbox_key = tuple(round(value, 6) for value in bbox)
            if bbox_key in region_boxes:
                raise ValueError(
                    f"silhouette region {region_id} duplicates another bbox")
            region_boxes.add(bbox_key)
            left = max(0, min(width - 1, int(math.floor(bbox[0] * width))))
            top = max(0, min(height - 1, int(math.floor(bbox[1] * height))))
            right = max(left + 1, min(width, int(math.ceil(bbox[2] * width))))
            bottom = max(top + 1, min(height, int(math.ceil(bbox[3] * height))))
            reference_crop = reference_fg[top:bottom, left:right]
            render_crop = render_fg[top:bottom, left:right]
            occupancy = float(reference_crop.mean())
            if not 0.02 <= occupancy <= 0.98:
                raise ValueError(
                    f"silhouette region {region_id} is not contour-informative "
                    f"(reference occupancy {occupancy:.3f}; need 0.02..0.98)")
            intersection = int(np.logical_and(reference_crop, render_crop).sum())
            union = int(np.logical_or(reference_crop, render_crop).sum())
            region_iou = float(intersection / union) if union else 0.0
            reference_area = int(reference_crop.sum())
            render_area = int(render_crop.sum())
            area_error = (abs(render_area / reference_area - 1.0)
                          if reference_area else math.inf)
            min_iou = policy.floor(
                f"silhouette region {region_id}.min_iou",
                entry.get("min_iou", thresholds.get("region_min_iou")),
                policy.region_iou_floor())
            max_area = policy.ceiling(
                f"silhouette region {region_id}.max_area_ratio_error",
                entry.get("max_area_ratio_error",
                          thresholds.get("region_area_ratio_max_error")),
                "region_area_ratio_max_error")
            if spec_version == 2 and not 0.0 <= min_iou <= 1.0:
                raise ValueError(
                    f"silhouette region {region_id}.min_iou must be within "
                    "[0.00, 1.00]")
            if spec_version == 2 and not 0.0 <= max_area <= 0.35:
                raise ValueError(
                    f"silhouette region {region_id}.max_area_ratio_error must "
                    "be within [0.00, 0.35] for fit spec version 2")
            fit_gate(gates, f"{region_id}.iou", "silhouette_region",
                     f">= {min_iou:.4f}", round(region_iou, 6),
                     region_iou >= min_iou)
            fit_gate(gates, f"{region_id}.area", "silhouette_region",
                     f"<= {max_area:.4f}", round(area_error, 6),
                     area_error <= max_area)
            region_records.append({
                "id": region_id,
                "bbox_uv": bbox,
                "reference_occupancy": occupancy,
                "iou": region_iou,
                "area_ratio_error": area_error,
            })
            draw_box(overlay, bbox, (1.0, 0.2, 0.9))
        base_report["silhouette_regions"] = region_records

        dg = depsgraph()
        landmark_records = []
        landmark_map = {}
        default_landmark_error = thresholds.get("landmark_max_error")
        landmarks = spec.get("landmarks", [])
        if not isinstance(landmarks, list):
            raise ValueError("landmarks must be a list")
        for index, entry in enumerate(landmarks):
            if not isinstance(entry, dict):
                raise ValueError(f"landmarks[{index}] must be an object")
            landmark_id = str(entry.get("id", f"landmark_{index + 1}"))
            if landmark_id in landmark_map:
                raise ValueError(f"duplicate landmark id: {landmark_id}")
            reference_uv = finite_values(
                entry.get("reference_uv"), 2, f"landmark {landmark_id}.reference_uv")
            try:
                render_uv = landmark_uv(bpy.context.scene, camera, dg, entry)
                error = math.hypot(
                    reference_uv[0] - render_uv[0], reference_uv[1] - render_uv[1])
                record = {
                    "id": landmark_id,
                    "reference_uv": reference_uv,
                    "render_uv": render_uv,
                    "error": error,
                }
                landmark_map[landmark_id] = record
                if bool(entry.get("gate", True)):
                    maximum = policy.ceiling(
                        f"landmark {landmark_id}.max_error",
                        entry.get("max_error", default_landmark_error),
                        "landmark_max_error")
                    fit_gate(gates, landmark_id, "landmark", f"<= {maximum:.4f}",
                             round(error, 6), error <= maximum)
                draw_cross(overlay, reference_uv, (1.0, 0.85, 0.05))
                draw_cross(overlay, render_uv, (0.0, 0.95, 1.0))
            except ValueError as exc:
                record = {"id": landmark_id, "reference_uv": reference_uv,
                          "error": str(exc)}
                fit_gate(gates, landmark_id, "landmark", "resolvable",
                         "unmeasurable", False, str(exc))
            landmark_records.append(record)
        base_report["landmarks"] = landmark_records

        # A landmark read off an image carries irreducible estimation error.
        # Twenty of them agreeing with the render to six decimal places did not
        # come from the image: they were back-filled by projecting the model
        # through its own camera, which turns every landmark, ratio, frame-axis,
        # and pose-chain gate into the model grading itself.
        gated = [record for record in landmark_records
                 if isinstance(record.get("error"), (int, float))]
        if len(gated) >= LANDMARK_PROVENANCE_MIN_SAMPLES:
            errors = sorted(record["error"] for record in gated)
            median = errors[len(errors) // 2]
            base_report["landmark_provenance"] = {
                "median_error": round(median, 9),
                "max_error": round(errors[-1], 9),
                "floor": LANDMARK_PROVENANCE_FLOOR,
            }
            if median < LANDMARK_PROVENANCE_FLOOR:
                fit_gate(
                    gates, "landmark_provenance", "policy",
                    f"median error >= {LANDMARK_PROVENANCE_FLOOR:g}",
                    round(median, 9), False,
                    f"{len(gated)} landmarks agree with the render to a median "
                    f"of {median:.2g} — reference_uv was projected from the "
                    "model, not observed in the image, so every landmark, "
                    "ratio, and pose gate here is vacuous; re-read each point "
                    "off the saved reference")

        ratio_records = []
        ratios = spec.get("ratios", [])
        if not isinstance(ratios, list):
            raise ValueError("ratios must be a list")
        for index, entry in enumerate(ratios):
            if not isinstance(entry, dict):
                raise ValueError(f"ratios[{index}] must be an object")
            ratio_id = str(entry.get("id", f"ratio_{index + 1}"))
            numerator = entry.get("numerator")
            denominator = entry.get("denominator")
            try:
                if (not isinstance(numerator, list) or len(numerator) != 2
                        or not isinstance(denominator, list) or len(denominator) != 2):
                    raise ValueError("numerator and denominator need two landmark ids")
                points = [landmark_map.get(str(item))
                          for item in numerator + denominator]
                if any(point is None for point in points):
                    raise ValueError("ratio references an unresolved landmark")
                axis = str(entry.get("axis", "distance"))
                ref_num = uv_distance(points[0]["reference_uv"],
                                      points[1]["reference_uv"], axis)
                ref_den = uv_distance(points[2]["reference_uv"],
                                      points[3]["reference_uv"], axis)
                got_num = uv_distance(points[0]["render_uv"],
                                      points[1]["render_uv"], axis)
                got_den = uv_distance(points[2]["render_uv"],
                                      points[3]["render_uv"], axis)
                if ref_den <= 1e-9 or got_den <= 1e-9 or ref_num <= 1e-9:
                    raise ValueError("ratio contains a zero-length segment")
                target_ratio = ref_num / ref_den
                measured_ratio = got_num / got_den
                relative_error = abs(measured_ratio / target_ratio - 1.0)
                maximum = policy.ceiling(
                    f"ratio {ratio_id}.max_relative_error",
                    entry.get("max_relative_error",
                              thresholds.get("ratio_max_relative_error")),
                    "ratio_max_relative_error")
                record = {
                    "id": ratio_id,
                    "target": target_ratio,
                    "measured": measured_ratio,
                    "relative_error": relative_error,
                }
                fit_gate(gates, ratio_id, "ratio", f"relative error <= {maximum:.4f}",
                         round(relative_error, 6), relative_error <= maximum,
                         f"target={target_ratio:.4f}, measured={measured_ratio:.4f}")
            except ValueError as exc:
                record = {"id": ratio_id, "error": str(exc)}
                fit_gate(gates, ratio_id, "ratio", "measurable", "unmeasurable",
                         False, str(exc))
            ratio_records.append(record)
        base_report["ratios"] = ratio_records

        pose_records = {"mode": None, "frame_axes": [], "chains": []}
        if isinstance(pose_config, dict):
            pose_records["mode"] = str(pose_config.get("mode", ""))
            pose_ids = set()
            default_axis_error = thresholds.get("pose_axis_max_angle_error_deg")
            for index, entry in enumerate(pose_config.get("frame_axes", [])):
                if not isinstance(entry, dict):
                    raise ValueError(f"pose.frame_axes[{index}] must be an object")
                axis_id = str(entry.get("id", f"frame_axis_{index + 1}"))
                if axis_id in pose_ids:
                    raise ValueError(f"duplicate pose gate id: {axis_id}")
                pose_ids.add(axis_id)
                ids = entry.get("landmarks")
                try:
                    if not isinstance(ids, list) or len(ids) != 2:
                        raise ValueError("frame axis needs exactly two landmark ids")
                    points = [landmark_map.get(str(item)) for item in ids]
                    if any(point is None for point in points):
                        raise ValueError("frame axis references an unresolved landmark")
                    reference_angle = directed_angle_deg(
                        points[0]["reference_uv"], points[1]["reference_uv"])
                    render_angle = directed_angle_deg(
                        points[0]["render_uv"], points[1]["render_uv"])
                    error = angle_error_deg(reference_angle, render_angle)
                    maximum = policy.ceiling(
                        f"pose frame axis {axis_id}.max_angle_error_deg",
                        entry.get("max_angle_error_deg", default_axis_error),
                        "pose_axis_max_angle_error_deg")
                    if maximum <= 0.0:
                        raise ValueError(
                            "pose frame-axis max_angle_error_deg must be positive")
                    record = {
                        "id": axis_id,
                        "landmarks": ids,
                        "reference_angle_deg": reference_angle,
                        "render_angle_deg": render_angle,
                        "error_deg": error,
                    }
                    fit_gate(gates, axis_id, "pose_axis",
                             f"angle error <= {maximum:.3f} deg",
                             round(error, 6), error <= maximum,
                             f"reference={reference_angle:.2f}, render={render_angle:.2f}")
                except ValueError as exc:
                    record = {"id": axis_id, "landmarks": ids, "error": str(exc)}
                    fit_gate(gates, axis_id, "pose_axis", "measurable",
                             "unmeasurable", False, str(exc))
                pose_records["frame_axes"].append(record)

            default_segment_error = thresholds.get(
                "pose_segment_max_angle_error_deg")
            default_joint_error = thresholds.get(
                "pose_joint_max_angle_error_deg")
            default_length_error = thresholds.get(
                "pose_length_fraction_max_error")
            for index, entry in enumerate(pose_config.get("chains", [])):
                if not isinstance(entry, dict):
                    raise ValueError(f"pose.chains[{index}] must be an object")
                chain_id = str(entry.get("id", f"pose_chain_{index + 1}"))
                if chain_id in pose_ids:
                    raise ValueError(f"duplicate pose gate id: {chain_id}")
                pose_ids.add(chain_id)
                ids = entry.get("landmarks")
                try:
                    if not isinstance(ids, list) or len(ids) < 3:
                        raise ValueError("pose chain needs at least three landmark ids")
                    points = [landmark_map.get(str(item)) for item in ids]
                    if any(point is None for point in points):
                        raise ValueError("pose chain references an unresolved landmark")
                    reference_points = [point["reference_uv"] for point in points]
                    render_points = [point["render_uv"] for point in points]
                    reference_segments = [
                        directed_angle_deg(a, b)
                        for a, b in zip(reference_points, reference_points[1:])]
                    render_segments = [
                        directed_angle_deg(a, b)
                        for a, b in zip(render_points, render_points[1:])]
                    segment_errors = [
                        angle_error_deg(a, b)
                        for a, b in zip(reference_segments, render_segments)]
                    reference_joints = [
                        joint_angle_deg(reference_points[i - 1],
                                        reference_points[i], reference_points[i + 1])
                        for i in range(1, len(reference_points) - 1)]
                    render_joints = [
                        joint_angle_deg(render_points[i - 1],
                                        render_points[i], render_points[i + 1])
                        for i in range(1, len(render_points) - 1)]
                    joint_errors = [abs(a - b)
                                    for a, b in zip(reference_joints, render_joints)]
                    reference_lengths = [uv_distance(a, b, "distance")
                                         for a, b in zip(
                                             reference_points, reference_points[1:])]
                    render_lengths = [uv_distance(a, b, "distance")
                                      for a, b in zip(render_points, render_points[1:])]
                    reference_total = sum(reference_lengths)
                    render_total = sum(render_lengths)
                    if reference_total <= 1e-9 or render_total <= 1e-9:
                        raise ValueError("pose chain has zero total length")
                    length_errors = [
                        abs(a / reference_total - b / render_total)
                        for a, b in zip(reference_lengths, render_lengths)]
                    maximum_segment = policy.ceiling(
                        f"pose chain {chain_id}.max_segment_angle_error_deg",
                        entry.get("max_segment_angle_error_deg",
                                  default_segment_error),
                        "pose_segment_max_angle_error_deg")
                    maximum_joint = policy.ceiling(
                        f"pose chain {chain_id}.max_joint_angle_error_deg",
                        entry.get("max_joint_angle_error_deg",
                                  default_joint_error),
                        "pose_joint_max_angle_error_deg")
                    maximum_length = policy.ceiling(
                        f"pose chain {chain_id}.max_length_fraction_error",
                        entry.get("max_length_fraction_error",
                                  default_length_error),
                        "pose_length_fraction_max_error")
                    if min(maximum_segment, maximum_joint, maximum_length) <= 0.0:
                        raise ValueError(
                            "pose chain error tolerances must be positive")
                    measured_segment = max(segment_errors, default=0.0)
                    measured_joint = max(joint_errors, default=0.0)
                    measured_length = max(length_errors, default=0.0)
                    fit_gate(gates, f"{chain_id}.segments", "pose_chain",
                             f"angle error <= {maximum_segment:.3f} deg",
                             round(measured_segment, 6),
                             measured_segment <= maximum_segment)
                    fit_gate(gates, f"{chain_id}.joints", "pose_chain",
                             f"bend error <= {maximum_joint:.3f} deg",
                             round(measured_joint, 6), measured_joint <= maximum_joint)
                    fit_gate(gates, f"{chain_id}.lengths", "pose_chain",
                             f"fraction error <= {maximum_length:.4f}",
                             round(measured_length, 6),
                             measured_length <= maximum_length)
                    record = {
                        "id": chain_id,
                        "landmarks": ids,
                        "segment_angle_errors_deg": segment_errors,
                        "joint_angle_errors_deg": joint_errors,
                        "length_fraction_errors": length_errors,
                    }
                except ValueError as exc:
                    record = {"id": chain_id, "landmarks": ids,
                              "error": str(exc)}
                    fit_gate(gates, chain_id, "pose_chain", "measurable",
                             "unmeasurable", False, str(exc))
                pose_records["chains"].append(record)
        base_report["pose"] = pose_records

        instance_records = []
        instance_map = {}
        instances = spec.get("instances", [])
        if not isinstance(instances, list):
            raise ValueError("instances must be a list")
        for index, entry in enumerate(instances):
            if not isinstance(entry, dict):
                raise ValueError(f"instances[{index}] must be an object")
            instance_id = str(entry.get("id", f"instance_{index + 1}"))
            if instance_id in instance_map:
                raise ValueError(f"duplicate instance id: {instance_id}")
            pattern = entry.get("pattern")
            reference_bbox = finite_values(
                entry.get("reference_bbox_uv"), 4,
                f"instance {instance_id}.reference_bbox_uv")
            if reference_bbox[0] >= reference_bbox[2] or reference_bbox[1] >= reference_bbox[3]:
                raise ValueError(f"instance {instance_id} has an invalid reference bbox")
            reference_centroid = finite_values(
                entry.get("reference_centroid_uv", [
                    (reference_bbox[0] + reference_bbox[2]) / 2.0,
                    (reference_bbox[1] + reference_bbox[3]) / 2.0,
                ]), 2, f"instance {instance_id}.reference_centroid_uv")
            try:
                observation = projected_instance(
                    bpy.context.scene, camera, dg, str(pattern))
                observation.update({
                    "id": instance_id,
                    "reference_bbox_uv": reference_bbox,
                    "reference_centroid_uv": reference_centroid,
                })
                bbox_error = max(abs(a - b) for a, b in zip(
                    reference_bbox, observation["bbox_uv"]))
                centroid_error = math.hypot(
                    reference_centroid[0] - observation["centroid_uv"][0],
                    reference_centroid[1] - observation["centroid_uv"][1])
                observation["bbox_error"] = bbox_error
                observation["centroid_error"] = centroid_error
                max_bbox = policy.ceiling(
                    f"instance {instance_id}.max_bbox_error",
                    entry.get("max_bbox_error",
                              thresholds.get("instance_bbox_max_error")),
                    "instance_bbox_max_error")
                max_centroid = policy.ceiling(
                    f"instance {instance_id}.max_centroid_error",
                    entry.get("max_centroid_error",
                              thresholds.get("instance_centroid_max_error")),
                    "instance_centroid_max_error")
                fit_gate(gates, f"{instance_id}.bbox", "instance",
                         f"<= {max_bbox:.4f}", round(bbox_error, 6),
                         bbox_error <= max_bbox)
                fit_gate(gates, f"{instance_id}.centroid", "instance",
                         f"<= {max_centroid:.4f}", round(centroid_error, 6),
                         centroid_error <= max_centroid)
                instance_map[instance_id] = observation
            except ValueError as exc:
                observation = {"id": instance_id, "pattern": pattern,
                               "reference_bbox_uv": reference_bbox,
                               "reference_centroid_uv": reference_centroid,
                               "error": str(exc)}
                fit_gate(gates, instance_id, "instance", "resolvable",
                         "unmeasurable", False, str(exc))
            instance_records.append(observation)
        base_report["instances"] = instance_records

        relation_records = []
        relations = spec.get("relations", [])
        if not isinstance(relations, list):
            raise ValueError("relations must be a list")
        for index, entry in enumerate(relations):
            if not isinstance(entry, dict):
                raise ValueError(f"relations[{index}] must be an object")
            relation_id = str(entry.get("id", f"relation_{index + 1}"))
            relation_type = str(entry.get("type", "relative_position"))
            try:
                if relation_type == "depth_order":
                    front_id, behind_id = str(entry.get("front")), str(entry.get("behind"))
                    front = instance_map.get(front_id)
                    behind = instance_map.get(behind_id)
                    if front is None or behind is None:
                        raise ValueError("depth_order references an unresolved instance")
                    margin = float(entry.get("min_margin_m", 0.0))
                    measured = behind["camera_depth_m"] - front["camera_depth_m"]
                    passed = measured > margin
                    target = f"> {margin:.4f} m"
                    note = f"{front_id} in front of {behind_id}"
                else:
                    a_id, b_id = str(entry.get("a")), str(entry.get("b"))
                    a = instance_map.get(a_id)
                    b = instance_map.get(b_id)
                    if a is None or b is None:
                        raise ValueError("relation references an unresolved instance")
                    maximum = policy.ceiling(
                        f"relation {relation_id}.max_error",
                        entry.get("max_error",
                                  thresholds.get("relation_max_error")),
                        "relation_max_error")
                    if relation_type == "relative_position":
                        target_delta = [
                            b["reference_centroid_uv"][axis]
                            - a["reference_centroid_uv"][axis] for axis in range(2)]
                        render_delta = [
                            b["centroid_uv"][axis] - a["centroid_uv"][axis]
                            for axis in range(2)]
                        measured = math.hypot(
                            target_delta[0] - render_delta[0],
                            target_delta[1] - render_delta[1])
                        note = f"target_delta={target_delta}, render_delta={render_delta}"
                    elif relation_type == "bbox_iou":
                        target_iou = bbox_iou(
                            a["reference_bbox_uv"], b["reference_bbox_uv"])
                        render_iou = bbox_iou(a["bbox_uv"], b["bbox_uv"])
                        measured = abs(target_iou - render_iou)
                        note = f"target_iou={target_iou:.4f}, render_iou={render_iou:.4f}"
                    else:
                        raise ValueError(
                            "relation.type must be relative_position, bbox_iou, or depth_order")
                    passed = measured <= maximum
                    target = f"error <= {maximum:.4f}"
                record = {
                    "id": relation_id,
                    "type": relation_type,
                    "measured": measured,
                    "pass": passed,
                    "note": note,
                }
                fit_gate(gates, relation_id, "relation", target,
                         round(measured, 6), passed, note)
            except ValueError as exc:
                record = {"id": relation_id, "type": relation_type,
                          "error": str(exc)}
                fit_gate(gates, relation_id, "relation", "measurable",
                         "unmeasurable", False, str(exc))
            relation_records.append(record)
        base_report["relations"] = relation_records

        if not gates:
            raise ValueError(
                "fit spec defines no gates; add mask, silhouette regions, landmarks, "
                "pose, instances, or relations")
        base_report["threshold_policy"] = {
            "complexity_class": policy.complexity,
            "reference_views": policy.views,
            "view_adjustment": policy.view_adjustment(),
            "single_view_reconstruction": policy.views == 1,
            "reference_thinness": policy.thinness,
            "mask_iou_floor": policy.mask_iou_floor(),
            "region_iou_floor": policy.region_iou_floor(),
            "loosening_attempts": policy.violations,
        }
        if policy.views == 1:
            print(f"{WARN}:SINGLE_VIEW] one reference view: depth is "
                  "unobservable, so registration floors are relaxed by "
                  f"{-FIT_VIEW_ADJUSTMENT[1]:.2f}. Anything this run cannot "
                  "verify must be reported as approximate, and more views "
                  "raise the bar rather than lower it")
        if policy.violations:
            fit_gate(
                gates, "threshold_policy", "policy", "spec may only tighten",
                f"{len(policy.violations)} loosened threshold(s)", False,
                "; ".join(policy.violations[:6]))
        save_rgba(renders_dir / "reference_overlay.png", overlay)
        passed_count = sum(1 for gate in gates if gate["pass"])
        base_report.update({
            "reference_image": reference_path.name,
            "camera": normalized_camera,
            "gates": gates,
            "summary": {
                "passed": passed_count,
                "total": len(gates),
                "failures": len(gates) - passed_count,
            },
            "passed": passed_count == len(gates),
            "inputs": input_hashes,
        })
        report_path.write_text(json.dumps(base_report, indent=2))
        print(f"ProcAgen3D fit — {reference_path.name}")
        for gate in gates:
            verdict = "PASS" if gate["pass"] else "FAIL"
            note = f"  ({gate['note']})" if gate.get("note") else ""
            print(f"  {gate['id']:<28} target {str(gate['target']):<24} "
                  f"measured {str(gate['measured']):>12}  {verdict}{note}")
        print(f"  -> {passed_count}/{len(gates)} fit gates passed")
        if passed_count != len(gates):
            print(f"{FAIL}:REFERENCE_FIT] {len(gates) - passed_count} fit gate(s) failed")
            finish(1)
        print(f"{OK} registered reference fit passed; overlay -> "
              f"{renders_dir / 'reference_overlay.png'}")
        finish(0)
    except Exception as exc:
        base_report["error"] = str(exc)
        report_path.write_text(json.dumps(base_report, indent=2))
        print(f"{FAIL}:FIT_SPEC] {exc}")
        finish(1)


def stage_solve_camera(args):
    """Resect the reference camera from observed landmarks instead of guessing it."""
    out_dir = Path(args.out).resolve()
    blend = out_dir / "scene.blend"
    spec_path = Path(args.spec).resolve()
    if not blend.is_file():
        print(f"{FAIL}:NO_SCENE] {blend} not found (run build first)")
        finish(1)
    try:
        spec = json.loads(spec_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"{FAIL}:FIT_SPEC] {exc}")
        finish(1)

    bpy.ops.wm.open_mainfile(filepath=str(blend))
    scene = bpy.context.scene
    data = bpy.data.cameras.new("ProcAgen3D_SolveCam")
    data.sensor_fit = "VERTICAL"
    data.type = "PERSP"
    camera = bpy.data.objects.new("ProcAgen3D_SolveCam", data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    reference = out_dir / str(spec.get("reference_image", ""))
    if reference.is_file():
        rgba = load_rgba(reference)
        scene.render.resolution_y = rgba.shape[0]
        scene.render.resolution_x = rgba.shape[1]
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = 1.0

    dg = depsgraph()
    correspondences = []
    skipped = []
    for index, entry in enumerate(spec.get("landmarks") or []):
        if not isinstance(entry, dict):
            continue
        landmark_id = str(entry.get("id", f"landmark_{index + 1}"))
        observed = entry.get("reference_uv")
        if not (isinstance(observed, list) and len(observed) == 2):
            skipped.append(f"{landmark_id} (no reference_uv)")
            continue
        try:
            if "world_point_m" in entry:
                point = Vector(finite_values(
                    entry["world_point_m"], 3, "landmark.world_point_m"))
            else:
                matched, _ = matched_geometry(scene, str(entry.get("pattern", "")))
                if len(matched) != 1:
                    raise ValueError("pattern must match exactly one object")
                point = matched[0].matrix_world.translation.copy()
        except ValueError as exc:
            skipped.append(f"{landmark_id} ({exc})")
            continue
        correspondences.append(
            (landmark_id, point, (float(observed[0]), float(observed[1]))))

    if len(correspondences) < 6:
        print(f"{FAIL}:CAMERA_SOLVE] need at least 6 resolvable landmarks with "
              f"reference_uv, found {len(correspondences)}; skipped: {skipped[:6]}")
        finish(1)

    declared = spec.get("camera") or {}
    if str(declared.get("projection", "perspective")).lower() != "perspective":
        print(f"{FAIL}:CAMERA_SOLVE] only perspective cameras can be resected")
        finish(1)
    target = declared.get("target_m") or [0.0, 0.0, 0.0]
    initial = {
        "azimuth_deg": float(declared.get("azimuth_deg", 0.0)),
        "elevation_deg": float(declared.get("elevation_deg", 0.0)),
        "roll_deg": float(declared.get("roll_deg", 0.0)),
        "fov_y_deg": float(declared.get("fov_y_deg", 40.0)),
        "distance_m": float(declared.get("distance_m", 0.0)) or 10.0,
        "target_x": float(target[0]), "target_y": float(target[1]),
        "target_z": float(target[2]),
        "shift_x": float(declared.get("shift_x", 0.0)),
        "shift_y": float(declared.get("shift_y", 0.0)),
    }
    initial.update({"root_pitch_deg": 0.0, "root_yaw_deg": 0.0,
                    "root_roll_deg": 0.0})
    free = ["azimuth_deg", "elevation_deg", "roll_deg", "fov_y_deg",
            "distance_m", "target_x", "target_y", "target_z"]
    if args.free_shift:
        free.extend(["shift_x", "shift_y"])
    if args.solve_root:
        free.extend(["root_pitch_deg", "root_yaw_deg", "root_roll_deg"])
    if args.fix:
        free = [name for name in free if name not in set(args.fix)]
    if not free:
        print(f"{FAIL}:CAMERA_SOLVE] every parameter is fixed")
        finish(1)

    params, residuals, rms, linear = solve_camera(
        scene, camera, correspondences, initial, free,
        scene.render.resolution_x, scene.render.resolution_y)
    solved = {
        "projection": "perspective",
        "azimuth_deg": round(params["azimuth_deg"], 4),
        "elevation_deg": round(params["elevation_deg"], 4),
        "roll_deg": round(params["roll_deg"], 4),
        "fov_y_deg": round(params["fov_y_deg"], 4),
        "distance_m": round(params["distance_m"], 6),
        "target_m": [round(params["target_x"], 6), round(params["target_y"], 6),
                     round(params["target_z"], 6)],
        "shift_x": round(params["shift_x"], 6),
        "shift_y": round(params["shift_y"], 6),
    }
    root_pose = {name: round(params.get(name, 0.0), 4)
                 for name in ("root_pitch_deg", "root_yaw_deg", "root_roll_deg")}
    report = {
        "procagen3d_camera_solve_version": 1,
        "landmarks_used": len(correspondences),
        "root_pose_deg": root_pose if args.solve_root else None,
        "landmarks_skipped": skipped,
        "free_parameters": free,
        "declared_camera": declared,
        "linear_resection": ({k: round(v, 4) for k, v in linear.items()}
                             if linear else None),
        "solved_camera": solved,
        "rms_uv_error": round(rms, 6),
        "max_rms": args.max_rms,
        "residuals": {name: round(error, 6) for name, error in residuals},
    }
    (out_dir / "camera_solution.json").write_text(json.dumps(report, indent=2))

    print(f"ProcAgen3D camera solve — {len(correspondences)} landmarks")
    for name, error in sorted(residuals, key=lambda item: -item[1]):
        print(f"  {name:<28} reprojection error {error:.4f}")
    print(f"  -> RMS {rms:.4f} uv")
    for key in ("azimuth_deg", "elevation_deg", "roll_deg", "fov_y_deg",
                "distance_m"):
        was = declared.get(key)
        print(f"  {key:<16} declared {str(was):>10}   solved {solved[key]:>10}")
    print(f"  target_m         declared {declared.get('target_m')}   "
          f"solved {solved['target_m']}")
    if args.solve_root:
        print(f"  root lean        pitch {root_pose['root_pitch_deg']:+.2f}  "
              f"yaw {root_pose['root_yaw_deg']:+.2f}  "
              f"roll {root_pose['root_roll_deg']:+.2f} deg")
    if rms > args.max_rms:
        print(f"{FAIL}:CAMERA_SOLVE] RMS {rms:.4f} exceeds {args.max_rms:.4f}: no "
              "single camera explains these landmarks, so the proportions or the "
              "landmark readings are wrong, not the viewpoint")
        finish(1)
    print(f"{OK} camera solved; paste solved_camera into reconstruction_plan"
          f".camera_solve.camera and fit_spec.camera -> "
          f"{out_dir / 'camera_solution.json'}")
    finish(0)


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

    # Invalid/non-finite camera metadata would make export_extras emit
    # non-compliant JSON. Keep the geometry build usable, warn, and omit only
    # the invalid optional contract from derivative artifacts.
    reference_camera_contract(bpy.context.scene, discard_invalid=True)

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
        use_renderable=True,
    )
    bpy.ops.wm.save_as_mainfile(filepath=str(out_dir / "scene.blend"))

    if not args.no_render:
        _, silhouettes = render_views(
            out_dir, args.size, args.engine, args.form_diagnostics)
        if silhouettes:
            graph["view_silhouettes"] = silhouettes
            (out_dir / "scene_graph.json").write_text(json.dumps(graph, indent=2))

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
    _, silhouettes = render_views(
        out_dir, args.size, args.engine, args.form_diagnostics)
    graph_path = out_dir / "scene_graph.json"
    if silhouettes and graph_path.is_file():
        graph = json.loads(graph_path.read_text())
        graph["view_silhouettes"] = silhouettes
        graph_path.write_text(json.dumps(graph, indent=2))
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
    p_build.add_argument("--form-diagnostics", action="store_true")

    p_render = sub.add_parser("render")
    p_render.add_argument("--out", required=True)
    p_render.add_argument("--size", type=int, default=512)
    p_render.add_argument("--engine", default="workbench",
                          choices=["workbench", "eevee", "cycles"])
    p_render.add_argument("--form-diagnostics", action="store_true")

    p_fit = sub.add_parser("fit")
    p_fit.add_argument("--out", required=True)
    p_fit.add_argument("--spec", required=True)
    p_fit.add_argument("--engine", default="workbench",
                       choices=["workbench", "eevee", "cycles"])

    p_solve = sub.add_parser("solve-camera")
    p_solve.add_argument("--out", required=True)
    p_solve.add_argument("--spec", required=True)
    p_solve.add_argument("--max-rms", type=float, default=0.02)
    p_solve.add_argument("--free-shift", action="store_true")
    p_solve.add_argument("--fix", nargs="*", default=[])
    p_solve.add_argument("--solve-root", action="store_true")

    p_joints = sub.add_parser("joints")
    p_joints.add_argument("--out", required=True)
    p_joints.add_argument("--strict", action="store_true")

    args = parser.parse_args(script_args())
    {"build": stage_build, "render": stage_render, "fit": stage_fit,
     "solve-camera": stage_solve_camera,
     "joints": stage_joints}[args.stage](args)


if __name__ == "__main__":
    main()
