"""Canonical six-view renders, optional form sheet, reference-camera contract."""

import math
from pathlib import Path

import bpy
from mathutils import Vector

from .runtime import FAIL, OK, WARN, depsgraph, mesh_objects
from .scene import union_bbox


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
    return written


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
