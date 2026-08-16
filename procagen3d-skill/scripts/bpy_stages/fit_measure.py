"""Mask, camera, projection, and landmark measurements for image-fit."""

import fnmatch
import math

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Quaternion, Vector

from .fit_io import finite_values, fit_path, load_rgba
from .scene import union_bbox


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
