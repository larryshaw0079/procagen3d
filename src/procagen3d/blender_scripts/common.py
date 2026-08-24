"""Shared Blender-side geometry, camera, reporting, and render helpers."""

from __future__ import annotations

import base64
import json
import math
from pathlib import Path

import bmesh
import bpy
from mathutils import Matrix, Vector


VIEW_DIRECTIONS = (
    ("front", (0.0, -1.0, 0.0)),
    ("back", (0.0, 1.0, 0.0)),
    ("left", (-1.0, 0.0, 0.0)),
    ("right", (1.0, 0.0, 0.0)),
    ("top", (0.0, 0.0, 1.0)),
    ("iso", (1.0, -1.0, 0.85)),
)
CANONICAL_VIEW_NAMES = tuple(spec[0] for spec in VIEW_DIRECTIONS)
_FACTORY_VIEW_SETTINGS = {
    name: getattr(bpy.context.scene.view_settings, name)
    for name in ("view_transform", "look", "exposure", "gamma", "use_curve_mapping")
    if hasattr(bpy.context.scene.view_settings, name)
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def reset_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in list(bpy.data.collections):
        bpy.data.collections.remove(collection)
    for datablocks in (
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.materials,
        bpy.data.cameras,
        bpy.data.lights,
        bpy.data.armatures,
        bpy.data.metaballs,
        bpy.data.images,
    ):
        for datablock in list(datablocks):
            if datablock.users == 0:
                datablocks.remove(datablock)


def mesh_objects():
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH" and is_renderable(obj)]


def is_renderable(obj):
    if obj.hide_render:
        return False
    collections = tuple(obj.users_collection)
    return not collections or any(not collection.hide_render for collection in collections)


def geometry_objects():
    supported = {"MESH", "CURVE", "SURFACE", "FONT", "META"}
    return [obj for obj in bpy.context.scene.objects if obj.type in supported and is_renderable(obj)]


def validate_drawable_scene():
    """Return renderable geometry after finite/evaluated triangle validation."""

    bpy.context.view_layer.update()
    objects = geometry_objects()
    if not objects:
        raise RuntimeError("scene contains no renderable geometry objects")
    depsgraph = bpy.context.evaluated_depsgraph_get()
    total_vertices = total_triangles = 0
    for obj in objects:
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        try:
            total_vertices += len(mesh.vertices)
            mesh.calc_loop_triangles()
            total_triangles += len(mesh.loop_triangles)
            if not all(math.isfinite(value) for row in evaluated.matrix_world for value in row):
                raise RuntimeError(f"geometry {obj.name!r} has a non-finite world transform")
            for vertex in mesh.vertices:
                point = evaluated.matrix_world @ vertex.co
                if not all(math.isfinite(value) for value in point):
                    raise RuntimeError(f"geometry {obj.name!r} has non-finite world vertices")
        finally:
            evaluated.to_mesh_clear()
    if total_vertices == 0:
        raise RuntimeError("scene contains geometry with no evaluated vertices")
    if total_triangles == 0:
        raise RuntimeError("scene contains geometry with no drawable triangles")
    return objects


def _evaluated_mesh(obj):
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    return evaluated, mesh


def world_vertices(objects=None):
    objects = objects or geometry_objects()
    values = []
    for obj in objects:
        evaluated, mesh = _evaluated_mesh(obj)
        try:
            matrix = evaluated.matrix_world
            values.extend(matrix @ vertex.co for vertex in mesh.vertices)
        finally:
            evaluated.to_mesh_clear()
    return values


def bounds(objects=None):
    objects = objects or geometry_objects()
    points = []
    for obj in objects:
        evaluated, mesh = _evaluated_mesh(obj)
        try:
            matrix = evaluated.matrix_world
            points.extend(matrix @ vertex.co for vertex in mesh.vertices)
        finally:
            evaluated.to_mesh_clear()
    if not points:
        raise RuntimeError("scene contains no renderable mesh bounds")
    minimum = Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
    maximum = Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
    dimensions = maximum - minimum
    center = (minimum + maximum) * 0.5
    return minimum, maximum, dimensions, center


def normalize_objects(objects, longest_dimension=2.0, imported_objects=None):
    minimum, maximum, dimensions, _ = bounds(objects)
    largest = max(dimensions)
    if largest <= 1.0e-9:
        raise RuntimeError("reference GLB has degenerate bounds")
    scale = float(longest_dimension) / largest
    cx = (minimum.x + maximum.x) * 0.5
    cy = (minimum.y + maximum.y) * 0.5
    transform = Matrix.Translation(Vector((-cx * scale, -cy * scale, -minimum.z * scale))) @ Matrix.Scale(scale, 4)

    imported = set(imported_objects or objects)
    roots = [obj for obj in imported if obj.parent not in imported]
    anchor = bpy.data.objects.new("ReferenceNormalization", None)
    bpy.context.scene.collection.objects.link(anchor)
    for obj in roots:
        world = obj.matrix_world.copy()
        obj.parent = anchor
        obj.matrix_world = world
    anchor.matrix_world = transform
    bpy.context.view_layer.update()
    return {
        "source_min": list(minimum),
        "source_max": list(maximum),
        "scale": scale,
        "translation": [-cx * scale, -cy * scale, -minimum.z * scale],
        "longest_dimension": longest_dimension,
    }


def _has_descendant(root, candidates):
    stack = list(root.children)
    while stack:
        current = stack.pop()
        if current in candidates:
            return True
        stack.extend(current.children)
    return False


def object_report(obj):
    evaluated, mesh = _evaluated_mesh(obj)
    matrix = evaluated.matrix_world
    points = [matrix @ vertex.co for vertex in mesh.vertices]
    if not points:
        evaluated.to_mesh_clear()
        raise RuntimeError(f"geometry object {obj.name!r} has no evaluated vertices")
    minimum = [min(point[i] for point in points) for i in range(3)]
    maximum = [max(point[i] for point in points) for i in range(3)]
    try:
        mesh.calc_loop_triangles()
        return {
            "name": obj.name,
            "type": obj.type,
            "vertices": len(mesh.vertices),
            "triangles": len(mesh.loop_triangles),
            "materials": [slot.material.name if slot.material else None for slot in obj.material_slots],
            "bounds": {"min": minimum, "max": maximum, "dimensions": [maximum[i] - minimum[i] for i in range(3)]},
        }
    finally:
        evaluated.to_mesh_clear()


def welded_components(obj, merge_distance=1.0e-5, limit=32):
    evaluated, mesh = _evaluated_mesh(obj)
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        bm.transform(evaluated.matrix_world)
        if bm.verts:
            bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=merge_distance)
        bm.verts.ensure_lookup_table()
        seen = set()
        components = []
        for start in bm.verts:
            if start.index in seen:
                continue
            stack = [start]
            seen.add(start.index)
            points = []
            while stack:
                vertex = stack.pop()
                points.append(vertex.co.copy())
                for edge in vertex.link_edges:
                    other = edge.other_vert(vertex)
                    if other.index not in seen:
                        seen.add(other.index)
                        stack.append(other)
            minimum = [min(point[i] for point in points) for i in range(3)]
            maximum = [max(point[i] for point in points) for i in range(3)]
            components.append(
                {
                    "vertices": len(points),
                    "bounds": {
                        "min": minimum,
                        "max": maximum,
                        "dimensions": [maximum[i] - minimum[i] for i in range(3)],
                    },
                }
            )
        components.sort(key=lambda item: item["vertices"], reverse=True)
        return {"count": len(components), "largest": components[:limit]}
    finally:
        bm.free()
        evaluated.to_mesh_clear()


def section_report(points, *, levels=21, angles=24, band_fraction=0.025):
    if not points:
        return []
    minimum_z = min(point.z for point in points)
    maximum_z = max(point.z for point in points)
    height = maximum_z - minimum_z
    if height <= 1.0e-9:
        return []
    band = max(height * band_fraction, 1.0e-4)
    sections = []
    for level in range(levels):
        z = minimum_z + height * level / (levels - 1)
        selected = [point for point in points if abs(point.z - z) <= band]
        if len(selected) < 3:
            continue
        cx = sum(point.x for point in selected) / len(selected)
        cy = sum(point.y for point in selected) / len(selected)
        radial = []
        half_bin = math.pi / angles
        for index in range(angles):
            target = -math.pi + 2.0 * math.pi * index / angles
            candidates = []
            for point in selected:
                angle = math.atan2(point.y - cy, point.x - cx)
                delta = abs((angle - target + math.pi) % (2.0 * math.pi) - math.pi)
                if delta <= half_bin:
                    candidates.append(math.hypot(point.x - cx, point.y - cy))
            radial.append(max(candidates) if candidates else 0.0)
        sections.append(
            {
                "z": z,
                "samples": len(selected),
                "centroid_xy": [cx, cy],
                "bounds_xy": [
                    min(point.x for point in selected),
                    min(point.y for point in selected),
                    max(point.x for point in selected),
                    max(point.y for point in selected),
                ],
                "radial_outline": radial,
            }
        )
    return sections


def geometry_report(*, include_components=True):
    objects = geometry_objects()
    minimum, maximum, dimensions, center = bounds(objects)
    points = world_vertices(objects)
    if len(points) > 120000:
        stride = math.ceil(len(points) / 120000)
        points = points[::stride]
    report = {
        "coordinate_system": {"up": "+Z", "width": "X", "depth": "Y", "ground": "Z=0"},
        "bounds": {
            "min": list(minimum),
            "max": list(maximum),
            "dimensions": list(dimensions),
            "center": list(center),
        },
        "geometry_object_count": len(objects),
        "mesh_count": sum(obj.type == "MESH" for obj in objects),
        "objects": [object_report(obj) for obj in objects],
        "cross_sections_z": section_report(points),
    }
    if include_components:
        report["welded_components"] = {
            obj.name: welded_components(obj, merge_distance=max(dimensions) * 1.0e-5) for obj in objects
        }
    return report


def camera_contract(size=256, points=None, margin=1.12):
    points = list(points or world_vertices())
    if not points:
        raise RuntimeError("cannot derive cameras without evaluated geometry")
    minimum = Vector(tuple(min(point[axis] for point in points) for axis in range(3)))
    maximum = Vector(tuple(max(point[axis] for point in points) for axis in range(3)))
    target = (minimum + maximum) * 0.5
    views = []
    for name, direction_values in VIEW_DIRECTIONS:
        direction = Vector(direction_values).normalized()
        location = target + direction * 4.5
        rotation = (target - location).to_track_quat("-Z", "Y").to_matrix()
        screen_x = rotation @ Vector((1.0, 0.0, 0.0))
        screen_y = rotation @ Vector((0.0, 1.0, 0.0))
        projected_x = [(point - target).dot(screen_x) for point in points]
        projected_y = [(point - target).dot(screen_y) for point in points]
        projected_extent = max(max(projected_x) - min(projected_x), max(projected_y) - min(projected_y))
        views.append(
            {
                "name": name,
                "location": list(location),
                "target": list(target),
                "ortho_scale": max(0.1, projected_extent * float(margin)),
            }
        )
    return {
        "projection": "ORTHO",
        "resolution": [int(size), int(size)],
        "framing": "evaluated-reference-projection",
        "margin": float(margin),
        "views": views,
    }


def _aim(camera, target):
    direction = Vector(target) - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def add_camera(spec, ortho_scale):
    data = bpy.data.cameras.new(f"EvidenceCamera_{spec['name']}")
    data.type = "ORTHO"
    data.ortho_scale = float(ortho_scale)
    data.clip_start = 0.01
    data.clip_end = 100.0
    camera = bpy.data.objects.new(data.name, data)
    bpy.context.scene.collection.objects.link(camera)
    camera.location = spec["location"]
    _aim(camera, spec["target"])
    return camera


def add_studio_lights(target):
    world = bpy.data.worlds.new("ProcAgen3D_World")
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    nodes.clear()
    background = nodes.new("ShaderNodeBackground")
    background.inputs["Color"].default_value = (0.035, 0.035, 0.05, 1.0)
    background.inputs["Strength"].default_value = 0.25
    output = nodes.new("ShaderNodeOutputWorld")
    world.node_tree.links.new(background.outputs["Background"], output.inputs["Surface"])
    specs = (
        ("Key", (3.0, -4.0, 5.0), 900.0, 4.0),
        ("Fill", (-4.0, -1.0, 3.0), 550.0, 3.0),
        ("Rim", (2.0, 4.0, 4.0), 700.0, 3.0),
    )
    for name, location, energy, size in specs:
        data = bpy.data.lights.new(name, "AREA")
        data.energy = energy
        data.shape = "DISK"
        data.size = size
        light = bpy.data.objects.new(name, data)
        bpy.context.scene.collection.objects.link(light)
        light.location = location
        _aim(light, target)


def configure_render(size):
    scene = bpy.context.scene
    scene.use_nodes = False
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = int(size)
    scene.render.resolution_y = int(size)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    scene.render.image_settings.color_depth = "8"
    for name, value in _FACTORY_VIEW_SETTINGS.items():
        try:
            setattr(scene.view_settings, name, value)
        except (AttributeError, TypeError, ValueError):
            pass


def _pack_render_data(beauty_path, silhouette_path, threshold=0.02):
    beauty = bpy.data.images.load(str(beauty_path), check_existing=False)
    silhouette = bpy.data.images.load(str(silhouette_path), check_existing=False)
    try:
        pixels = list(beauty.pixels)
        silhouette_pixels = list(silhouette.pixels)
        width = int(beauty.size[0])
        height = int(beauty.size[1])
        pixel_count = width * height
        if (
            silhouette.size[:] != beauty.size[:]
            or len(pixels) != pixel_count * 4
            or len(silhouette_pixels) != pixel_count * 4
        ):
            raise RuntimeError(
                "beauty and silhouette renders must be aligned RGBA images"
            )
        bits = bytearray((pixel_count + 7) // 8)
        rgb = bytearray(pixel_count * 3)
        foreground = 0
        for pixel_index in range(pixel_count):
            offset = pixel_index * 4
            red, green, blue, _beauty_alpha = pixels[offset : offset + 4]
            alpha = silhouette_pixels[offset + 3]
            if alpha > threshold:
                bits[pixel_index // 8] |= 1 << (7 - pixel_index % 8)
                foreground += 1
            for channel_index, value in enumerate((red, green, blue)):
                if not math.isfinite(value):
                    raise RuntimeError("rendered image contains a non-finite RGB channel")
                clamped = min(1.0, max(0.0, float(value)))
                rgb[pixel_index * 3 + channel_index] = int(clamped * 255.0 + 0.5)
        return {
            "width": width,
            "height": height,
            "foreground_pixels": foreground,
            "encoding": "base64-msb-packbits",
            "data": base64.b64encode(bytes(bits)).decode("ascii"),
            "rgb_encoding": "base64-rgb8",
            "rgb_data": base64.b64encode(bytes(rgb)).decode("ascii"),
        }
    finally:
        bpy.data.images.remove(beauty)
        bpy.data.images.remove(silhouette)


def _remove_existing_lights_and_cameras():
    for obj in list(bpy.context.scene.objects):
        if obj.type in {"LIGHT", "CAMERA"}:
            bpy.data.objects.remove(obj, do_unlink=True)


def _silhouette_material():
    material = bpy.data.materials.new("ProcAgen3D_Silhouette")
    material.diffuse_color = (1.0, 1.0, 1.0, 1.0)
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    if principled:
        principled.inputs["Base Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        principled.inputs["Alpha"].default_value = 1.0
        principled.inputs["Roughness"].default_value = 1.0
    return material


def render_views(output_dir, contract):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    names = [spec["name"] for spec in contract["views"]]
    if len(names) != len(set(names)) or set(names) != set(CANONICAL_VIEW_NAMES):
        raise RuntimeError(
            "camera contract must contain each canonical view exactly once: "
            + ", ".join(CANONICAL_VIEW_NAMES)
        )
    configure_render(contract["resolution"][0])
    _remove_existing_lights_and_cameras()
    target = contract["views"][0]["target"]
    add_studio_lights(target)
    silhouette_material = _silhouette_material()
    view_layer = bpy.context.view_layer
    view_layer.material_override = None
    masks = {"schema_version": 2, "views": {}}
    for spec in contract["views"]:
        camera = add_camera(spec, spec["ortho_scale"])
        bpy.context.scene.camera = camera
        output = output_dir / f"{spec['name']}.png"
        bpy.context.scene.render.filepath = str(output)
        if "FINISHED" not in bpy.ops.render.render(write_still=True) or not output.is_file():
            raise RuntimeError(f"beauty render failed for canonical view {spec['name']}")
        silhouette = output_dir / f".{spec['name']}.silhouette.png"
        view_layer.material_override = silhouette_material
        bpy.context.scene.render.filepath = str(silhouette)
        if "FINISHED" not in bpy.ops.render.render(write_still=True) or not silhouette.is_file():
            raise RuntimeError(f"silhouette render failed for canonical view {spec['name']}")
        view_layer.material_override = None
        masks["views"][spec["name"]] = _pack_render_data(output, silhouette)
        silhouette.unlink(missing_ok=True)
        bpy.data.objects.remove(camera, do_unlink=True)
    write_json(output_dir / "masks.json", masks)
    return masks
