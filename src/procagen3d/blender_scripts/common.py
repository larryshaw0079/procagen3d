"""Shared Blender-side geometry, camera, reporting, and render helpers."""

from __future__ import annotations

import base64
import json
import math
from pathlib import Path

import bmesh
import bpy
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree


VIEW_DIRECTIONS = (
    ("front", (0.0, -1.0, 0.0)),
    ("back", (0.0, 1.0, 0.0)),
    ("left", (-1.0, 0.0, 0.0)),
    ("right", (1.0, 0.0, 0.0)),
    ("top", (0.0, 0.0, 1.0)),
    ("iso", (1.0, -1.0, 0.85)),
)
CANONICAL_VIEW_NAMES = tuple(spec[0] for spec in VIEW_DIRECTIONS)
STRUCTURE_RECORD_LIMIT = 64
CONTACT_RECORD_LIMIT = 512
CONTACT_VERTEX_SAMPLE_LIMIT = 96
SELF_INTERSECTION_TRIANGLE_LIMIT = 50_000
INTER_OBJECT_OVERLAP_PRODUCT_LIMIT = 25_000_000
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


def validate_drawable_scene(objects=None):
    """Return renderable geometry after finite/evaluated triangle validation."""

    bpy.context.view_layer.update()
    objects = list(objects) if objects is not None else geometry_objects()
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


def _triangle_quality(a, b, c):
    """Return a scale-independent triangle quality in [0, 1]."""

    ab = (b - a).length_squared
    bc = (c - b).length_squared
    ca = (a - c).length_squared
    denominator = ab + bc + ca
    if denominator <= 0.0:
        return 0.0
    doubled_area = (b - a).cross(c - a).length
    return min(1.0, max(0.0, 2.0 * math.sqrt(3.0) * doubled_area / denominator))


def _component_records(bm, *, limit=STRUCTURE_RECORD_LIMIT):
    bm.verts.ensure_lookup_table()
    seen = set()
    components = []
    for start in bm.verts:
        if start.index in seen:
            continue
        stack = [start]
        seen.add(start.index)
        vertices = []
        faces = set()
        while stack:
            vertex = stack.pop()
            vertices.append(vertex)
            faces.update(vertex.link_faces)
            for edge in vertex.link_edges:
                other = edge.other_vert(vertex)
                if other.index not in seen:
                    seen.add(other.index)
                    stack.append(other)
        points = [vertex.co for vertex in vertices]
        minimum = [min(point[axis] for point in points) for axis in range(3)]
        maximum = [max(point[axis] for point in points) for axis in range(3)]
        components.append(
            {
                "vertices": len(vertices),
                "faces": len(faces),
                "bounds": {
                    "min": minimum,
                    "max": maximum,
                    "dimensions": [maximum[i] - minimum[i] for i in range(3)],
                },
            }
        )
    components.sort(key=lambda item: (item["vertices"], item["faces"]), reverse=True)
    return {"count": len(components), "largest": components[:limit]}


def _self_intersection_proxy(points, triangles):
    """Return bounded non-adjacent triangle-overlap diagnostics.

    BVH overlap is a practical broad/narrow-phase proxy. Adjacent triangles are
    excluded because their shared boundary is intentional. Very large objects
    are skipped rather than making evidence generation unbounded.
    """

    if not triangles:
        return {"status": "empty", "triangle_pairs": 0, "examples": []}
    if len(triangles) > SELF_INTERSECTION_TRIANGLE_LIMIT:
        return {
            "status": "skipped-triangle-limit",
            "triangle_limit": SELF_INTERSECTION_TRIANGLE_LIMIT,
            "triangles": len(triangles),
            "triangle_pairs": None,
            "examples": [],
        }
    tree = BVHTree.FromPolygons(points, triangles, all_triangles=True, epsilon=1.0e-9)
    if tree is None:
        return {"status": "bvh-unavailable", "triangle_pairs": None, "examples": []}
    minimum = Vector(tuple(min(point[axis] for point in points) for axis in range(3)))
    maximum = Vector(tuple(max(point[axis] for point in points) for axis in range(3)))
    boundary_epsilon_squared = max((maximum - minimum).length * 1.0e-7, 1.0e-9) ** 2

    def shared_geometric_vertices(first_triangle, second_triangle):
        matches = set()
        for first_vertex in first_triangle:
            for second_vertex in second_triangle:
                if second_vertex in matches:
                    continue
                if (
                    points[first_vertex] - points[second_vertex]
                ).length_squared <= boundary_epsilon_squared:
                    matches.add(second_vertex)
                    break
        return len(matches)

    examples = []
    count = 0
    for first, second in tree.overlap(tree):
        if first >= second:
            continue
        if set(triangles[first]) & set(triangles[second]):
            continue
        # GLB import commonly splits vertices along UV/material seams. Treat
        # one- or two-vertex geometric contact as ordinary adjacency even when
        # the triangle indices no longer share vertices. Three shared points
        # remain reportable because coincident duplicate faces are structural
        # defects rather than a seam.
        if shared_geometric_vertices(triangles[first], triangles[second]) in (1, 2):
            continue
        count += 1
        if len(examples) < STRUCTURE_RECORD_LIMIT:
            examples.append([first, second])
    return {"status": "measured", "triangle_pairs": count, "examples": examples}


def _object_structure(mesh, matrix, points, dimensions):
    mesh.calc_loop_triangles()
    triangles = [tuple(triangle.vertices) for triangle in mesh.loop_triangles]
    scale = max(max(dimensions), 1.0e-9)
    degenerate_area_epsilon = max(1.0e-18, scale * scale * 1.0e-12)
    areas = []
    qualities = []
    degenerate = 0
    for a_index, b_index, c_index in triangles:
        a, b, c = points[a_index], points[b_index], points[c_index]
        area = 0.5 * (b - a).cross(c - a).length
        quality = _triangle_quality(a, b, c)
        if not math.isfinite(area) or area <= degenerate_area_epsilon:
            degenerate += 1
        else:
            areas.append(area)
        qualities.append(quality)

    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        bm.transform(matrix)
        bm.normal_update()
        if bm.verts:
            bmesh.ops.remove_doubles(
                bm,
                verts=list(bm.verts),
                dist=max(scale * 1.0e-5, 1.0e-8),
            )
        boundary_edges = sum(edge.is_boundary for edge in bm.edges)
        wire_edges = sum(edge.is_wire for edge in bm.edges)
        manifold_edges = sum(edge.is_manifold for edge in bm.edges)
        non_manifold_edges = sum(not edge.is_manifold for edge in bm.edges)
        inconsistent_winding_edges = sum(
            edge.is_manifold and not edge.is_contiguous for edge in bm.edges
        )
        loose_vertices = sum(not vertex.link_edges for vertex in bm.verts)
        welded_vertices = len(bm.verts)
        welded_edges = len(bm.edges)
        components = _component_records(bm)
        closed = bool(bm.faces) and non_manifold_edges == 0
        signed_volume = float(bm.calc_volume(signed=True)) if closed else None
    finally:
        bm.free()

    ordered_quality = sorted(qualities)
    quality_p05 = (
        ordered_quality[max(0, math.ceil(len(ordered_quality) * 0.05) - 1)]
        if ordered_quality
        else None
    )
    return {
        "connected_components": components,
        "topology": {
            "source_edges": len(mesh.edges),
            "welded_vertices": welded_vertices,
            "edges": welded_edges,
            "boundary_edges": boundary_edges,
            "wire_edges": wire_edges,
            "manifold_edges": manifold_edges,
            "non_manifold_edges": non_manifold_edges,
            "inconsistent_winding_edges": inconsistent_winding_edges,
            "loose_vertices": loose_vertices,
            "closed_manifold_proxy": closed,
        },
        "normal_consistency": {
            "manifold_edge_consistency": (
                (manifold_edges - inconsistent_winding_edges) / manifold_edges
                if manifold_edges
                else None
            ),
            "signed_volume": signed_volume,
            "outward_orientation_proxy": signed_volume is None or signed_volume >= 0.0,
        },
        "triangle_quality": {
            "surface_area": math.fsum(areas),
            "degenerate_area_epsilon": degenerate_area_epsilon,
            "degenerate_triangles": degenerate,
            "quality_min": min(qualities) if qualities else None,
            "quality_mean": math.fsum(qualities) / len(qualities) if qualities else None,
            "quality_p05": quality_p05,
            "low_quality_below_0_05": sum(value < 0.05 for value in qualities),
        },
        "self_intersection_proxy": _self_intersection_proxy(points, triangles),
    }


def object_report(obj):
    evaluated, mesh = _evaluated_mesh(obj)
    matrix = evaluated.matrix_world
    points = [matrix @ vertex.co for vertex in mesh.vertices]
    if not points:
        evaluated.to_mesh_clear()
        raise RuntimeError(f"geometry object {obj.name!r} has no evaluated vertices")
    minimum = [min(point[i] for point in points) for i in range(3)]
    maximum = [max(point[i] for point in points) for i in range(3)]
    dimensions = [maximum[i] - minimum[i] for i in range(3)]
    try:
        mesh.calc_loop_triangles()
        return {
            "name": obj.name,
            "type": obj.type,
            "vertices": len(mesh.vertices),
            "triangles": len(mesh.loop_triangles),
            "materials": [slot.material.name if slot.material else None for slot in obj.material_slots],
            "bounds": {"min": minimum, "max": maximum, "dimensions": dimensions},
            "structure": _object_structure(mesh, matrix, points, dimensions),
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


def _surface_snapshot(obj):
    evaluated, mesh = _evaluated_mesh(obj)
    try:
        matrix = evaluated.matrix_world
        points = [matrix @ vertex.co for vertex in mesh.vertices]
        mesh.calc_loop_triangles()
        triangles = [tuple(triangle.vertices) for triangle in mesh.loop_triangles]
        if not points or not triangles:
            return None
        minimum = [min(point[axis] for point in points) for axis in range(3)]
        maximum = [max(point[axis] for point in points) for axis in range(3)]
        tree = BVHTree.FromPolygons(points, triangles, all_triangles=True, epsilon=1.0e-9)
        return {
            "name": obj.name,
            "points": points,
            "triangles": triangles,
            "bvh": tree,
            "min": minimum,
            "max": maximum,
        }
    finally:
        evaluated.to_mesh_clear()


def _aabb_gap(first, second):
    axis_gaps = [
        max(0.0, first["min"][axis] - second["max"][axis], second["min"][axis] - first["max"][axis])
        for axis in range(3)
    ]
    return math.sqrt(math.fsum(value * value for value in axis_gaps)), axis_gaps


def _sampled_surface_gap(first, second):
    distances = []
    for source, target in ((first, second), (second, first)):
        points = source["points"]
        stride = max(1, math.ceil(len(points) / CONTACT_VERTEX_SAMPLE_LIMIT))
        for point in points[::stride]:
            nearest = target["bvh"].find_nearest(point) if target["bvh"] else None
            if nearest is not None and nearest[3] is not None and math.isfinite(nearest[3]):
                distances.append(float(nearest[3]))
    return min(distances) if distances else None


def _global_welded_components(snapshots, merge_distance, *, limit=STRUCTURE_RECORD_LIMIT):
    points = []
    point_objects = []
    edges = []
    for snapshot in snapshots:
        offset = len(points)
        points.extend(snapshot["points"])
        point_objects.extend([snapshot["name"]] * len(snapshot["points"]))
        edge_set = set()
        for triangle in snapshot["triangles"]:
            for first, second in (
                (triangle[0], triangle[1]),
                (triangle[1], triangle[2]),
                (triangle[2], triangle[0]),
            ):
                edge_set.add((min(first, second), max(first, second)))
        edges.extend((offset + first, offset + second) for first, second in edge_set)
    if not points:
        return {"count": 0, "largest": [], "merge_distance": merge_distance}

    parents = list(range(len(points)))
    sizes = [1] * len(points)

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first, second):
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            return
        if sizes[first_root] < sizes[second_root]:
            first_root, second_root = second_root, first_root
        parents[second_root] = first_root
        sizes[first_root] += sizes[second_root]

    for first, second in edges:
        union(first, second)

    cell_size = max(float(merge_distance), 1.0e-12)
    cells = {}
    for index, point in enumerate(points):
        cell = tuple(math.floor(float(point[axis]) / cell_size) for axis in range(3))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for other in cells.get((cell[0] + dx, cell[1] + dy, cell[2] + dz), ()):
                        if (point - points[other]).length <= cell_size:
                            union(index, other)
        cells.setdefault(cell, []).append(index)

    groups = {}
    for index in range(len(points)):
        groups.setdefault(find(index), []).append(index)
    records = []
    for indices in groups.values():
        component_points = [points[index] for index in indices]
        names = sorted({point_objects[index] for index in indices})
        minimum = [min(point[axis] for point in component_points) for axis in range(3)]
        maximum = [max(point[axis] for point in component_points) for axis in range(3)]
        records.append(
            {
                "vertices": len(indices),
                "object_count": len(names),
                "objects": names[:16],
                "objects_truncated": max(0, len(names) - 16),
                "bounds": {
                    "min": minimum,
                    "max": maximum,
                    "dimensions": [maximum[axis] - minimum[axis] for axis in range(3)],
                },
            }
        )
    records.sort(key=lambda item: (item["vertices"], item["object_count"]), reverse=True)
    return {"count": len(records), "largest": records[:limit], "merge_distance": merge_distance}


def structural_diagnostics(objects, *, scene_dimensions):
    """Measure topology and bounded inter-object contact/intersection proxies."""

    scale = max(max(scene_dimensions), 1.0e-9)
    merge_distance = max(scale * 1.0e-5, 1.0e-8)
    contact_distance = max(scale * 5.0e-3, 1.0e-5)
    snapshots = [snapshot for obj in objects if (snapshot := _surface_snapshot(obj)) is not None]
    contacts = []
    intersections = []
    overlap_without_intersection = []
    contact_count = 0
    intersection_count = 0
    overlap_without_intersection_count = 0
    skipped_overlap_tests = 0
    contact_names = set()
    tested_pairs = 0
    broad_phase_pairs = 0
    for first_index, first in enumerate(snapshots):
        for second in snapshots[first_index + 1 :]:
            tested_pairs += 1
            aabb_distance, axis_gaps = _aabb_gap(first, second)
            if aabb_distance > contact_distance:
                continue
            broad_phase_pairs += 1
            overlap_measured = (
                first["bvh"] is not None
                and second["bvh"] is not None
                and len(first["triangles"]) * len(second["triangles"])
                <= INTER_OBJECT_OVERLAP_PRODUCT_LIMIT
            )
            if overlap_measured:
                overlap_pairs = first["bvh"].overlap(second["bvh"])
            else:
                overlap_pairs = []
                skipped_overlap_tests += 1
            surface_gap = _sampled_surface_gap(first, second)
            record = {
                "objects": [first["name"], second["name"]],
                "aabb_distance": aabb_distance,
                "aabb_axis_gaps": axis_gaps,
                "sampled_surface_gap": surface_gap,
                "triangle_overlap_status": (
                    "measured" if overlap_measured else "skipped-product-limit"
                ),
            }
            if overlap_pairs:
                record["triangle_overlap_pairs"] = len(overlap_pairs)
                intersection_count += 1
                if len(intersections) < CONTACT_RECORD_LIMIT:
                    intersections.append(record)
                contact_names.update(record["objects"])
            elif overlap_measured and aabb_distance == 0.0:
                overlap_without_intersection_count += 1
                if len(overlap_without_intersection) < CONTACT_RECORD_LIMIT:
                    overlap_without_intersection.append(record)
            if surface_gap is not None and surface_gap <= contact_distance:
                contact_count += 1
                if len(contacts) < CONTACT_RECORD_LIMIT:
                    contacts.append(record)
                contact_names.update(record["objects"])
    names = {snapshot["name"] for snapshot in snapshots}
    return {
        "schema_version": 1,
        "global_welded_components": _global_welded_components(
            snapshots, merge_distance
        ),
        "contact_intersection_proxy": {
            "contact_distance": contact_distance,
            "object_pairs_tested": tested_pairs,
            "broad_phase_pairs": broad_phase_pairs,
            "near_contact_pair_count": contact_count,
            "near_contact_pairs": contacts,
            "triangle_intersection_pair_count": intersection_count,
            "triangle_intersection_pairs": intersections,
            "triangle_overlap_tests_skipped": skipped_overlap_tests,
            "triangle_overlap_product_limit": INTER_OBJECT_OVERLAP_PRODUCT_LIMIT,
            "aabb_overlap_without_triangle_intersection_count": (
                overlap_without_intersection_count
            ),
            "aabb_overlap_without_triangle_intersection_pairs": overlap_without_intersection,
            "isolated_objects": sorted(names - contact_names),
            "records_truncated_at": CONTACT_RECORD_LIMIT,
            "method": (
                "world-AABB broad phase; BVH triangle-overlap intersection proxy; "
                "bounded bidirectional vertex-to-surface contact proxy"
            ),
        },
    }


def section_report(points, *, axis=2, levels=21, angles=24, band_fraction=0.025):
    if not points:
        return []
    if axis not in (0, 1, 2):
        raise ValueError("section axis must be 0, 1, or 2")
    plane_axes = tuple(index for index in range(3) if index != axis)
    axis_name = "xyz"[axis]
    plane_name = "".join("xyz"[index] for index in plane_axes)
    minimum_axis = min(point[axis] for point in points)
    maximum_axis = max(point[axis] for point in points)
    span = maximum_axis - minimum_axis
    if span <= 1.0e-9:
        return []
    band = max(span * band_fraction, 1.0e-4)
    sections = []
    for level in range(levels):
        coordinate = minimum_axis + span * level / (levels - 1)
        selected = [point for point in points if abs(point[axis] - coordinate) <= band]
        if len(selected) < 3:
            continue
        center_a = sum(point[plane_axes[0]] for point in selected) / len(selected)
        center_b = sum(point[plane_axes[1]] for point in selected) / len(selected)
        radial = []
        half_bin = math.pi / angles
        for index in range(angles):
            target = -math.pi + 2.0 * math.pi * index / angles
            candidates = []
            for point in selected:
                delta_a = point[plane_axes[0]] - center_a
                delta_b = point[plane_axes[1]] - center_b
                angle = math.atan2(delta_b, delta_a)
                delta = abs(
                    (angle - target + math.pi) % (2.0 * math.pi) - math.pi
                )
                if delta <= half_bin:
                    candidates.append(math.hypot(delta_a, delta_b))
            radial.append(max(candidates) if candidates else 0.0)
        sections.append(
            {
                axis_name: coordinate,
                "samples": len(selected),
                f"centroid_{plane_name}": [center_a, center_b],
                f"bounds_{plane_name}": [
                    min(point[plane_axes[0]] for point in selected),
                    min(point[plane_axes[1]] for point in selected),
                    max(point[plane_axes[0]] for point in selected),
                    max(point[plane_axes[1]] for point in selected),
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
    object_reports = [object_report(obj) for obj in objects]
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
        "objects": object_reports,
        "cross_sections_x": section_report(points, axis=0),
        "cross_sections_y": section_report(points, axis=1),
        "cross_sections_z": section_report(points, axis=2),
        "structure": structural_diagnostics(objects, scene_dimensions=list(dimensions)),
    }
    if include_components:
        report["welded_components"] = {
            item["name"]: item["structure"]["connected_components"]
            for item in object_reports
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


def _emission_material(name):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    emission = nodes.new("ShaderNodeEmission")
    output = nodes.new("ShaderNodeOutputMaterial")
    material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material, emission


def _normal_material():
    material, emission = _emission_material("ProcAgen3D_WorldNormal")
    nodes = material.node_tree.nodes
    geometry = nodes.new("ShaderNodeNewGeometry")
    remap = nodes.new("ShaderNodeVectorMath")
    remap.operation = "MULTIPLY_ADD"
    remap.inputs[1].default_value = (0.5, 0.5, 0.5)
    remap.inputs[2].default_value = (0.5, 0.5, 0.5)
    material.node_tree.links.new(geometry.outputs["Normal"], remap.inputs[0])
    material.node_tree.links.new(remap.outputs["Vector"], emission.inputs["Color"])
    emission.inputs["Strength"].default_value = 1.0
    return material


def _object_id_material():
    material, emission = _emission_material("ProcAgen3D_ObjectID")
    info = material.node_tree.nodes.new("ShaderNodeObjectInfo")
    material.node_tree.links.new(info.outputs["Color"], emission.inputs["Color"])
    emission.inputs["Strength"].default_value = 1.0
    return material


def _depth_material():
    material, emission = _emission_material("ProcAgen3D_LinearDepth")
    nodes = material.node_tree.nodes
    camera = nodes.new("ShaderNodeCameraData")
    remap = nodes.new("ShaderNodeMapRange")
    remap.clamp = True
    remap.inputs["To Min"].default_value = 1.0
    remap.inputs["To Max"].default_value = 0.0
    material.node_tree.links.new(camera.outputs["View Z Depth"], remap.inputs["Value"])
    material.node_tree.links.new(remap.outputs["Result"], emission.inputs["Color"])
    emission.inputs["Strength"].default_value = 1.0
    return material, remap


def _diagnostic_color(index):
    # Odd multiplication is bijective modulo 2^24, giving stable unique IDs
    # while distributing early object indices into visually distinct colors.
    # Boundary pixels remain visual diagnostics because PNG antialiasing blends.
    encoded = (int(index) * 0x9E3779B1) & 0xFFFFFF
    red = encoded & 0xFF
    green = (encoded >> 8) & 0xFF
    blue = (encoded >> 16) & 0xFF
    return red, green, blue


def _diagnostic_render_settings():
    scene = bpy.context.scene
    saved = {
        name: getattr(scene.view_settings, name)
        for name in ("view_transform", "look", "exposure", "gamma")
        if hasattr(scene.view_settings, name)
    }
    for name, value in (
        ("view_transform", "Raw"),
        ("look", "None"),
        ("exposure", 0.0),
        ("gamma", 1.0),
    ):
        try:
            setattr(scene.view_settings, name, value)
        except (AttributeError, TypeError, ValueError):
            pass
    return saved


def _restore_view_settings(saved):
    for name, value in saved.items():
        try:
            setattr(bpy.context.scene.view_settings, name, value)
        except (AttributeError, TypeError, ValueError):
            pass


def _render_diagnostics(output_dir, contract):
    diagnostic_root = Path(output_dir) / "diagnostics"
    paths = {
        kind: diagnostic_root / kind for kind in ("depth", "normal", "object_id")
    }
    for directory in paths.values():
        directory.mkdir(parents=True, exist_ok=True)

    objects = sorted(geometry_objects(), key=lambda obj: obj.name)
    original_colors = {obj.name: tuple(obj.color) for obj in objects}
    object_ids = []
    for index, obj in enumerate(objects, 1):
        rgb = _diagnostic_color(index)
        obj.color = (*(channel / 255.0 for channel in rgb), 1.0)
        object_ids.append({"id": index, "rgb8": list(rgb), "object": obj.name})

    normal_material = _normal_material()
    object_material = _object_id_material()
    depth_material, depth_remap = _depth_material()
    points = world_vertices(objects)
    view_layer = bpy.context.view_layer
    saved_settings = _diagnostic_render_settings()
    views = {}
    try:
        for spec in contract["views"]:
            camera = add_camera(spec, spec["ortho_scale"])
            bpy.context.scene.camera = camera
            location = Vector(spec["location"])
            direction = (Vector(spec["target"]) - location).normalized()
            depths = [(point - location).dot(direction) for point in points]
            minimum_depth = min(depths)
            maximum_depth = max(depths)
            if maximum_depth - minimum_depth <= 1.0e-9:
                maximum_depth = minimum_depth + 1.0e-9
            depth_remap.inputs["From Min"].default_value = minimum_depth
            depth_remap.inputs["From Max"].default_value = maximum_depth

            view_record = {
                "depth_range": [minimum_depth, maximum_depth],
                "depth": f"diagnostics/depth/{spec['name']}.png",
                "normal": f"diagnostics/normal/{spec['name']}.png",
                "object_id": f"diagnostics/object_id/{spec['name']}.png",
            }
            for kind, material in (
                ("depth", depth_material),
                ("normal", normal_material),
                ("object_id", object_material),
            ):
                output = paths[kind] / f"{spec['name']}.png"
                view_layer.material_override = material
                bpy.context.scene.render.filepath = str(output)
                if (
                    "FINISHED" not in bpy.ops.render.render(write_still=True)
                    or not output.is_file()
                ):
                    raise RuntimeError(
                        f"{kind} diagnostic render failed for canonical view {spec['name']}"
                    )
            view_layer.material_override = None
            views[spec["name"]] = view_record
            bpy.data.objects.remove(camera, do_unlink=True)
    finally:
        view_layer.material_override = None
        for obj in objects:
            if obj.name in original_colors:
                obj.color = original_colors[obj.name]
        _restore_view_settings(saved_settings)

    manifest = {
        "schema_version": 1,
        "path_base": "canonical-render-root",
        "depth_encoding": (
            "8-bit display-linear grayscale; white is nearest and black is farthest "
            "within each recorded depth_range"
        ),
        "normal_encoding": "world-space XYZ remapped from [-1, 1] to [0, 1]",
        "object_id_encoding": (
            "manifest-mapped unique 24-bit RGB object ID; antialiased boundaries"
        ),
        "objects": object_ids,
        "views": views,
    }
    write_json(diagnostic_root / "manifest.json", manifest)
    return manifest


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
    diagnostics = _render_diagnostics(output_dir, contract)
    masks["diagnostics"] = {
        "schema_version": diagnostics["schema_version"],
        "manifest": "diagnostics/manifest.json",
    }
    write_json(output_dir / "masks.json", masks)
    return masks
