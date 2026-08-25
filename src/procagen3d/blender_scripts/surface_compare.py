"""Measure bidirectional, normal-aware surface agreement between two GLBs."""

from __future__ import annotations

import argparse
import heapq
import math
import sys
from array import array
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from common import VIEW_DIRECTIONS, geometry_objects, normalize_objects, reset_scene, write_json


MAX_SAMPLES = 1_000_000
WORST_SAMPLE_LIMIT = 32
VISIBILITY_SAMPLE_BUDGET = 16_384
HEATMAP_SIZE = 256
COVERAGE_THRESHOLDS = (0.005, 0.010, 0.020, 0.040, 0.080)
NORMAL_ALIGNMENT_DEGREES = 30.0
NORMAL_DISTANCE_SCALE = 0.02


def _sample_count(value):
    try:
        count = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if not 1 <= count <= MAX_SAMPLES:
        raise argparse.ArgumentTypeError(f"must be between 1 and {MAX_SAMPLES}")
    return count


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-glb", type=Path, required=True)
    parser.add_argument("--candidate-glb", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=_sample_count, required=True)
    raw = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(raw)


def _import_glb(path, *, label):
    if not path.is_file():
        raise RuntimeError(f"{label} GLB does not exist: {path}")
    before = set(bpy.context.scene.objects)
    if "FINISHED" not in bpy.ops.import_scene.gltf(filepath=str(path)):
        raise RuntimeError(f"{label} GLB import did not finish")
    imported = [obj for obj in bpy.context.scene.objects if obj not in before]
    bpy.context.scene.frame_set(0)
    for obj in imported:
        if obj.type == "ARMATURE":
            obj.data.pose_position = "REST"
    bpy.context.view_layer.update()
    imported_set = set(imported)
    geometry = [obj for obj in geometry_objects() if obj in imported_set]
    if not geometry:
        raise RuntimeError(f"{label} GLB imported without geometry objects")
    return imported, geometry


def _finite_point(point, *, label):
    values = tuple(float(value) for value in point)
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"{label} contains a non-finite surface point")
    return values


def _surface(objects, *, label):
    """Return deterministic evaluated surface data with face/object identity."""

    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    vertices = []
    triangles = []
    normals = []
    areas = []
    triangle_metadata = []
    object_records = []
    for object_index, obj in enumerate(sorted(objects, key=lambda item: item.name)):
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        try:
            vertex_offset = len(vertices)
            triangle_offset = len(triangles)
            matrix = evaluated.matrix_world
            object_vertices = [matrix @ vertex.co for vertex in mesh.vertices]
            for point in object_vertices:
                _finite_point(point, label=label)
            vertices.extend(object_vertices)
            mesh.calc_loop_triangles()
            object_area = 0.0
            for local_triangle_index, triangle in enumerate(mesh.loop_triangles):
                a_index, b_index, c_index = (
                    vertex_offset + index for index in triangle.vertices
                )
                cross = (vertices[b_index] - vertices[a_index]).cross(
                    vertices[c_index] - vertices[a_index]
                )
                area = 0.5 * cross.length
                if not math.isfinite(area):
                    raise RuntimeError(f"{label} contains a non-finite triangle area")
                if area <= 1.0e-15:
                    continue
                triangles.append((a_index, b_index, c_index))
                normals.append(cross.normalized())
                areas.append(area)
                object_area += area
                triangle_metadata.append(
                    {
                        "object": obj.name,
                        "object_index": object_index,
                        "polygon_index": int(triangle.polygon_index),
                        "triangle_index_in_object": local_triangle_index,
                    }
                )
            object_records.append(
                {
                    "name": obj.name,
                    "object_index": object_index,
                    "vertices": len(object_vertices),
                    "triangles": len(triangles) - triangle_offset,
                    "area": object_area,
                }
            )
        finally:
            evaluated.to_mesh_clear()
    if not vertices or not triangles:
        raise RuntimeError(f"{label} GLB has no non-degenerate evaluated triangles")
    total_area = math.fsum(areas)
    if not math.isfinite(total_area) or total_area <= 0.0:
        raise RuntimeError(f"{label} GLB has invalid evaluated surface area")
    bvh = BVHTree.FromPolygons(vertices, triangles, all_triangles=True)
    if bvh is None:
        raise RuntimeError(f"could not construct {label} surface BVH")
    minimum = Vector(tuple(min(point[axis] for point in vertices) for axis in range(3)))
    maximum = Vector(tuple(max(point[axis] for point in vertices) for axis in range(3)))
    return {
        "vertices": vertices,
        "triangles": triangles,
        "normals": normals,
        "areas": areas,
        "triangle_metadata": triangle_metadata,
        "objects": object_records,
        "total_area": total_area,
        "bounds_min": minimum,
        "bounds_max": maximum,
        "bounds_diagonal": max((maximum - minimum).length, 1.0e-9),
        "bvh": bvh,
    }


def _radical_inverse(index, base):
    inverse = 1.0 / base
    factor = inverse
    value = 0.0
    while index:
        index, digit = divmod(index, base)
        value += digit * factor
        factor *= inverse
    return value


def _area_weighted_samples(surface, count):
    """Yield exactly *count* deterministic, uniform-by-area surface samples."""

    vertices = surface["vertices"]
    triangles = surface["triangles"]
    areas = surface["areas"]
    total_area = surface["total_area"]
    cumulative = areas[0]
    triangle_index = 0
    for sample_index in range(count):
        target_area = (sample_index + 0.5) * total_area / count
        while cumulative < target_area and triangle_index + 1 < len(triangles):
            triangle_index += 1
            cumulative += areas[triangle_index]
        a_index, b_index, c_index = triangles[triangle_index]
        u = _radical_inverse(sample_index + 1, 2)
        v = _radical_inverse(sample_index + 1, 3)
        root_u = math.sqrt(u)
        weight_a = 1.0 - root_u
        weight_b = root_u * (1.0 - v)
        weight_c = root_u * v
        yield (
            sample_index,
            triangle_index,
            vertices[a_index] * weight_a
            + vertices[b_index] * weight_b
            + vertices[c_index] * weight_c,
        )


def _percentile(sorted_values, quantile):
    if not sorted_values:
        raise RuntimeError("cannot compute a percentile without samples")
    position = (len(sorted_values) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _distance_summary(distances):
    if not distances:
        raise RuntimeError("surface comparison produced no distance samples")
    mean = math.fsum(distances) / len(distances)
    rms = math.sqrt(math.fsum(value * value for value in distances) / len(distances))
    ordered = sorted(distances)
    return {
        "mean": mean,
        "rms": rms,
        "p95": _percentile(ordered, 0.95),
        "max": ordered[-1],
    }


def _optional_summary(values):
    return _distance_summary(values) if values else None


def _coverage_summary(distances, normal_angles, source_area):
    sample_count = len(distances)
    normal_aligned = sum(value <= NORMAL_ALIGNMENT_DEGREES for value in normal_angles)
    thresholds = []
    for threshold in COVERAGE_THRESHOLDS:
        within = sum(value <= threshold for value in distances)
        aligned_within = sum(
            distance <= threshold and angle <= NORMAL_ALIGNMENT_DEGREES
            for distance, angle in zip(distances, normal_angles, strict=True)
        )
        fraction = within / sample_count if sample_count else None
        thresholds.append(
            {
                "distance": threshold,
                "samples_within": within,
                "fraction_within": fraction,
                "estimated_area_within": source_area * fraction if fraction is not None else None,
                "distance_and_normal_aligned_samples": aligned_within,
                "distance_and_normal_aligned_fraction": (
                    aligned_within / sample_count if sample_count else None
                ),
            }
        )
    return {
        "samples": sample_count,
        "normal_alignment_threshold_degrees": NORMAL_ALIGNMENT_DEGREES,
        "normal_aligned_fraction": normal_aligned / sample_count if sample_count else None,
        "thresholds": thresholds,
    }


def _visible_views(surface, point):
    diagonal = surface["bounds_diagonal"]
    ray_distance = max(diagonal * 2.0, 4.0)
    tolerance = max(diagonal * 2.0e-5, 2.0e-6)
    visible = []
    for name, raw_direction in VIEW_DIRECTIONS:
        direction = Vector(raw_direction).normalized()
        origin = point + direction * ray_distance
        hit, _normal, _face_index, _distance = surface["bvh"].ray_cast(
            origin, -direction, ray_distance + diagonal
        )
        if hit is not None and (hit - point).length <= tolerance:
            visible.append(name)
    return visible


def _identity(metadata, triangle_index):
    return {
        **metadata,
        "surface_triangle_index": triangle_index,
    }


def _directed_distances(source, target, count):
    # Compact double arrays keep the million-sample upper bound practical in
    # Blender processes that already hold two detailed evaluated scenes.
    distances = array("d")
    normal_angles = array("d")
    unoriented_normal_angles = array("d")
    point_to_plane_distances = array("d")
    normal_aware_distances = array("d")
    worst = []
    diagnostic_samples = []
    visibility_stride = max(1, math.ceil(count / VISIBILITY_SAMPLE_BUDGET))
    visibility_distances = array("d")
    visibility_angles = array("d")
    per_view_distances = {
        name: array("d") for name, _direction in VIEW_DIRECTIONS
    }
    per_object = {
        record["object_index"]: {
            "identity": record,
            "distances": array("d"),
            "angles": array("d"),
            "normal_aware": array("d"),
        }
        for record in source["objects"]
    }
    target_bvh = target["bvh"]
    for sample_index, source_triangle_index, point in _area_weighted_samples(source, count):
        nearest, _nearest_normal, target_triangle_index, distance = target_bvh.find_nearest(point)
        if (
            nearest is None
            or target_triangle_index is None
            or distance is None
            or not math.isfinite(distance)
            or not 0 <= target_triangle_index < len(target["triangles"])
        ):
            raise RuntimeError("surface BVH returned an invalid nearest point")
        distance = float(distance)
        source_normal = source["normals"][source_triangle_index]
        target_normal = target["normals"][target_triangle_index]
        cosine = max(-1.0, min(1.0, source_normal.dot(target_normal)))
        normal_angle = math.degrees(math.acos(cosine))
        unoriented_angle = math.degrees(math.acos(abs(cosine)))
        signed_plane_offset = float((point - nearest).dot(target_normal))
        point_to_plane = abs(signed_plane_offset)
        normal_aware = math.hypot(distance, NORMAL_DISTANCE_SCALE * (1.0 - cosine) * 0.5)

        distances.append(distance)
        normal_angles.append(normal_angle)
        unoriented_normal_angles.append(unoriented_angle)
        point_to_plane_distances.append(point_to_plane)
        normal_aware_distances.append(normal_aware)

        source_metadata = source["triangle_metadata"][source_triangle_index]
        target_metadata = target["triangle_metadata"][target_triangle_index]
        bucket = per_object[source_metadata["object_index"]]
        bucket["distances"].append(distance)
        bucket["angles"].append(normal_angle)
        bucket["normal_aware"].append(normal_aware)

        record = {
            "sample_index": sample_index,
            "distance": distance,
            "source": list(_finite_point(point, label="sample")),
            "nearest": list(_finite_point(nearest, label="nearest surface")),
            "source_identity": _identity(source_metadata, source_triangle_index),
            "target_identity": _identity(target_metadata, target_triangle_index),
            "source_normal": list(_finite_point(source_normal, label="source normal")),
            "target_normal": list(_finite_point(target_normal, label="target normal")),
            "normal_cosine": cosine,
            "normal_angle_degrees": normal_angle,
            "unoriented_normal_angle_degrees": unoriented_angle,
            "signed_target_plane_offset": signed_plane_offset,
            "point_to_plane_distance": point_to_plane,
            "normal_aware_distance": normal_aware,
        }
        item = (distance, -sample_index, record)
        if len(worst) < WORST_SAMPLE_LIMIT:
            heapq.heappush(worst, item)
        elif item[:2] > worst[0][:2]:
            heapq.heapreplace(worst, item)

        if sample_index % visibility_stride == 0:
            visible_from = _visible_views(source, point)
            diagnostic_samples.append(
                {
                    "point": list(_finite_point(point, label="diagnostic sample")),
                    "distance": distance,
                    "normal_angle_degrees": normal_angle,
                    "visible_from": visible_from,
                }
            )
            if visible_from:
                visibility_distances.append(distance)
                visibility_angles.append(normal_angle)
            for view_name in visible_from:
                per_view_distances[view_name].append(distance)

    worst_samples = []
    for _distance, _negative_index, record in sorted(
        worst, key=lambda value: (-value[0], -value[1])
    ):
        record["visible_from"] = _visible_views(source, Vector(record["source"]))
        worst_samples.append(record)

    per_object_reports = []
    for object_index in sorted(per_object):
        bucket = per_object[object_index]
        object_distances = bucket["distances"]
        object_angles = bucket["angles"]
        identity = bucket["identity"]
        per_object_reports.append(
            {
                **identity,
                "samples": len(object_distances),
                "distance": _optional_summary(object_distances),
                "normal_angle_degrees": _optional_summary(object_angles),
                "normal_aware_distance": _optional_summary(bucket["normal_aware"]),
                "coverage": _coverage_summary(object_distances, object_angles, identity["area"]),
            }
        )

    visibility_report = {
        "method": "bounded canonical-view first-hit ray proxy",
        "canonical_views": [name for name, _direction in VIEW_DIRECTIONS],
        "sample_stride": visibility_stride,
        "tested_samples": len(diagnostic_samples),
        "visible_from_any_view_samples": len(visibility_distances),
        "visible_from_any_view_fraction": (
            len(visibility_distances) / len(diagnostic_samples)
            if diagnostic_samples
            else None
        ),
        "estimated_visible_surface_area": (
            source["total_area"] * len(visibility_distances) / len(diagnostic_samples)
            if diagnostic_samples
            else None
        ),
        "distance": _optional_summary(visibility_distances),
        "coverage": _coverage_summary(
            visibility_distances,
            visibility_angles,
            source["total_area"]
            * len(visibility_distances)
            / max(1, len(diagnostic_samples)),
        ),
        "by_view": {
            name: {
                "visible_samples": len(per_view_distances[name]),
                "fraction_of_tested_samples": (
                    len(per_view_distances[name]) / len(diagnostic_samples)
                    if diagnostic_samples
                    else None
                ),
                "distance": _optional_summary(per_view_distances[name]),
            }
            for name, _direction in VIEW_DIRECTIONS
        },
    }
    report = {
        "samples": len(distances),
        "source_surface_area": source["total_area"],
        **_distance_summary(distances),
        "worst_samples": worst_samples,
        "normal_metrics": {
            "normal_distance_scale": NORMAL_DISTANCE_SCALE,
            "oriented_cosine_mean": math.fsum(
                math.cos(math.radians(value)) for value in normal_angles
            )
            / len(normal_angles),
            "normal_angle_degrees": _distance_summary(normal_angles),
            "unoriented_normal_angle_degrees": _distance_summary(
                unoriented_normal_angles
            ),
            "point_to_plane_distance": _distance_summary(point_to_plane_distances),
            "normal_aware_distance": _distance_summary(normal_aware_distances),
        },
        "coverage": _coverage_summary(distances, normal_angles, source["total_area"]),
        "per_source_object": per_object_reports,
        "visible_external_proxy": visibility_report,
    }
    raw = {
        "normal_angles": normal_angles,
        "unoriented_normal_angles": unoriented_normal_angles,
        "normal_aware_distances": normal_aware_distances,
        "visibility_distances": visibility_distances,
    }
    return distances, report, diagnostic_samples, raw


def _projection_frames(reference, candidate):
    minimum = Vector(
        tuple(
            min(reference["bounds_min"][axis], candidate["bounds_min"][axis])
            for axis in range(3)
        )
    )
    maximum = Vector(
        tuple(
            max(reference["bounds_max"][axis], candidate["bounds_max"][axis])
            for axis in range(3)
        )
    )
    corners = [
        Vector((x, y, z))
        for x in (minimum.x, maximum.x)
        for y in (minimum.y, maximum.y)
        for z in (minimum.z, maximum.z)
    ]
    frames = {}
    for name, raw_direction in VIEW_DIRECTIONS:
        direction = Vector(raw_direction).normalized()
        forward = -direction
        up_hint = Vector((0.0, 0.0, 1.0))
        if abs(forward.dot(up_hint)) > 0.98:
            up_hint = Vector((0.0, 1.0, 0.0))
        right = forward.cross(up_hint).normalized()
        up = right.cross(forward).normalized()
        horizontal = [point.dot(right) for point in corners]
        vertical = [point.dot(up) for point in corners]
        h_min, h_max = min(horizontal), max(horizontal)
        v_min, v_max = min(vertical), max(vertical)
        padding = 0.025 * max(h_max - h_min, v_max - v_min, 1.0e-6)
        frames[name] = {
            "right": right,
            "up": up,
            "horizontal": (h_min - padding, h_max + padding),
            "vertical": (v_min - padding, v_max + padding),
        }
    return frames


def _heat_color(value):
    value = max(0.0, min(1.0, value))
    if value <= 1.0 / 3.0:
        fraction = value * 3.0
        return (0.0, fraction, 1.0, 1.0)
    if value <= 2.0 / 3.0:
        fraction = (value - 1.0 / 3.0) * 3.0
        return (fraction, 1.0, 1.0 - fraction, 1.0)
    fraction = (value - 2.0 / 3.0) * 3.0
    return (1.0, 1.0 - fraction, 0.0, 1.0)


def _save_heatmap(path, records, frame, field, normalization):
    values = [-1.0] * (HEATMAP_SIZE * HEATMAP_SIZE)
    h_min, h_max = frame["horizontal"]
    v_min, v_max = frame["vertical"]
    for record in records:
        point = Vector(record["point"])
        x = round((point.dot(frame["right"]) - h_min) / (h_max - h_min) * (HEATMAP_SIZE - 1))
        y = round((point.dot(frame["up"]) - v_min) / (v_max - v_min) * (HEATMAP_SIZE - 1))
        if not 0 <= x < HEATMAP_SIZE or not 0 <= y < HEATMAP_SIZE:
            continue
        value = float(record[field])
        for offset_y in (-1, 0, 1):
            for offset_x in (-1, 0, 1):
                px, py = x + offset_x, y + offset_y
                if 0 <= px < HEATMAP_SIZE and 0 <= py < HEATMAP_SIZE:
                    index = py * HEATMAP_SIZE + px
                    values[index] = max(values[index], value)
    pixels = [0.0] * (HEATMAP_SIZE * HEATMAP_SIZE * 4)
    denominator = max(float(normalization), 1.0e-12)
    for index, value in enumerate(values):
        if value < 0.0:
            continue
        pixels[index * 4 : index * 4 + 4] = _heat_color(value / denominator)
    image = bpy.data.images.new(
        f"Procagen3DResidual-{path.stem}",
        width=HEATMAP_SIZE,
        height=HEATMAP_SIZE,
        alpha=True,
    )
    try:
        image.pixels.foreach_set(pixels)
        image.file_format = "PNG"
        image.filepath_raw = str(path)
        image.save()
    finally:
        bpy.data.images.remove(image)


def _write_residual_artifacts(output, frames, directed_records):
    root = output.parent / "surface_residuals"
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "path_base": "surface-comparison-report-directory",
        "image_size": [HEATMAP_SIZE, HEATMAP_SIZE],
        "color_map": "transparent background; blue=0, cyan=1/3, yellow=2/3, red>=range",
        "projection_bounds": "shared reference/candidate world-space bounds per canonical view",
        "directions": {},
    }
    for direction_name, payload in directed_records.items():
        records = payload["records"]
        distance_values = [record["distance"] for record in records]
        distance_range = _optional_summary(distance_values)
        distance_normalization = (
            distance_range["p95"]
            if distance_range and distance_range["p95"] > 0.0
            else max(distance_values, default=1.0)
        )
        metrics = {
            "distance": {
                "field": "distance",
                "range": distance_normalization,
                "units": "normalized-scene-units",
            },
            "normal_angle": {
                "field": "normal_angle_degrees",
                "range": 90.0,
                "units": "degrees",
            },
        }
        direction_manifest = {}
        for metric_name, metric in metrics.items():
            view_paths = {}
            directory = root / direction_name / metric_name
            directory.mkdir(parents=True, exist_ok=True)
            for view_name, _direction in VIEW_DIRECTIONS:
                path = directory / f"{view_name}.png"
                _save_heatmap(
                    path,
                    records,
                    frames[view_name],
                    metric["field"],
                    metric["range"],
                )
                view_paths[view_name] = path.relative_to(output.parent).as_posix()
            direction_manifest[metric_name] = {
                "normalization_max": metric["range"],
                "units": metric["units"],
                "views": view_paths,
            }
        manifest["directions"][direction_name] = direction_manifest
    manifest_path = root / "manifest.json"
    write_json(manifest_path, manifest)
    return {
        "manifest": manifest_path.relative_to(output.parent).as_posix(),
        "root": root.relative_to(output.parent).as_posix(),
    }


def main():
    args = arguments()
    reset_scene()

    reference_imported, reference_objects = _import_glb(
        args.reference_glb, label="reference"
    )
    normalization = normalize_objects(reference_objects, imported_objects=reference_imported)
    bpy.context.view_layer.update()
    candidate_imported, candidate_objects = _import_glb(
        args.candidate_glb, label="candidate"
    )
    del candidate_imported

    reference = _surface(reference_objects, label="reference")
    candidate = _surface(candidate_objects, label="candidate")
    (
        candidate_distances,
        candidate_to_reference,
        candidate_records,
        candidate_raw,
    ) = _directed_distances(candidate, reference, args.samples)
    (
        reference_distances,
        reference_to_candidate,
        reference_records,
        reference_raw,
    ) = _directed_distances(reference, candidate, args.samples)
    symmetric = _distance_summary(candidate_distances + reference_distances)
    frames = _projection_frames(reference, candidate)
    residual_artifacts = _write_residual_artifacts(
        args.output,
        frames,
        {
            "candidate_to_reference": {"records": candidate_records},
            "reference_to_candidate": {"records": reference_records},
        },
    )

    reference_area = reference["total_area"]
    candidate_area = candidate["total_area"]
    all_normal_angles = candidate_raw["normal_angles"] + reference_raw["normal_angles"]
    all_unoriented_angles = (
        candidate_raw["unoriented_normal_angles"]
        + reference_raw["unoriented_normal_angles"]
    )
    all_visible_distances = (
        candidate_raw["visibility_distances"] + reference_raw["visibility_distances"]
    )
    report = {
        "schema_version": 1,
        "units": "normalized-scene-units",
        "pose_policy": "frame-0, armatures-in-rest-position",
        "sampling": {
            "strategy": "deterministic-stratified-area-low-discrepancy-barycentric",
            "requested_samples_per_direction": args.samples,
            "percentile_method": "linear interpolation at (n - 1) * q",
            "worst_sample_limit_per_direction": WORST_SAMPLE_LIMIT,
            "visibility_sample_budget_per_direction": VISIBILITY_SAMPLE_BUDGET,
        },
        "reference_normalization": normalization,
        "surfaces": {
            "reference": {
                "vertices": len(reference["vertices"]),
                "triangles": len(reference["triangles"]),
                "area": reference_area,
                "objects": reference["objects"],
            },
            "candidate": {
                "vertices": len(candidate["vertices"]),
                "triangles": len(candidate["triangles"]),
                "area": candidate_area,
                "objects": candidate["objects"],
            },
        },
        "area_comparison": {
            "reference": reference_area,
            "candidate": candidate_area,
            "candidate_to_reference_ratio": candidate_area / reference_area,
            "absolute_difference": abs(candidate_area - reference_area),
            "relative_absolute_difference": abs(candidate_area - reference_area) / reference_area,
        },
        "candidate_to_reference": candidate_to_reference,
        "reference_to_candidate": reference_to_candidate,
        "symmetric": symmetric,
        "normal_aware": {
            "normal_distance_scale": NORMAL_DISTANCE_SCALE,
            "distance": _distance_summary(
                candidate_raw["normal_aware_distances"]
                + reference_raw["normal_aware_distances"]
            ),
            "normal_angle_degrees": _distance_summary(all_normal_angles),
            "unoriented_normal_angle_degrees": _distance_summary(
                all_unoriented_angles
            ),
        },
        "visible_external_proxy": {
            "method": "symmetric union of directed canonical-view first-hit samples",
            "samples": len(all_visible_distances),
            "distance": _optional_summary(all_visible_distances),
            "candidate_to_reference": candidate_to_reference["visible_external_proxy"],
            "reference_to_candidate": reference_to_candidate["visible_external_proxy"],
        },
        "residual_artifacts": residual_artifacts,
    }
    write_json(args.output, report)
    print("PROCAGEN3D_SURFACE_COMPARISON_READY")


if __name__ == "__main__":
    main()
