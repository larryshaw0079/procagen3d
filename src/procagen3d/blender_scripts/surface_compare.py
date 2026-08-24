"""Measure bidirectional surface distance between a reference and candidate GLB."""

from __future__ import annotations

import argparse
import heapq
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bpy
from mathutils.bvhtree import BVHTree

from common import geometry_objects, normalize_objects, reset_scene, write_json


MAX_SAMPLES = 1_000_000
WORST_SAMPLE_LIMIT = 32


def _sample_count(value):
    try:
        count = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if not 1 <= count <= MAX_SAMPLES:
        raise argparse.ArgumentTypeError(
            f"must be between 1 and {MAX_SAMPLES}"
        )
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
    """Return evaluated world vertices, indexed triangles, and triangle areas."""

    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    vertices = []
    triangles = []
    areas = []
    for obj in objects:
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        try:
            offset = len(vertices)
            matrix = evaluated.matrix_world
            object_vertices = [matrix @ vertex.co for vertex in mesh.vertices]
            for point in object_vertices:
                _finite_point(point, label=label)
            vertices.extend(object_vertices)
            mesh.calc_loop_triangles()
            for triangle in mesh.loop_triangles:
                a, b, c = (offset + index for index in triangle.vertices)
                area = 0.5 * (vertices[b] - vertices[a]).cross(
                    vertices[c] - vertices[a]
                ).length
                if not math.isfinite(area):
                    raise RuntimeError(f"{label} contains a non-finite triangle area")
                if area <= 1.0e-15:
                    continue
                triangles.append((a, b, c))
                areas.append(area)
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
    return {
        "vertices": vertices,
        "triangles": triangles,
        "areas": areas,
        "total_area": total_area,
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
        # Bases two and three form a deterministic low-discrepancy sequence.
        # The square root converts it to a uniform barycentric triangle sample.
        u = _radical_inverse(sample_index + 1, 2)
        v = _radical_inverse(sample_index + 1, 3)
        root_u = math.sqrt(u)
        weight_a = 1.0 - root_u
        weight_b = root_u * (1.0 - v)
        weight_c = root_u * v
        yield (
            sample_index,
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


def _directed_distances(source, target, count):
    distances = []
    worst = []
    target_bvh = target["bvh"]
    for sample_index, point in _area_weighted_samples(source, count):
        nearest, _normal, _face_index, distance = target_bvh.find_nearest(point)
        if nearest is None or distance is None or not math.isfinite(distance):
            raise RuntimeError("surface BVH returned an invalid nearest point")
        distance = float(distance)
        distances.append(distance)
        source_point = _finite_point(point, label="sample")
        nearest_point = _finite_point(nearest, label="nearest surface")
        # Larger distance is worse. For ties, retain the earliest sample so the
        # bounded diagnostics are stable across platforms.
        item = (distance, -sample_index, sample_index, source_point, nearest_point)
        if len(worst) < WORST_SAMPLE_LIMIT:
            heapq.heappush(worst, item)
        elif item[:2] > worst[0][:2]:
            heapq.heapreplace(worst, item)
    worst_samples = [
        {
            "sample_index": item[2],
            "distance": item[0],
            "source": list(item[3]),
            "nearest": list(item[4]),
        }
        for item in sorted(worst, key=lambda value: (-value[0], value[2]))
    ]
    return distances, {
        "samples": len(distances),
        "source_surface_area": source["total_area"],
        **_distance_summary(distances),
        "worst_samples": worst_samples,
    }


def main():
    args = arguments()
    reset_scene()

    reference_imported, reference_objects = _import_glb(
        args.reference_glb, label="reference"
    )
    normalization = normalize_objects(
        reference_objects, imported_objects=reference_imported
    )
    bpy.context.view_layer.update()
    candidate_imported, candidate_objects = _import_glb(
        args.candidate_glb, label="candidate"
    )
    del candidate_imported

    reference = _surface(reference_objects, label="reference")
    candidate = _surface(candidate_objects, label="candidate")
    candidate_distances, candidate_to_reference = _directed_distances(
        candidate, reference, args.samples
    )
    reference_distances, reference_to_candidate = _directed_distances(
        reference, candidate, args.samples
    )
    symmetric = _distance_summary(candidate_distances + reference_distances)

    report = {
        "schema_version": 1,
        "units": "normalized-scene-units",
        "pose_policy": "frame-0, armatures-in-rest-position",
        "sampling": {
            "strategy": "deterministic-stratified-area-low-discrepancy-barycentric",
            "requested_samples_per_direction": args.samples,
            "percentile_method": "linear interpolation at (n - 1) * q",
            "worst_sample_limit_per_direction": WORST_SAMPLE_LIMIT,
        },
        "reference_normalization": normalization,
        "surfaces": {
            "reference": {
                "vertices": len(reference["vertices"]),
                "triangles": len(reference["triangles"]),
                "area": reference["total_area"],
            },
            "candidate": {
                "vertices": len(candidate["vertices"]),
                "triangles": len(candidate["triangles"]),
                "area": candidate["total_area"],
            },
        },
        "candidate_to_reference": candidate_to_reference,
        "reference_to_candidate": reference_to_candidate,
        "symmetric": symmetric,
    }
    write_json(args.output, report)
    print("PROCAGEN3D_SURFACE_COMPARISON_READY")


if __name__ == "__main__":
    main()
