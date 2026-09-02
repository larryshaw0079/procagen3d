"""Validate split URDF geometry at rest and kinematics at nonzero poses."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bpy
from mathutils import Matrix, Vector

from common import geometry_objects, reset_scene, write_json


GLTF_VISUAL_RPY = (math.pi / 2.0, 0.0, 0.0)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-glb", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--assembly-transforms", type=Path, required=True)
    parser.add_argument("--parts-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1.0e-5)
    raw = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(raw)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def vector3(value, label):
    if not isinstance(value, str):
        raise RuntimeError(f"{label} must contain three numbers")
    fields = value.split()
    if len(fields) != 3:
        raise RuntimeError(f"{label} must contain three numbers")
    try:
        result = tuple(float(field) for field in fields)
    except ValueError as exc:
        raise RuntimeError(f"{label} must contain three numbers") from exc
    if not all(math.isfinite(component) for component in result):
        raise RuntimeError(f"{label} must contain finite numbers")
    return result


def origin_matrix(element, label):
    if element is None:
        raise RuntimeError(f"{label} is missing an origin")
    xyz = vector3(element.get("xyz"), f"{label} xyz")
    roll, pitch, yaw = vector3(element.get("rpy"), f"{label} rpy")
    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)
    rotation_x = Matrix(
        ((1.0, 0.0, 0.0, 0.0), (0.0, cx, -sx, 0.0), (0.0, sx, cx, 0.0), (0.0, 0.0, 0.0, 1.0))
    )
    rotation_y = Matrix(
        ((cy, 0.0, sy, 0.0), (0.0, 1.0, 0.0, 0.0), (-sy, 0.0, cy, 0.0), (0.0, 0.0, 0.0, 1.0))
    )
    rotation_z = Matrix(
        ((cz, -sz, 0.0, 0.0), (sz, cz, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    )
    return Matrix.Translation(xyz) @ rotation_z @ rotation_y @ rotation_x


def matrices_close(left, right, tolerance):
    error = max(
        abs(float(left[row][column]) - float(right[row][column]))
        for row in range(4)
        for column in range(4)
    )
    return error <= tolerance


def matrix4(value, label):
    try:
        matrix = Matrix(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} must be a 4x4 matrix") from exc
    if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
        raise RuntimeError(f"{label} must be a 4x4 matrix")
    if not all(
        math.isfinite(float(matrix[row][column]))
        for row in range(4)
        for column in range(4)
    ):
        raise RuntimeError(f"{label} must contain finite numbers")
    return matrix


def assembly_parts(path):
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"assembly transform document is unreadable: {exc}") from exc
    version = document.get("schema_version")
    placement = document.get("placement")
    if (version, placement) not in {(1, "host-solved"), (2, "urdf-link")}:
        raise RuntimeError("URDF validation requires a supported link-frame document")
    values = document.get("parts")
    if not isinstance(values, list) or not values:
        raise RuntimeError("assembly transform document must contain parts")
    parts = {}
    object_owner = {}
    for index, value in enumerate(values):
        if not isinstance(value, dict):
            raise RuntimeError(f"assembly part {index} must be an object")
        part_id = value.get("id")
        names = value.get("object_names")
        if not isinstance(part_id, str) or not part_id or part_id in parts:
            raise RuntimeError(f"assembly part {index} has a duplicate or invalid id")
        if (
            not isinstance(names, list)
            or not names
            or not all(isinstance(name, str) and name for name in names)
        ):
            raise RuntimeError(f"assembly part {part_id!r} has invalid object_names")
        if version == 2:
            part_world = matrix4(
                value.get("part_world_matrix"),
                f"assembly part {part_id!r} part_world_matrix",
            )
            part_from_link = matrix4(
                value.get("part_from_link_matrix"),
                f"assembly part {part_id!r} part_from_link_matrix",
            )
            link_world = matrix4(
                value.get("link_world_matrix"),
                f"assembly part {part_id!r} link_world_matrix",
            )
            if not matrices_close(part_world @ part_from_link, link_world, 1.0e-8):
                raise RuntimeError(
                    f"assembly part {part_id!r} has inconsistent part and link frames"
                )
        else:
            part_world = matrix4(
                value.get("world_matrix"),
                f"assembly part {part_id!r} world_matrix",
            )
            part_from_link = Matrix.Identity(4)
            link_world = part_world
        for name in names:
            if name in object_owner:
                raise RuntimeError(f"object {name!r} belongs to multiple assembly parts")
            object_owner[name] = part_id
        parts[part_id] = {
            "object_names": tuple(names),
            "part_world_matrix": part_world,
            "part_from_link_matrix": part_from_link,
            "link_world_matrix": link_world,
        }

    probes = []
    raw_probes = document.get("motion_probes", []) if version == 2 else []
    if not isinstance(raw_probes, list):
        raise RuntimeError("URDF link-frame document has invalid motion_probes")
    seen_probe_ids = set()
    for index, value in enumerate(raw_probes):
        if not isinstance(value, dict):
            raise RuntimeError(f"motion probe {index} must be an object")
        mate_id = value.get("mate_id")
        joint_type = value.get("joint_type")
        assembly_parameter = value.get("assembly_parameter")
        urdf_position = value.get("urdf_position")
        expected_values = value.get("expected_link_world_matrices")
        if (
            not isinstance(mate_id, str)
            or not mate_id
            or mate_id in seen_probe_ids
        ):
            raise RuntimeError(f"motion probe {index} has an invalid or duplicate mate_id")
        if joint_type not in {"revolute", "prismatic"}:
            raise RuntimeError(f"motion probe {mate_id!r} has an unsupported joint type")
        if not all(
            isinstance(number, (int, float))
            and not isinstance(number, bool)
            and math.isfinite(float(number))
            for number in (assembly_parameter, urdf_position)
        ):
            raise RuntimeError(f"motion probe {mate_id!r} has invalid parameters")
        if abs(float(urdf_position)) <= 1.0e-12:
            raise RuntimeError(f"motion probe {mate_id!r} must be nonzero")
        if not isinstance(expected_values, dict) or set(expected_values) != set(parts):
            raise RuntimeError(
                f"motion probe {mate_id!r} must cover every assembly link exactly"
            )
        expected = {
            part_id: matrix4(
                expected_values[part_id],
                f"motion probe {mate_id!r} link {part_id!r}",
            )
            for part_id in parts
        }
        seen_probe_ids.add(mate_id)
        probes.append(
            {
                "mate_id": mate_id,
                "joint_type": joint_type,
                "assembly_parameter": float(assembly_parameter),
                "urdf_position": float(urdf_position),
                "expected": expected,
            }
        )
    if version == 2 and not probes:
        raise RuntimeError("URDF link-frame document must contain nonzero motion probes")
    return parts, object_owner, probes


def urdf_link_transforms(path, parts, tolerance):
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise RuntimeError(f"URDF is unreadable: {exc}") from exc
    if root.tag != "robot":
        raise RuntimeError("URDF root element must be <robot>")

    links = {}
    expected_visual = origin_matrix(
        ET.fromstring(
            f'<origin xyz="0 0 0" rpy="{GLTF_VISUAL_RPY[0]} 0 0"/>'
        ),
        "expected glTF visual",
    )
    for element in root.findall("link"):
        name = element.get("name")
        if not isinstance(name, str) or not name or name in links:
            raise RuntimeError("URDF contains a duplicate or unnamed link")
        visuals = element.findall("visual")
        if len(visuals) != 1:
            raise RuntimeError(f"URDF link {name!r} must contain exactly one visual")
        visual = visuals[0]
        visual_matrix = origin_matrix(visual.find("origin"), f"URDF link {name!r} visual")
        if not matrices_close(visual_matrix, expected_visual, tolerance):
            raise RuntimeError(
                f"URDF link {name!r} GLB visual must rotate Y-up to Z-up with Rx(+pi/2)"
            )
        mesh = visual.find("./geometry/mesh")
        expected_filename = f"urdf_parts/{name}.glb"
        if mesh is None or mesh.get("filename") != expected_filename:
            raise RuntimeError(
                f"URDF link {name!r} must reference {expected_filename!r}"
            )
        links[name] = visual_matrix
    if set(links) != set(parts):
        raise RuntimeError(
            "URDF links do not cover the host-solved assembly exactly; "
            f"missing={sorted(set(parts) - set(links))}, extra={sorted(set(links) - set(parts))}"
        )

    incoming = {}
    joints = []
    joint_names = set()
    for index, joint in enumerate(root.findall("joint")):
        name = joint.get("name")
        kind = joint.get("type")
        if not isinstance(name, str) or not name or name in joint_names:
            raise RuntimeError(f"URDF joint {index} is unnamed or duplicated")
        if kind not in {"fixed", "revolute", "continuous", "prismatic"}:
            raise RuntimeError(f"URDF joint {name!r} has unsupported type {kind!r}")
        parent_element = joint.find("parent")
        child_element = joint.find("child")
        parent = parent_element.get("link") if parent_element is not None else None
        child = child_element.get("link") if child_element is not None else None
        if parent not in links or child not in links:
            raise RuntimeError(f"URDF joint {index} references an unknown link")
        if child in incoming:
            raise RuntimeError(f"URDF link {child!r} has multiple parent joints")
        axis = None
        if kind != "fixed":
            axis_element = joint.find("axis")
            axis_value = vector3(
                axis_element.get("xyz") if axis_element is not None else None,
                f"URDF joint {name!r} axis",
            )
            axis = Vector(axis_value)
            if axis.length <= 1.0e-12:
                raise RuntimeError(f"URDF joint {name!r} axis must be nonzero")
            axis.normalize()
        incoming[child] = name
        joint_names.add(name)
        joints.append(
            {
                "name": name,
                "kind": kind,
                "parent": parent,
                "child": child,
                "origin": origin_matrix(
                    joint.find("origin"), f"URDF joint {name!r}"
                ),
                "axis": axis,
            }
        )
    if len(joints) != len(links) - 1:
        raise RuntimeError("URDF joint graph is not a tree")
    roots = sorted(set(links) - set(incoming))
    if len(roots) != 1:
        raise RuntimeError(f"URDF must have exactly one root link; found {roots!r}")

    model = {"links": set(links), "joints": joints, "root": roots[0]}
    world = urdf_forward_kinematics(model, {})
    for part_id, part in parts.items():
        if not matrices_close(
            world[part_id], part["link_world_matrix"], tolerance
        ):
            raise RuntimeError(
                f"URDF zero-pose transform for link {part_id!r} does not match "
                "the connector-centred assembly frame"
            )
    return world, model


def joint_delta(kind, axis, position):
    if kind == "fixed":
        return Matrix.Identity(4)
    if axis is None:
        raise RuntimeError(f"URDF {kind} joint is missing its axis")
    if kind in {"revolute", "continuous"}:
        return Matrix.Rotation(position, 4, axis)
    if kind == "prismatic":
        return Matrix.Translation(axis * position)
    raise RuntimeError(f"unsupported URDF joint type {kind!r}")


def urdf_forward_kinematics(model, joint_values):
    joints = model["joints"]
    known = {joint["name"] for joint in joints}
    unknown = sorted(set(joint_values) - known)
    if unknown:
        raise RuntimeError(f"motion probe references unknown URDF joints: {unknown!r}")
    world = {model["root"]: Matrix.Identity(4)}
    unresolved = list(joints)
    while unresolved:
        progressed = False
        remaining = []
        for joint in unresolved:
            parent = joint["parent"]
            if parent not in world:
                remaining.append(joint)
                continue
            position = float(joint_values.get(joint["name"], 0.0))
            if not math.isfinite(position):
                raise RuntimeError(f"URDF joint {joint['name']!r} has a non-finite position")
            if joint["kind"] == "fixed" and abs(position) > 1.0e-12:
                raise RuntimeError(f"fixed URDF joint {joint['name']!r} cannot move")
            relative = joint["origin"] @ joint_delta(
                joint["kind"], joint["axis"], position
            )
            world[joint["child"]] = world[parent] @ relative
            progressed = True
        if not progressed:
            raise RuntimeError("URDF joint graph is cyclic or disconnected")
        unresolved = remaining
    if set(world) != model["links"]:
        raise RuntimeError("URDF forward kinematics did not cover every link")
    return world


def import_glb(path):
    reset_scene()
    before = set(bpy.context.scene.objects)
    if "FINISHED" not in bpy.ops.import_scene.gltf(filepath=str(path)):
        raise RuntimeError(f"GLB import did not finish: {path}")
    bpy.context.view_layer.update()
    imported = set(bpy.context.scene.objects) - before
    objects = [obj for obj in geometry_objects() if obj in imported]
    if not objects:
        raise RuntimeError(f"GLB contains no renderable geometry: {path}")
    names = [obj.name for obj in objects]
    if len(names) != len(set(names)):
        raise RuntimeError(f"GLB contains duplicate geometry names: {path}")
    return objects


def object_measurement(obj, prefix):
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        matrix = prefix @ evaluated.matrix_world
        points = [matrix @ vertex.co for vertex in mesh.vertices]
        if not points:
            raise RuntimeError(f"geometry object {obj.name!r} has no vertices")
        if not all(all(math.isfinite(float(value)) for value in point) for point in points):
            raise RuntimeError(f"geometry object {obj.name!r} has non-finite vertices")
        mesh.calc_loop_triangles()
        return {
            "vertices": len(mesh.vertices),
            "triangles": len(mesh.loop_triangles),
            "min": [min(float(point[axis]) for point in points) for axis in range(3)],
            "max": [max(float(point[axis]) for point in points) for axis in range(3)],
        }
    finally:
        evaluated.to_mesh_clear()


def measurements(path, prefix):
    return {
        obj.name: object_measurement(obj, prefix)
        for obj in import_glb(path)
    }


def max_bounds_error(expected, actual):
    return max(
        abs(float(expected[bound][axis]) - float(actual[bound][axis]))
        for bound in ("min", "max")
        for axis in range(3)
    )


def matrix_error(expected, actual):
    return max(
        abs(float(expected[row][column]) - float(actual[row][column]))
        for row in range(4)
        for column in range(4)
    )


def translation_error(expected, actual):
    return math.sqrt(
        sum(
            (float(expected[row][3]) - float(actual[row][3])) ** 2
            for row in range(3)
        )
    )


def rotation_error(expected, actual):
    expected_rotation = expected.to_3x3()
    actual_rotation = actual.to_3x3()
    relative = expected_rotation.transposed() @ actual_rotation
    trace = sum(float(relative[index][index]) for index in range(3))
    cosine = max(-1.0, min(1.0, (trace - 1.0) * 0.5))
    return math.acos(cosine)


def main():
    args = arguments()
    for label, path in (
        ("model GLB", args.model_glb),
        ("URDF", args.urdf),
        ("assembly transforms", args.assembly_transforms),
    ):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"{label} must be a regular file")
    if args.parts_dir.is_symlink() or not args.parts_dir.is_dir():
        raise RuntimeError("URDF parts directory must be a regular directory")
    if not math.isfinite(args.tolerance) or args.tolerance <= 0.0:
        raise RuntimeError("zero-pose tolerance must be a positive finite number")

    parts, object_owner, motion_probes = assembly_parts(args.assembly_transforms)
    link_world, urdf_model = urdf_link_transforms(
        args.urdf, parts, args.tolerance
    )
    baseline = measurements(args.model_glb, Matrix.Identity(4))
    if set(baseline) != set(object_owner):
        raise RuntimeError(
            "compiled model object names do not cover the assembly exactly; "
            f"missing={sorted(set(object_owner) - set(baseline))}, "
            f"extra={sorted(set(baseline) - set(object_owner))}"
        )
    scene_min = [min(value["min"][axis] for value in baseline.values()) for axis in range(3)]
    scene_max = [max(value["max"][axis] for value in baseline.values()) for axis in range(3)]
    scene_scale = max(1.0, *(scene_max[axis] - scene_min[axis] for axis in range(3)))
    absolute_tolerance = args.tolerance * scene_scale

    joints_by_name = {joint["name"]: joint for joint in urdf_model["joints"]}
    motion_records = []
    max_motion_matrix_error = 0.0
    max_motion_translation_error = 0.0
    max_motion_rotation_error = 0.0
    for probe in motion_probes:
        mate_id = probe["mate_id"]
        joint = joints_by_name.get(mate_id)
        if joint is None:
            raise RuntimeError(f"motion probe references missing URDF joint {mate_id!r}")
        if joint["kind"] != probe["joint_type"]:
            raise RuntimeError(
                f"motion probe {mate_id!r} expected {probe['joint_type']!r}, "
                f"but URDF declares {joint['kind']!r}"
            )
        posed = urdf_forward_kinematics(
            urdf_model, {mate_id: probe["urdf_position"]}
        )
        probe_matrix_error = 0.0
        probe_translation_error = 0.0
        probe_rotation_error = 0.0
        affected_links = []
        for part_id, expected in probe["expected"].items():
            actual = posed[part_id]
            current_matrix_error = matrix_error(expected, actual)
            current_translation_error = translation_error(expected, actual)
            current_rotation_error = rotation_error(expected, actual)
            probe_matrix_error = max(probe_matrix_error, current_matrix_error)
            probe_translation_error = max(
                probe_translation_error, current_translation_error
            )
            probe_rotation_error = max(
                probe_rotation_error, current_rotation_error
            )
            if not matrices_close(
                expected, parts[part_id]["link_world_matrix"], args.tolerance
            ):
                affected_links.append(part_id)
            if current_matrix_error > args.tolerance:
                raise RuntimeError(
                    f"URDF motion probe {mate_id!r} disagrees with the assembly solver "
                    f"for link {part_id!r}: matrix error {current_matrix_error:.9g} "
                    f"exceeds tolerance {args.tolerance:.9g}"
                )
        max_motion_matrix_error = max(
            max_motion_matrix_error, probe_matrix_error
        )
        max_motion_translation_error = max(
            max_motion_translation_error, probe_translation_error
        )
        max_motion_rotation_error = max(
            max_motion_rotation_error, probe_rotation_error
        )
        motion_records.append(
            {
                "mate_id": mate_id,
                "joint_type": probe["joint_type"],
                "assembly_parameter": probe["assembly_parameter"],
                "urdf_position": probe["urdf_position"],
                "affected_links": sorted(affected_links),
                "max_matrix_error": probe_matrix_error,
                "max_translation_error": probe_translation_error,
                "max_rotation_error_rad": probe_rotation_error,
            }
        )

    records = []
    largest_error = 0.0
    reconstructed_names = set()
    reconstructed_vertex_count = 0
    reconstructed_triangle_count = 0
    vertex_count_changed_objects = 0
    for part_id, part in parts.items():
        path = args.parts_dir / f"{part_id}.glb"
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"URDF link GLB is missing: {path.name}")
        actual = measurements(path, link_world[part_id])
        expected_names = set(part["object_names"])
        if set(actual) != expected_names:
            raise RuntimeError(
                f"URDF link {part_id!r} object names do not match its plan ownership; "
                f"missing={sorted(expected_names - set(actual))}, "
                f"extra={sorted(set(actual) - expected_names)}"
            )
        part_error = 0.0
        part_source_vertices = 0
        part_reconstructed_vertices = 0
        part_triangles = 0
        for name, value in actual.items():
            expected = baseline[name]
            if value["triangles"] != expected["triangles"]:
                raise RuntimeError(
                    f"URDF zero-pose triangle count mismatch for object {name!r} "
                    f"in link {part_id!r}: "
                    f"{value['triangles']} != {expected['triangles']}"
                )
            part_source_vertices += expected["vertices"]
            part_reconstructed_vertices += value["vertices"]
            part_triangles += value["triangles"]
            if value["vertices"] != expected["vertices"]:
                vertex_count_changed_objects += 1
            error = max_bounds_error(expected, value)
            part_error = max(part_error, error)
            if error > absolute_tolerance:
                raise RuntimeError(
                    f"URDF zero-pose bounds mismatch for object {name!r} in link {part_id!r}: "
                    f"error {error:.9g} exceeds tolerance {absolute_tolerance:.9g}"
                )
            reconstructed_names.add(name)
        largest_error = max(largest_error, part_error)
        reconstructed_vertex_count += part_reconstructed_vertices
        reconstructed_triangle_count += part_triangles
        records.append(
            {
                "part_id": part_id,
                "path": path.name,
                "sha256": sha256(path),
                "object_count": len(actual),
                "source_vertex_count": part_source_vertices,
                "reconstructed_vertex_count": part_reconstructed_vertices,
                "triangle_count": part_triangles,
                "max_bounds_error": part_error,
            }
        )
    if reconstructed_names != set(baseline):
        raise RuntimeError("URDF zero-pose reconstruction did not cover every compiled object")

    write_json(
        args.out,
        {
            "schema_version": 2,
            "status": "passed",
            "model_sha256": sha256(args.model_glb),
            "urdf_sha256": sha256(args.urdf),
            "assembly_sha256": sha256(args.assembly_transforms),
            "part_count": len(parts),
            "object_count": len(baseline),
            "source_vertex_count": sum(
                value["vertices"] for value in baseline.values()
            ),
            "reconstructed_vertex_count": reconstructed_vertex_count,
            "source_triangle_count": sum(
                value["triangles"] for value in baseline.values()
            ),
            "reconstructed_triangle_count": reconstructed_triangle_count,
            "vertex_count_changed_object_count": vertex_count_changed_objects,
            "relative_tolerance": args.tolerance,
            "absolute_tolerance": absolute_tolerance,
            "max_bounds_error": largest_error,
            "motion_probe_count": len(motion_records),
            "movable_joint_count": len(motion_records),
            "max_motion_matrix_error": max_motion_matrix_error,
            "max_motion_translation_error": max_motion_translation_error,
            "max_motion_rotation_error_rad": max_motion_rotation_error,
            "visual_rpy": list(GLTF_VISUAL_RPY),
            "motion_probes": motion_records,
            "parts": records,
        },
    )
    print("PROCAGEN3D_URDF_ZERO_POSE_VALIDATED")


if __name__ == "__main__":
    main()
