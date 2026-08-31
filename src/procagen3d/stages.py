"""Pure helpers for the structured, incremental ProcAgen3D pipeline.

The Blender subprocess orchestration remains in :mod:`procagen3d.pipeline`.
This module owns the deterministic decisions around part checkpoints and
targeted repairs so those decisions are independently testable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


PIPELINE_MODES = ("structured", "legacy")

_MATERIAL_GATES = frozenset(
    {
        "mean_spatial_rgb_similarity",
        "mean_palette_similarity",
        "default_white_primitive_fraction",
    }
)

_GATE_PRIORITY = {
    "source_build": 0,
    "mean_surface_distance": 10,
    "p95_surface_distance": 11,
    "visible_surface_coverage": 12,
    "mean_normal_angle_degrees": 13,
    "minimum_surface_area_ratio": 14,
    "maximum_surface_area_ratio": 15,
    "minimum_view_silhouette_iou": 20,
    "mean_silhouette_iou": 21,
    "minimum_view_area_similarity": 22,
    "center_distance": 23,
    "ground_offset": 24,
    "semantic_part_coverage": 30,
    "triangle_richness": 31,
    "unjoined_attachment_fraction": 40,
    "intersection_fraction": 41,
    "loose_component_fraction": 42,
    "boundary_edge_fraction": 43,
    "non_manifold_edge_fraction": 44,
    "inverted_normal_fraction": 45,
    "degenerate_triangle_fraction": 46,
    "low_quality_triangle_fraction": 47,
    "mean_spatial_rgb_similarity": 60,
    "mean_palette_similarity": 61,
    "default_white_primitive_fraction": 62,
}


class StructuredStageError(ValueError):
    """Raised when an incremental checkpoint violates the part contract."""


@dataclass(frozen=True, slots=True)
class RepairTarget:
    """One deterministic gate diagnosis passed to a fixer agent."""

    gate: str
    message: str
    value: Any = None
    threshold: Any = None
    operator: str | None = None
    view: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "gate": self.gate,
                "message": self.message,
                "value": self.value,
                "threshold": self.threshold,
                "operator": self.operator,
                "view": self.view,
            }.items()
            if value is not None
        }


def validate_pipeline_mode(value: str) -> str:
    if value not in PIPELINE_MODES:
        raise ValueError(f"pipeline_mode must be one of: {', '.join(PIPELINE_MODES)}")
    return value


def is_structured_plan(plan: Mapping[str, Any]) -> bool:
    assembly = plan.get("assembly")
    return (
        isinstance(assembly, Mapping)
        and assembly.get("version") == 1
        and isinstance(assembly.get("part_order"), list)
        and bool(assembly["part_order"])
    )


def structured_part_order(plan: Mapping[str, Any]) -> tuple[str, ...]:
    if not is_structured_plan(plan):
        raise StructuredStageError("plan does not contain a version-1 assembly graph")
    assembly = plan["assembly"]
    assert isinstance(assembly, Mapping)
    raw_order = assembly["part_order"]
    assert isinstance(raw_order, list)
    order = tuple(raw_order)
    if not all(isinstance(item, str) and item for item in order):
        raise StructuredStageError("assembly.part_order must contain non-empty part IDs")
    part_ids = {
        part.get("id")
        for part in plan.get("parts", [])
        if isinstance(part, Mapping) and isinstance(part.get("id"), str)
    }
    if len(order) != len(set(order)):
        raise StructuredStageError("assembly.part_order must not contain duplicates")
    missing = part_ids - set(order)
    unknown = set(order) - part_ids
    if missing or unknown:
        details = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if unknown:
            details.append("unknown " + ", ".join(sorted(unknown)))
        raise StructuredStageError("assembly.part_order does not match parts: " + "; ".join(details))
    return order


def part_by_id(plan: Mapping[str, Any], part_id: str) -> Mapping[str, Any]:
    for part in plan.get("parts", []):
        if isinstance(part, Mapping) and part.get("id") == part_id:
            return part
    raise StructuredStageError(f"unknown part id {part_id!r}")


def part_object_names(plan: Mapping[str, Any], part_ids: Sequence[str]) -> tuple[str, ...]:
    names: list[str] = []
    for part_id in part_ids:
        part = part_by_id(plan, part_id)
        raw_names = part.get("object_names")
        if not isinstance(raw_names, list) or not all(
            isinstance(name, str) and name.strip() for name in raw_names
        ):
            raise StructuredStageError(f"part {part_id!r} has invalid object_names")
        names.extend(raw_names)
    return tuple(names)


def normalized_object_key(value: str) -> str:
    """Return the stable object identity used across GLB checkpoint gates."""

    value = re.sub(r"\.\d{3}$", "", value.strip().lower())
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_")


def _unique_object_keys(names: Sequence[str], *, label: str) -> dict[str, str]:
    keyed: dict[str, str] = {}
    for name in names:
        key = normalized_object_key(name)
        if not key:
            raise StructuredStageError(
                f"{label} object name {name!r} has no stable normalized identity"
            )
        prior = keyed.get(key)
        if prior is not None:
            raise StructuredStageError(
                f"{label} object names {prior!r} and {name!r} collide as {key!r}"
            )
        keyed[key] = name
    return keyed


def _probe_object_records(probe: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    nodes = probe.get("nodes")
    meshes = probe.get("meshes")
    if not isinstance(nodes, list) or not isinstance(meshes, list):
        raise StructuredStageError("compiled GLB probe lacks node or mesh records")
    mesh_by_index = {
        mesh.get("index"): mesh
        for mesh in meshes
        if isinstance(mesh, Mapping) and isinstance(mesh.get("index"), int)
    }
    instance_by_node: dict[int, Mapping[str, Any]] = {}
    instances = probe.get("instances")
    if isinstance(instances, list):
        for instance in instances:
            if not isinstance(instance, Mapping) or not isinstance(
                instance.get("node"), int
            ):
                continue
            node_index = int(instance["node"])
            if node_index in instance_by_node:
                raise StructuredStageError(
                    f"compiled GLB probe contains multiple world instances for node {node_index}"
                )
            instance_by_node[node_index] = instance
    records: dict[str, dict[str, Any]] = {}
    for fallback_index, node in enumerate(nodes):
        if not isinstance(node, Mapping) or not isinstance(node.get("name"), str):
            continue
        mesh_index = node.get("mesh")
        mesh = mesh_by_index.get(mesh_index)
        if not isinstance(mesh, Mapping):
            continue
        name = str(node["name"])
        key = normalized_object_key(name)
        if not key:
            raise StructuredStageError(
                f"compiled GLB object name {name!r} has no stable normalized identity"
            )
        prior = records.get(key)
        if prior is not None:
            raise StructuredStageError(
                "compiled GLB object names "
                f"{prior['name']!r} and {name!r} collide as {key!r}"
            )
        primitives = []
        for primitive in mesh.get("primitives", []):
            if not isinstance(primitive, Mapping):
                continue
            primitives.append(
                {
                    key: primitive.get(key)
                    for key in (
                        "mode",
                        "vertex_count",
                        "element_count",
                        "triangle_count",
                        "position_bounds",
                        "position_sha256",
                        "indices_sha256",
                        "geometry_sha256",
                    )
                }
            )
        node_index = node.get("index")
        if not isinstance(node_index, int):
            node_index = fallback_index
        instance = instance_by_node.get(node_index)
        record = {
            "name": name,
            "local_matrix": node.get("local_matrix"),
            "bounds": mesh.get("bounds"),
            "primitives": primitives,
        }
        if instance is not None:
            record["world_matrix"] = instance.get("world_matrix")
            record["parent_identity"] = {
                "has_parent": instance.get("parent_node") is not None,
                "name": instance.get("parent_name"),
            }
        elif node.get("world_matrix") is not None:
            record["world_matrix"] = node.get("world_matrix")
            record["parent_identity"] = {
                "has_parent": node.get("parent_node") is not None,
                "name": node.get("parent_name"),
            }
        records[key] = record
    return records


def geometry_signature(
    probe: Mapping[str, Any],
    *,
    object_names: Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return material-independent geometry records keyed by normalized name."""

    records = _probe_object_records(probe)
    if object_names is None:
        return records
    requested = set(_unique_object_keys(object_names, label="requested"))
    return {key: value for key, value in records.items() if key in requested}


def validate_incremental_probe(
    *,
    plan: Mapping[str, Any],
    completed_part_ids: Sequence[str],
    probe: Mapping[str, Any],
    previous_signature: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate one cleanly exported incremental part checkpoint.

    All completed objects must exist, future objects must not leak into the
    build, and already accepted geometry must remain byte-for-byte equivalent
    at the probe-record level.
    """

    order = structured_part_order(plan)
    completed = tuple(completed_part_ids)
    if completed != order[: len(completed)]:
        raise StructuredStageError("completed parts must be a prefix of assembly.part_order")
    expected_names = part_object_names(plan, completed)
    future_names = part_object_names(plan, order[len(completed) :])
    declared_keys = _unique_object_keys(
        part_object_names(plan, order), label="declared"
    )
    records = _probe_object_records(probe)
    actual_keys = set(records)
    expected_name_set = set(expected_names)
    future_name_set = set(future_names)
    expected_keys = {
        key for key, name in declared_keys.items() if name in expected_name_set
    }
    future_keys = {
        key for key, name in declared_keys.items() if name in future_name_set
    }
    missing = sorted(expected_keys - actual_keys)
    leaked = sorted(future_keys & actual_keys)
    if missing or leaked:
        details = []
        if missing:
            details.append("missing completed objects: " + ", ".join(missing))
        if leaked:
            details.append("future objects built early: " + ", ".join(leaked))
        raise StructuredStageError("; ".join(details))

    digest_fields = ("position_sha256", "indices_sha256", "geometry_sha256")
    unhashed = sorted(
        key
        for key in expected_keys
        if any(
            any(
                not isinstance(primitive.get(field), str)
                or re.fullmatch(r"[0-9a-f]{64}", primitive[field]) is None
                for field in digest_fields
            )
            for primitive in records[key].get("primitives", [])
            if isinstance(primitive, Mapping)
        )
        or not records[key].get("primitives")
    )
    if unhashed:
        raise StructuredStageError(
            "completed objects lack deterministic geometry content hashes: "
            + ", ".join(unhashed)
        )

    unplaced = sorted(
        key
        for key in expected_keys
        if "world_matrix" not in records[key]
        or "parent_identity" not in records[key]
    )
    if unplaced:
        raise StructuredStageError(
            "completed objects lack deterministic world placement records: "
            + ", ".join(unplaced)
        )

    signature = {key: records[key] for key in sorted(expected_keys)}
    changed_previous: list[str] = []
    if previous_signature is not None:
        for key, prior in previous_signature.items():
            if key not in signature or signature[key] != prior:
                changed_previous.append(key)
    if changed_previous:
        raise StructuredStageError(
            "previously accepted part geometry changed: "
            + ", ".join(sorted(changed_previous))
        )
    return {
        "completed_parts": list(completed),
        "expected_objects": list(expected_names),
        "future_objects": list(future_names),
        "geometry_signature": signature,
    }


def hard_gate_failures(
    comparison: Mapping[str, Any], *, include_materials: bool = True
) -> tuple[Mapping[str, Any], ...]:
    hard_gates = comparison.get("hard_gates")
    failures = hard_gates.get("failures") if isinstance(hard_gates, Mapping) else None
    if not isinstance(failures, list):
        return ()
    result = []
    for failure in failures:
        if not isinstance(failure, Mapping) or not isinstance(failure.get("gate"), str):
            continue
        if not include_materials and failure["gate"] in _MATERIAL_GATES:
            continue
        result.append(failure)
    return tuple(result)


def geometry_gates_passed(comparison: Mapping[str, Any]) -> bool:
    return not hard_gate_failures(comparison, include_materials=False)


def material_gate_failures(
    comparison: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    """Return only deterministic appearance/PBR gate failures."""

    return tuple(
        failure
        for failure in hard_gate_failures(comparison, include_materials=True)
        if failure["gate"] in _MATERIAL_GATES
    )


def material_gates_passed(comparison: Mapping[str, Any]) -> bool:
    return not material_gate_failures(comparison)


def select_repair_target(
    comparison: Mapping[str, Any], *, include_materials: bool = True
) -> RepairTarget | None:
    """Choose exactly one failed gate for the next repair transaction."""

    failures = hard_gate_failures(comparison, include_materials=include_materials)
    if not failures:
        return None
    selected = min(
        enumerate(failures),
        key=lambda item: (_GATE_PRIORITY.get(str(item[1]["gate"]), 1000), item[0]),
    )[1]
    gate = str(selected["gate"])
    return RepairTarget(
        gate=gate,
        message=str(selected.get("message") or f"hard gate {gate} failed"),
        value=selected.get("value"),
        threshold=selected.get("threshold"),
        operator=(str(selected["operator"]) if selected.get("operator") is not None else None),
        view=(str(selected["view"]) if selected.get("view") is not None else None),
    )
