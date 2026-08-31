from __future__ import annotations

from copy import deepcopy

import pytest

from procagen3d.stages import (
    StructuredStageError,
    geometry_gates_passed,
    geometry_signature,
    hard_gate_failures,
    is_structured_plan,
    material_gate_failures,
    material_gates_passed,
    select_repair_target,
    structured_part_order,
    validate_incremental_probe,
    validate_pipeline_mode,
)


def _structured_plan() -> dict:
    return {
        "parts": [
            {"id": "base", "object_names": ["Base Shell"]},
            {"id": "arm", "object_names": ["Left-Wheel", "Wheel Pin"]},
            {"id": "tool", "object_names": ["Tool.Head"]},
        ],
        "assembly": {
            "version": 1,
            "part_order": ["base", "arm", "tool"],
        },
    }


def _identity() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _probe(*names: str) -> dict:
    nodes = []
    meshes = []
    instances = []
    for index, name in enumerate(names):
        bounds = {
            "min": [float(index), 0.0, 0.0],
            "max": [float(index + 1), 1.0, 1.0],
        }
        nodes.append(
            {
                "index": index,
                "name": name,
                "mesh": index,
                "local_matrix": _identity(),
            }
        )
        instances.append(
            {
                "node": index,
                "node_name": name,
                "mesh": index,
                "world_matrix": _identity(),
                "path": [index],
                "parent_node": None,
                "parent_name": None,
            }
        )
        meshes.append(
            {
                "index": index,
                "bounds": bounds,
                "primitives": [
                    {
                        "mode": 4,
                        "vertex_count": 8 + index,
                        "element_count": 36,
                        "triangle_count": 12,
                        "position_bounds": bounds,
                        "position_sha256": "1" * 64,
                        "indices_sha256": "2" * 64,
                        "geometry_sha256": "3" * 64,
                        # Geometry signatures deliberately ignore material data.
                        "material": 100 + index,
                    }
                ],
            }
        )
    return {"nodes": nodes, "meshes": meshes, "instances": instances}


def _comparison(*failures: object) -> dict:
    return {"hard_gates": {"passed": not failures, "failures": list(failures)}}


def test_pipeline_mode_and_structured_plan_detection_are_strict() -> None:
    assert validate_pipeline_mode("structured") == "structured"
    assert validate_pipeline_mode("legacy") == "legacy"
    with pytest.raises(ValueError, match="pipeline_mode"):
        validate_pipeline_mode("auto")

    plan = _structured_plan()
    assert is_structured_plan(plan)
    assert not is_structured_plan({"assembly": {"version": 1, "part_order": []}})
    assert not is_structured_plan(
        {"assembly": {"version": "1", "part_order": ["base"]}}
    )


def test_structured_part_order_matches_all_declared_parts() -> None:
    plan = _structured_plan()
    assert structured_part_order(plan) == ("base", "arm", "tool")

    duplicate = deepcopy(plan)
    duplicate["assembly"]["part_order"] = ["base", "arm", "arm"]
    with pytest.raises(StructuredStageError, match="must not contain duplicates"):
        structured_part_order(duplicate)

    mismatch = deepcopy(plan)
    mismatch["assembly"]["part_order"] = ["base", "ghost"]
    with pytest.raises(StructuredStageError) as error:
        structured_part_order(mismatch)
    assert "missing arm, tool" in str(error.value)
    assert "unknown ghost" in str(error.value)

    invalid_item = deepcopy(plan)
    invalid_item["assembly"]["part_order"] = ["base", "", "tool"]
    with pytest.raises(StructuredStageError, match="non-empty part IDs"):
        structured_part_order(invalid_item)

    with pytest.raises(StructuredStageError, match="version-1 assembly graph"):
        structured_part_order({"parts": plan["parts"]})


def test_incremental_probe_accepts_completed_objects_and_normalizes_names() -> None:
    validation = validate_incremental_probe(
        plan=_structured_plan(),
        completed_part_ids=["base", "arm"],
        probe=_probe("base-shell.001", "LEFT WHEEL", "Wheel_Pin", "helper"),
    )

    assert validation["completed_parts"] == ["base", "arm"]
    assert validation["expected_objects"] == [
        "Base Shell",
        "Left-Wheel",
        "Wheel Pin",
    ]
    assert validation["future_objects"] == ["Tool.Head"]
    assert set(validation["geometry_signature"]) == {
        "base_shell",
        "left_wheel",
        "wheel_pin",
    }
    assert "helper" not in validation["geometry_signature"]


def test_incremental_probe_rejects_non_prefix_missing_and_future_objects() -> None:
    plan = _structured_plan()

    with pytest.raises(StructuredStageError, match="must be a prefix"):
        validate_incremental_probe(
            plan=plan,
            completed_part_ids=["arm"],
            probe=_probe("Left-Wheel", "Wheel Pin"),
        )

    with pytest.raises(StructuredStageError, match="missing completed objects: wheel_pin"):
        validate_incremental_probe(
            plan=plan,
            completed_part_ids=["base", "arm"],
            probe=_probe("Base Shell", "Left-Wheel"),
        )

    with pytest.raises(StructuredStageError, match="future objects built early: tool_head"):
        validate_incremental_probe(
            plan=plan,
            completed_part_ids=["base"],
            probe=_probe("Base Shell", "Tool.Head"),
        )

    with pytest.raises(StructuredStageError) as error:
        validate_incremental_probe(
            plan=plan,
            completed_part_ids=["base", "arm"],
            probe=_probe("Base Shell", "Tool.Head"),
        )
    assert "missing completed objects: left_wheel, wheel_pin" in str(error.value)
    assert "future objects built early: tool_head" in str(error.value)


def test_incremental_probe_freezes_previously_accepted_geometry() -> None:
    plan = _structured_plan()
    first = validate_incremental_probe(
        plan=plan,
        completed_part_ids=["base"],
        probe=_probe("Base Shell"),
    )
    second_probe = _probe("Base Shell", "Left-Wheel", "Wheel Pin")

    second = validate_incremental_probe(
        plan=plan,
        completed_part_ids=["base", "arm"],
        probe=second_probe,
        previous_signature=first["geometry_signature"],
    )
    assert set(second["geometry_signature"]) == {
        "base_shell",
        "left_wheel",
        "wheel_pin",
    }

    changed = deepcopy(second_probe)
    changed["meshes"][0]["primitives"][0]["triangle_count"] = 99
    with pytest.raises(
        StructuredStageError,
        match="previously accepted part geometry changed: base_shell",
    ):
        validate_incremental_probe(
            plan=plan,
            completed_part_ids=["base", "arm"],
            probe=changed,
            previous_signature=first["geometry_signature"],
        )


def test_incremental_probe_freezes_content_world_transform_and_parent() -> None:
    plan = _structured_plan()
    first_probe = _probe("Base Shell")
    first = validate_incremental_probe(
        plan=plan,
        completed_part_ids=["base"],
        probe=first_probe,
    )

    for field, value in (
        ("geometry_sha256", "4" * 64),
        ("position_sha256", "5" * 64),
    ):
        changed = deepcopy(first_probe)
        changed["meshes"][0]["primitives"][0][field] = value
        with pytest.raises(
            StructuredStageError,
            match="previously accepted part geometry changed: base_shell",
        ):
            validate_incremental_probe(
                plan=plan,
                completed_part_ids=["base"],
                probe=changed,
                previous_signature=first["geometry_signature"],
            )

    moved = deepcopy(first_probe)
    moved["instances"][0]["world_matrix"][0][3] = 2.0
    with pytest.raises(StructuredStageError, match="base_shell"):
        validate_incremental_probe(
            plan=plan,
            completed_part_ids=["base"],
            probe=moved,
            previous_signature=first["geometry_signature"],
        )

    reparented = deepcopy(first_probe)
    reparented["instances"][0].update(parent_node=7, parent_name="FixtureRoot")
    with pytest.raises(StructuredStageError, match="base_shell"):
        validate_incremental_probe(
            plan=plan,
            completed_part_ids=["base"],
            probe=reparented,
            previous_signature=first["geometry_signature"],
        )


def test_incremental_probe_requires_hashes_and_rejects_name_collisions() -> None:
    plan = _structured_plan()
    unhashed = _probe("Base Shell")
    del unhashed["meshes"][0]["primitives"][0]["geometry_sha256"]
    with pytest.raises(StructuredStageError, match="lack deterministic geometry"):
        validate_incremental_probe(
            plan=plan,
            completed_part_ids=["base"],
            probe=unhashed,
        )

    unplaced = _probe("Base Shell")
    unplaced.pop("instances")
    with pytest.raises(StructuredStageError, match="world placement records"):
        validate_incremental_probe(
            plan=plan,
            completed_part_ids=["base"],
            probe=unplaced,
        )

    with pytest.raises(StructuredStageError, match="collide as 'bolt'"):
        geometry_signature(_probe("Bolt", "Bolt.001"))

    colliding_plan = deepcopy(plan)
    colliding_plan["parts"][0]["object_names"] = ["Bolt", "Bolt.001"]
    with pytest.raises(StructuredStageError, match="collide as 'bolt'"):
        validate_incremental_probe(
            plan=colliding_plan,
            completed_part_ids=["base"],
            probe=_probe("Bolt"),
        )


def test_geometry_signature_is_material_independent_and_filterable() -> None:
    probe = _probe("Base Shell", "Arm")
    changed_material = deepcopy(probe)
    changed_material["meshes"][0]["primitives"][0]["material"] = 999

    assert geometry_signature(probe) == geometry_signature(changed_material)
    assert set(geometry_signature(probe, object_names=["base-shell"])) == {
        "base_shell"
    }


def test_material_gate_filtering_preserves_geometry_failures() -> None:
    comparison = _comparison(
        {
            "gate": "mean_palette_similarity",
            "value": 0.3,
            "threshold": 0.8,
        },
        {"gate": "intersection_fraction", "value": 0.2, "threshold": 0.03},
        {"message": "malformed record"},
        {"gate": "default_white_primitive_fraction", "value": 1.0},
        {"gate": "mean_spatial_rgb_similarity", "value": 0.1},
    )

    assert [failure["gate"] for failure in hard_gate_failures(comparison)] == [
        "mean_palette_similarity",
        "intersection_fraction",
        "default_white_primitive_fraction",
        "mean_spatial_rgb_similarity",
    ]
    assert [
        failure["gate"]
        for failure in hard_gate_failures(comparison, include_materials=False)
    ] == ["intersection_fraction"]
    assert not geometry_gates_passed(comparison)
    assert [failure["gate"] for failure in material_gate_failures(comparison)] == [
        "mean_palette_similarity",
        "default_white_primitive_fraction",
        "mean_spatial_rgb_similarity",
    ]
    assert not material_gates_passed(comparison)

    material_only = _comparison(
        {"gate": "mean_palette_similarity"},
        {"gate": "default_white_primitive_fraction"},
    )
    assert geometry_gates_passed(material_only)
    assert not material_gates_passed(material_only)
    assert material_gates_passed(_comparison({"gate": "intersection_fraction"}))
    assert hard_gate_failures({"hard_gates": {"failures": "invalid"}}) == ()


def test_repair_target_uses_deterministic_gate_priority() -> None:
    comparison = _comparison(
        {"gate": "mean_palette_similarity", "message": "palette"},
        {"gate": "unknown_custom_gate", "message": "unknown"},
        {"gate": "intersection_fraction", "message": "intersections"},
        {"gate": "mean_silhouette_iou", "message": "silhouette"},
        {
            "gate": "p95_surface_distance",
            "message": "surface tail",
            "value": 0.08,
            "threshold": 0.04,
            "operator": "<=",
            "view": "iso",
        },
    )

    target = select_repair_target(comparison)
    assert target is not None
    assert target.gate == "p95_surface_distance"
    assert target.as_dict() == {
        "gate": "p95_surface_distance",
        "message": "surface tail",
        "value": 0.08,
        "threshold": 0.04,
        "operator": "<=",
        "view": "iso",
    }

    materials_and_unknown = _comparison(
        {"gate": "mean_palette_similarity", "message": "palette"},
        {"gate": "custom_b", "message": "second unknown"},
        {"gate": "custom_a", "message": "first unknown by order"},
    )
    geometry_target = select_repair_target(
        materials_and_unknown, include_materials=False
    )
    assert geometry_target is not None
    assert geometry_target.gate == "custom_b"
    assert select_repair_target(
        _comparison({"gate": "mean_palette_similarity"}),
        include_materials=False,
    ) is None
    assert select_repair_target(_comparison()) is None
