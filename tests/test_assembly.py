from __future__ import annotations

import copy
import math

import pytest

from procagen3d.assembly import (
    AssemblyError,
    frame_matrix,
    identity_matrix,
    multiply_matrices,
    solve_assembly_transforms,
    solve_child_transform,
    solve_mate_transform,
    topological_part_order,
    transform_point,
    validate_assembly,
)
from procagen3d.plan_schema import PLAN_SCHEMA, PlanSchemaError, validate_plan_document


def _frame(
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict[str, list[float]]:
    return {
        "origin": list(origin),
        "x_axis": [1.0, 0.0, 0.0],
        "y_axis": [0.0, 1.0, 0.0],
        "z_axis": [0.0, 0.0, 1.0],
    }


def _attachment(parent_id: str, kind: str) -> dict[str, object]:
    return {
        "parent_id": parent_id,
        "type": kind,
        "contact_region": {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]},
        "max_gap": 0.01,
        "max_penetration": 0.01,
        "min_contact_area": 0.0,
    }


def _part(identifier: str, parent_id: str, kind: str) -> dict[str, object]:
    return {
        "id": identifier,
        "name": identifier.replace("_", " ").title(),
        "shape_family": "hard-surface",
        "approximate_bounds": {
            "min": [0.0, 0.0, 0.0],
            "max": [1.0, 1.0, 1.0],
        },
        "visual_role": "assembly part",
        "object_names": [identifier.title().replace("_", "")],
        "attachment": _attachment(parent_id, kind),
    }


def _connector(
    identifier: str,
    part_id: str,
    *,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    interface: str = "cylindrical",
    role: str = "neutral",
) -> dict[str, object]:
    return {
        "id": identifier,
        "part_id": part_id,
        "interface": interface,
        "role": role,
        "frame": _frame(origin),
        "nominal_dimensions": {"diameter": 0.1},
    }


def _mate(
    identifier: str,
    parent_connector_id: str,
    child_connector_id: str,
    kind: str,
) -> dict[str, object]:
    mate: dict[str, object] = {
        "id": identifier,
        "type": kind,
        "parent_connector_id": parent_connector_id,
        "child_connector_id": child_connector_id,
        "fit": "clearance",
        "clearance": 0.002,
        "fit_offset": [0.0, 0.0, 0.0],
        "nominal_dimensions": {"diameter": 0.1},
    }
    if kind in {"revolute", "prismatic"}:
        mate.update(rest=0.0, limits={"lower": -2.0, "upper": 2.0})
    elif kind == "spherical":
        mate.update(
            rest=[0.0, 0.0, 0.0],
            limits={"lower": [-1.0, -1.0, -1.0], "upper": [1.0, 1.0, 1.0]},
        )
    return mate


def _plan() -> dict[str, object]:
    return {
        "subject": "hinged fixture",
        "subject_kind": "object",
        "coordinate_frame": {"up": "+Z"},
        "dimensions": [2.0, 2.0, 2.0],
        "parts": [
            _part("base", "__root__", "root"),
            _part("door", "base", "articulated"),
        ],
        "assembly": {
            "version": 1,
            "part_order": ["base", "door"],
            "connectors": [
                _connector("base.hinge", "base", origin=(1.0, 0.0, 0.5), role="female"),
                _connector("door.hinge", "door", role="male"),
            ],
            "mates": [
                {
                    **_mate("door_joint", "base.hinge", "door.hinge", "revolute"),
                    "rest": 0.25,
                    "limits": {"lower": -1.0, "upper": 1.0},
                }
            ],
        },
        "materials": [],
        "construction_strategy": "build in assembly order",
        "identity_features": [],
        "limitations": [],
    }


def _flatten(matrix: object) -> tuple[float, ...]:
    assert isinstance(matrix, tuple)
    return tuple(value for row in matrix for value in row)


def test_schema_exposes_optional_host_solved_assembly_contract() -> None:
    assembly = PLAN_SCHEMA["properties"]["assembly"]
    assert "assembly" not in PLAN_SCHEMA["required"]
    assert assembly["properties"]["version"]["enum"] == [1]
    assert assembly["properties"]["mates"]["items"]["properties"]["type"]["enum"] == [
        "rigid",
        "revolute",
        "prismatic",
        "spherical",
    ]


def test_legacy_plan_synthesizes_a_topological_empty_assembly_view() -> None:
    plan = _plan()
    del plan["assembly"]
    # Existing plans were allowed to store parts in an order different from the
    # attachment graph.  Compatibility normalization must not reject them.
    plan["parts"] = [plan["parts"][1], plan["parts"][0]]

    normalized = validate_plan_document(plan)

    assert normalized["assembly"] == {
        "version": 1,
        "part_order": ["base", "door"],
        "connectors": [],
        "mates": [],
    }
    assert topological_part_order(normalized) == ("base", "door")
    assert solve_assembly_transforms(normalized) == {
        "base": identity_matrix(),
        "door": identity_matrix(),
    }


def test_explicit_assembly_receives_only_backward_compatible_defaults() -> None:
    plan = _plan()
    assembly = plan["assembly"]
    assert isinstance(assembly, dict)
    connector = assembly["connectors"][0]
    mate = assembly["mates"][0]
    assert isinstance(connector, dict) and isinstance(mate, dict)
    del connector["role"]
    del connector["nominal_dimensions"]
    del mate["fit"]
    del mate["clearance"]
    del mate["fit_offset"]
    del mate["nominal_dimensions"]
    del mate["rest"]

    normalized = validate_plan_document(plan)
    normalized_connector = normalized["assembly"]["connectors"][0]
    normalized_mate = normalized["assembly"]["mates"][0]
    assert normalized_connector["role"] == "neutral"
    assert normalized_connector["nominal_dimensions"] == {}
    assert normalized_mate["fit"] == "none"
    assert normalized_mate["clearance"] == 0.0
    assert normalized_mate["fit_offset"] == [0.0, 0.0, 0.0]
    assert normalized_mate["nominal_dimensions"] == {}
    assert normalized_mate["rest"] == 0.0


def test_revolute_solver_aligns_connectors_and_accepts_joint_override() -> None:
    plan = validate_plan_document(_plan())
    plan["assembly"]["mates"][0]["limits"] = {"lower": -2.0, "upper": 2.0}
    at_rest = solve_child_transform(plan, "door_joint")
    overridden = solve_child_transform(plan, "door_joint", joint_parameter=math.pi / 2)

    assert transform_point(at_rest, (0.0, 0.0, 0.0)) == pytest.approx((1.0, 0.0, 0.5))
    assert transform_point(overridden, (0.0, 0.0, 0.0)) == pytest.approx((1.0, 0.0, 0.5))
    assert transform_point(overridden, (1.0, 0.0, 0.0)) == pytest.approx((1.0, 1.0, 0.5))

    connector = plan["assembly"]["connectors"][1]
    child_connector_world = multiply_matrices(overridden, frame_matrix(connector["frame"]))
    expected = multiply_matrices(
        frame_matrix(plan["assembly"]["connectors"][0]["frame"]),
        (
            (0.0, -1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )
    assert _flatten(child_connector_world) == pytest.approx(_flatten(expected))


def test_prismatic_fit_offset_and_child_local_frame_are_composed_in_order() -> None:
    mate = _mate("slide", "parent", "child", "prismatic")
    mate["fit_offset"] = [0.0, 0.0, 0.1]
    solved = solve_mate_transform(
        identity_matrix(),
        _frame((1.0, 2.0, 3.0)),
        _frame((0.0, 0.0, 1.0)),
        mate,
        joint_parameter=2.0,
    )
    assert transform_point(solved, (0.0, 0.0, 0.0)) == pytest.approx((1.0, 2.0, 4.1))


def test_spherical_solver_uses_deterministic_xyz_euler_angles() -> None:
    mate = _mate("ball", "parent", "child", "spherical")
    mate["limits"] = {
        "lower": [-2.0, -2.0, -2.0],
        "upper": [2.0, 2.0, 2.0],
    }
    solved = solve_mate_transform(
        identity_matrix(),
        _frame(),
        _frame(),
        mate,
        joint_parameter=(0.0, 0.0, math.pi / 2),
    )
    assert transform_point(solved, (1.0, 0.0, 0.0)) == pytest.approx((0.0, 1.0, 0.0), abs=1e-8)


def test_complete_solver_propagates_parent_world_transforms_and_joint_values() -> None:
    plan = validate_plan_document(_plan())
    root = (
        (1.0, 0.0, 0.0, 10.0),
        (0.0, 1.0, 0.0, -2.0),
        (0.0, 0.0, 1.0, 4.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    transforms = solve_assembly_transforms(
        plan,
        {"door_joint": 0.0},
        root_transform=root,
    )
    assert transforms["base"] == root
    assert transform_point(transforms["door"], (0.0, 0.0, 0.0)) == pytest.approx(
        (11.0, -2.0, 4.5)
    )
    with pytest.raises(AssemblyError, match="unknown mate ids"):
        solve_assembly_transforms(plan, {"missing": 0.0})


def test_all_supported_mate_types_validate_in_one_ordered_graph() -> None:
    plan = _plan()
    part_specs = [
        ("base", "__root__", "root"),
        ("fixed", "base", "fused"),
        ("rotor", "fixed", "articulated"),
        ("slider", "rotor", "articulated"),
        ("ball", "slider", "articulated"),
    ]
    plan["parts"] = [_part(*spec) for spec in part_specs]
    connectors: list[dict[str, object]] = []
    mates: list[dict[str, object]] = []
    for parent, child, kind in (
        ("base", "fixed", "rigid"),
        ("fixed", "rotor", "revolute"),
        ("rotor", "slider", "prismatic"),
        ("slider", "ball", "spherical"),
    ):
        parent_connector = f"{parent}.to.{child}"
        child_connector = f"{child}.from.{parent}"
        connectors.extend(
            [_connector(parent_connector, parent), _connector(child_connector, child)]
        )
        mates.append(_mate(f"{parent}-{child}", parent_connector, child_connector, kind))
    plan["assembly"] = {
        "version": 1,
        "part_order": [spec[0] for spec in part_specs],
        "connectors": connectors,
        "mates": mates,
    }

    normalized = validate_plan_document(plan)
    report = validate_assembly(normalized)
    assert report.valid
    assert report.ordered_part_ids == ("base", "fixed", "rotor", "slider", "ball")
    assert set(solve_assembly_transforms(normalized)) == set(report.ordered_part_ids)


def test_validation_reports_frame_reference_order_and_joint_errors_together() -> None:
    plan = validate_plan_document(_plan())
    broken = copy.deepcopy(plan)
    assembly = broken["assembly"]
    assembly["part_order"] = ["door", "door", "unknown"]
    assembly["connectors"][0]["frame"]["z_axis"] = [0.0, 0.0, -1.0]
    assembly["connectors"][1]["id"] = "base.hinge"
    mate = assembly["mates"][0]
    mate["parent_connector_id"] = "missing"
    mate["rest"] = 2.0

    report = validate_assembly(broken)
    keywords = {issue.keyword for issue in report.issues}
    assert {
        "completePartOrder",
        "knownPart",
        "orthonormalFrame",
        "uniqueConnectorId",
        "knownConnector",
        "jointConstraints",
        "uniquePartOrder",
    } <= keywords
    assert not report.valid
    assert report.as_dict()["valid"] is False
    with pytest.raises(AssemblyError, match="assembly is invalid"):
        report.raise_for_errors()


def test_plan_validation_rejects_non_topological_order_and_attachment_mismatch() -> None:
    plan = _plan()
    assembly = plan["assembly"]
    assembly["part_order"] = ["door", "base"]
    plan["parts"][1]["attachment"]["parent_id"] = "door"

    with pytest.raises(PlanSchemaError) as caught:
        validate_plan_document(plan)

    rendered = str(caught.value)
    assert "topological" in rendered or "must appear after" in rendered
    assert "mate parent part must match" in rendered


def test_validation_rejects_left_handed_frames_incompatible_roles_and_interfaces() -> None:
    plan = validate_plan_document(_plan())
    assembly = plan["assembly"]
    assembly["connectors"][0]["frame"]["z_axis"] = [0.0, 0.0, -1.0]
    assembly["connectors"][0]["role"] = "male"
    assembly["connectors"][1]["role"] = "male"
    assembly["connectors"][1]["interface"] = "tab-slot"

    report = validate_assembly(plan)
    keywords = {issue.keyword for issue in report.issues}
    assert "orthonormalFrame" in keywords
    assert "compatibleConnectorRole" in keywords
    assert "compatibleInterface" in keywords


def test_solver_rejects_out_of_limit_and_rigid_joint_parameters() -> None:
    plan = validate_plan_document(_plan())
    with pytest.raises(AssemblyError, match="outside"):
        solve_child_transform(plan, "door_joint", joint_parameter=1.5)

    rigid = _mate("fixed", "parent", "child", "rigid")
    with pytest.raises(AssemblyError, match="does not accept"):
        solve_mate_transform(
            identity_matrix(), _frame(), _frame(), rigid, joint_parameter=0.1
        )


def test_multiple_consistent_mates_are_allowed_but_overconstraint_is_detected() -> None:
    plan = _plan()
    assembly = plan["assembly"]
    assembly["connectors"].extend(
        [
            _connector("base.second", "base", origin=(1.0, 0.0, 0.5)),
            _connector("door.second", "door", origin=(0.0, 0.0, 0.0)),
        ]
    )
    assembly["mates"].append(
        {
            **_mate("door_second", "base.second", "door.second", "revolute"),
            "rest": 0.25,
            "limits": {"lower": -1.0, "upper": 1.0},
        }
    )
    normalized = validate_plan_document(plan)
    transforms = solve_assembly_transforms(normalized)
    assert "door" in transforms

    normalized["assembly"]["connectors"][2]["frame"]["origin"] = [2.0, 0.0, 0.5]
    with pytest.raises(AssemblyError, match="over-constrain"):
        solve_assembly_transforms(normalized)
