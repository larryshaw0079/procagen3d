from __future__ import annotations

import json
from pathlib import Path

import pytest

import procagen3d.pipeline as pipeline
from procagen3d.plan_schema import (
    PLAN_SCHEMA,
    PlanSchemaError,
    plan_schema_violations,
    plan_schema_text,
    validate_plan_document,
)


def _valid_plan() -> dict[str, object]:
    return {
        "subject": "fixture",
        "subject_kind": "object",
        "coordinate_frame": {"up": "+Z"},
        "dimensions": [1.0, 1.0, 1.0],
        "parts": [{"semantic_name": "Body"}],
        "materials": [],
        "construction_strategy": "one primitive",
        "identity_features": [],
        "limitations": [],
    }


def _connector(connector_id: str, part_id: str, role: str) -> dict[str, object]:
    return {
        "id": connector_id,
        "part_id": part_id,
        "interface": "generic",
        "role": role,
        "frame": {
            "origin": [0.0, 0.0, 0.0],
            "x_axis": [1.0, 0.0, 0.0],
            "y_axis": [0.0, 1.0, 0.0],
            "z_axis": [0.0, 0.0, 1.0],
        },
        "nominal_dimensions": {},
    }


def _plan_with_mate(mate_type: str) -> dict[str, object]:
    plan = _valid_plan()
    plan["parts"] = [
        {
            "id": "base",
            "name": "Base",
            "attachment": {"parent_id": "__root__", "type": "root"},
        },
        {
            "id": "child",
            "name": "Child",
            "attachment": {"parent_id": "base", "type": "articulated"},
        },
    ]
    plan["assembly"] = {
        "version": 1,
        "part_order": ["base", "child"],
        "connectors": [
            _connector("base.connector", "base", "female"),
            _connector("child.connector", "child", "male"),
        ],
        "mates": [
            {
                "id": "child.mate",
                "type": mate_type,
                "parent_connector_id": "base.connector",
                "child_connector_id": "child.connector",
                "fit": "none",
                "clearance": 0.0,
                "fit_offset": [0.0, 0.0, 0.0],
                "nominal_dimensions": {},
            }
        ],
    }
    return plan


def test_schema_declares_modes_and_old_plans_default_to_procedural() -> None:
    mode = PLAN_SCHEMA["properties"]["reconstruction_mode"]
    assert mode["enum"] == ["procedural", "glb-ref"]
    assert mode["default"] == "procedural"

    original = _valid_plan()
    normalized = validate_plan_document(original)
    assert "reconstruction_mode" not in original
    assert normalized["reconstruction_mode"] == "procedural"
    assert normalized["granularity"] == "medium"
    assert normalized["quality_profile"] == {
        "surface_fidelity": "off",
        "detail_richness": "standard",
        "material_fidelity": "faithful",
        "structural_coherence": "coherent",
    }

    granularity = PLAN_SCHEMA["properties"]["granularity"]
    assert granularity["enum"] == ["coarse", "medium", "fine", "surface"]
    assert granularity["default"] == "medium"

    original["reconstruction_mode"] = "glb-ref"
    original["granularity"] = "fine"
    assert validate_plan_document(original)["reconstruction_mode"] == "glb-ref"
    assert validate_plan_document(original)["granularity"] == "fine"


def test_schema_documents_and_enforces_the_reserved_root_parent() -> None:
    attachment_schema = PLAN_SCHEMA["properties"]["parts"]["items"]["properties"][
        "attachment"
    ]
    assert "__root__" in attachment_schema["properties"]["parent_id"]["description"]
    assert attachment_schema["allOf"][0]["then"]["properties"]["parent_id"][
        "enum"
    ] == ["__root__"]
    assert attachment_schema["allOf"][0]["else"]["properties"]["parent_id"][
        "not"
    ] == {"enum": ["__root__"]}

    normalized = validate_plan_document(_valid_plan())
    normalized["parts"][0]["attachment"]["parent_id"] = "world"
    schema_errors = plan_schema_violations(normalized)
    assert any(
        violation.contract == "schema"
        and violation.path == ("parts", 0, "attachment", "parent_id")
        and violation.keyword == "enum"
        for violation in schema_errors
    )

    with pytest.raises(PlanSchemaError) as caught:
        validate_plan_document(normalized)

    assert "the JSON Schema and semantic contract" in str(caught.value)
    assert {violation.contract for violation in caught.value.violations} == {
        "schema",
        "semantic",
    }
    assert "[schema] $.parts[0].attachment.parent_id" in str(caught.value)
    assert "[semantic] $.parts[0].attachment.parent_id" in str(caught.value)

    normalized = validate_plan_document(_valid_plan())
    normalized["parts"].append(
        {
            "id": "child",
            "name": "Child",
            "shape_family": "box",
            "approximate_bounds": {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]},
            "visual_role": "child",
            "object_names": ["Child"],
            "attachment": {
                "parent_id": "__root__",
                "type": "fused",
                "contact_region": {
                    "min": [0.0, 0.0, 0.0],
                    "max": [1.0, 1.0, 1.0],
                },
                "max_gap": 0.0,
                "max_penetration": 0.0,
                "min_contact_area": 0.0,
            },
        }
    )
    sentinel_errors = plan_schema_violations(normalized)
    assert any(
        violation.path == ("parts", 1, "attachment", "parent_id")
        and violation.keyword == "not"
        for violation in sentinel_errors
    )


def test_legacy_parts_receive_deterministic_typed_attachments() -> None:
    plan = _valid_plan()
    plan["dimensions"] = [4.0, 2.0, 1.5]
    plan["parts"] = [
        {
            "name": "Main Body",
            "approximate_bounds": [[-2, -1, 0], [2, 1, 1]],
        },
        {"semantic_name": "Roof"},
    ]

    normalized = validate_plan_document(plan)
    body, roof = normalized["parts"]
    assert body["id"] == "main_body"
    assert body["approximate_bounds"] == {
        "min": [-2.0, -1.0, 0.0],
        "max": [2.0, 1.0, 1.0],
    }
    assert body["attachment"]["parent_id"] == "__root__"
    assert body["attachment"]["type"] == "root"
    assert roof["id"] == "roof"
    assert roof["attachment"]["parent_id"] == "main_body"
    assert roof["attachment"]["type"] == "surface-contact"
    assert roof["object_names"] == ["Roof"]


@pytest.mark.parametrize(
    ("mate_type", "expected_rest"),
    [
        ("revolute", 0.0),
        ("prismatic", 0.0),
        ("spherical", [0.0, 0.0, 0.0]),
    ],
)
def test_joint_conditionals_preserve_legacy_rest_defaults(
    mate_type: str, expected_rest: object
) -> None:
    normalized = validate_plan_document(_plan_with_mate(mate_type))
    assert normalized["assembly"]["mates"][0]["rest"] == expected_rest


def test_schema_conditionally_forbids_rigid_rest_and_limits() -> None:
    mate_schema = PLAN_SCHEMA["properties"]["assembly"]["properties"]["mates"][
        "items"
    ]
    assert "Rigid mates omit rest and limits" in mate_schema["properties"]["type"][
        "description"
    ]

    normalized = validate_plan_document(_plan_with_mate("rigid"))
    mate = normalized["assembly"]["mates"][0]
    mate["rest"] = 0.0
    mate["limits"] = {"lower": 0.0, "upper": 0.0}

    with pytest.raises(PlanSchemaError) as caught:
        validate_plan_document(normalized)

    schema_paths = {
        violation.path
        for violation in caught.value.violations
        if violation.contract == "schema" and violation.keyword == "not"
    }
    assert schema_paths == {
        ("assembly", "mates", 0, "limits"),
        ("assembly", "mates", 0, "rest"),
    }
    assert any(
        violation.contract == "semantic"
        and violation.keyword == "jointConstraints"
        and "must not declare rest or limits" in violation.message
        for violation in caught.value.violations
    )


@pytest.mark.parametrize(
    ("mate_type", "invalid_rest", "invalid_limits", "expected_paths"),
    [
        (
            "revolute",
            [0.0, 0.0, 0.0],
            {"lower": [-1.0, -1.0, -1.0], "upper": [1.0, 1.0, 1.0]},
            {
                ("assembly", "mates", 0, "rest"),
                ("assembly", "mates", 0, "limits", "lower"),
                ("assembly", "mates", 0, "limits", "upper"),
            },
        ),
        (
            "spherical",
            0.0,
            {"lower": -1.0, "upper": 1.0},
            {
                ("assembly", "mates", 0, "rest"),
                ("assembly", "mates", 0, "limits", "lower"),
                ("assembly", "mates", 0, "limits", "upper"),
            },
        ),
    ],
)
def test_schema_matches_joint_rest_and_limit_shapes_to_mate_type(
    mate_type: str,
    invalid_rest: object,
    invalid_limits: object,
    expected_paths: set[tuple[str | int, ...]],
) -> None:
    normalized = validate_plan_document(_plan_with_mate(mate_type))
    mate = normalized["assembly"]["mates"][0]
    mate["rest"] = invalid_rest
    mate["limits"] = invalid_limits

    schema_errors = plan_schema_violations(normalized)
    type_paths = {
        violation.path
        for violation in schema_errors
        if violation.contract == "schema" and violation.keyword == "type"
    }
    assert expected_paths <= type_paths


def test_typed_part_ids_and_attachment_references_are_not_rewritten() -> None:
    plan = _valid_plan()
    plan["parts"] = [
        {
            "id": "body-shell",
            "name": "Body shell",
            "shape_family": "loft",
            "approximate_bounds": {"min": [-1, -2, 0], "max": [1, 2, 1]},
            "visual_role": "primary mass",
            "object_names": ["BodyShell"],
            "attachment": {
                "parent_id": "__root__",
                "type": "root",
                "contact_region": {"min": [-1, -2, 0], "max": [1, 2, 1]},
                "max_gap": 0,
                "max_penetration": 0,
                "min_contact_area": 0,
            },
        },
        {
            "id": "windshield.glass",
            "name": "Windshield",
            "shape_family": "curved panel",
            "approximate_bounds": {"min": [-0.8, -0.2, 0.6], "max": [0.8, 0, 1]},
            "visual_role": "glazing",
            "object_names": ["WindshieldGlass"],
            "attachment": {
                "parent_id": "body-shell",
                "type": "embedded",
                "contact_region": {"min": [-0.8, -0.2, 0.6], "max": [0.8, 0, 1]},
                "max_gap": 0.005,
                "max_penetration": 0.01,
                "min_contact_area": 0.1,
            },
        },
    ]

    normalized = validate_plan_document(plan)
    assert [part["id"] for part in normalized["parts"]] == [
        "body-shell",
        "windshield.glass",
    ]
    assert normalized["parts"][1]["attachment"]["parent_id"] == "body-shell"


def test_typed_attachment_semantics_are_hard_validation_errors() -> None:
    plan = _valid_plan()
    plan["parts"] = [
        {"id": "duplicate", "name": "Body"},
        {
            "id": "duplicate",
            "name": "Window",
            "attachment": {
                "parent_id": "missing-parent",
                "type": "surface-contact",
                "max_gap": -0.1,
            },
        },
    ]

    with pytest.raises(PlanSchemaError) as caught:
        validate_plan_document(plan)

    rendered = "\n".join(str(item) for item in caught.value.violations)
    assert "unique" in rendered
    assert "missing-parent" in rendered or "declared part id" in rendered
    assert "max_gap" in rendered


def test_typed_attachment_graph_rejects_cycles_and_reversed_contact_bounds() -> None:
    plan = _valid_plan()
    plan["parts"] = [
        {
            "id": "root",
            "name": "Root",
        },
        {
            "id": "left",
            "name": "Left",
            "attachment": {
                "parent_id": "right",
                "type": "fused",
                "contact_region": {"min": [1, 0, 0], "max": [0, 1, 1]},
            },
        },
        {
            "id": "right",
            "name": "Right",
            "attachment": {"parent_id": "left", "type": "fused"},
        },
    ]

    with pytest.raises(PlanSchemaError) as caught:
        validate_plan_document(plan)

    rendered = "\n".join(str(item) for item in caught.value.violations)
    assert "must not form a cycle" in rendered
    assert "contact_region.min[0]" in rendered


def test_invalid_attachment_parent_type_is_reported_without_validator_crash() -> None:
    plan = _valid_plan()
    plan["parts"] = [
        {"id": "root", "name": "Root"},
        {
            "id": "child",
            "name": "Child",
            "attachment": {"parent_id": {}, "type": "fused"},
        },
    ]

    with pytest.raises(PlanSchemaError) as caught:
        validate_plan_document(plan)

    assert "attachment.parent_id" in str(caught.value)


def test_prompt_ready_schema_text_is_the_exact_authoritative_document() -> None:
    assert json.loads(plan_schema_text()) == PLAN_SCHEMA


def test_semantic_only_failure_is_not_mislabeled_as_json_schema() -> None:
    plan = _valid_plan()
    plan["material_plan"] = {
        "schema_version": 1,
        "materials": [
            {
                "id": "paint",
                "base_color_rgba": [0.2, 0.3, 0.4, 1.0],
                "metallic": 0.0,
                "roughness": 0.5,
            },
            {
                "id": "trim",
                "base_color_rgba": [0.05, 0.05, 0.05, 1.0],
                "metallic": 0.0,
                "roughness": 0.8,
            },
        ],
        "assignments": [
            {"part_id": "body", "material_id": "paint"},
            {"part_id": "body", "material_id": "trim"},
        ],
    }

    with pytest.raises(PlanSchemaError) as caught:
        validate_plan_document(plan)

    assert {violation.contract for violation in caught.value.violations} == {"semantic"}
    assert str(caught.value).startswith("plan violates the semantic contract (1 error):")
    assert "duplicate material assignment target 'body'" in str(caught.value)


def test_validation_collects_all_independent_schema_errors() -> None:
    plan = _valid_plan()
    plan.update(
        {
            "subject": "   ",
            "subject_kind": "hybrid",
            "coordinate_frame": [],
            "dimensions": [0.0, False, "wide", 2.0],
            "parts": [],
            "materials": {},
            "construction_strategy": "",
            "identity_features": "crest",
            "reconstruction_mode": "mesh-copy",
            "granularity": "microscopic",
            "character_analysis": {
                "pose": " ",
                "proportions": [],
                "held_props": {},
            },
        }
    )
    del plan["limitations"]

    with pytest.raises(PlanSchemaError) as caught:
        validate_plan_document(plan)

    rendered = "\n".join(str(item) for item in caught.value.violations)
    assert len(caught.value.violations) >= 15
    for expected in (
        "character_analysis.clothing_layers",
        "character_analysis.facial_landmarks",
        "character_analysis.held_props",
        "character_analysis.pose",
        "character_analysis.proportions",
        "construction_strategy",
        "coordinate_frame",
        "dimensions[0]",
        "dimensions[1]",
        "dimensions[2]",
        "identity_features",
        "limitations",
        "materials",
        "parts",
        "reconstruction_mode",
        "granularity",
        "subject",
    ):
        assert expected in rendered


def test_pipeline_reports_all_schema_errors_in_one_failure(tmp_path: Path) -> None:
    plan = _valid_plan()
    plan["dimensions"] = [0, "wide"]
    plan["construction_strategy"] = " "
    plan["materials"] = "steel"
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(pipeline.PipelineError) as caught:
        pipeline._validate_plan(path)

    message = str(caught.value)
    assert "violates the JSON Schema" in message
    assert "$.dimensions" in message
    assert "$.dimensions[0]" in message
    assert "$.dimensions[1]" in message
    assert "$.construction_strategy" in message
    assert "$.materials" in message


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_pipeline_rejects_non_standard_json_numbers(tmp_path: Path, constant: str) -> None:
    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps(_valid_plan()).replace("[1.0, 1.0, 1.0]", f"[1.0, {constant}, 1.0]"),
        encoding="utf-8",
    )

    with pytest.raises(pipeline.PipelineError, match="invalid JSON"):
        pipeline._validate_plan(path)
