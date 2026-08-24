from __future__ import annotations

import json
from pathlib import Path

import pytest

import procagen3d.pipeline as pipeline
from procagen3d.plan_schema import (
    PLAN_SCHEMA,
    PlanSchemaError,
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


def test_schema_declares_modes_and_old_plans_default_to_procedural() -> None:
    mode = PLAN_SCHEMA["properties"]["reconstruction_mode"]
    assert mode["enum"] == ["procedural", "glb-ref"]
    assert mode["default"] == "procedural"

    original = _valid_plan()
    normalized = validate_plan_document(original)
    assert "reconstruction_mode" not in original
    assert normalized["reconstruction_mode"] == "procedural"
    assert normalized["granularity"] == "medium"

    granularity = PLAN_SCHEMA["properties"]["granularity"]
    assert granularity["enum"] == ["coarse", "medium", "fine", "surface"]
    assert granularity["default"] == "medium"

    original["reconstruction_mode"] = "glb-ref"
    original["granularity"] = "fine"
    assert validate_plan_document(original)["reconstruction_mode"] == "glb-ref"
    assert validate_plan_document(original)["granularity"] == "fine"


def test_prompt_ready_schema_text_is_the_exact_authoritative_document() -> None:
    assert json.loads(plan_schema_text()) == PLAN_SCHEMA


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
