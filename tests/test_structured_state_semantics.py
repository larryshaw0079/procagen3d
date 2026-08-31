from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import procagen3d.pipeline as pipeline
from procagen3d.pipeline import PipelineError
from procagen3d.workspace import Workspace, write_json


_KEYS_BY_PART = {
    "base": ("base_housing",),
    "arm": ("primary_arm", "pivot_pin"),
}


def _structured_plan() -> dict[str, Any]:
    identity_frame = {
        "origin": [0.0, 0.0, 0.0],
        "x_axis": [1.0, 0.0, 0.0],
        "y_axis": [0.0, 1.0, 0.0],
        "z_axis": [0.0, 0.0, 1.0],
    }
    return {
        "subject": "state semantics fixture",
        "subject_kind": "object",
        "coordinate_frame": {"up": "+Z"},
        "dimensions": [2.0, 1.0, 1.0],
        "parts": [
            {
                "id": "base",
                "name": "Base",
                "object_names": ["Base Housing"],
                "attachment": {"type": "root"},
            },
            {
                "id": "arm",
                "name": "Arm",
                "object_names": ["Primary Arm", "Pivot.Pin.001"],
                "attachment": {"type": "articulated", "parent_id": "base"},
            },
        ],
        "assembly": {
            "version": 1,
            "placement": "host-solved",
            "part_order": ["base", "arm"],
            "connectors": [
                {
                    "id": "base_hinge",
                    "part_id": "base",
                    "interface": "cylindrical",
                    "role": "female",
                    "frame": identity_frame,
                    "nominal_dimensions": {"diameter": 0.1},
                },
                {
                    "id": "arm_hinge",
                    "part_id": "arm",
                    "interface": "cylindrical",
                    "role": "male",
                    "frame": identity_frame,
                    "nominal_dimensions": {"diameter": 0.1},
                },
            ],
            "mates": [
                {
                    "id": "arm_joint",
                    "type": "revolute",
                    "parent_connector_id": "base_hinge",
                    "child_connector_id": "arm_hinge",
                    "fit": "clearance",
                    "clearance": 0.002,
                    "fit_offset": [0.0, 0.0, 0.0],
                    "nominal_dimensions": {"diameter": 0.1},
                    "rest": 0.0,
                    "limits": {"lower": -1.0, "upper": 1.0},
                }
            ],
        },
        "materials": [],
        "construction_strategy": "incremental parts",
        "identity_features": [],
        "limitations": [],
    }


def _workspace(tmp_path: Path) -> Workspace:
    source = tmp_path / "source"
    source.mkdir()
    image = source / "reference.png"
    glb = source / "reference.glb"
    image.write_bytes(b"image")
    glb.write_bytes(b"glb")
    workspace = Workspace.create(
        base=tmp_path / "outputs",
        slug="structured-state-semantics",
        image=image,
        glb=glb,
        prompt="fixture",
        backend="codex",
    )
    write_json(workspace.plan_path, _structured_plan())
    workspace.program_path.write_text("def build():\n    pass\n", encoding="utf-8")
    return workspace


@pytest.fixture(autouse=True)
def _isolate_state_semantics(monkeypatch: pytest.MonkeyPatch) -> None:
    # Artifact provenance has its own tests. These cases intentionally exercise
    # the cheaper state checks before Blender/GLB checkpoint inspection.
    monkeypatch.setattr(
        pipeline,
        "_validate_structured_resume_artifacts",
        lambda *args, **kwargs: None,
    )


def _signature_for(completed: list[str]) -> dict[str, dict[str, str]]:
    return {
        key: {"fixture": key}
        for part_id in completed
        for key in _KEYS_BY_PART[part_id]
    }


def _set_progress(state: dict[str, Any], completed: list[str]) -> None:
    state["completed_parts"] = completed
    state["geometry_signature"] = _signature_for(completed)
    state["checkpoints"] = [
        {
            "part_id": part_id,
            "path": f"checkpoints/{index + 1:03d}-{part_id}",
            "iteration": index,
        }
        for index, part_id in enumerate(completed)
    ]


def _state_for_phase(
    workspace: Workspace,
    *,
    phase: str,
    geometry_passed: Any,
    materials_status: Any,
) -> dict[str, Any]:
    state = pipeline._new_structured_state(workspace, _structured_plan())
    completed = ["base"] if phase == "parts" else ["base", "arm"]
    _set_progress(state, completed)
    state.update(
        phase=phase,
        geometry_passed=geometry_passed,
        materials_status=materials_status,
    )
    if materials_status == "failed":
        state["material_error"] = "fixture material failure"
    return state


def _load(workspace: Workspace, state: dict[str, Any]) -> dict[str, Any] | None:
    write_json(pipeline._structured_state_path(workspace), state)
    return pipeline._load_structured_state(workspace)


@pytest.mark.parametrize(
    ("phase", "geometry_passed", "materials_status"),
    [
        pytest.param("parts", False, "pending", id="parts"),
        pytest.param("geometry", False, "pending", id="geometry"),
        pytest.param("materials", True, "pending", id="materials"),
        pytest.param("final", False, "blocked-geometry", id="geometry-blocked"),
        pytest.param("final", True, "applied", id="materials-applied"),
        pytest.param("final", True, "skipped", id="materials-skipped"),
        pytest.param("final", True, "failed", id="materials-failed"),
    ],
)
def test_resume_accepts_reachable_phase_status_combinations(
    tmp_path: Path,
    phase: str,
    geometry_passed: bool,
    materials_status: str,
) -> None:
    workspace = _workspace(tmp_path)
    state = _state_for_phase(
        workspace,
        phase=phase,
        geometry_passed=geometry_passed,
        materials_status=materials_status,
    )

    assert _load(workspace, state) == state


@pytest.mark.parametrize(
    ("phase", "geometry_passed", "materials_status"),
    [
        pytest.param("parts", True, "pending", id="parts-after-geometry"),
        pytest.param("parts", False, "applied", id="parts-after-materials"),
        pytest.param("geometry", True, "pending", id="geometry-already-passed"),
        pytest.param("geometry", False, "skipped", id="geometry-materials-skipped"),
        pytest.param("materials", False, "pending", id="materials-before-geometry"),
        pytest.param("materials", True, "applied", id="materials-already-applied"),
        pytest.param("final", False, "pending", id="final-geometry-unresolved"),
        pytest.param("final", False, "failed", id="final-wrong-geometry-failure"),
        pytest.param("final", True, "pending", id="final-materials-unresolved"),
        pytest.param("final", True, "blocked-geometry", id="final-wrong-blocker"),
    ],
)
def test_resume_rejects_unreachable_phase_status_combinations(
    tmp_path: Path,
    phase: str,
    geometry_passed: bool,
    materials_status: str,
) -> None:
    workspace = _workspace(tmp_path)
    state = _state_for_phase(
        workspace,
        phase=phase,
        geometry_passed=geometry_passed,
        materials_status=materials_status,
    )

    with pytest.raises(PipelineError) as error:
        _load(workspace, state)

    message = str(error.value).lower()
    assert any(word in message for word in ("phase", "geometry", "material", "status"))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("geometry_passed", 0, id="non-boolean-geometry-result"),
        pytest.param("materials_status", "unknown", id="unknown-material-status"),
    ],
)
def test_resume_rejects_invalid_phase_status_field_values(
    tmp_path: Path, field: str, value: Any
) -> None:
    workspace = _workspace(tmp_path)
    state = _state_for_phase(
        workspace,
        phase="geometry",
        geometry_passed=False,
        materials_status="pending",
    )
    state[field] = value

    with pytest.raises(PipelineError):
        _load(workspace, state)


def test_parts_phase_rejects_an_already_complete_part_order(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    state = _state_for_phase(
        workspace,
        phase="parts",
        geometry_passed=False,
        materials_status="pending",
    )
    _set_progress(state, ["base", "arm"])

    with pytest.raises(PipelineError, match="phase|complete"):
        _load(workspace, state)


@pytest.mark.parametrize(
    ("phase", "completed", "expected_keys"),
    [
        pytest.param("parts", [], set(), id="no-completed-parts"),
        pytest.param("parts", ["base"], {"base_housing"}, id="completed-prefix"),
        pytest.param(
            "geometry",
            ["base", "arm"],
            {"base_housing", "primary_arm", "pivot_pin"},
            id="all-completed-parts",
        ),
    ],
)
def test_resume_accepts_exact_normalized_object_signature_keys(
    tmp_path: Path,
    phase: str,
    completed: list[str],
    expected_keys: set[str],
) -> None:
    workspace = _workspace(tmp_path)
    state = _state_for_phase(
        workspace,
        phase=phase,
        geometry_passed=False,
        materials_status="pending",
    )
    _set_progress(state, completed)

    loaded = _load(workspace, state)

    assert loaded is not None
    assert set(loaded["geometry_signature"]) == expected_keys


@pytest.mark.parametrize(
    ("phase", "completed", "signature_keys"),
    [
        pytest.param("parts", ["base"], set(), id="missing-completed-object"),
        pytest.param(
            "parts",
            ["base"],
            {"base_housing", "primary_arm"},
            id="future-object-leak",
        ),
        pytest.param(
            "geometry",
            ["base", "arm"],
            {"base_housing", "primary_arm"},
            id="missing-secondary-object",
        ),
        pytest.param(
            "geometry",
            ["base", "arm"],
            {"base_housing", "primary_arm", "pivot_pin", "helper"},
            id="unexpected-helper-object",
        ),
        pytest.param(
            "geometry",
            ["base", "arm"],
            {"base", "arm"},
            id="part-ids-are-not-object-keys",
        ),
    ],
)
def test_resume_rejects_non_exact_geometry_signature_key_coverage(
    tmp_path: Path,
    phase: str,
    completed: list[str],
    signature_keys: set[str],
) -> None:
    workspace = _workspace(tmp_path)
    state = _state_for_phase(
        workspace,
        phase=phase,
        geometry_passed=False,
        materials_status="pending",
    )
    _set_progress(state, completed)
    state["geometry_signature"] = {
        key: {"fixture": key} for key in signature_keys
    }

    with pytest.raises(PipelineError, match="signature|geometry"):
        _load(workspace, state)
