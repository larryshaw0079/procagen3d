from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import procagen3d.pipeline as pipeline
from procagen3d.pipeline import PipelineConfig, PipelineError
from procagen3d.quality import resolve_quality_profile
from procagen3d.workspace import Workspace, sha256, write_json


def _frame() -> dict[str, list[float]]:
    return {
        "origin": [0.0, 0.0, 0.0],
        "x_axis": [1.0, 0.0, 0.0],
        "y_axis": [0.0, 1.0, 0.0],
        "z_axis": [0.0, 0.0, 1.0],
    }


def _plan() -> dict[str, Any]:
    return {
        "subject": "transaction fixture",
        "subject_kind": "object",
        "coordinate_frame": {"up": "+Z"},
        "dimensions": [2.0, 1.0, 1.0],
        "parts": [
            {
                "id": "base",
                "name": "Base",
                "object_names": ["Base"],
                "attachment": {"type": "root"},
            },
            {
                "id": "arm",
                "name": "Arm",
                "object_names": ["Arm"],
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
                    "frame": _frame(),
                    "nominal_dimensions": {"diameter": 0.1},
                },
                {
                    "id": "arm_hinge",
                    "part_id": "arm",
                    "interface": "cylindrical",
                    "role": "male",
                    "frame": _frame(),
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
        slug="geometry-transaction",
        image=image,
        glb=glb,
        prompt="fixture",
        backend="codex",
    )
    write_json(workspace.plan_path, _plan())
    workspace.program_path.write_text("def build():\n    pass\n", encoding="utf-8")
    return workspace


def _geometry_state(workspace: Workspace) -> dict[str, Any]:
    state = pipeline._new_structured_state(
        workspace, pipeline._validate_plan(workspace.plan_path)
    )
    order = list(state["part_order"])
    state.update(
        phase="geometry",
        completed_parts=order,
        geometry_signature={part_id: {} for part_id in order},
        checkpoints=[
            {
                "part_id": part_id,
                "path": f"checkpoints/{index:03d}-{part_id}",
                "iteration": index - 1,
            }
            for index, part_id in enumerate(order, start=1)
        ],
    )
    write_json(pipeline._structured_state_path(workspace), state)
    return state


def _failed_comparison() -> dict[str, Any]:
    return {
        "score": 0.4,
        "passed": False,
        "hard_gates": {
            "passed": False,
            "failures": [
                {"gate": "intersection_fraction", "message": "parts overlap"}
            ],
        },
    }


def _record_agent_trajectory(
    workspace: Workspace, *, iteration: int
) -> None:
    trajectory = workspace.trajectory_dir(iteration)
    (trajectory / "program.py").write_bytes(workspace.program_path.read_bytes())
    (trajectory / "plan.json").write_bytes(workspace.plan_path.read_bytes())


def _run_geometry(
    workspace: Workspace,
    state: dict[str, Any],
    *,
    stages: list[dict[str, Any]],
) -> tuple[int, int | None, dict[str, Any], dict[str, Any]]:
    return pipeline._run_geometry_acceptance(
        workspace,
        object(),  # type: ignore[arg-type]
        PipelineConfig(max_geometry_repairs=1, min_structured_parts=2),
        reconstruction_mode="procedural",
        granularity="medium",
        quality_profile=resolve_quality_profile("medium"),
        user_prompt="fixture",
        state=state,
        next_iteration=10,
        active_iteration=None,
        agent_runs=[],
        structured_stages=stages,
        progress=None,
    )


def test_failed_repair_preflight_rolls_back_source_and_keeps_resume_safe(
    tmp_path: Path, monkeypatch: Any
) -> None:
    workspace = _workspace(tmp_path)
    state = _geometry_state(workspace)
    accepted_program = workspace.program_path.read_bytes()
    accepted_plan = workspace.plan_path.read_bytes()
    accepted_bindings = pipeline._structured_plan_bindings(
        workspace, pipeline._validate_plan(workspace.plan_path)
    )
    accepted_builds = 0

    def build(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal accepted_builds
        if workspace.program_path.read_bytes() != accepted_program:
            raise PipelineError("repair candidate failed its clean preflight build")
        accepted_builds += 1
        return _failed_comparison()

    def break_candidate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        updated = pipeline._validate_plan(workspace.plan_path)
        updated["assembly"]["connectors"][0]["frame"]["origin"] = [
            0.05,
            0.0,
            0.0,
        ]
        write_json(workspace.plan_path, updated)
        workspace.program_path.write_text(
            "def build():\n    raise RuntimeError('broken repair')\n",
            encoding="utf-8",
        )
        _record_agent_trajectory(workspace, iteration=kwargs["iteration"])
        return {"provider_success": True}

    monkeypatch.setattr(pipeline, "build_workspace", build)
    monkeypatch.setattr(pipeline, "_invoke_agent", break_candidate)
    monkeypatch.setattr(
        pipeline,
        "_structured_geometry_signature",
        lambda *args, **kwargs: {"base": {}, "arm": {}},
    )
    # Resume-artifact validation is orthogonal to this transaction test.
    monkeypatch.setattr(
        pipeline,
        "_validate_structured_resume_artifacts",
        lambda *args, **kwargs: None,
        raising=False,
    )
    stages: list[dict[str, Any]] = []

    _, _, updated_state, comparison = _run_geometry(
        workspace, state, stages=stages
    )

    assert comparison["passed"] is False
    assert accepted_builds == 1
    assert updated_state["phase"] == "final"
    assert updated_state["geometry_passed"] is False
    assert workspace.program_path.read_bytes() == accepted_program
    assert workspace.plan_path.read_bytes() == accepted_plan
    assert all(updated_state[key] == value for key, value in accepted_bindings.items())
    assert any(
        stage.get("phase") == "geometry-repair"
        and stage.get("rolled_back") is True
        and "preflight build" in str(stage.get("error"))
        for stage in stages
    )
    assert pipeline._load_structured_state(workspace) == updated_state


def test_geometry_repair_cannot_change_pbr_declarations(
    tmp_path: Path, monkeypatch: Any
) -> None:
    workspace = _workspace(tmp_path)
    state = _geometry_state(workspace)
    accepted_program = workspace.program_path.read_bytes()
    accepted_plan = workspace.plan_path.read_bytes()
    accepted_bindings = pipeline._structured_plan_bindings(
        workspace, pipeline._validate_plan(workspace.plan_path)
    )
    build_count = 0

    def build(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal build_count
        build_count += 1
        return _failed_comparison()

    def rewrite_pbr(*args: Any, **kwargs: Any) -> dict[str, Any]:
        updated = pipeline._validate_plan(workspace.plan_path)
        updated["materials"] = [{"id": "unauthorized-paint"}]
        updated["material_plan"] = {
            "schema_version": 1,
            "materials": [
                {
                    "id": "unauthorized-paint",
                    "base_color_rgba": [0.8, 0.1, 0.1, 1.0],
                    "metallic": 0.0,
                    "roughness": 0.5,
                }
            ],
            "assignments": [
                {"part_id": part_id, "material_id": "unauthorized-paint"}
                for part_id in ("base", "arm")
            ],
        }
        write_json(workspace.plan_path, updated)
        workspace.program_path.write_text(
            "def build():\n    pass\n# unauthorized PBR rewrite\n",
            encoding="utf-8",
        )
        _record_agent_trajectory(workspace, iteration=kwargs["iteration"])
        return {"provider_success": True}

    monkeypatch.setattr(pipeline, "build_workspace", build)
    monkeypatch.setattr(pipeline, "_invoke_agent", rewrite_pbr)
    monkeypatch.setattr(
        pipeline,
        "_structured_geometry_signature",
        lambda *args, **kwargs: {"base": {}, "arm": {}},
    )
    monkeypatch.setattr(
        pipeline,
        "_validate_structured_resume_artifacts",
        lambda *args, **kwargs: None,
        raising=False,
    )
    stages: list[dict[str, Any]] = []

    _, _, updated_state, comparison = _run_geometry(
        workspace, state, stages=stages
    )

    assert comparison["passed"] is False
    assert build_count == 1
    assert updated_state["phase"] == "final"
    assert updated_state["geometry_passed"] is False
    assert workspace.program_path.read_bytes() == accepted_program
    assert workspace.plan_path.read_bytes() == accepted_plan
    assert all(updated_state[key] == value for key, value in accepted_bindings.items())
    assert any(
        stage.get("phase") == "geometry-repair"
        and stage.get("rolled_back") is True
        for stage in stages
    )
    assert pipeline._load_structured_state(workspace) == updated_state
