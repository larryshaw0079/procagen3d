from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

import procagen3d.pipeline as pipeline
from procagen3d.pipeline import PipelineConfig, PipelineError
from procagen3d.quality import resolve_quality_profile
from procagen3d.workspace import Workspace, sha256, write_json


def _legacy_plan() -> dict[str, Any]:
    return {
        "subject": "fixture",
        "subject_kind": "object",
        "coordinate_frame": {"up": "+Z"},
        "dimensions": [1.0, 1.0, 1.0],
        "parts": [{"name": "Body"}],
        "materials": [],
        "construction_strategy": "primitive",
        "identity_features": [],
        "limitations": [],
    }


def _structured_plan() -> dict[str, Any]:
    identity_frame = {
        "origin": [0.0, 0.0, 0.0],
        "x_axis": [1.0, 0.0, 0.0],
        "y_axis": [0.0, 1.0, 0.0],
        "z_axis": [0.0, 0.0, 1.0],
    }
    return {
        "subject": "fixture mechanism",
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
            {
                "id": "badge",
                "name": "Badge",
                "object_names": ["Badge"],
                "attachment": {"type": "intentional-gap", "parent_id": "base"},
            },
        ],
        "assembly": {
            "version": 1,
            "placement": "host-solved",
            "part_order": ["base", "arm", "badge"],
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
                {
                    "id": "base_badge_mount",
                    "part_id": "base",
                    "interface": "planar",
                    "role": "neutral",
                    "frame": identity_frame,
                    "nominal_dimensions": {"width": 0.2},
                },
                {
                    "id": "badge_mount",
                    "part_id": "badge",
                    "interface": "planar",
                    "role": "neutral",
                    "frame": identity_frame,
                    "nominal_dimensions": {"width": 0.2},
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
                },
                {
                    "id": "badge_offset",
                    "type": "rigid",
                    "parent_connector_id": "base_badge_mount",
                    "child_connector_id": "badge_mount",
                    "fit": "none",
                    "clearance": 0.0,
                    "fit_offset": [0.0, 0.0, 0.0],
                    "nominal_dimensions": {"width": 0.2},
                },
            ],
        },
        "materials": [],
        "construction_strategy": "incremental parts",
        "identity_features": [],
        "limitations": [],
    }


def _workspace(tmp_path: Path, plan: dict[str, Any]) -> Workspace:
    source = tmp_path / "source"
    source.mkdir()
    image = source / "reference.png"
    glb = source / "reference.glb"
    image.write_bytes(b"image")
    glb.write_bytes(b"glb")
    workspace = Workspace.create(
        base=tmp_path / "outputs",
        slug="structured-fixture",
        image=image,
        glb=glb,
        prompt="fixture",
        backend="codex",
    )
    write_json(workspace.plan_path, plan)
    workspace.program_path.write_text("def build():\n    pass\n", encoding="utf-8")
    return workspace


def _prepared_probe() -> dict[str, Any]:
    return {
        "scene": {"vertex_count": 8, "triangle_count": 12},
        "semantic_decomposition": {"status": "sufficient"},
    }


def _geometry_report() -> dict[str, Any]:
    objects = []
    for index, name in enumerate(("Base", "Arm", "Badge")):
        objects.append(
            {
                "name": name,
                "type": "MESH",
                "vertices": 8,
                "triangles": 12,
                "bounds": {
                    "min": [float(index), 0.0, 0.0],
                    "max": [float(index + 1), 1.0, 1.0],
                },
            }
        )
    return {
        "geometry_object_count": 3,
        "mesh_count": 3,
        "bounds": {"min": [0.0, 0.0, 0.0], "max": [3.0, 1.0, 1.0]},
        "objects": objects,
    }


def _successful_build(workspace: Workspace):
    def fake_build(*args: Any, **kwargs: Any) -> dict[str, Any]:
        workspace.artifacts_dir.mkdir(parents=True, exist_ok=True)
        (workspace.artifacts_dir / "model.glb").write_bytes(b"compiled")
        (workspace.artifacts_dir / "scene.blend").write_bytes(b"blend")
        write_json(
            workspace.artifacts_dir / "build_manifest.json",
            {
                "program_sha256": sha256(workspace.program_path),
                "plan_sha256": sha256(workspace.plan_path),
            },
        )
        return {
            "score": 1.0,
            "score_passed": True,
            "passed": True,
            "hard_gates": {"passed": True, "failures": []},
        }

    return fake_build


def _unexpected_stage(name: str):
    def fail(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"{name} should not run")

    return fail


def test_explicit_assembly_contract_requires_host_placement_for_every_part() -> None:
    plan = _structured_plan()

    pipeline._validate_explicit_assembly_contract(plan, min_parts=2)

    missing_mate = deepcopy(plan)
    missing_mate["assembly"]["mates"] = []
    with pytest.raises(PipelineError, match="host-solved mates: arm") as error:
        pipeline._validate_explicit_assembly_contract(missing_mate, min_parts=2)

    # Intentional visual separation still needs a rigid spatial mate so a
    # locally authored part receives a deterministic world transform.
    assert "badge" in str(error.value)


def test_explicit_assembly_contract_rejects_missing_structure_and_bad_minimum() -> None:
    plan = _structured_plan()

    with pytest.raises(ValueError, match="positive integer"):
        pipeline._validate_explicit_assembly_contract(plan, min_parts=True)
    with pytest.raises(PipelineError, match="at least 4 semantic parts"):
        pipeline._validate_explicit_assembly_contract(plan, min_parts=4)

    no_assembly = deepcopy(plan)
    no_assembly.pop("assembly")
    with pytest.raises(PipelineError, match="explicit assembly object"):
        pipeline._validate_explicit_assembly_contract(no_assembly, min_parts=2)

    bad_arrays = deepcopy(plan)
    bad_arrays["assembly"]["connectors"] = {}
    with pytest.raises(PipelineError, match="connector and mate arrays"):
        pipeline._validate_explicit_assembly_contract(bad_arrays, min_parts=2)

    unknown_child_connector = deepcopy(plan)
    unknown_child_connector["assembly"]["mates"][0]["child_connector_id"] = "ghost"
    with pytest.raises(PipelineError, match="host-solved mates: arm"):
        pipeline._validate_explicit_assembly_contract(
            unknown_child_connector, min_parts=2
        )


def test_structured_state_is_created_and_loaded_for_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _structured_plan()
    workspace = _workspace(tmp_path, plan)
    normalized_plan = pipeline._validate_plan(workspace.plan_path)

    state = pipeline._new_structured_state(workspace, plan)
    assert state == {
        "schema_version": 2,
        "phase": "parts",
        "program_sha256": sha256(workspace.program_path),
        "plan_sha256": sha256(workspace.plan_path),
        "contract_sha256": pipeline._structured_contract_sha256(normalized_plan),
        "pbr_sha256": pipeline._structured_pbr_sha256(normalized_plan),
        "acceptance_sha256": pipeline._structured_acceptance_sha256(normalized_plan),
        "topology_sha256": pipeline._structured_topology_sha256(normalized_plan),
        "part_order": ["base", "arm", "badge"],
        "completed_parts": [],
        "geometry_signature": {},
        "checkpoints": [],
        "geometry_passed": False,
        "materials_status": "pending",
    }

    state.update(
        completed_parts=["base"],
        geometry_signature={"base": {"triangle_count": 12}},
        checkpoints=[{"part_id": "base", "path": "checkpoints/001-base"}],
    )
    write_json(pipeline._structured_state_path(workspace), state)
    monkeypatch.setattr(
        pipeline, "_validate_structured_resume_artifacts", lambda *args, **kwargs: None
    )

    assert pipeline._load_structured_state(workspace) == state


def test_incremental_failure_is_persisted_and_replayed_exactly_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _structured_plan()
    workspace = _workspace(tmp_path, plan)
    state = pipeline._new_structured_state(workspace, plan)
    traceback = (
        "Blender incremental part 001-base exited with 1.\n"
        "stderr:\nTraceback (most recent call last):\n"
        "  File \"program.py\", line 63, in _mesh_object\n"
        "    bpy.context.collection.objects.link(obj)\n"
        "AttributeError: 'NoneType' object has no attribute 'objects'\n"
    )

    monkeypatch.setattr(
        pipeline,
        "_invoke_agent",
        lambda *args, **kwargs: {"provider_success": True},
    )

    def fail_checkpoint(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise PipelineError(traceback)

    monkeypatch.setattr(pipeline, "_build_incremental_checkpoint", fail_checkpoint)

    with pytest.raises(PipelineError, match="failed after 1 attempt"):
        pipeline._run_incremental_authoring(
            workspace,
            object(),  # type: ignore[arg-type]
            PipelineConfig(max_part_repairs=0),
            reconstruction_mode="procedural",
            granularity="medium",
            quality_profile=resolve_quality_profile("medium"),
            user_prompt="fixture",
            state=state,
            next_iteration=0,
            agent_runs=[],
            structured_stages=[],
            progress=None,
        )

    persisted = pipeline._load_structured_state(workspace)
    assert persisted is not None
    assert persisted["pending_part_failure"] == {
        "part_id": "base",
        "error": traceback,
    }
    assert persisted["program_sha256"] == sha256(workspace.program_path)
    assert persisted["plan_sha256"] == sha256(workspace.plan_path)

    invocation: dict[str, Any] = {}
    resumed_stages: list[dict[str, Any]] = []

    def capture_resume(*args: Any, **kwargs: Any) -> dict[str, Any]:
        invocation.update(kwargs)
        raise PipelineError("resume agent stopped")

    monkeypatch.setattr(pipeline, "_invoke_agent", capture_resume)
    with pytest.raises(PipelineError, match="resume agent stopped"):
        pipeline._run_incremental_authoring(
            workspace,
            object(),  # type: ignore[arg-type]
            PipelineConfig(max_part_repairs=0),
            reconstruction_mode="procedural",
            granularity="medium",
            quality_profile=resolve_quality_profile("medium"),
            user_prompt="fixture",
            state=persisted,
            next_iteration=1,
            agent_runs=[],
            structured_stages=resumed_stages,
            progress=None,
        )

    assert traceback in invocation["prompt"]
    assert invocation["is_repair"] is True
    assert resumed_stages[0]["phase"] == "part-repair"
    transitioned = pipeline._load_structured_state(workspace)
    assert transitioned is not None
    assert transitioned["pending_part_failure"] == {
        "part_id": "base",
        "error": "resume agent stopped",
    }


def test_incremental_resume_recovers_matching_legacy_run_report_failure_prompt_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _structured_plan()
    workspace = _workspace(tmp_path, plan)
    state = pipeline._new_structured_state(workspace, plan)
    traceback = (
        "Blender incremental part 001-base exited with 1.\n"
        "AttributeError: 'NoneType' object has no attribute 'objects'"
    )
    write_json(
        workspace.root / "run_report.json",
        {
            "status": "failed",
            "stage": "structured-authoring",
            "pipeline_mode": "structured",
            "workspace": str(workspace.root),
            "structured_stages": [
                {
                    "phase": "part-repair",
                    "part_id": "base",
                    "passed": False,
                    "error": traceback,
                }
            ],
        },
    )
    invocation: dict[str, Any] = {}
    stages: list[dict[str, Any]] = []

    def capture_resume(*args: Any, **kwargs: Any) -> dict[str, Any]:
        invocation.update(kwargs)
        on_disk = json.loads(
            pipeline._structured_state_path(workspace).read_text(encoding="utf-8")
        )
        assert "pending_part_failure" not in on_disk
        raise PipelineError("new retry failure")

    monkeypatch.setattr(pipeline, "_invoke_agent", capture_resume)
    with pytest.raises(PipelineError, match="new retry failure"):
        pipeline._run_incremental_authoring(
            workspace,
            object(),  # type: ignore[arg-type]
            PipelineConfig(max_part_repairs=0),
            reconstruction_mode="procedural",
            granularity="medium",
            quality_profile=resolve_quality_profile("medium"),
            user_prompt="fixture",
            state=state,
            next_iteration=3,
            agent_runs=[],
            structured_stages=stages,
            progress=None,
        )

    assert traceback in invocation["prompt"]
    assert invocation["is_repair"] is True
    assert stages[0]["phase"] == "part-repair"
    persisted = pipeline._load_structured_state(workspace)
    assert persisted is not None
    assert persisted["pending_part_failure"] == {
        "part_id": "base",
        "error": "new retry failure",
    }


def test_successful_incremental_checkpoint_clears_pending_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _structured_plan()
    workspace = _workspace(tmp_path, plan)
    state = pipeline._new_structured_state(workspace, plan)
    prior_failure = "exact prior Blender traceback"
    state["pending_part_failure"] = {"part_id": "base", "error": prior_failure}
    write_json(pipeline._structured_state_path(workspace), state)
    invocations: list[dict[str, Any]] = []

    def fake_agent(*args: Any, **kwargs: Any) -> dict[str, Any]:
        invocations.append(kwargs)
        if len(invocations) == 1:
            trajectory = workspace.trajectory_dir(kwargs["iteration"])
            (trajectory / "program.py").write_bytes(workspace.program_path.read_bytes())
            (trajectory / "plan.json").write_bytes(workspace.plan_path.read_bytes())
            return {"provider_success": True}
        on_disk = json.loads(
            pipeline._structured_state_path(workspace).read_text(encoding="utf-8")
        )
        assert "pending_part_failure" not in on_disk
        raise PipelineError("stop after observing cleared state")

    monkeypatch.setattr(pipeline, "_invoke_agent", fake_agent)
    monkeypatch.setattr(
        pipeline,
        "_build_incremental_checkpoint",
        lambda *args, **kwargs: {
            "validation": {"geometry_signature": {"base": {}}}
        },
    )

    with pytest.raises(PipelineError, match="stop after observing cleared state"):
        pipeline._run_incremental_authoring(
            workspace,
            object(),  # type: ignore[arg-type]
            PipelineConfig(max_part_repairs=0),
            reconstruction_mode="procedural",
            granularity="medium",
            quality_profile=resolve_quality_profile("medium"),
            user_prompt="fixture",
            state=state,
            next_iteration=0,
            agent_runs=[],
            structured_stages=[],
            progress=None,
        )

    assert prior_failure in invocations[0]["prompt"]
    assert invocations[0]["is_repair"] is True


def test_structured_state_rejects_pending_failure_for_a_non_next_part(
    tmp_path: Path,
) -> None:
    plan = _structured_plan()
    workspace = _workspace(tmp_path, plan)
    state = pipeline._new_structured_state(workspace, plan)
    state["pending_part_failure"] = {"part_id": "arm", "error": "wrong part"}
    write_json(pipeline._structured_state_path(workspace), state)

    with pytest.raises(PipelineError, match="does not match the next part"):
        pipeline._load_structured_state(workspace)


def test_incremental_runtime_document_contains_only_the_completed_prefix() -> None:
    plan = pipeline.validate_plan_document(_structured_plan())

    document = pipeline._assembly_runtime_document(plan, part_ids=["base"])

    assert document is not None
    assert [part["id"] for part in document["parts"]] == ["base"]
    with pytest.raises(PipelineError, match="assembly-order prefix"):
        pipeline._assembly_runtime_document(plan, part_ids=["base", "badge"])


def test_structured_topology_hash_allows_frame_tuning_but_freezes_graph() -> None:
    plan = pipeline.validate_plan_document(_structured_plan())
    baseline = pipeline._structured_topology_sha256(plan)

    tuned = deepcopy(plan)
    tuned["assembly"]["connectors"][0]["frame"]["origin"] = [0.2, 0.0, 0.0]
    assert pipeline._structured_topology_sha256(tuned) == baseline

    rewired = deepcopy(plan)
    rewired["assembly"]["mates"][0]["child_connector_id"] = "badge_mount"
    assert pipeline._structured_topology_sha256(rewired) != baseline


def test_material_scene_identity_freezes_hierarchy_but_ignores_primitives() -> None:
    identity = [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    before = {
        "base": {
            "name": "Base",
            "local_matrix": identity,
            "world_matrix": identity,
            "parent_identity": {"has_parent": False, "name": None},
            "bounds": {"min": [0, 0, 0], "max": [1, 1, 1]},
            "primitives": [{"geometry_sha256": "a" * 64}],
        }
    }
    repartitioned = deepcopy(before)
    repartitioned["base"]["primitives"] = [
        {"geometry_sha256": "b" * 64},
        {"geometry_sha256": "c" * 64},
    ]
    pipeline._assert_material_scene_identity(
        before, repartitioned, label="fixture"
    )

    reparented = deepcopy(repartitioned)
    reparented["base"]["parent_identity"] = {
        "has_parent": True,
        "name": "HiddenParent",
    }
    with pytest.raises(PipelineError, match="parent_identity"):
        pipeline._assert_material_scene_identity(
            before, reparented, label="fixture"
        )


def _geometry_phase_state(
    workspace: Workspace, plan: dict[str, Any]
) -> dict[str, Any]:
    state = pipeline._new_structured_state(workspace, plan)
    order = list(state["part_order"])
    state.update(
        phase="geometry",
        completed_parts=order,
        checkpoints=[
            {"part_id": part_id, "path": f"checkpoints/{index:03d}-{part_id}"}
            for index, part_id in enumerate(order, start=1)
        ],
        geometry_signature={part_id: {} for part_id in order},
    )
    write_json(pipeline._structured_state_path(workspace), state)
    return state


def test_rejected_incremental_plan_edit_restores_resumable_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _structured_plan()
    workspace = _workspace(tmp_path, plan)
    original_program = workspace.program_path.read_bytes()
    original_plan = workspace.plan_path.read_bytes()
    state = pipeline._new_structured_state(workspace, plan)

    def edit_frozen_plan(*args: Any, **kwargs: Any) -> dict[str, Any]:
        changed = deepcopy(plan)
        changed["subject"] = "unauthorized rewrite"
        write_json(workspace.plan_path, changed)
        workspace.program_path.write_text("def build():\n    raise RuntimeError()\n")
        return {"provider_success": True}

    monkeypatch.setattr(pipeline, "_invoke_agent", edit_frozen_plan)

    with pytest.raises(PipelineError, match="edited the frozen"):
        pipeline._run_incremental_authoring(
            workspace,
            object(),  # type: ignore[arg-type]
            PipelineConfig(max_part_repairs=0),
            reconstruction_mode="procedural",
            granularity="medium",
            quality_profile=resolve_quality_profile("medium"),
            user_prompt="fixture",
            state=state,
            next_iteration=0,
            agent_runs=[],
            structured_stages=[],
            progress=None,
        )

    assert workspace.program_path.read_bytes() == original_program
    assert workspace.plan_path.read_bytes() == original_plan
    assert pipeline._load_structured_state(workspace) == state


def test_rejected_geometry_topology_edit_rolls_back_and_remains_resumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _structured_plan()
    workspace = _workspace(tmp_path, plan)
    state = _geometry_phase_state(workspace, plan)
    original_program = workspace.program_path.read_bytes()
    original_plan = workspace.plan_path.read_bytes()
    comparison = {
        "score": 0.4,
        "passed": False,
        "hard_gates": {
            "passed": False,
            "failures": [{"gate": "intersection_fraction", "message": "overlap"}],
        },
    }

    def rewire(*args: Any, **kwargs: Any) -> dict[str, Any]:
        changed = deepcopy(plan)
        changed["assembly"]["mates"][0]["id"] = "rewired_joint"
        write_json(workspace.plan_path, changed)
        workspace.program_path.write_text("def build():\n    pass\n# changed\n")
        return {"provider_success": True}

    monkeypatch.setattr(pipeline, "_invoke_agent", rewire)
    monkeypatch.setattr(pipeline, "build_workspace", lambda *args, **kwargs: comparison)
    monkeypatch.setattr(
        pipeline,
        "_structured_geometry_signature",
        lambda *args, **kwargs: {"base": {}, "arm": {}, "badge": {}},
    )
    monkeypatch.setattr(
        pipeline, "_validate_structured_resume_artifacts", lambda *args, **kwargs: None
    )
    stages: list[dict[str, Any]] = []

    _, _, updated, _ = pipeline._run_geometry_acceptance(
        workspace,
        object(),  # type: ignore[arg-type]
        PipelineConfig(max_geometry_repairs=1),
        reconstruction_mode="procedural",
        granularity="medium",
        quality_profile=resolve_quality_profile("medium"),
        user_prompt="fixture",
        state=state,
        next_iteration=0,
        active_iteration=None,
        agent_runs=[],
        structured_stages=stages,
        progress=None,
    )

    assert workspace.program_path.read_bytes() == original_program
    assert workspace.plan_path.read_bytes() == original_plan
    assert any(stage.get("rolled_back") for stage in stages)
    assert updated["phase"] == "final"
    assert pipeline._load_structured_state(workspace) == updated


def test_geometry_repair_may_tune_connector_placement_without_changing_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _structured_plan()
    workspace = _workspace(tmp_path, plan)
    state = _geometry_phase_state(workspace, plan)
    original_contract = state["contract_sha256"]
    original_acceptance = state["acceptance_sha256"]
    comparisons = iter(
        [
            {
                "score": 0.5,
                "passed": False,
                "hard_gates": {
                    "passed": False,
                    "failures": [
                        {"gate": "intersection_fraction", "message": "overlap"}
                    ],
                },
            },
            {
                "score": 0.9,
                "passed": True,
                "hard_gates": {"passed": True, "failures": []},
            },
        ]
    )

    def tune_frame(*args: Any, **kwargs: Any) -> dict[str, Any]:
        updated = pipeline._validate_plan(workspace.plan_path)
        updated["assembly"]["connectors"][0]["frame"]["origin"] = [0.1, 0.0, 0.0]
        write_json(workspace.plan_path, updated)
        workspace.program_path.write_text("def build():\n    pass\n# placement repair\n")
        trajectory = workspace.trajectory_dir(kwargs["iteration"])
        (trajectory / "program.py").write_bytes(workspace.program_path.read_bytes())
        (trajectory / "plan.json").write_bytes(workspace.plan_path.read_bytes())
        return {"provider_success": True}

    monkeypatch.setattr(pipeline, "_invoke_agent", tune_frame)
    monkeypatch.setattr(pipeline, "build_workspace", lambda *args, **kwargs: next(comparisons))
    monkeypatch.setattr(
        pipeline,
        "_structured_geometry_signature",
        lambda *args, **kwargs: {
            "base": {"geometry_sha256": "current"},
            "arm": {"geometry_sha256": "current"},
            "badge": {"geometry_sha256": "current"},
        },
    )
    monkeypatch.setattr(
        pipeline, "_validate_structured_resume_artifacts", lambda *args, **kwargs: None
    )

    _, _, updated, comparison = pipeline._run_geometry_acceptance(
        workspace,
        object(),  # type: ignore[arg-type]
        PipelineConfig(max_geometry_repairs=1),
        reconstruction_mode="procedural",
        granularity="medium",
        quality_profile=resolve_quality_profile("medium"),
        user_prompt="fixture",
        state=state,
        next_iteration=0,
        active_iteration=None,
        agent_runs=[],
        structured_stages=[],
        progress=None,
    )

    assert comparison["passed"] is True
    assert updated["phase"] == "materials"
    assert updated["contract_sha256"] != original_contract
    assert updated["acceptance_sha256"] == original_acceptance
    assert set(updated["geometry_signature"]) == {"base", "arm", "badge"}
    assert pipeline._load_structured_state(workspace) == updated


def test_structured_state_rejects_same_order_plan_changes(tmp_path: Path) -> None:
    plan = _structured_plan()
    workspace = _workspace(tmp_path, plan)
    pipeline._new_structured_state(workspace, plan)

    changed = deepcopy(plan)
    changed["subject"] = "a different mechanism"
    write_json(workspace.plan_path, changed)

    with pytest.raises(PipelineError, match="does not match the current"):
        pipeline._load_structured_state(workspace)


def test_structured_state_rejects_program_changes(tmp_path: Path) -> None:
    plan = _structured_plan()
    workspace = _workspace(tmp_path, plan)
    pipeline._new_structured_state(workspace, plan)

    workspace.program_path.write_text("def build():\n    raise RuntimeError('changed')\n")

    with pytest.raises(PipelineError, match="current src/program.py"):
        pipeline._load_structured_state(workspace)


@pytest.mark.parametrize(
    "field, message",
    [
        ("contract_sha256", "contract binding"),
        ("pbr_sha256", "PBR binding"),
        ("acceptance_sha256", "acceptance binding"),
        ("topology_sha256", "topology binding"),
    ],
)
def test_structured_state_rejects_corrupted_semantic_bindings(
    tmp_path: Path, field: str, message: str
) -> None:
    plan = _structured_plan()
    workspace = _workspace(tmp_path, plan)
    state = pipeline._new_structured_state(workspace, plan)
    state[field] = "0" * 64
    write_json(pipeline._structured_state_path(workspace), state)

    with pytest.raises(PipelineError, match=message):
        pipeline._load_structured_state(workspace)


@pytest.mark.parametrize(
    "document, message",
    [
        ({"schema_version": 3}, "unsupported schema"),
        (
            {
                "schema_version": 2,
                "phase": "parts",
                "completed_parts": [],
                "geometry_signature": [],
            },
            "lacks part progress",
        ),
    ],
)
def test_invalid_structured_state_is_not_resumed(
    tmp_path: Path, document: dict[str, Any], message: str
) -> None:
    workspace = _workspace(tmp_path, _structured_plan())
    write_json(pipeline._structured_state_path(workspace), document)

    with pytest.raises(PipelineError, match=message):
        pipeline._load_structured_state(workspace)


def test_run_pipeline_resumes_existing_structured_part_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _structured_plan()
    workspace = _workspace(tmp_path, plan)
    state = pipeline._new_structured_state(workspace, plan)
    state.update(
        completed_parts=["base"],
        geometry_signature={"base": {"triangle_count": 12}},
        checkpoints=[
            {
                "part_id": "base",
                "path": "checkpoints/001-base",
                "iteration": 0,
            }
        ],
    )
    write_json(pipeline._structured_state_path(workspace), state)

    seen: list[dict[str, Any]] = []

    def fake_incremental(*args: Any, **kwargs: Any):
        resumed = deepcopy(kwargs["state"])
        seen.append(resumed)
        completed = dict(resumed)
        completed.update(
            phase="final",
            completed_parts=["base", "arm", "badge"],
            geometry_passed=True,
            materials_status="skipped",
        )
        return kwargs["next_iteration"], None, completed

    monkeypatch.setattr(pipeline.BlenderRuntime, "discover", lambda explicit=None: object())
    monkeypatch.setattr(
        pipeline, "prepare_reference", lambda *args, **kwargs: _prepared_probe()
    )
    monkeypatch.setattr(pipeline, "_run_incremental_authoring", fake_incremental)
    monkeypatch.setattr(
        pipeline, "_run_geometry_acceptance", _unexpected_stage("geometry acceptance")
    )
    monkeypatch.setattr(
        pipeline, "_run_material_stage", _unexpected_stage("material stage")
    )
    monkeypatch.setattr(
        pipeline, "_new_structured_state", _unexpected_stage("state recreation")
    )
    monkeypatch.setattr(
        pipeline, "_validate_structured_resume_artifacts", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(pipeline, "build_workspace", _successful_build(workspace))

    report = pipeline.run_pipeline(
        workspace,
        PipelineConfig(max_repairs=0, max_fidelity_repairs=1, min_score=0.9),
    )

    assert len(seen) == 1
    assert seen[0]["completed_parts"] == ["base"]
    assert seen[0]["geometry_signature"] == {"base": {"triangle_count": 12}}
    assert report["structured_state"]["completed_parts"] == ["base", "arm", "badge"]
    assert report["status"] == "complete"


def test_existing_structured_plan_without_state_starts_incremental_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path, _structured_plan())
    seen: list[dict[str, Any]] = []

    def fake_incremental(*args: Any, **kwargs: Any):
        created = deepcopy(kwargs["state"])
        seen.append(created)
        completed = dict(created)
        completed.update(
            phase="final",
            completed_parts=["base", "arm", "badge"],
            geometry_passed=True,
            materials_status="skipped",
        )
        return kwargs["next_iteration"], None, completed

    monkeypatch.setattr(pipeline.BlenderRuntime, "discover", lambda explicit=None: object())
    monkeypatch.setattr(
        pipeline, "prepare_reference", lambda *args, **kwargs: _prepared_probe()
    )
    monkeypatch.setattr(pipeline, "_run_incremental_authoring", fake_incremental)
    monkeypatch.setattr(
        pipeline, "_run_geometry_acceptance", _unexpected_stage("geometry acceptance")
    )
    monkeypatch.setattr(
        pipeline, "_run_material_stage", _unexpected_stage("material stage")
    )
    monkeypatch.setattr(pipeline, "build_workspace", _successful_build(workspace))

    report = pipeline.run_pipeline(
        workspace,
        PipelineConfig(max_repairs=0, max_fidelity_repairs=1, min_score=0.9),
    )

    assert seen[0]["completed_parts"] == []
    assert seen[0]["part_order"] == ["base", "arm", "badge"]
    assert pipeline._structured_state_path(workspace).is_file()
    assert report["status"] == "complete"


def test_failed_geometry_gates_block_the_material_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path, _structured_plan())
    state = pipeline._new_structured_state(workspace, _structured_plan())
    state["phase"] = "geometry"
    comparison = {
        "score": 0.4,
        "passed": False,
        "hard_gates": {
            "passed": False,
            "failures": [
                {
                    "gate": "intersection_fraction",
                    "message": "parts intersect unexpectedly",
                }
            ],
        },
    }
    monkeypatch.setattr(pipeline, "build_workspace", lambda *args, **kwargs: comparison)
    monkeypatch.setattr(
        pipeline,
        "_structured_geometry_signature",
        lambda *args, **kwargs: {"base": {"geometry_sha256": "fixture"}},
    )

    _, _, updated, returned = pipeline._run_geometry_acceptance(
        workspace,
        object(),  # type: ignore[arg-type]
        PipelineConfig(max_geometry_repairs=0),
        reconstruction_mode="procedural",
        granularity="medium",
        quality_profile=resolve_quality_profile("medium"),
        user_prompt="fixture",
        state=state,
        next_iteration=0,
        active_iteration=None,
        agent_runs=[],
        structured_stages=[],
        progress=None,
    )

    assert returned is comparison
    assert updated["geometry_passed"] is False
    assert updated["phase"] == "final"
    assert updated["materials_status"] == "blocked-geometry"


def test_material_quality_gate_rejects_and_rolls_back_dedicated_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _structured_plan()
    workspace = _workspace(tmp_path, plan)
    write_json(workspace.evidence_dir / "glb_probe.json", {"materials": []})
    write_json(workspace.artifacts_dir / "scene_report.json", _geometry_report())
    original_plan = workspace.plan_path.read_bytes()
    state = pipeline._new_structured_state(workspace, plan)
    state.update(
        phase="materials",
        geometry_passed=True,
        completed_parts=["base", "arm", "badge"],
        geometry_signature={"base": {}, "arm": {}, "badge": {}},
        checkpoints=[
            {
                "part_id": part_id,
                "path": f"checkpoints/{index:03d}-{part_id}",
                "iteration": index - 1,
            }
            for index, part_id in enumerate(("base", "arm", "badge"), start=1)
        ],
    )

    def fake_agent(*args: Any, **kwargs: Any) -> dict[str, Any]:
        updated = deepcopy(plan)
        updated["material_plan"] = {
            "schema_version": 1,
            "materials": [
                {
                    "id": "paint",
                    "base_color_rgba": [0.2, 0.3, 0.4, 1.0],
                    "metallic": 0.0,
                    "roughness": 0.5,
                }
            ],
            "assignments": [
                {"part_id": part_id, "material_id": "paint"}
                for part_id in ("base", "arm", "badge")
            ],
        }
        updated["materials"] = [{"id": "paint"}]
        write_json(workspace.plan_path, updated)
        trajectory = workspace.trajectory_dir(kwargs["iteration"])
        (trajectory / "program.py").write_bytes(workspace.program_path.read_bytes())
        (trajectory / "plan.json").write_bytes(workspace.plan_path.read_bytes())
        return {"provider_success": True}

    monkeypatch.setattr(pipeline, "_invoke_agent", fake_agent)
    monkeypatch.setattr(
        pipeline, "_validate_structured_resume_artifacts", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        pipeline,
        "_structured_geometry_signature",
        lambda *args, **kwargs: {"base": {}, "arm": {}, "badge": {}},
    )
    monkeypatch.setattr(
        pipeline, "_assert_material_scene_identity", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        pipeline,
        "_build_incremental_checkpoint",
        lambda *args, **kwargs: {
            "scene": {},
            "validation": {"geometry_signature": {"base": {}}},
        },
    )
    def fake_material_build(*args: Any, **kwargs: Any) -> dict[str, Any]:
        model = workspace.artifacts_dir / "model.glb"
        model.write_bytes(b"final-material-model")
        trajectory = kwargs.get("trajectory_dir")
        assert isinstance(trajectory, Path)
        (trajectory / "model.glb").write_bytes(model.read_bytes())
        return {
            "score": 0.8,
            "passed": False,
            "hard_gates": {
                "passed": False,
                "failures": [
                    {
                        "gate": "mean_palette_similarity",
                        "message": "palette mismatch",
                    }
                ],
            },
        }

    monkeypatch.setattr(pipeline, "build_workspace", fake_material_build)

    _, _, updated_state, guard, failure = pipeline._run_material_stage(
        workspace,
        object(),  # type: ignore[arg-type]
        PipelineConfig(max_material_repairs=0),
        reconstruction_mode="procedural",
        granularity="medium",
        quality_profile=resolve_quality_profile("medium"),
        user_prompt="fixture",
        state=state,
        next_iteration=0,
        active_iteration=None,
        agent_runs=[],
        structured_stages=[],
        progress=None,
    )

    assert guard is not None and guard["passed"] is True
    assert failure is not None and "palette mismatch" in failure
    assert updated_state["materials_status"] == "failed"
    assert workspace.plan_path.read_bytes() == original_plan


def test_legacy_raw_plan_does_not_start_structured_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path, _legacy_plan())

    monkeypatch.setattr(pipeline.BlenderRuntime, "discover", lambda explicit=None: object())
    monkeypatch.setattr(
        pipeline, "prepare_reference", lambda *args, **kwargs: _prepared_probe()
    )
    monkeypatch.setattr(
        pipeline, "_run_incremental_authoring", _unexpected_stage("incremental stage")
    )
    monkeypatch.setattr(
        pipeline, "_run_geometry_acceptance", _unexpected_stage("geometry stage")
    )
    monkeypatch.setattr(
        pipeline, "_run_material_stage", _unexpected_stage("material stage")
    )
    monkeypatch.setattr(pipeline, "build_workspace", _successful_build(workspace))

    report = pipeline.run_pipeline(
        workspace,
        PipelineConfig(
            pipeline_mode="structured",
            max_repairs=0,
            max_fidelity_repairs=1,
            min_score=0.9,
        ),
    )

    assert report["pipeline_mode"] == "structured"
    assert report["structured_state"] is None
    assert report["structured_stages"] == []
    assert not pipeline._structured_state_path(workspace).exists()
    assert report["status"] == "complete"
