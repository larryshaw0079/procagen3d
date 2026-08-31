from __future__ import annotations

import json
from pathlib import Path

import pytest

import procagen3d.pipeline as pipeline
from procagen3d.blender import BlenderError
from procagen3d.pipeline import PipelineConfig, PipelineError
from procagen3d.progress import ProgressEvent, emit_progress
from procagen3d.workspace import Workspace, sha256, write_json


def _workspace(tmp_path: Path) -> Workspace:
    image = tmp_path / "reference.png"
    glb = tmp_path / "reference.glb"
    image.write_bytes(b"image")
    glb.write_bytes(b"glb")
    workspace = Workspace.create(
        base=tmp_path / "outputs",
        slug="fixture",
        image=image,
        glb=glb,
        prompt="fixture",
        backend="codex",
    )
    write_json(
        workspace.plan_path,
        {
            "subject": "fixture",
            "subject_kind": "object",
            "coordinate_frame": {"up": "+Z"},
            "dimensions": [1.0, 1.0, 1.0],
            "parts": [{"name": "Body"}],
            "materials": [],
            "construction_strategy": "primitive",
            "identity_features": [],
            "limitations": [],
        },
    )
    workspace.program_path.write_text("def build():\n    pass\n", encoding="utf-8")
    return workspace


def _prepared_probe() -> dict:
    return {
        "scene": {"vertex_count": 3, "triangle_count": 1},
        "semantic_decomposition": {"status": "insufficient"},
    }


def test_plan_only_failed_repair_restores_source_matching_last_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path)
    original_plan = workspace.plan_path.read_bytes()
    original_program = workspace.program_path.read_bytes()
    runtime = object()
    monkeypatch.setattr(pipeline.BlenderRuntime, "discover", lambda explicit=None: runtime)
    monkeypatch.setattr(pipeline, "prepare_reference", lambda *args, **kwargs: _prepared_probe())

    build_calls = 0

    def fake_build(*args, **kwargs):
        nonlocal build_calls
        build_calls += 1
        if build_calls == 2:
            raise PipelineError("repaired plan is invalid")
        write_json(
            workspace.artifacts_dir / "build_manifest.json",
            {
                "program_sha256": sha256(workspace.program_path),
                "plan_sha256": sha256(workspace.plan_path),
            },
        )
        return {"score": 0.2, "passed": False}

    def fake_agent(*args, **kwargs):
        workspace.plan_path.write_text("{}", encoding="utf-8")
        return {"backend": "fake", "model": "fake", "files_modified": ["src/plan.json"]}

    monkeypatch.setattr(pipeline, "build_workspace", fake_build)
    monkeypatch.setattr(pipeline, "_invoke_agent", fake_agent)

    report = pipeline.run_pipeline(
        workspace,
        PipelineConfig(max_repairs=0, max_fidelity_repairs=1, min_score=0.9),
    )

    assert report["status"] == "needs-review"
    assert "source was restored" in report["warning"]
    assert workspace.program_path.read_bytes() == original_program
    assert workspace.plan_path.read_bytes() == original_plan
    artifact = json.loads(
        (workspace.artifacts_dir / "build_manifest.json").read_text(encoding="utf-8")
    )
    assert artifact["program_sha256"] == sha256(workspace.program_path)
    assert artifact["plan_sha256"] == sha256(workspace.plan_path)
    assert (workspace.root / "trajectories" / "iter_00" / "rejected_plan.json").is_file()


def test_reference_failure_replaces_stale_complete_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path)
    workspace.update_manifest(status="complete")
    write_json(workspace.root / "run_report.json", {"status": "complete"})

    def missing_blender(explicit=None):
        raise BlenderError("Blender missing")

    monkeypatch.setattr(pipeline.BlenderRuntime, "discover", missing_blender)

    with pytest.raises(BlenderError, match="missing"):
        pipeline.run_pipeline(workspace, PipelineConfig())

    assert workspace.manifest()["status"] == "failed"
    report = json.loads((workspace.root / "run_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["stage"] == "reference"


def test_cached_prepare_only_updates_manifest_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path)
    workspace.update_manifest(status="complete")
    monkeypatch.setattr(pipeline.BlenderRuntime, "discover", lambda explicit=None: object())
    monkeypatch.setattr(pipeline, "prepare_reference", lambda *args, **kwargs: _prepared_probe())

    report = pipeline.run_pipeline(
        workspace,
        PipelineConfig(granularity="fine"),
        prepare_only=True,
    )

    assert report["status"] == "prepared"
    assert report["granularity"] == "fine"
    assert workspace.manifest()["status"] == "prepared"
    assert workspace.manifest()["granularity"] == "fine"


def test_run_pipeline_forwards_progress_through_a_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path)
    runtime = object()
    events: list[ProgressEvent] = []
    build_calls = 0
    repair_calls = 0

    monkeypatch.setattr(pipeline.BlenderRuntime, "discover", lambda explicit=None: runtime)

    def fake_prepare(*args, **kwargs):
        emit_progress(kwargs.get("progress"), "info", "fake-reference", "Reference ready")
        return _prepared_probe()

    def fake_build(*args, **kwargs):
        nonlocal build_calls
        build_calls += 1
        emit_progress(kwargs.get("progress"), "info", "fake-build", "Build finished")
        write_json(
            workspace.artifacts_dir / "build_manifest.json",
            {
                "program_sha256": sha256(workspace.program_path),
                "plan_sha256": sha256(workspace.plan_path),
            },
        )
        return {
            "score": 0.2 if build_calls == 1 else 1.0,
            "passed": build_calls > 1,
        }

    def fake_agent(*args, **kwargs):
        nonlocal repair_calls
        repair_calls += 1
        assert kwargs["is_repair"] is True
        emit_progress(kwargs.get("progress"), "info", "fake-repair", "Repair finished")
        return {"backend": "fake", "model": "fake", "files_modified": []}

    monkeypatch.setattr(pipeline, "prepare_reference", fake_prepare)
    monkeypatch.setattr(pipeline, "build_workspace", fake_build)
    monkeypatch.setattr(pipeline, "_invoke_agent", fake_agent)

    report = pipeline.run_pipeline(
        workspace,
        PipelineConfig(max_repairs=1, min_score=0.9),
        progress=events.append,
    )

    assert report["status"] == "complete"
    assert build_calls == 2
    assert repair_calls == 1
    assert [event.stage for event in events if event.stage.startswith("fake-")] == [
        "fake-reference",
        "fake-build",
        "fake-repair",
        "fake-build",
    ]
    assert [
        event.message for event in events if event.stage == "build-attempt" and event.kind == "info"
    ] == ["Build attempt 1 of at most 3", "Build attempt 2 of at most 3"]
    assert events[-1].kind == "success"
    assert events[-1].stage == "pipeline"


def test_schema_build_and_post_render_repairs_have_independent_budgets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path)
    monkeypatch.setattr(pipeline.BlenderRuntime, "discover", lambda explicit=None: object())
    monkeypatch.setattr(pipeline, "prepare_reference", lambda *args, **kwargs: _prepared_probe())
    build_calls = 0
    repair_prompts: list[dict] = []

    def fake_build(*args, **kwargs):
        nonlocal build_calls
        build_calls += 1
        if build_calls == 1:
            raise PipelineError("schema failed")
        write_json(
            workspace.artifacts_dir / "build_manifest.json",
            {
                "program_sha256": sha256(workspace.program_path),
                "plan_sha256": sha256(workspace.plan_path),
            },
        )
        return {
            "score": 0.2 if build_calls == 2 else 1.0,
            "score_passed": build_calls == 3,
            "passed": build_calls == 3,
            "hard_gates": {"failures": []},
        }

    def fake_agent(*args, **kwargs):
        repair_prompts.append(kwargs)
        return {"backend": "fake", "model": "fake", "files_modified": []}

    monkeypatch.setattr(pipeline, "build_workspace", fake_build)
    monkeypatch.setattr(pipeline, "_invoke_agent", fake_agent)

    report = pipeline.run_pipeline(
        workspace,
        PipelineConfig(max_repairs=1, max_fidelity_repairs=1, min_score=0.9),
    )

    assert report["status"] == "complete"
    assert build_calls == 3
    assert len(repair_prompts) == 2
    assert repair_prompts[0]["include_candidate"] is False
    assert repair_prompts[1]["include_candidate"] is True
    assert [attempt["phase"] for attempt in report["build_attempts"]] == [
        "initial",
        "schema-build-repair",
        "post-render-repair",
    ]
    assert report["retry_budget"] == {
        "initial_agent": {"used": 0, "maximum": 1},
        "schema_build": {"used": 1, "maximum": 1},
        "post_render": {"used": 1, "maximum": 1},
    }


def test_initial_agent_failure_retries_in_a_new_trajectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path)
    plan = workspace.plan_path.read_bytes()
    program = workspace.program_path.read_bytes()
    workspace.plan_path.unlink()
    workspace.program_path.unlink()
    monkeypatch.setattr(pipeline.BlenderRuntime, "discover", lambda explicit=None: object())
    monkeypatch.setattr(pipeline, "prepare_reference", lambda *args, **kwargs: _prepared_probe())
    invocations: list[int] = []

    def fake_agent(*args, **kwargs):
        iteration = kwargs["iteration"]
        invocations.append(iteration)
        trajectory = workspace.trajectory_dir(iteration)
        if len(invocations) == 1:
            write_json(trajectory / "result.json", {"success": False, "exit_reason": "timeout"})
            raise PipelineError("initial provider timeout")
        workspace.program_path.write_bytes(program)
        workspace.plan_path.write_bytes(plan)
        (trajectory / "program.py").write_bytes(program)
        (trajectory / "plan.json").write_bytes(plan)
        return {"backend": "fake", "model": "fake", "files_modified": ["src/program.py"]}

    def fake_build(*args, **kwargs):
        write_json(
            workspace.artifacts_dir / "build_manifest.json",
            {
                "program_sha256": sha256(workspace.program_path),
                "plan_sha256": sha256(workspace.plan_path),
            },
        )
        return {"score": 1.0, "score_passed": True, "passed": True}

    monkeypatch.setattr(pipeline, "_invoke_agent", fake_agent)
    monkeypatch.setattr(pipeline, "build_workspace", fake_build)

    report = pipeline.run_pipeline(
        workspace,
        PipelineConfig(
            pipeline_mode="legacy",
            max_initial_agent_retries=1,
            max_repairs=0,
        ),
    )

    assert invocations == [0, 1]
    assert [run["phase"] for run in report["agent_runs"]] == ["initial", "initial-retry"]
    assert report["agent_runs"][0]["error"] == "initial provider timeout"
    assert report["retry_budget"]["initial_agent"] == {"used": 1, "maximum": 1}
    assert report["status"] == "complete"


def test_structured_initial_retry_receives_exact_validation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path)
    workspace.plan_path.unlink()
    workspace.program_path.unlink()
    monkeypatch.setattr(pipeline.BlenderRuntime, "discover", lambda explicit=None: object())
    monkeypatch.setattr(pipeline, "prepare_reference", lambda *args, **kwargs: _prepared_probe())

    prompts: list[str] = []
    invocation_modes: list[tuple[bool, bool]] = []

    def fake_agent(*args, **kwargs):
        prompts.append(kwargs["prompt"])
        invocation_modes.append(
            (kwargs["is_repair"], kwargs["include_candidate"])
        )
        workspace.program_path.write_text("def build():\n    pass\n", encoding="utf-8")
        write_json(workspace.plan_path, {"assembly": {}})
        return {
            "backend": "fake",
            "model": "fake",
            "provider_success": True,
            "files_modified": ["src/plan.json", "src/program.py"],
        }

    validation_failure = (
        "HOST-VALIDATION-42: $.assembly.mates[0]: "
        "rigid mates must not declare rest or limits"
    )
    validation_calls = 0

    def fake_validate(path: Path) -> dict:
        nonlocal validation_calls
        validation_calls += 1
        if validation_calls == 1:
            raise PipelineError(validation_failure)
        return {"assembly": {}, "parts": []}

    def stop_after_planning(*args, **kwargs):
        raise PipelineError("stop after structured planning")

    monkeypatch.setattr(pipeline, "_invoke_agent", fake_agent)
    monkeypatch.setattr(pipeline, "_validate_plan", fake_validate)
    monkeypatch.setattr(
        pipeline,
        "_validate_explicit_assembly_contract",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(pipeline, "_load_structured_state", stop_after_planning)

    with pytest.raises(PipelineError, match="stop after structured planning"):
        pipeline.run_pipeline(
            workspace,
            PipelineConfig(
                pipeline_mode="structured",
                max_initial_agent_retries=1,
            ),
        )

    assert len(prompts) == 2
    assert validation_failure not in prompts[0]
    assert validation_failure in prompts[1]
    assert prompts[1] != prompts[0]
    assert invocation_modes == [(False, False), (True, True)]


def test_run_repairs_preserved_invalid_structured_plan_before_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path)
    write_json(workspace.plan_path, {"assembly": {}})
    monkeypatch.setattr(pipeline.BlenderRuntime, "discover", lambda explicit=None: object())
    monkeypatch.setattr(pipeline, "prepare_reference", lambda *args, **kwargs: _prepared_probe())

    validation_failure = (
        "src/plan.json plan violates the validation contract (1 error):\n"
        "- $.parts[0].attachment.parent_id: a root attachment must use '__root__'"
    )
    validation_calls = 0

    def fake_validate(path: Path) -> dict:
        nonlocal validation_calls
        validation_calls += 1
        if validation_calls == 1:
            raise PipelineError(validation_failure)
        return {"assembly": {}, "parts": []}

    invocations: list[dict] = []

    def fake_agent(*args, **kwargs):
        invocations.append(kwargs)
        return {
            "backend": "fake",
            "model": "fake",
            "provider_success": True,
            "files_modified": ["src/plan.json", "src/program.py"],
        }

    def stop_after_planning(*args, **kwargs):
        raise PipelineError("stop after preserved planning repair")

    monkeypatch.setattr(pipeline, "_validate_plan", fake_validate)
    monkeypatch.setattr(
        pipeline,
        "_validate_explicit_assembly_contract",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(pipeline, "_invoke_agent", fake_agent)
    monkeypatch.setattr(pipeline, "_load_structured_state", stop_after_planning)

    with pytest.raises(PipelineError, match="stop after preserved planning repair"):
        pipeline.run_pipeline(
            workspace,
            PipelineConfig(
                pipeline_mode="structured",
                max_initial_agent_retries=1,
            ),
        )

    assert len(invocations) == 1
    assert validation_failure in invocations[0]["prompt"]
    assert invocations[0]["is_repair"] is True
    assert invocations[0]["include_candidate"] is True


def test_post_render_repairs_stop_on_stall_and_restore_best_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path)
    monkeypatch.setattr(pipeline.BlenderRuntime, "discover", lambda explicit=None: object())
    monkeypatch.setattr(pipeline, "prepare_reference", lambda *args, **kwargs: _prepared_probe())
    scores = [0.5, 0.7, 0.6]
    build_calls = 0
    repair_calls = 0

    def fake_build(*args, **kwargs):
        nonlocal build_calls
        score = scores[build_calls]
        build_calls += 1
        (workspace.artifacts_dir / "candidate.txt").write_text(
            str(build_calls), encoding="utf-8"
        )
        write_json(
            workspace.artifacts_dir / "build_manifest.json",
            {
                "program_sha256": sha256(workspace.program_path),
                "plan_sha256": sha256(workspace.plan_path),
            },
        )
        return {
            "score": score,
            "score_passed": False,
            "passed": False,
            "hard_gates": {"passed": False, "failures": []},
        }

    def fake_agent(*args, **kwargs):
        nonlocal repair_calls
        repair_calls += 1
        iteration = kwargs["iteration"]
        workspace.program_path.write_text(
            f"def build():\n    return {repair_calls}\n", encoding="utf-8"
        )
        trajectory = workspace.trajectory_dir(iteration)
        (trajectory / "program.py").write_bytes(workspace.program_path.read_bytes())
        (trajectory / "plan.json").write_bytes(workspace.plan_path.read_bytes())
        return {"backend": "fake", "model": "fake", "files_modified": ["src/program.py"]}

    monkeypatch.setattr(pipeline, "build_workspace", fake_build)
    monkeypatch.setattr(pipeline, "_invoke_agent", fake_agent)

    report = pipeline.run_pipeline(
        workspace,
        PipelineConfig(max_repairs=0, max_fidelity_repairs=3, min_score=0.9),
    )

    assert build_calls == 3
    assert repair_calls == 2
    assert report["score"] == 0.7
    assert report["best_candidate"] == {
        "build_attempt": 1,
        "trajectory_iteration": 0,
        "score": 0.7,
        "passed": False,
        "restored": True,
    }
    assert [attempt["selected"] for attempt in report["build_attempts"]] == [
        False,
        True,
        False,
    ]
    assert report["retry_budget"]["post_render"] == {"used": 2, "maximum": 3}
    assert "improvement stalled" in report["warning"].lower()
    assert workspace.program_path.read_text(encoding="utf-8").endswith("return 1\n")
    assert (workspace.artifacts_dir / "candidate.txt").read_text(encoding="utf-8") == "2"


def test_resume_routes_glb_to_the_trajectory_matching_restored_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path)
    iter_00 = workspace.root / "trajectories" / "iter_00"
    iter_01 = workspace.root / "trajectories" / "iter_01"
    iter_00.mkdir()
    iter_01.mkdir()
    (iter_00 / "program.py").write_bytes(workspace.program_path.read_bytes())
    (iter_00 / "plan.json").write_bytes(workspace.plan_path.read_bytes())
    (iter_01 / "program.py").write_text("def rejected():\n    pass\n", encoding="utf-8")
    (iter_01 / "plan.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(pipeline.BlenderRuntime, "discover", lambda explicit=None: object())
    monkeypatch.setattr(pipeline, "prepare_reference", lambda *args, **kwargs: _prepared_probe())
    seen_trajectory: list[Path | None] = []

    def fake_build(*args, **kwargs):
        trajectory = kwargs.get("trajectory_dir")
        seen_trajectory.append(trajectory)
        assert trajectory == iter_00
        (iter_00 / "model.glb").write_bytes(b"restored source model")
        write_json(
            workspace.artifacts_dir / "build_manifest.json",
            {
                "program_sha256": sha256(workspace.program_path),
                "plan_sha256": sha256(workspace.plan_path),
            },
        )
        return {"score": 1.0, "passed": True}

    monkeypatch.setattr(pipeline, "build_workspace", fake_build)

    report = pipeline.run_pipeline(
        workspace,
        PipelineConfig(max_repairs=0, min_score=0.9),
    )

    assert seen_trajectory == [iter_00]
    assert report["build_attempts"][0]["trajectory_glb"] == (
        "trajectories/iter_00/model.glb"
    )
    assert not (iter_01 / "model.glb").exists()
