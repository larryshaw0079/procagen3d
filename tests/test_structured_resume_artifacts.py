from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import procagen3d.pipeline as pipeline
from procagen3d.pipeline import PipelineError
from procagen3d.workspace import Workspace, sha256, write_json


_SIGNATURE = {"body": {"triangle_count": 12}}


@dataclass
class ResumeArtifacts:
    workspace: Workspace
    plan: dict[str, Any]
    state: dict[str, Any]
    checkpoint_dir: Path
    trajectory_dir: Path
    probe_calls: list[tuple[Path, tuple[str, ...], str]]


@pytest.fixture
def resume_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> ResumeArtifacts:
    source = tmp_path / "source"
    source.mkdir()
    image = source / "reference.png"
    glb = source / "reference.glb"
    image.write_bytes(b"image")
    glb.write_bytes(b"glb")
    workspace = Workspace.create(
        base=tmp_path / "outputs",
        slug="structured-resume-artifacts",
        image=image,
        glb=glb,
        prompt="fixture",
        backend="codex",
    )
    plan = {
        "parts": [
            {
                "id": "base",
                "object_names": ["Body"],
                "attachment": {"type": "root"},
            }
        ],
        "assembly": {
            "version": 1,
            "placement": "host-solved",
            "part_order": ["base"],
            "connectors": [],
            "mates": [],
        },
    }
    write_json(workspace.plan_path, plan)
    workspace.program_path.write_text("def build():\n    pass\n", encoding="utf-8")

    iteration = 4
    checkpoint_name = "001-base"
    checkpoint_dir = workspace.root / "checkpoints" / checkpoint_name
    checkpoint_dir.mkdir(parents=True)
    checkpoint_model = checkpoint_dir / "model.glb"
    checkpoint_model.write_bytes(b"checkpoint model")
    checkpoint_scene_report = checkpoint_dir / "scene_report.json"
    write_json(checkpoint_scene_report, {"geometry_object_count": 1})

    trajectory_dir = workspace.root / "trajectories" / f"iter_{iteration:02d}"
    trajectory_dir.mkdir(parents=True)
    trajectory_program = trajectory_dir / "program.py"
    trajectory_plan = trajectory_dir / "plan.json"
    trajectory_program.write_text("def build():\n    pass\n", encoding="utf-8")
    write_json(trajectory_plan, plan)
    (trajectory_dir / "model.glb").write_bytes(checkpoint_model.read_bytes())

    checkpoint_report = {
        "schema_version": 1,
        "checkpoint": checkpoint_name,
        "program_sha256": sha256(trajectory_program),
        "plan_sha256": sha256(trajectory_plan),
        "model_sha256": sha256(checkpoint_model),
        "scene_report_sha256": sha256(checkpoint_scene_report),
        "validation": {
            "completed_parts": ["base"],
            "geometry_signature": deepcopy(_SIGNATURE),
        },
    }
    write_json(checkpoint_dir / "checkpoint.json", checkpoint_report)
    write_json(
        checkpoint_dir / "build_manifest.json",
        {
            "schema_version": 1,
            "kind": "incremental-part-checkpoint",
            "checkpoint": checkpoint_name,
            "program_sha256": checkpoint_report["program_sha256"],
            "plan_sha256": checkpoint_report["plan_sha256"],
            "model_sha256": checkpoint_report["model_sha256"],
            "scene_report_sha256": checkpoint_report["scene_report_sha256"],
            "clean_room": True,
            "compiled_glb_verified_in_separate_process": True,
        },
    )

    published_model = workspace.artifacts_dir / "model.glb"
    published_model.write_bytes(b"published model")
    published_scene_report = workspace.artifacts_dir / "scene_report.json"
    write_json(published_scene_report, {"geometry_object_count": 1})
    state = {
        "completed_parts": ["base"],
        "checkpoints": [
            {
                "part_id": "base",
                "path": f"checkpoints/{checkpoint_name}",
                "iteration": iteration,
            }
        ],
        "program_sha256": sha256(workspace.program_path),
        "plan_sha256": sha256(workspace.plan_path),
        "geometry_signature": deepcopy(_SIGNATURE),
    }
    write_json(
        workspace.artifacts_dir / "build_manifest.json",
        {
            "schema_version": 1,
            "program_sha256": state["program_sha256"],
            "plan_sha256": state["plan_sha256"],
            "model_sha256": sha256(published_model),
            "scene_report_sha256": sha256(published_scene_report),
            "clean_room": True,
            "compiled_glb_verified_in_separate_process": True,
        },
    )

    probe_calls: list[tuple[Path, tuple[str, ...], str]] = []

    def fake_probe_signature(
        model_path: Path,
        *,
        plan: dict[str, Any],
        completed_part_ids: list[str],
        label: str,
    ) -> dict[str, Any]:
        probe_calls.append((model_path, tuple(completed_part_ids), label))
        return deepcopy(_SIGNATURE)

    monkeypatch.setattr(pipeline, "_probe_signature_for_parts", fake_probe_signature)
    return ResumeArtifacts(
        workspace=workspace,
        plan=plan,
        state=state,
        checkpoint_dir=checkpoint_dir,
        trajectory_dir=trajectory_dir,
        probe_calls=probe_calls,
    )


def _validate(fixture: ResumeArtifacts) -> None:
    pipeline._validate_structured_resume_artifacts(
        fixture.workspace,
        plan=fixture.plan,
        state=fixture.state,
    )


def test_validates_checkpoint_trajectory_and_published_bindings(
    resume_artifacts: ResumeArtifacts,
) -> None:
    _validate(resume_artifacts)

    assert resume_artifacts.probe_calls == [
        (
            resume_artifacts.checkpoint_dir / "model.glb",
            ("base",),
            "structured checkpoint 'base'",
        ),
        (
            resume_artifacts.workspace.artifacts_dir / "model.glb",
            ("base",),
            "published structured build",
        ),
    ]


def test_rejects_missing_checkpoint_directory(
    resume_artifacts: ResumeArtifacts,
) -> None:
    resume_artifacts.checkpoint_dir.rename(
        resume_artifacts.checkpoint_dir.with_name("001-base-missing")
    )

    with pytest.raises(PipelineError, match="checkpoint 'base' directory is unavailable"):
        _validate(resume_artifacts)


def test_rejects_tampered_trajectory_source(
    resume_artifacts: ResumeArtifacts,
) -> None:
    (resume_artifacts.trajectory_dir / "program.py").write_text(
        "def build():\n    raise RuntimeError('tampered')\n",
        encoding="utf-8",
    )

    with pytest.raises(PipelineError, match="checkpoint 'base' program.py binding is invalid"):
        _validate(resume_artifacts)


def test_rejects_tampered_checkpoint_model_hash(
    resume_artifacts: ResumeArtifacts,
) -> None:
    (resume_artifacts.checkpoint_dir / "model.glb").write_bytes(
        b"tampered checkpoint model"
    )

    with pytest.raises(PipelineError, match="checkpoint 'base' model binding is invalid"):
        _validate(resume_artifacts)


def test_rejects_tampered_checkpoint_trajectory_model(
    resume_artifacts: ResumeArtifacts,
) -> None:
    (resume_artifacts.trajectory_dir / "model.glb").write_bytes(
        b"tampered trajectory model"
    )

    with pytest.raises(
        PipelineError,
        match="checkpoint 'base' trajectory model binding is invalid",
    ):
        _validate(resume_artifacts)


def test_rejects_tampered_checkpoint_scene_report(
    resume_artifacts: ResumeArtifacts,
) -> None:
    write_json(
        resume_artifacts.checkpoint_dir / "scene_report.json",
        {"geometry_object_count": 99},
    )

    with pytest.raises(
        PipelineError,
        match="checkpoint 'base' scene report binding is invalid",
    ):
        _validate(resume_artifacts)


def test_rejects_invalid_checkpoint_manifest_policy(
    resume_artifacts: ResumeArtifacts,
) -> None:
    manifest_path = resume_artifacts.checkpoint_dir / "build_manifest.json"
    manifest = pipeline._read_regular_json(manifest_path, label="fixture manifest")
    manifest["clean_room"] = False
    write_json(manifest_path, manifest)

    with pytest.raises(
        PipelineError,
        match="checkpoint 'base' manifest binding is invalid",
    ):
        _validate(resume_artifacts)


def test_rejects_stale_published_source_hash(
    resume_artifacts: ResumeArtifacts,
) -> None:
    manifest_path = resume_artifacts.workspace.artifacts_dir / "build_manifest.json"
    manifest = pipeline._read_regular_json(manifest_path, label="fixture manifest")
    manifest["program_sha256"] = "0" * 64
    write_json(manifest_path, manifest)

    with pytest.raises(
        PipelineError,
        match="published structured build has a stale program binding",
    ):
        _validate(resume_artifacts)


def test_rejects_invalid_published_manifest_policy(
    resume_artifacts: ResumeArtifacts,
) -> None:
    manifest_path = resume_artifacts.workspace.artifacts_dir / "build_manifest.json"
    manifest = pipeline._read_regular_json(manifest_path, label="fixture manifest")
    manifest["compiled_glb_verified_in_separate_process"] = False
    write_json(manifest_path, manifest)

    with pytest.raises(
        PipelineError,
        match="published structured build manifest policy is invalid",
    ):
        _validate(resume_artifacts)


def test_rejects_tampered_published_scene_report(
    resume_artifacts: ResumeArtifacts,
) -> None:
    write_json(
        resume_artifacts.workspace.artifacts_dir / "scene_report.json",
        {"geometry_object_count": 99},
    )

    with pytest.raises(
        PipelineError,
        match="published structured scene report binding is invalid",
    ):
        _validate(resume_artifacts)


def test_rejects_state_vs_published_signature_mismatch(
    resume_artifacts: ResumeArtifacts,
) -> None:
    resume_artifacts.state["geometry_signature"] = {
        "body": {"triangle_count": 99}
    }

    with pytest.raises(
        PipelineError,
        match="published structured geometry does not match structured_state.json",
    ):
        _validate(resume_artifacts)
