from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

import procagen3d.pipeline as pipeline
from procagen3d.workspace import Workspace


def make_workspace(tmp_path: Path) -> Workspace:
    image = tmp_path / "image.png"
    glb = tmp_path / "model.glb"
    image.write_bytes(b"image")
    glb.write_bytes(b"glb")
    return Workspace.create(
        base=tmp_path / "outputs",
        slug="asset",
        image=image,
        glb=glb,
        prompt="",
        backend="codex",
    )


class FakeBackend:
    def __init__(
        self,
        *,
        unauthorized: bool = False,
        omit: str | None = None,
        resolve_reported_paths: bool = False,
        external_reported_path: Path | None = None,
    ):
        self.unauthorized = unauthorized
        self.omit = omit
        self.resolve_reported_paths = resolve_reported_paths
        self.external_reported_path = external_reported_path
        self.seen_workspace: Path | None = None

    def run(self, *, prompt, workspace, trajectory_dir, image_paths, timeout_s):
        self.seen_workspace = workspace
        plan = {
            "subject": "fixture",
            "subject_kind": "object",
            "coordinate_frame": {"up": "+Z"},
            "dimensions": [1, 1, 1],
            "parts": [{"name": "Body"}],
            "materials": [],
            "construction_strategy": "cube",
            "identity_features": [],
            "limitations": [],
        }
        modified = []
        if self.omit != "plan.json":
            (workspace / "src" / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
            modified.append(workspace / "src" / "plan.json")
        if self.omit != "program.py":
            (workspace / "src" / "program.py").write_text(
                "def build():\n    pass\n", encoding="utf-8"
            )
            modified.append(workspace / "src" / "program.py")
        if self.unauthorized:
            (workspace / "manifest.json").write_text("corrupted", encoding="utf-8")
            modified.append(workspace / "manifest.json")
        if self.resolve_reported_paths:
            modified = [path.resolve() for path in modified]
        if self.external_reported_path is not None:
            modified.append(self.external_reported_path)
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        paths = {}
        for name in ("prompt.txt", "transcript.jsonl", "stderr.log", "result.json"):
            path = trajectory_dir / name
            path.write_text(prompt if name == "prompt.txt" else "", encoding="utf-8")
            paths[name] = path
        (trajectory_dir / "final_message.txt").write_text("done", encoding="utf-8")
        return SimpleNamespace(
            ok=True,
            error=None,
            stderr="",
            final_message="done",
            exit_reason="completed",
            files_modified=tuple(modified),
            prompt_path=paths["prompt.txt"],
            transcript_path=paths["transcript.jsonl"],
            stderr_path=paths["stderr.log"],
            result_path=paths["result.json"],
            backend="fake",
            model="fake-model",
            duration_s=0.1,
            usage={},
        )


class SymlinkSourceBackend(FakeBackend):
    def __init__(self, external_src: Path):
        super().__init__()
        self.external_src = external_src

    def run(self, *, prompt, workspace, trajectory_dir, image_paths, timeout_s):
        staged_src = workspace / "src"
        staged_src.rmdir()
        staged_src.symlink_to(self.external_src, target_is_directory=True)
        result = super().run(
            prompt=prompt,
            workspace=workspace,
            trajectory_dir=trajectory_dir,
            image_paths=image_paths,
            timeout_s=timeout_s,
        )
        # Match the backend snapshot edge case: rglob sees the directory link,
        # but does not descend to report its externally located children.
        result.files_modified = ()
        return result


class ActivityBackend(FakeBackend):
    name = "codex"

    def run(
        self,
        *,
        prompt,
        workspace,
        trajectory_dir,
        image_paths,
        timeout_s,
        on_activity=None,
        heartbeat_interval_s=None,
    ):
        self.heartbeat_interval_s = heartbeat_interval_s
        if on_activity is not None:
            on_activity("Provider is inspecting reference evidence")
        return super().run(
            prompt=prompt,
            workspace=workspace,
            trajectory_dir=trajectory_dir,
            image_paths=image_paths,
            timeout_s=timeout_s,
        )


def test_agent_runs_in_disposable_copy_and_promotes_only_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = make_workspace(tmp_path)
    fake = FakeBackend()
    monkeypatch.setattr(pipeline, "create_backend", lambda name: fake)

    result = pipeline._invoke_agent(
        workspace,
        backend_name="codex",
        prompt="author",
        iteration=0,
        timeout_s=10,
    )

    assert fake.seen_workspace is not None
    assert fake.seen_workspace != workspace.root
    assert not fake.seen_workspace.exists()
    assert workspace.program_path.read_text(encoding="utf-8").startswith("def build")
    assert json.loads(workspace.plan_path.read_text(encoding="utf-8"))["subject"] == "fixture"
    assert (workspace.root / "trajectories" / "iter_00" / "transcript.jsonl").is_file()
    assert result["model"] == "fake-model"


def test_cli_backend_activity_is_forwarded_without_closing_the_agent_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = make_workspace(tmp_path)
    fake = ActivityBackend()
    events = []
    monkeypatch.setattr(pipeline, "create_backend", lambda name: fake)
    # The production branch is intentionally based on CLIBackend so ordinary
    # duck-typed test/custom backends keep their existing fixed run signature.
    monkeypatch.setattr(pipeline, "CLIBackend", ActivityBackend)

    pipeline._invoke_agent(
        workspace,
        backend_name="codex",
        prompt="author",
        iteration=0,
        timeout_s=10,
        progress=events.append,
    )

    activity = [
        event
        for event in events
        if event.kind == "info" and "inspecting reference" in event.message
    ]
    assert len(activity) == 1
    assert activity[0].stage == "agent-00"
    assert activity[0].elapsed_s is None
    assert fake.heartbeat_interval_s is None
    assert events[0].kind == "start"
    assert events[-1].kind == "success"


def test_unauthorized_agent_write_is_discarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = make_workspace(tmp_path)
    original_manifest = workspace.manifest_path.read_bytes()
    fake = FakeBackend(unauthorized=True)
    monkeypatch.setattr(pipeline, "create_backend", lambda name: fake)

    with pytest.raises(pipeline.PipelineError, match="outside src"):
        pipeline._invoke_agent(
            workspace,
            backend_name="codex",
            prompt="author",
            iteration=0,
            timeout_s=10,
        )

    assert workspace.manifest_path.read_bytes() == original_manifest
    assert not workspace.program_path.exists()
    assert not workspace.plan_path.exists()
    rejected = workspace.root / "trajectories" / "iter_00"
    assert (rejected / "rejected_program.py").is_file()
    assert (rejected / "rejected_plan.json").is_file()


def test_equivalent_macos_var_private_var_paths_are_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resolved provider paths must match the lexical /var temp-root alias."""

    workspace = make_workspace(tmp_path)
    private_var = tmp_path / "private" / "var"
    private_var.mkdir(parents=True)
    var_alias = tmp_path / "var"
    try:
        var_alias.symlink_to(private_var, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")
    aliased_agent_root = var_alias / "procagen3d-agent-fixed"
    original_temporary_directory = pipeline.tempfile.TemporaryDirectory

    @contextmanager
    def aliased_temporary_directory(*args, **kwargs):
        if kwargs.get("prefix") == "procagen3d-agent-" and kwargs.get("dir") is None:
            aliased_agent_root.mkdir()
            yield str(aliased_agent_root)
            return
        with original_temporary_directory(*args, **kwargs) as directory:
            yield directory

    fake = FakeBackend(resolve_reported_paths=True)
    monkeypatch.setattr(pipeline.tempfile, "TemporaryDirectory", aliased_temporary_directory)
    monkeypatch.setattr(pipeline, "create_backend", lambda name: fake)

    result = pipeline._invoke_agent(
        workspace,
        backend_name="codex",
        prompt="author",
        iteration=0,
        timeout_s=10,
    )

    assert fake.seen_workspace is not None
    assert fake.seen_workspace == aliased_agent_root.resolve()
    assert fake.seen_workspace != aliased_agent_root
    assert result["files_modified"] == ["src/plan.json", "src/program.py"]
    assert workspace.program_path.is_file()
    assert workspace.plan_path.is_file()


def test_true_out_of_workspace_reported_path_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = make_workspace(tmp_path)
    external = tmp_path / "genuinely-external.py"
    external.write_text("changed", encoding="utf-8")
    fake = FakeBackend(
        resolve_reported_paths=True,
        external_reported_path=external.resolve(),
    )
    monkeypatch.setattr(pipeline, "create_backend", lambda name: fake)

    with pytest.raises(pipeline.PipelineError, match="outside src"):
        pipeline._invoke_agent(
            workspace,
            backend_name="codex",
            prompt="author",
            iteration=0,
            timeout_s=10,
        )

    assert external.read_text(encoding="utf-8") == "changed"
    assert not workspace.program_path.exists()
    assert not workspace.plan_path.exists()
    rejected = workspace.root / "trajectories" / "iter_00"
    assert (rejected / "rejected_program.py").is_file()
    assert (rejected / "rejected_plan.json").is_file()


def test_symlinked_agent_src_directory_is_rejected_without_retention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = make_workspace(tmp_path)
    external_src = tmp_path / "external-src"
    external_src.mkdir()
    fake = SymlinkSourceBackend(external_src)
    monkeypatch.setattr(pipeline, "create_backend", lambda name: fake)

    with pytest.raises(pipeline.PipelineError, match="src/ must remain"):
        pipeline._invoke_agent(
            workspace,
            backend_name="codex",
            prompt="author",
            iteration=0,
            timeout_s=10,
        )

    assert (external_src / "program.py").is_file()
    assert (external_src / "plan.json").is_file()
    assert not workspace.program_path.exists()
    assert not workspace.plan_path.exists()
    rejected = workspace.root / "trajectories" / "iter_00"
    assert not (rejected / "rejected_program.py").exists()
    assert not (rejected / "rejected_plan.json").exists()


@pytest.mark.parametrize("missing", ["program.py", "plan.json"])
def test_successful_agent_missing_required_deliverable_marks_run_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
) -> None:
    workspace = make_workspace(tmp_path)
    fake = FakeBackend(omit=missing)
    monkeypatch.setattr(pipeline, "create_backend", lambda name: fake)
    monkeypatch.setattr(pipeline.BlenderRuntime, "discover", lambda explicit=None: object())
    monkeypatch.setattr(
        pipeline,
        "prepare_reference",
        lambda *args, **kwargs: {
            "scene": {"vertex_count": 3, "triangle_count": 1},
            "semantic_decomposition": {"status": "insufficient"},
        },
    )

    with pytest.raises(pipeline.PipelineError, match=f"agent src/{missing}"):
        pipeline.run_pipeline(workspace, pipeline.PipelineConfig())

    assert workspace.manifest()["status"] == "failed"
    report = json.loads((workspace.root / "run_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert f"agent src/{missing}" in report["error"]
    assert not workspace.program_path.exists()
    assert not workspace.plan_path.exists()
