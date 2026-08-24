from __future__ import annotations

import json
from pathlib import Path

import procagen3d.cli as cli
from procagen3d.backends import create_backend
from procagen3d.cli import _command_examples, build_parser
from procagen3d.progress import emit_progress
from procagen3d.workspace import Workspace


def test_make_defaults_to_requested_codex_configuration() -> None:
    args = build_parser().parse_args(["make", "image.png", "model.glb"])
    assert args.backend == "codex"
    assert args.max_repairs == 2
    assert args.max_fidelity_repairs == 1
    assert args.reconstruction_mode == "procedural"
    backend = create_backend(args.backend)
    assert backend.model == "gpt-5.6-sol"
    assert backend.reasoning_effort == "max"


def test_run_uses_manifest_backend_when_override_is_absent() -> None:
    args = build_parser().parse_args(["run", "workspace"])
    assert args.workspace == Path("workspace")
    assert args.backend is None
    assert args.reconstruction_mode is None


def test_make_accepts_reference_derived_mode() -> None:
    args = build_parser().parse_args(
        ["make", "image.png", "model.glb", "--mode", "reference-derived"]
    )
    assert args.reconstruction_mode == "reference-derived"


def test_long_running_commands_accept_no_progress() -> None:
    parser = build_parser()

    make = parser.parse_args(["make", "image.png", "model.glb", "--no-progress"])
    run = parser.parse_args(["run", "workspace", "--no-progress"])
    build = parser.parse_args(["build", "workspace", "--no-progress"])

    assert make.command == "make"
    assert run.command == "run"
    assert build.command == "build"


def test_make_routes_progress_to_stderr_and_can_disable_it(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    workspace = Workspace(tmp_path / "workspace")

    monkeypatch.setattr(cli.Workspace, "create", lambda **kwargs: workspace)

    def fake_pipeline(*args, **kwargs):
        emit_progress(kwargs.get("progress"), "info", "fake-pipeline", "Pipeline active")
        return {"status": "prepared"}

    monkeypatch.setattr(cli, "run_pipeline", fake_pipeline)
    parser = build_parser()

    args = parser.parse_args(
        ["make", "image.png", "model.glb", "--name", "fixture", "--prepare-only"]
    )
    assert cli._command_make(args) == 0
    captured = capsys.readouterr()
    assert "Pipeline active" in captured.err
    assert "Creating workspace" in captured.err
    assert "Status" in captured.out
    assert "Pipeline active" not in captured.out

    quiet_args = parser.parse_args(
        [
            "make",
            "image.png",
            "model.glb",
            "--name",
            "fixture",
            "--prepare-only",
            "--no-progress",
        ]
    )
    assert cli._command_make(quiet_args) == 0
    quiet = capsys.readouterr()
    assert quiet.err == ""
    assert "Status" in quiet.out


def test_redirected_build_summary_retains_comparison_json(tmp_path: Path, capsys) -> None:
    workspace = Workspace(tmp_path / "workspace")
    report = {"score": 0.75, "passed": True, "summary": {}}
    deliverables = {
        "program": "src/program.py",
        "plan": "src/plan.json",
        "glb": "artifacts/model.glb",
        "blend": "artifacts/scene.blend",
        "comparison": "artifacts/comparison.json",
        "build_manifest": "artifacts/build_manifest.json",
    }

    cli._build_summary(
        workspace,
        report,
        status="complete",
        deliverables=deliverables,
    )

    output = capsys.readouterr().out
    json_output, paths = output.split("\nBlender source:", maxsplit=1)
    assert json.loads(json_output) == report
    assert "compiled GLB:" in paths


def test_build_archives_against_the_matching_agent_trajectory(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    image = tmp_path / "reference.png"
    glb = tmp_path / "reference.glb"
    image.write_bytes(b"image")
    glb.write_bytes(b"glb")
    workspace = Workspace.create(
        base=tmp_path / "outputs",
        slug="fixture",
        image=image,
        glb=glb,
        prompt="",
        backend="codex",
    )
    workspace.program_path.write_text("def build():\n    pass\n", encoding="utf-8")
    workspace.plan_path.write_text("{}\n", encoding="utf-8")
    trajectory = workspace.trajectory_dir(0)
    (trajectory / "program.py").write_bytes(workspace.program_path.read_bytes())
    (trajectory / "plan.json").write_bytes(workspace.plan_path.read_bytes())

    monkeypatch.setattr(cli.Workspace, "locate", lambda *args, **kwargs: workspace)
    monkeypatch.setattr(
        cli.BlenderRuntime,
        "discover",
        lambda explicit=None: type("Runtime", (), {"executable": Path("/fake/blender")})(),
    )
    monkeypatch.setattr(cli, "prepare_reference", lambda *args, **kwargs: {})
    seen: list[Path | None] = []

    def fake_build(*args, **kwargs):
        seen.append(kwargs.get("trajectory_dir"))
        return {"score": 1.0, "passed": True, "summary": {}}

    monkeypatch.setattr(cli, "build_workspace", fake_build)
    args = build_parser().parse_args(["build", str(workspace.root), "--no-progress"])

    assert cli._command_build(args) == 0
    assert seen == [trajectory]
    assert "compiled GLB:" in capsys.readouterr().out


def test_cursor_and_grok_defaults_are_explicit() -> None:
    cursor = create_backend("cursor")
    grok = create_backend("grok")
    assert cursor.model == "cursor-grok-4.6-xhigh-fast"
    assert grok.model == "grok-4.6"
    assert grok.reasoning_effort == "xhigh"


def test_examples_scans_arbitrary_glb_names(tmp_path: Path, capsys) -> None:
    sample = tmp_path / "character"
    sample.mkdir()
    (sample / "reference.webp").write_bytes(b"image")
    (sample / "tripo-output.glb").write_bytes(b"glb")

    assert _command_examples(type("Args", (), {"root": tmp_path})()) == 0
    output = capsys.readouterr().out
    assert "tripo-output.glb" in output
    assert "reference.webp" in output


def test_examples_missing_root_is_an_actionable_error(tmp_path: Path) -> None:
    args = type("Args", (), {"root": tmp_path / "missing"})()
    try:
        _command_examples(args)
    except FileNotFoundError as exc:
        assert "pass --root" in str(exc)
    else:  # pragma: no cover - explicit failure message without another dependency
        raise AssertionError("missing root should fail")
