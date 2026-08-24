"""Command-line interface for the standalone ProcAgen3D application."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence

from rich.console import Console
from rich.text import Text

from . import __version__
from .backends import BACKEND_NAMES, create_backend
from .blender import BlenderError, BlenderRuntime
from .glb_probe import GLBProbeError, probe_glb
from .pipeline import PipelineConfig, PipelineError, build_workspace, prepare_reference, run_pipeline
from .process import run_process
from .progress import progress_step
from .rich_ui import RichProgressReporter, print_comparison, print_workspace_summary
from .workspace import Workspace, slugify, write_json


def _positive_int(value: str) -> int:
    converted = int(value)
    if converted <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return converted


def _render_size(value: str) -> int:
    converted = int(value)
    if not 64 <= converted <= 2048:
        raise argparse.ArgumentTypeError("must be between 64 and 2048")
    return converted


def _score(value: str) -> float:
    converted = float(value)
    if not 0.0 <= converted <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return converted


def _add_runtime_options(parser: argparse.ArgumentParser, *, backend_default: str | None) -> None:
    parser.add_argument("--backend", choices=BACKEND_NAMES, default=backend_default)
    parser.add_argument("--blender", type=Path, help="path to the Blender executable")
    parser.add_argument("--max-repairs", type=int, choices=range(0, 11), default=2)
    parser.add_argument("--min-score", type=_score, default=0.35)
    parser.add_argument("--render-size", type=_render_size, default=256)
    parser.add_argument("--llm-timeout", type=_positive_int, default=1800, metavar="SECONDS")
    parser.add_argument("--blender-timeout", type=_positive_int, default=900, metavar="SECONDS")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable Rich stage progress on stderr",
    )


def _config(args: argparse.Namespace, *, backend: str) -> PipelineConfig:
    return PipelineConfig(
        backend=backend,
        blender=args.blender,
        max_repairs=args.max_repairs,
        min_score=args.min_score,
        render_size=args.render_size,
        llm_timeout_s=args.llm_timeout,
        blender_timeout_s=args.blender_timeout,
    )


def _workspace_summary(workspace: Workspace, report: dict[str, Any]) -> None:
    print_workspace_summary(Console(), workspace=workspace.root, report=report)


def _build_summary(
    workspace: Workspace,
    report: dict[str, Any],
    *,
    status: str,
    deliverables: dict[str, str],
) -> None:
    console = Console()
    if console.is_terminal:
        print_comparison(console, report)
        print_workspace_summary(
            console,
            workspace=workspace.root,
            report={
                "status": status,
                "score": report["score"],
                "deliverables": deliverables,
            },
        )
        return
    # Preserve the pre-Rich redirected-output contract for scripts and CI.
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Blender source: {workspace.program_path}")
    print(f"compiled GLB: {workspace.artifacts_dir / 'model.glb'}")


def _progress_context(args: argparse.Namespace):
    if getattr(args, "no_progress", False):
        return nullcontext(None)
    return RichProgressReporter()


def _command_make(args: argparse.Namespace) -> int:
    slug = args.name or slugify(args.glb.expanduser().resolve().parent.name)
    with _progress_context(args) as progress:
        with progress_step(
            progress,
            "workspace-setup",
            "Creating workspace and copying input provenance",
        ) as stage:
            workspace = Workspace.create(
                base=args.output,
                slug=slug,
                image=args.image,
                glb=args.glb,
                prompt=args.prompt,
                backend=args.backend,
            )
            stage.complete(f"Workspace created — {workspace.root}")
        report = run_pipeline(
            workspace,
            _config(args, backend=args.backend),
            prepare_only=args.prepare_only,
            force_probe=args.force_probe,
            progress=progress,
        )
    _workspace_summary(workspace, report)
    return 0 if report["status"] in {"prepared", "complete"} else 2


def _command_run(args: argparse.Namespace) -> int:
    with _progress_context(args) as progress:
        with progress_step(
            progress,
            "workspace-load",
            "Loading workspace and verifying input provenance",
        ) as stage:
            workspace = Workspace.locate(args.workspace, base=args.output)
            backend = args.backend or str(workspace.manifest().get("backend") or "codex")
            stage.complete(f"Workspace ready — {workspace.root}")
        report = run_pipeline(
            workspace,
            _config(args, backend=backend),
            prepare_only=args.prepare_only,
            force_probe=args.force_probe,
            progress=progress,
        )
    _workspace_summary(workspace, report)
    return 0 if report["status"] in {"prepared", "complete"} else 2


def _command_build(args: argparse.Namespace) -> int:
    workspace: Workspace | None = None
    try:
        with _progress_context(args) as progress:
            with progress_step(
                progress,
                "workspace-load",
                "Loading workspace and verifying input provenance",
            ) as stage:
                workspace = Workspace.locate(args.workspace, base=args.output)
                stage.complete(f"Workspace ready — {workspace.root}")
            with progress_step(progress, "runtime", "Locating Blender") as stage:
                runtime = BlenderRuntime.discover(args.blender)
                stage.complete(f"Blender ready — {runtime.executable}")
            prepare_reference(
                workspace,
                runtime,
                render_size=args.render_size,
                timeout_s=args.blender_timeout,
                force=args.force_probe,
                progress=progress,
            )
            report = build_workspace(
                workspace,
                runtime,
                min_score=args.min_score,
                timeout_s=args.blender_timeout,
                progress=progress,
            )
    except (BlenderError, PipelineError, GLBProbeError, OSError, ValueError) as exc:
        if workspace is not None:
            workspace.update_manifest(status="failed")
            write_json(
                workspace.root / "run_report.json",
                {
                    "schema_version": 1,
                    "status": "failed",
                    "workspace": str(workspace.root),
                    "backend": None,
                    "stage": "build",
                    "error": str(exc),
                },
            )
        raise
    assert workspace is not None
    status = "complete" if report["passed"] else "needs-review"
    deliverables = {
        "program": "src/program.py",
        "plan": "src/plan.json",
        "glb": "artifacts/model.glb",
        "blend": "artifacts/scene.blend",
        "comparison": "artifacts/comparison.json",
        "build_manifest": "artifacts/build_manifest.json",
    }
    workspace.update_manifest(status=status, score=report["score"], deliverables=deliverables)
    write_json(
        workspace.root / "run_report.json",
        {
            "schema_version": 1,
            "status": status,
            "workspace": str(workspace.root),
            "backend": None,
            "score": report["score"],
            "passed": report["passed"],
            "deliverables": deliverables,
        },
    )
    _build_summary(
        workspace,
        report,
        status=status,
        deliverables=deliverables,
    )
    return 0 if report["passed"] else 2


def _command_probe(args: argparse.Namespace) -> int:
    report = probe_glb(args.glb)
    if args.out:
        write_json(args.out, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def _command_inspect(args: argparse.Namespace) -> int:
    workspace = Workspace.locate(args.workspace, base=args.output)
    manifest = workspace.manifest()
    run_report_path = workspace.root / "run_report.json"
    comparison_path = workspace.artifacts_dir / "comparison.json"
    value = {
        "workspace": str(workspace.root),
        "manifest": manifest,
        "deliverables": {
            "plan": workspace.plan_path.is_file(),
            "program": workspace.program_path.is_file(),
            "blend": (workspace.artifacts_dir / "scene.blend").is_file(),
            "glb": (workspace.artifacts_dir / "model.glb").is_file(),
        },
        "run_report": json.loads(run_report_path.read_text(encoding="utf-8"))
        if run_report_path.is_file()
        else None,
        "comparison": json.loads(comparison_path.read_text(encoding="utf-8"))
        if comparison_path.is_file()
        else None,
    }
    print(json.dumps(value, indent=2, ensure_ascii=False))
    return 0


def _version_result(command: list[str]) -> dict[str, Any]:
    executable = shutil.which(command[0])
    if not executable:
        return {"available": False, "command": command[0]}
    result = run_process([executable, *command[1:]], cwd=Path.cwd(), timeout_s=20)
    first_line = (result.stdout or result.stderr).strip().splitlines()
    return {
        "available": result.ok,
        "path": executable,
        "version": first_line[0] if first_line else None,
    }


def _command_doctor(args: argparse.Namespace) -> int:
    checks = {
        "python": {"available": True, "version": sys.version.split()[0], "path": sys.executable},
        "uv": _version_result(["uv", "--version"]),
        "codex": _version_result(["codex", "--version"]),
        "grok": _version_result(["grok", "--version"]),
        "cursor": _version_result(["cursor-agent", "--version"]),
    }
    try:
        runtime = BlenderRuntime.discover(args.blender)
        version = runtime.version()
        lines = (version.stdout or version.stderr).strip().splitlines()
        checks["blender"] = {
            "available": version.ok,
            "path": str(runtime.executable),
            "version": lines[0] if lines else None,
        }
    except Exception as exc:
        checks["blender"] = {"available": False, "error": str(exc)}
    checks["defaults"] = {
        name: {"model": create_backend(name).model}
        for name in BACKEND_NAMES
    }
    checks["defaults"]["codex"]["reasoning_effort"] = "max"
    checks["defaults"]["grok"].update(
        reasoning_effort="xhigh",
        fast_mode="not exposed by Grok Build CLI 1.0.5",
    )
    checks["defaults"]["cursor"]["mode"] = "Extra High Fast"
    print(json.dumps(checks, indent=2, ensure_ascii=False))
    has_agent = any(checks[name].get("available") for name in ("codex", "grok", "cursor"))
    return 0 if checks["blender"].get("available") and has_agent else 1


def _command_examples(args: argparse.Namespace) -> int:
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"example root not found: {root}; pass --root PATH to an image/GLB collection"
        )
    pairs = []
    for glb in sorted(root.rglob("*.glb")):
        images = sorted(
            path
            for suffix in ("*.png", "*.jpg", "*.jpeg", "*.webp")
            for path in glb.parent.glob(suffix)
        )
        if images:
            pairs.append({"name": glb.parent.name, "image": str(images[0]), "glb": str(glb)})
    print(json.dumps(pairs, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="procagen3d",
        description="Generate standalone Blender Python and compile it to GLB using image + GLB evidence.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    make = commands.add_parser("make", aliases=["generate"], help="create and execute a new workspace")
    make.add_argument("image", type=Path)
    make.add_argument("glb", type=Path)
    make.add_argument("--prompt", "-p", default="")
    make.add_argument("--name", help="workspace slug; defaults to the GLB parent directory")
    make.add_argument("--output", "-o", type=Path, default=Path("outputs"))
    make.add_argument("--prepare-only", action="store_true", help="build evidence without invoking an LLM")
    make.add_argument("--force-probe", action="store_true")
    _add_runtime_options(make, backend_default="codex")
    make.set_defaults(handler=_command_make)

    run = commands.add_parser("run", help="resume or repair an existing workspace")
    run.add_argument("workspace", type=Path)
    run.add_argument("--output", "-o", type=Path, default=Path("outputs"))
    run.add_argument("--prepare-only", action="store_true")
    run.add_argument("--force-probe", action="store_true")
    _add_runtime_options(run, backend_default=None)
    run.set_defaults(handler=_command_run)

    build = commands.add_parser("build", help="compile and verify the current src/program.py without an LLM")
    build.add_argument("workspace", type=Path)
    build.add_argument("--output", "-o", type=Path, default=Path("outputs"))
    build.add_argument("--blender", type=Path)
    build.add_argument("--min-score", type=_score, default=0.35)
    build.add_argument("--render-size", type=_render_size, default=256)
    build.add_argument("--blender-timeout", type=_positive_int, default=900)
    build.add_argument("--force-probe", action="store_true")
    build.add_argument(
        "--no-progress",
        action="store_true",
        help="disable Rich stage progress on stderr",
    )
    build.set_defaults(handler=_command_build)

    inspect = commands.add_parser("inspect", help="show workspace state and artifact metrics")
    inspect.add_argument("workspace", type=Path)
    inspect.add_argument("--output", "-o", type=Path, default=Path("outputs"))
    inspect.set_defaults(handler=_command_inspect)

    probe = commands.add_parser("probe", help="inspect a GLB container without Blender")
    probe.add_argument("glb", type=Path)
    probe.add_argument("--out", type=Path)
    probe.set_defaults(handler=_command_probe)

    doctor = commands.add_parser("doctor", help="check uv, Blender, and CLI-agent installations")
    doctor.add_argument("--blender", type=Path)
    doctor.set_defaults(handler=_command_doctor)

    examples = commands.add_parser("examples", help="list image + GLB pairs under a local root")
    examples.add_argument("--root", type=Path, default=Path("assets/3d_glb"))
    examples.set_defaults(handler=_command_examples)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        Console(stderr=True).print(Text("procagen3d: interrupted", style="bold yellow"))
        return 130
    except (
        BlenderError,
        PipelineError,
        GLBProbeError,
        OSError,
        ValueError,
    ) as exc:
        message = Text("procagen3d: error: ", style="bold red")
        message.append(str(exc))
        Console(stderr=True).print(message)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
