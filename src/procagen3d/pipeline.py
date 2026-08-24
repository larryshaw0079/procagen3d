"""GLB-guided agent → Blender source → compiled GLB pipeline."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import struct
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .backends import CLIBackend, create_backend
from .blender import BlenderError, BlenderRuntime, require_success
from .glb_probe import probe_glb
from .granularity import (
    DEFAULT_GRANULARITY,
    get_granularity_profile,
    validate_granularity,
)
from .metrics import SurfaceGateThresholds, compare_workspace, mask_metrics
from .plan_schema import PlanSchemaError, validate_plan_document
from .progress import ProgressReporter, emit_progress, progress_step
from .prompts import initial_prompt, repair_prompt
from .reconstruction import DEFAULT_RECONSTRUCTION_MODE, validate_reconstruction_mode
from .source_guard import SourceGuardError, assert_safe_source
from .workspace import Workspace, sha256, write_json


class PipelineError(RuntimeError):
    """A recoverable stage failure that prevents a valid deliverable."""


@dataclass(frozen=True)
class PipelineConfig:
    backend: str = "codex"
    blender: Path | None = None
    max_repairs: int = 2
    max_fidelity_repairs: int = 1
    min_score: float = 0.35
    reconstruction_mode: str = DEFAULT_RECONSTRUCTION_MODE
    granularity: str = DEFAULT_GRANULARITY
    render_size: int = 256
    llm_timeout_s: int = 1800
    blender_timeout_s: int = 900


CANONICAL_VIEWS = ("front", "back", "left", "right", "top", "iso")


def _replace_directory(staged: Path, target: Path) -> None:
    """Atomically promote a complete staged directory on the same filesystem."""

    backup = target.parent / f".{target.name}.backup-{uuid.uuid4().hex}"
    had_target = target.exists()
    if had_target:
        target.replace(backup)
    try:
        staged.replace(target)
    except Exception:
        if had_target and backup.exists():
            backup.replace(target)
        raise
    if backup.exists():
        # Promotion is already committed. A stale backup is preferable to
        # reporting failure after the new target became live (notably when an
        # antivirus/indexer briefly locks files on Windows).
        shutil.rmtree(backup, ignore_errors=True)


def _validate_positive_runtime(*, render_size: int, timeout_s: int) -> None:
    if not 64 <= render_size <= 2048:
        raise ValueError("render_size must be between 64 and 2048")
    if timeout_s <= 0:
        raise ValueError("timeout must be greater than zero")


def _complete_evidence(root: Path, *, render_size: int) -> bool:
    expected = [
        root / "glb_probe.json",
        root / "reference_scene.json",
        root / "camera_contract.json",
        root / "reference_views" / "masks.json",
        *(root / "reference_views" / f"{name}.png" for name in CANONICAL_VIEWS),
    ]
    if not all(not path.is_symlink() and path.is_file() for path in expected):
        return False
    try:
        glb_report = json.loads((root / "glb_probe.json").read_text(encoding="utf-8"))
        scene = json.loads((root / "reference_scene.json").read_text(encoding="utf-8"))
        camera = json.loads((root / "camera_contract.json").read_text(encoding="utf-8"))
        masks = json.loads((root / "reference_views" / "masks.json").read_text(encoding="utf-8"))
        if not all(isinstance(value, dict) for value in (glb_report, scene, camera, masks)):
            return False
        if not glb_report.get("self_contained") or glb_report.get("reference_readiness") != "pass":
            return False
        geometry_count = scene.get("geometry_object_count")
        bounds = scene.get("bounds")
        if type(geometry_count) is not int or geometry_count <= 0 or not isinstance(bounds, dict):
            return False
        for field in ("min", "max", "dimensions", "center"):
            vector = bounds.get(field)
            if not (
                isinstance(vector, list)
                and len(vector) == 3
                and all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                    for value in vector
                )
            ):
                return False

        camera_views = camera.get("views")
        mask_views = masks.get("views")
        if not isinstance(camera_views, list) or not isinstance(mask_views, dict):
            return False
        if masks.get("schema_version") != 2:
            return False
        if camera.get("resolution") != [render_size, render_size]:
            return False
        if [view.get("name") for view in camera_views if isinstance(view, dict)] != list(
            CANONICAL_VIEWS
        ):
            return False
        for view in camera_views:
            if not isinstance(view, dict):
                return False
            scale = view.get("ortho_scale")
            if (
                not isinstance(scale, (int, float))
                or isinstance(scale, bool)
                or not math.isfinite(scale)
                or scale <= 0.0
            ):
                return False
            for field in ("location", "target"):
                vector = view.get(field)
                if not (
                    isinstance(vector, list)
                    and len(vector) == 3
                    and all(
                        isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and math.isfinite(value)
                        for value in vector
                    )
                ):
                    return False
        if set(mask_views) != set(CANONICAL_VIEWS):
            return False
        for name in CANONICAL_VIEWS:
            record = mask_views[name]
            if not isinstance(record, dict):
                return False
            if record.get("width") != render_size or record.get("height") != render_size:
                return False
            mask_metrics(record, record)
            header = (root / "reference_views" / f"{name}.png").read_bytes()[:24]
            if (
                len(header) != 24
                or header[:8] != b"\x89PNG\r\n\x1a\n"
                or header[12:16] != b"IHDR"
                or struct.unpack(">II", header[16:24]) != (render_size, render_size)
            ):
                return False
        return True
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError, struct.error):
        return False


def _verified_glb_snapshot(workspace: Workspace) -> tuple[bytes, str]:
    path = workspace.glb_path
    value = path.read_bytes()
    digest = hashlib.sha256(value).hexdigest()
    manifest = workspace.manifest()
    expected = manifest["inputs"]["glb"]["sha256"]
    if digest != expected:
        raise PipelineError("reference GLB changed while it was being snapshotted")
    return value, digest


def prepare_reference(
    workspace: Workspace,
    runtime: BlenderRuntime,
    *,
    render_size: int = 256,
    timeout_s: int = 900,
    force: bool = False,
    progress: ProgressReporter | None = None,
) -> dict[str, Any]:
    """Validate, measure, normalize, and render the evidence GLB."""

    _validate_positive_runtime(render_size=render_size, timeout_s=timeout_s)
    with progress_step(
        progress,
        "reference-input",
        "Validating reference GLB provenance",
    ) as stage:
        manifest = workspace.manifest()
        source_glb, glb_digest = _verified_glb_snapshot(workspace)
        stage.complete(
            f"Reference GLB verified — {len(source_glb) / (1024 * 1024):.1f} MiB",
            glb_sha256=glb_digest,
        )
    cached = manifest.get("reference") if isinstance(manifest.get("reference"), dict) else {}
    cache_matches = (
        cached.get("glb_sha256") == glb_digest
        and cached.get("render_size") == render_size
    )
    cache_complete = False
    if not force and cache_matches:
        with progress_step(
            progress,
            "reference-cache",
            f"Validating cached {render_size}×{render_size} reference evidence",
        ) as stage:
            cache_complete = _complete_evidence(
                workspace.evidence_dir,
                render_size=render_size,
            )
            if cache_complete:
                stage.complete(
                    f"Cached {render_size}×{render_size} reference evidence is complete"
                )
            else:
                stage.complete(
                    "Cached reference evidence is incomplete; rebuilding it",
                    kind="warning",
                )
    if cache_complete:
        return json.loads((workspace.evidence_dir / "glb_probe.json").read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory(prefix=".procagen3d-evidence-", dir=workspace.root) as directory:
        stage_root = Path(directory)
        staged_glb = stage_root / "reference.glb"
        staged_glb.write_bytes(source_glb)
        with progress_step(
            progress,
            "reference-probe",
            "Inspecting GLB container, accessors, and semantic boundaries",
        ) as stage:
            report = probe_glb(staged_glb)
            if report["reference_readiness"] != "pass":
                raise PipelineError("the reference GLB is not a self-contained, drawable scene")
            scene = report.get("scene", {})
            stage.complete(
                "Reference geometry inspected — "
                f"{int(scene.get('vertex_count', 0)):,} vertices, "
                f"{int(scene.get('triangle_count', 0)):,} triangles",
                semantic_status=report.get("semantic_decomposition", {}).get("status"),
            )
        report["path"] = "inputs/reference.glb"
        staged_evidence = stage_root / "evidence"
        staged_evidence.mkdir()
        write_json(staged_evidence / "glb_probe.json", report)
        with progress_step(
            progress,
            "reference-render",
            f"Normalizing the GLB and rendering six {render_size}×{render_size} reference views",
        ) as stage:
            result = runtime.run_stage(
                "reference_probe",
                [
                    "--glb",
                    staged_glb,
                    "--evidence-dir",
                    staged_evidence,
                    "--size",
                    str(render_size),
                ],
                cwd=stage_root,
                timeout_s=timeout_s,
            )
            (staged_evidence / "reference_probe.stdout.log").write_text(
                result.stdout, encoding="utf-8"
            )
            (staged_evidence / "reference_probe.stderr.log").write_text(
                result.stderr, encoding="utf-8"
            )
            if not result.ok:
                failure = workspace.root / "trajectories" / "reference_probe_failure"
                failure.mkdir(parents=True, exist_ok=True)
                (failure / "stdout.log").write_text(result.stdout, encoding="utf-8")
                (failure / "stderr.log").write_text(result.stderr, encoding="utf-8")
            require_success(result, stage="reference probe")
            scene_path = staged_evidence / "reference_scene.json"
            scene_report = json.loads(scene_path.read_text(encoding="utf-8"))
            if not isinstance(scene_report, dict):
                raise PipelineError("Blender reference_scene.json must contain an object")
            scene_report["source"] = "inputs/reference.glb"
            write_json(scene_path, scene_report)
            if not _complete_evidence(staged_evidence, render_size=render_size):
                raise PipelineError(
                    "Blender reference probe produced an incomplete canonical evidence set"
                )
            stage.complete(
                "Reference normalized and six canonical views rendered",
                blender_duration_s=getattr(result, "duration_s", None),
            )
        current_glb, current_digest = _verified_glb_snapshot(workspace)
        if current_digest != glb_digest or current_glb != source_glb:
            raise PipelineError("reference GLB changed during evidence generation")
        _replace_directory(staged_evidence, workspace.evidence_dir)
    workspace.update_manifest(
        status="prepared",
        reference={
            "glb_sha256": glb_digest,
            "render_size": render_size,
            "scene": "evidence/reference_scene.json",
            "views": "evidence/reference_views",
        },
    )
    return report


def _validate_plan(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PipelineError("agent did not create src/plan.json")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-standard numeric constant {token!r}")
            ),
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise PipelineError(f"src/plan.json is invalid JSON: {exc}") from exc
    try:
        return validate_plan_document(value)
    except PlanSchemaError as exc:
        raise PipelineError(f"src/plan.json {exc}") from exc


def _guard_program(path: Path) -> None:
    if not path.is_file():
        raise PipelineError("agent did not create src/program.py")
    source = path.read_text(encoding="utf-8")
    try:
        assert_safe_source(source, filename="src/program.py")
    except SourceGuardError as exc:
        raise PipelineError(str(exc)) from exc


def _source_snapshot(path: Path, *, label: str) -> bytes:
    try:
        canonical_parent = path.parent.resolve(strict=True)
    except (OSError, RuntimeError):
        canonical_parent = None
    if (
        canonical_parent != path.parent
        or path.parent.is_symlink()
        or path.is_symlink()
        or not path.is_file()
    ):
        raise PipelineError(f"{label} must be a regular, non-symlink file")
    value = path.read_bytes()
    if len(value) > 2_000_000:
        raise PipelineError(f"{label} is unexpectedly large")
    return value


def _agent_images(root: Path, *, image_relative: Path, include_candidate: bool) -> tuple[Path, ...]:
    paths = [root / image_relative]
    paths.extend(root / "evidence" / "reference_views" / f"{name}.png" for name in CANONICAL_VIEWS)
    if include_candidate:
        paths.extend(root / "artifacts" / "renders" / f"{name}.png" for name in CANONICAL_VIEWS)
    return tuple(path for path in paths if path.is_file())


def _copy_agent_transcript(result: Any, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for source in (
        result.prompt_path,
        result.transcript_path,
        result.stderr_path,
        result.result_path,
        result.prompt_path.parent / "final_message.txt",
    ):
        if source.is_file():
            shutil.copy2(source, destination / source.name)


def _write_bytes_atomic(path: Path, value: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".restore-tmp")
    temporary.write_bytes(value)
    temporary.replace(path)


def _matching_source_iteration(workspace: Workspace) -> int | None:
    """Find the newest trajectory whose reviewed source is currently active."""

    if not workspace.program_path.is_file() or not workspace.plan_path.is_file():
        return None
    try:
        program = workspace.program_path.read_bytes()
        plan = workspace.plan_path.read_bytes()
    except OSError:
        return None
    for iteration in range(workspace.next_trajectory_iteration() - 1, -1, -1):
        directory = workspace.root / "trajectories" / f"iter_{iteration:02d}"
        candidate_program = directory / "program.py"
        candidate_plan = directory / "plan.json"
        if (
            directory.is_symlink()
            or not directory.is_dir()
            or candidate_program.is_symlink()
            or candidate_plan.is_symlink()
        ):
            continue
        try:
            if (
                candidate_program.read_bytes() == program
                and candidate_plan.read_bytes() == plan
            ):
                return iteration
        except OSError:
            continue
    return None


def matching_source_trajectory(workspace: Workspace) -> Path | None:
    """Return the newest regular trajectory owning the active source snapshot."""

    iteration = _matching_source_iteration(workspace)
    if iteration is None:
        return None
    return workspace.root / "trajectories" / f"iter_{iteration:02d}"


def _validated_source_trajectory(
    workspace: Workspace,
    trajectory: Path,
    *,
    program_snapshot: bytes,
    plan_snapshot: bytes,
) -> Path:
    """Validate that a trajectory owns the exact source being compiled."""

    trajectory = trajectory.expanduser()
    try:
        canonical = trajectory.resolve(strict=True)
        trajectories_root = (workspace.root / "trajectories").resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise PipelineError("source trajectory is unavailable") from exc
    suffix = canonical.name.removeprefix("iter_")
    if (
        canonical != trajectory
        or trajectory.is_symlink()
        or not trajectory.is_dir()
        or canonical.parent != trajectories_root
        or not suffix.isdigit()
    ):
        raise PipelineError("source trajectory must be a regular iter_XX directory")
    if (
        _source_snapshot(canonical / "program.py", label="trajectory program.py")
        != program_snapshot
        or _source_snapshot(canonical / "plan.json", label="trajectory plan.json")
        != plan_snapshot
    ):
        raise PipelineError("source trajectory does not match the compiled source snapshots")
    return canonical


def _archive_iteration_glb(source: Path, trajectory: Path) -> Path:
    """Atomically retain one fully validated intermediate GLB without clobbering it."""

    if not source.is_file() or source.is_symlink():
        raise PipelineError("validated model.glb is unavailable for trajectory archival")
    destination = trajectory / "model.glb"
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_file():
            raise PipelineError("trajectory model.glb must be a regular, non-symlink file")
        if sha256(destination) == sha256(source):
            return destination
        raise PipelineError("trajectory already contains a different model.glb")
    temporary = trajectory / f".model.glb.tmp-{uuid.uuid4().hex}"
    try:
        shutil.copy2(source, temporary)
        try:
            # A hard-link publication is atomic and refuses to replace a file
            # created by a concurrent build between validation and commit.
            os.link(temporary, destination)
        except FileExistsError:
            if (
                destination.is_symlink()
                or not destination.is_file()
                or sha256(destination) != sha256(source)
            ):
                raise PipelineError("trajectory concurrently acquired a different model.glb")
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
    return destination


def _invoke_agent(
    workspace: Workspace,
    *,
    backend_name: str,
    prompt: str,
    iteration: int,
    timeout_s: int,
    include_candidate: bool = False,
    is_repair: bool = False,
    progress: ProgressReporter | None = None,
) -> dict[str, Any]:
    backend = create_backend(backend_name)
    if timeout_s <= 0:
        raise ValueError("LLM timeout must be greater than zero")
    action = "Repairing" if is_repair else "Authoring"
    configured_model = getattr(backend, "model", None)
    model_label = f" ({configured_model})" if configured_model else ""
    stage_name = f"agent-{iteration:02d}"
    with progress_step(
        progress,
        stage_name,
        f"{action} Blender source with {backend_name}{model_label}",
    ) as stage:
        trajectory = workspace.trajectory_dir(iteration)
        image_relative = workspace.image_path.relative_to(workspace.root)
        with tempfile.TemporaryDirectory(prefix="procagen3d-agent-") as directory:
            # macOS exposes the same temporary directory through both /var and
            # /private/var. Backends canonicalize their workspace before they
            # report modified files, so keep this trust boundary canonical too.
            agent_root = Path(directory).resolve(strict=True)
            shutil.copytree(workspace.root / "inputs", agent_root / "inputs")
            shutil.copytree(workspace.evidence_dir, agent_root / "evidence")
            (agent_root / "src").mkdir()
            for current in (workspace.plan_path, workspace.program_path):
                if current.is_file():
                    shutil.copy2(current, agent_root / "src" / current.name)
            if include_candidate and workspace.artifacts_dir.is_dir():
                render_source = workspace.artifacts_dir / "renders"
                if render_source.is_dir():
                    shutil.copytree(render_source, agent_root / "artifacts" / "renders")
            agent_trajectory = agent_root / ".trajectory"
            run_arguments: dict[str, Any] = {
                "prompt": prompt,
                "workspace": agent_root,
                "trajectory_dir": agent_trajectory,
                "image_paths": _agent_images(
                    agent_root,
                    image_relative=image_relative,
                    include_candidate=include_candidate,
                ),
                "timeout_s": timeout_s,
            }
            if isinstance(backend, CLIBackend) and progress is not None:
                run_arguments["on_activity"] = lambda message: emit_progress(
                    progress,
                    "info",
                    stage_name,
                    message,
                )
            result = backend.run(
                **run_arguments,
            )
            _copy_agent_transcript(result, trajectory)
            if not result.ok:
                detail = (
                    result.error
                    or result.stderr[-2000:]
                    or result.final_message
                    or result.exit_reason
                )
                raise PipelineError(
                    f"{backend_name} agent failed ({result.exit_reason}): {detail}"
                )
            staged_src = agent_root / "src"
            try:
                canonical_src = staged_src.resolve(strict=True)
            except (OSError, RuntimeError):
                canonical_src = None
            if (
                canonical_src != staged_src
                or staged_src.is_symlink()
                or not staged_src.is_dir()
            ):
                raise PipelineError(
                    "agent src/ must remain a regular, non-symlink directory "
                    "inside the disposable workspace"
                )
            changed_relative: list[str] = []
            unauthorized: list[str] = []
            for path in result.files_modified:
                try:
                    candidate = Path(path).expanduser()
                    if not candidate.is_absolute():
                        candidate = agent_root / candidate
                    canonical_path = candidate.resolve(strict=False)
                    relative = canonical_path.relative_to(agent_root)
                except (OSError, RuntimeError, ValueError):
                    unauthorized.append(str(path))
                    continue
                changed_relative.append(relative.as_posix())
                if not relative.parts or relative.parts[0] != "src":
                    unauthorized.append(relative.as_posix())
            staged_program = staged_src / "program.py"
            staged_plan = staged_src / "plan.json"
            if unauthorized:
                # Keep bounded, regular candidates for forensic review after an
                # expensive rejected run, but never promote them into src/.
                for source, label, destination in (
                    (
                        staged_program,
                        "agent src/program.py",
                        trajectory / "rejected_program.py",
                    ),
                    (
                        staged_plan,
                        "agent src/plan.json",
                        trajectory / "rejected_plan.json",
                    ),
                ):
                    try:
                        candidate_value = _source_snapshot(source, label=label)
                    except PipelineError:
                        continue
                    destination.write_bytes(candidate_value)
                raise PipelineError(
                    "agent changed files outside src/: " + ", ".join(unauthorized)
                )
            # Snapshot both required deliverables before any source promotion.
            # The plan and AST contracts are checked in source-guard.
            program_value = _source_snapshot(staged_program, label="agent src/program.py")
            plan_value = _source_snapshot(staged_plan, label="agent src/plan.json")
            # Preserve the exact reviewed candidates before committing them to src/.
            # A trajectory-write failure must not leave source changed while the
            # invocation reports an error.
            (trajectory / "program.py").write_bytes(program_value)
            (trajectory / "plan.json").write_bytes(plan_value)
            with tempfile.TemporaryDirectory(
                prefix=".procagen3d-src-", dir=workspace.root
            ) as source_stage:
                promoted_src = Path(source_stage) / "src"
                promoted_src.mkdir()
                (promoted_src / "program.py").write_bytes(program_value)
                (promoted_src / "plan.json").write_bytes(plan_value)
                _replace_directory(promoted_src, workspace.src_dir)
            payload = {
                "backend": result.backend,
                "model": result.model,
                "duration_s": result.duration_s,
                "usage": result.usage,
                "files_modified": changed_relative,
            }
            stage.complete(
                f"{backend_name} produced plan.json and program.py",
                model=result.model,
                files_modified=changed_relative,
                provider_duration_s=result.duration_s,
            )
    return payload


def build_workspace(
    workspace: Workspace,
    runtime: BlenderRuntime,
    *,
    min_score: float = 0.35,
    reconstruction_mode: str = DEFAULT_RECONSTRUCTION_MODE,
    granularity: str = DEFAULT_GRANULARITY,
    timeout_s: int = 900,
    trajectory_dir: Path | None = None,
    progress: ProgressReporter | None = None,
) -> dict[str, Any]:
    """Compile the current source in a clean directory and compare it."""

    if not 0.0 <= min_score <= 1.0:
        raise ValueError("min_score must be between 0 and 1")
    reconstruction_mode = validate_reconstruction_mode(reconstruction_mode)
    granularity = validate_granularity(granularity)
    granularity_profile = get_granularity_profile(granularity)
    if timeout_s <= 0:
        raise ValueError("Blender timeout must be greater than zero")
    reference_snapshot: bytes | None = None
    reference_digest: str | None = None
    if reconstruction_mode == "glb-ref" or granularity_profile.surface_evaluation_enabled:
        reference_snapshot, reference_digest = _verified_glb_snapshot(workspace)
    source_trajectory: Path | None = None
    with progress_step(
        progress,
        "source-validation",
        "Snapshotting plan.json and program.py",
    ) as stage:
        program_snapshot = _source_snapshot(workspace.program_path, label="src/program.py")
        plan_snapshot = _source_snapshot(workspace.plan_path, label="src/plan.json")
        program_digest = hashlib.sha256(program_snapshot).hexdigest()
        plan_digest = hashlib.sha256(plan_snapshot).hexdigest()
        if trajectory_dir is not None:
            source_trajectory = _validated_source_trajectory(
                workspace,
                trajectory_dir,
                program_snapshot=program_snapshot,
                plan_snapshot=plan_snapshot,
            )
        stage.complete(
            f"Source snapshots captured — program {program_digest[:10]}",
            program_sha256=program_digest,
            plan_sha256=plan_digest,
        )
    with tempfile.TemporaryDirectory(prefix=".procagen3d-build-", dir=workspace.root) as directory:
        clean_root = Path(directory)
        staged_program = clean_root / "program.py"
        staged_plan = clean_root / "plan.json"
        staged_reference = clean_root / "reference.glb"
        staged_artifacts = clean_root / "artifacts"
        staged_program.write_bytes(program_snapshot)
        staged_plan.write_bytes(plan_snapshot)
        if reference_snapshot is not None:
            staged_reference.write_bytes(reference_snapshot)
        with progress_step(
            progress,
            "source-guard",
            "Applying the plan schema and Blender source guard",
        ) as stage:
            validated_plan = _validate_plan(staged_plan)
            if validated_plan.get("reconstruction_mode") != reconstruction_mode:
                raise PipelineError(
                    "src/plan.json reconstruction_mode must match the build mode: "
                    f"expected {reconstruction_mode!r}, found "
                    f"{validated_plan.get('reconstruction_mode')!r}"
                )
            if validated_plan.get("granularity") != granularity:
                raise PipelineError(
                    "src/plan.json granularity must match the build granularity: "
                    f"expected {granularity!r}, found "
                    f"{validated_plan.get('granularity')!r}"
                )
            _guard_program(staged_program)
            stage.complete("Plan schema and Blender source guard passed")
        with progress_step(
            progress,
            "blender-build",
            "Executing program.py in factory Blender and exporting model.glb",
        ) as stage:
            build_arguments = [
                "--program",
                staged_program,
                "--artifacts-dir",
                staged_artifacts,
                "--mode",
                reconstruction_mode,
                "--granularity",
                granularity,
            ]
            if reconstruction_mode == "glb-ref":
                build_arguments.extend(["--reference-glb", staged_reference])
            result = runtime.run_stage(
                "build_asset",
                build_arguments,
                cwd=clean_root,
                timeout_s=timeout_s,
            )
            staged_artifacts.mkdir(parents=True, exist_ok=True)
            (staged_artifacts / "build.stdout.log").write_text(
                result.stdout, encoding="utf-8"
            )
            (staged_artifacts / "build.stderr.log").write_text(
                result.stderr, encoding="utf-8"
            )
            if not result.ok:
                failure = workspace.root / "trajectories" / "build_failure"
                failure.mkdir(parents=True, exist_ok=True)
                (failure / "stdout.log").write_text(result.stdout, encoding="utf-8")
                (failure / "stderr.log").write_text(result.stderr, encoding="utf-8")
            require_success(result, stage="source build")
            stage.complete(
                "Blender source executed; scene.blend and model.glb exported",
                blender_duration_s=getattr(result, "duration_s", None),
            )
        model_path = staged_artifacts / "model.glb"
        with progress_step(
            progress,
            "export-validation",
            "Checking the exported GLB container and embedded resources",
        ) as stage:
            model_probe = probe_glb(model_path)
            if (
                not model_probe["self_contained"]
                or model_probe["reference_readiness"] != "pass"
            ):
                raise PipelineError("compiled model.glb is not a self-contained drawable scene")
            model_scene = model_probe.get("scene", {})
            stage.complete(
                "Exported GLB is self-contained — "
                f"{int(model_scene.get('mesh_count', 0)):,} meshes, "
                f"{int(model_scene.get('triangle_count', 0)):,} triangles",
            )
        with progress_step(
            progress,
            "compiled-probe",
            "Re-importing model.glb in a second Blender process and rendering six views",
        ) as stage:
            compiled_result = runtime.run_stage(
                "compiled_probe",
                [
                    "--glb",
                    model_path,
                    "--artifacts-dir",
                    staged_artifacts,
                    "--camera-contract",
                    workspace.evidence_dir / "camera_contract.json",
                ],
                cwd=clean_root,
                timeout_s=timeout_s,
            )
            (staged_artifacts / "compiled_probe.stdout.log").write_text(
                compiled_result.stdout, encoding="utf-8"
            )
            (staged_artifacts / "compiled_probe.stderr.log").write_text(
                compiled_result.stderr, encoding="utf-8"
            )
            if not compiled_result.ok:
                failure = workspace.root / "trajectories" / "compiled_probe_failure"
                failure.mkdir(parents=True, exist_ok=True)
                (failure / "build.stdout.log").write_text(result.stdout, encoding="utf-8")
                (failure / "build.stderr.log").write_text(result.stderr, encoding="utf-8")
                (failure / "stdout.log").write_text(
                    compiled_result.stdout, encoding="utf-8"
                )
                (failure / "stderr.log").write_text(
                    compiled_result.stderr, encoding="utf-8"
                )
            require_success(compiled_result, stage="compiled GLB probe")
            stage.complete(
                "Compiled GLB re-imported and six canonical views rendered",
                blender_duration_s=getattr(compiled_result, "duration_s", None),
            )
        surface_comparison_path: Path | None = None
        if granularity_profile.surface_evaluation_enabled:
            surface_comparison_path = staged_artifacts / "surface_comparison.json"
            with progress_step(
                progress,
                "surface-fidelity",
                "Measuring bidirectional 3D surface distance",
            ) as stage:
                surface_result = runtime.run_stage(
                    "surface_compare",
                    [
                        "--reference-glb",
                        staged_reference,
                        "--candidate-glb",
                        model_path,
                        "--output",
                        surface_comparison_path,
                        "--samples",
                        str(granularity_profile.surface_sample_budget),
                    ],
                    cwd=clean_root,
                    timeout_s=timeout_s,
                )
                (staged_artifacts / "surface_compare.stdout.log").write_text(
                    surface_result.stdout, encoding="utf-8"
                )
                (staged_artifacts / "surface_compare.stderr.log").write_text(
                    surface_result.stderr, encoding="utf-8"
                )
                if not surface_result.ok:
                    failure = workspace.root / "trajectories" / "surface_compare_failure"
                    failure.mkdir(parents=True, exist_ok=True)
                    (failure / "stdout.log").write_text(
                        surface_result.stdout, encoding="utf-8"
                    )
                    (failure / "stderr.log").write_text(
                        surface_result.stderr, encoding="utf-8"
                    )
                require_success(surface_result, stage="surface comparison")
                if not surface_comparison_path.is_file():
                    raise PipelineError(
                        "Blender surface comparison did not produce its JSON report"
                    )
                surface_report = json.loads(
                    surface_comparison_path.read_text(encoding="utf-8")
                )
                if not isinstance(surface_report, dict):
                    raise PipelineError("surface_comparison.json must contain an object")
                surface_report.update(
                    granularity=granularity,
                    reference="inputs/reference.glb",
                    candidate="artifacts/model.glb",
                )
                write_json(surface_comparison_path, surface_report)
                symmetric = surface_report.get("symmetric")
                if not isinstance(symmetric, dict):
                    raise PipelineError(
                        "surface_comparison.json must contain symmetric statistics"
                    )
                try:
                    mean_surface_distance = float(symmetric.get("mean"))
                    p95_surface_distance = float(symmetric.get("p95"))
                except (TypeError, ValueError, OverflowError) as exc:
                    raise PipelineError(
                        "surface_comparison.json has invalid symmetric distances"
                    ) from exc
                if (
                    not math.isfinite(mean_surface_distance)
                    or mean_surface_distance < 0.0
                    or not math.isfinite(p95_surface_distance)
                    or p95_surface_distance < 0.0
                ):
                    raise PipelineError(
                        "surface_comparison.json has invalid symmetric distances"
                    )
                stage.complete(
                    "Surface distance measured — "
                    f"mean {mean_surface_distance:.5f}, "
                    f"p95 {p95_surface_distance:.5f}",
                    blender_duration_s=getattr(surface_result, "duration_s", None),
                    mean_surface_distance=mean_surface_distance,
                    p95_surface_distance=p95_surface_distance,
                )
        with progress_step(
            progress,
            "artifact-provenance",
            "Recording GLB, Blender scene, and report provenance",
        ) as stage:
            # Reports survive promotion out of the disposable build directory,
            # so keep their provenance paths stable and workspace-relative.
            model_probe["path"] = "artifacts/model.glb"
            write_json(staged_artifacts / "model_probe.json", model_probe)
            scene_report_path = staged_artifacts / "scene_report.json"
            scene_report = json.loads(scene_report_path.read_text(encoding="utf-8"))
            if not isinstance(scene_report, dict):
                raise PipelineError("Blender scene_report.json must contain an object")
            scene_report["program"] = "src/program.py"
            write_json(scene_report_path, scene_report)
            model_digest = sha256(model_path)
            blend_digest = sha256(staged_artifacts / "scene.blend")
            write_json(
                staged_artifacts / "build_manifest.json",
                {
                    "schema_version": 1,
                    "program_sha256": program_digest,
                    "plan_sha256": plan_digest,
                    "model_sha256": model_digest,
                    "blend_sha256": blend_digest,
                    "clean_room": True,
                    "reconstruction_mode": reconstruction_mode,
                    "granularity": granularity,
                    "program_is_standalone_replay_source": (
                        reconstruction_mode == "procedural"
                    ),
                    "replay_source_of_truth": (
                        ["src/program.py", "inputs/reference.glb", "host-reference-preload-v1"]
                        if reconstruction_mode == "glb-ref"
                        else ["src/program.py"]
                    ),
                    "source_glb_imported_at_build_time": (
                        reconstruction_mode == "glb-ref"
                    ),
                    "reference_glb_sha256": reference_digest,
                    "reference_glb_used_for_surface_evaluation": (
                        granularity_profile.surface_evaluation_enabled
                    ),
                    "reference_contract": (
                        "host-imported-normalized-source; originals removed before export"
                        if reconstruction_mode == "glb-ref"
                        else (
                            "measurement-and-host-surface-evaluation-evidence-only"
                            if granularity_profile.surface_evaluation_enabled
                            else "measurement-evidence-only"
                        )
                    ),
                    "compiled_glb_verified_in_separate_process": True,
                    "surface_distance_evaluation": (
                        {
                            "report": "artifacts/surface_comparison.json",
                            "samples_per_direction": (
                                granularity_profile.surface_sample_budget
                            ),
                            "max_mean_distance": (
                                granularity_profile.max_mean_surface_distance
                            ),
                            "max_p95_distance": (
                                granularity_profile.max_p95_surface_distance
                            ),
                        }
                        if granularity_profile.surface_evaluation_enabled
                        else None
                    ),
                },
            )
            stage.complete(
                f"Artifact provenance recorded — GLB {model_digest[:10]}",
                model_sha256=model_digest,
                blend_sha256=blend_digest,
            )
        if source_trajectory is not None:
            if (
                _source_snapshot(workspace.program_path, label="src/program.py")
                != program_snapshot
                or _source_snapshot(workspace.plan_path, label="src/plan.json")
                != plan_snapshot
            ):
                raise PipelineError("src/program.py or src/plan.json changed during the build")
            # Blender may have run for several minutes since the first trust
            # check. Re-resolve the trajectory immediately before publication.
            source_trajectory = _validated_source_trajectory(
                workspace,
                source_trajectory,
                program_snapshot=program_snapshot,
                plan_snapshot=plan_snapshot,
            )
            relative_glb = (
                source_trajectory / "model.glb"
            ).relative_to(workspace.root).as_posix()
            with progress_step(
                progress,
                "trajectory-glb",
                f"Saving validated iteration GLB to {relative_glb}",
            ) as stage:
                archived = _archive_iteration_glb(model_path, source_trajectory)
                stage.complete(
                    f"Saved intermediate GLB — {archived.relative_to(workspace.root).as_posix()}",
                    model_sha256=model_digest,
                )
        with progress_step(
            progress,
            "fidelity",
            "Scoring renders, dimensions, centering, and configured surface gates",
        ) as stage:
            surface_thresholds = None
            if granularity_profile.surface_evaluation_enabled:
                assert granularity_profile.max_mean_surface_distance is not None
                assert granularity_profile.max_p95_surface_distance is not None
                surface_thresholds = SurfaceGateThresholds(
                    max_mean_surface_distance=(
                        granularity_profile.max_mean_surface_distance
                    ),
                    max_p95_surface_distance=(
                        granularity_profile.max_p95_surface_distance
                    ),
                )
            comparison = compare_workspace(
                reference_masks=workspace.evidence_dir / "reference_views" / "masks.json",
                candidate_masks=staged_artifacts / "renders" / "masks.json",
                reference_scene=workspace.evidence_dir / "reference_scene.json",
                candidate_scene=staged_artifacts / "scene_report.json",
                output=staged_artifacts / "comparison.json",
                min_score=min_score,
                surface_comparison=surface_comparison_path,
                surface_gate_thresholds=surface_thresholds,
            )
            comparison["granularity"] = granularity
            write_json(staged_artifacts / "comparison.json", comparison)
            if (
                _source_snapshot(workspace.program_path, label="src/program.py")
                != program_snapshot
                or _source_snapshot(workspace.plan_path, label="src/plan.json")
                != plan_snapshot
            ):
                raise PipelineError("src/program.py or src/plan.json changed during the build")
            verdict = "passed" if comparison["passed"] else "needs review"
            gate_verdict = (
                "hard gates passed"
                if comparison.get("hard_gates", {}).get("passed")
                else "hard gates failed"
            )
            stage.complete(
                f"Fidelity score {comparison['score']:.4f}; {gate_verdict} — {verdict}",
                score=comparison["score"],
                passed=comparison["passed"],
                hard_gates_passed=comparison.get("hard_gates", {}).get("passed"),
            )
        _replace_directory(staged_artifacts, workspace.artifacts_dir)
        return comparison


def run_pipeline(
    workspace: Workspace,
    config: PipelineConfig,
    *,
    prepare_only: bool = False,
    force_probe: bool = False,
    progress: ProgressReporter | None = None,
) -> dict[str, Any]:
    if config.max_repairs < 0:
        raise ValueError("max_repairs cannot be negative")
    if config.max_fidelity_repairs < 1:
        raise ValueError("max_fidelity_repairs must be at least one")
    reconstruction_mode = validate_reconstruction_mode(config.reconstruction_mode)
    granularity = validate_granularity(config.granularity)
    workspace.update_manifest(
        reconstruction_mode=reconstruction_mode,
        granularity=granularity,
    )
    if not 0.0 <= config.min_score <= 1.0:
        raise ValueError("min_score must be between 0 and 1")
    _validate_positive_runtime(render_size=config.render_size, timeout_s=config.blender_timeout_s)
    if config.llm_timeout_s <= 0:
        raise ValueError("LLM timeout must be greater than zero")
    try:
        with progress_step(progress, "runtime", "Locating Blender") as stage:
            runtime = BlenderRuntime.discover(config.blender)
            executable = getattr(runtime, "executable", "configured executable")
            stage.complete(f"Blender ready — {executable}")
        probe = prepare_reference(
            workspace,
            runtime,
            render_size=config.render_size,
            timeout_s=config.blender_timeout_s,
            force=force_probe,
            progress=progress,
        )
    except (PipelineError, BlenderError, OSError, ValueError) as exc:
        workspace.update_manifest(status="failed")
        write_json(
            workspace.root / "run_report.json",
            {
                "schema_version": 1,
                "status": "failed",
                "workspace": str(workspace.root),
                "backend": config.backend,
                "reconstruction_mode": reconstruction_mode,
                "granularity": granularity,
                "stage": "reference",
                "agent_runs": [],
                "build_attempts": [],
                "error": str(exc),
            },
        )
        raise
    if prepare_only:
        report = {
            "schema_version": 1,
            "status": "prepared",
            "workspace": str(workspace.root),
            "reconstruction_mode": reconstruction_mode,
            "granularity": granularity,
            "reference": {
                "vertices": probe["scene"]["vertex_count"],
                "triangles": probe["scene"]["triangle_count"],
                "semantic_status": probe["semantic_decomposition"]["status"],
            },
        }
        write_json(workspace.root / "run_report.json", report)
        workspace.update_manifest(
            status="prepared",
            reconstruction_mode=reconstruction_mode,
            granularity=granularity,
        )
        emit_progress(
            progress,
            "info",
            "pipeline",
            "Reference evidence prepared; stopping before agent invocation",
        )
        return report

    manifest = workspace.manifest()
    user_prompt = str(manifest.get("prompt") or "")
    agent_runs: list[dict[str, Any]] = []
    build_attempts: list[dict[str, Any]] = []
    next_iteration = workspace.next_trajectory_iteration()
    active_iteration = _matching_source_iteration(workspace)
    if not workspace.program_path.is_file() or not workspace.plan_path.is_file():
        try:
            run = _invoke_agent(
                workspace,
                backend_name=config.backend,
                prompt=initial_prompt(
                    root=workspace.root,
                    image=workspace.image_path,
                    user_prompt=user_prompt,
                    reconstruction_mode=reconstruction_mode,
                    granularity=granularity,
                ),
                iteration=next_iteration,
                timeout_s=config.llm_timeout_s,
                progress=progress,
            )
        except PipelineError as exc:
            workspace.update_manifest(status="failed")
            write_json(
                workspace.root / "run_report.json",
                {
                    "schema_version": 1,
                    "status": "failed",
                    "workspace": str(workspace.root),
                    "backend": config.backend,
                    "reconstruction_mode": reconstruction_mode,
                    "granularity": granularity,
                    "agent_runs": [],
                    "build_attempts": [],
                    "error": str(exc),
                },
            )
            raise
        agent_runs.append(run)
        next_iteration += 1
        active_iteration = _matching_source_iteration(workspace)

    last_failure: str | None = None
    comparison: dict[str, Any] | None = None
    valid_artifact = False
    last_valid_program: bytes | None = None
    last_valid_plan: bytes | None = None
    last_valid_comparison: dict[str, Any] | None = None
    build_repairs_used = 0
    fidelity_repairs_used = 0
    build_attempt = 0
    build_phase = "initial"
    maximum_builds = 1 + config.max_repairs + config.max_fidelity_repairs
    while True:
        emit_progress(
            progress,
            "info",
            "build-attempt",
            f"Build attempt {build_attempt + 1} of at most {maximum_builds}",
            phase=build_phase,
            build_repairs_used=build_repairs_used,
            fidelity_repairs_used=fidelity_repairs_used,
        )
        try:
            comparison = build_workspace(
                workspace,
                runtime,
                min_score=config.min_score,
                reconstruction_mode=reconstruction_mode,
                granularity=granularity,
                timeout_s=config.blender_timeout_s,
                trajectory_dir=(
                    workspace.root / "trajectories" / f"iter_{active_iteration:02d}"
                    if active_iteration is not None
                    else None
                ),
                progress=progress,
            )
            valid_artifact = True
            last_valid_program = workspace.program_path.read_bytes()
            last_valid_plan = workspace.plan_path.read_bytes()
            last_valid_comparison = comparison
            built_attempt: dict[str, Any] = {
                "attempt": build_attempt,
                "phase": build_phase,
                "built": True,
                "comparison": comparison,
            }
            if active_iteration is not None:
                archived_glb = (
                    workspace.root
                    / "trajectories"
                    / f"iter_{active_iteration:02d}"
                    / "model.glb"
                )
                if archived_glb.is_file() and not archived_glb.is_symlink():
                    built_attempt["trajectory_glb"] = archived_glb.relative_to(
                        workspace.root
                    ).as_posix()
            build_attempts.append(built_attempt)
            if comparison["passed"]:
                last_failure = None
            else:
                hard_gate_failures = comparison.get("hard_gates", {}).get("failures", [])
                failed_names = [
                    str(item.get("gate"))
                    for item in hard_gate_failures
                    if isinstance(item, dict) and item.get("gate")
                ]
                score_failure = not bool(comparison.get("score_passed", comparison["passed"]))
                failure_parts = []
                if score_failure:
                    failure_parts.append(
                        f"aggregate score {comparison['score']:.4f} below {config.min_score:.4f}"
                    )
                if failed_names:
                    failure_parts.append("hard gates failed: " + ", ".join(failed_names))
                last_failure = "; ".join(failure_parts) or "fidelity acceptance failed"
                emit_progress(
                    progress,
                    "warning",
                    "build-attempt",
                    last_failure,
                )
        except (PipelineError, BlenderError, OSError, ValueError) as exc:
            last_failure = str(exc)
            build_attempts.append(
                {
                    "attempt": build_attempt,
                    "phase": build_phase,
                    "built": False,
                    "error": last_failure,
                }
            )
            next_action = (
                "preparing a schema/build repair"
                if build_repairs_used < config.max_repairs
                else "no schema/build repairs remain"
            )
            emit_progress(
                progress,
                "warning",
                "build-attempt",
                f"Build attempt {build_attempt + 1} failed; {next_action}",
                error=last_failure,
            )
            if build_repairs_used >= config.max_repairs:
                break
            try:
                run = _invoke_agent(
                    workspace,
                    backend_name=config.backend,
                    prompt=repair_prompt(
                        root=workspace.root,
                        user_prompt=user_prompt,
                        failure=last_failure,
                        comparison=comparison,
                        iteration=next_iteration,
                        reconstruction_mode=reconstruction_mode,
                        granularity=granularity,
                    ),
                    iteration=next_iteration,
                    timeout_s=config.llm_timeout_s,
                    include_candidate=valid_artifact,
                    is_repair=True,
                    progress=progress,
                )
            except PipelineError as repair_exc:
                last_failure = str(repair_exc)
                build_attempts.append(
                    {
                        "attempt": build_attempt,
                        "phase": "agent-build-repair",
                        "built": False,
                        "stage": "agent-repair",
                        "error": last_failure,
                    }
                )
                break
            agent_runs.append(run)
            build_repairs_used += 1
            build_attempt += 1
            build_phase = "schema-build-repair"
            next_iteration += 1
            active_iteration = _matching_source_iteration(workspace)
            continue

        # A successful render (and configured surface evaluation) is the boundary
        # between the independent retry budgets. The first fidelity repair is
        # mandatory, even for a passing candidate, so aggregate metrics cannot
        # short-circuit visual/surface review.
        if fidelity_repairs_used >= config.max_fidelity_repairs:
            break
        try:
            run = _invoke_agent(
                workspace,
                backend_name=config.backend,
                prompt=repair_prompt(
                    root=workspace.root,
                    user_prompt=user_prompt,
                    failure=last_failure,
                    comparison=comparison,
                    iteration=next_iteration,
                    reconstruction_mode=reconstruction_mode,
                    granularity=granularity,
                ),
                iteration=next_iteration,
                timeout_s=config.llm_timeout_s,
                include_candidate=True,
                is_repair=True,
                progress=progress,
            )
        except PipelineError as exc:
            last_failure = str(exc)
            build_attempts.append(
                {
                    "attempt": build_attempt,
                    "phase": "agent-fidelity-repair",
                    "built": False,
                    "stage": "agent-repair",
                    "error": last_failure,
                }
            )
            break
        agent_runs.append(run)
        fidelity_repairs_used += 1
        build_attempt += 1
        build_phase = "post-render-repair"
        next_iteration += 1
        active_iteration = _matching_source_iteration(workspace)

    if not valid_artifact:
        workspace.update_manifest(status="failed")
        report = {
            "schema_version": 1,
            "status": "failed",
            "workspace": str(workspace.root),
            "backend": config.backend,
            "reconstruction_mode": reconstruction_mode,
            "granularity": granularity,
            "agent_runs": agent_runs,
            "build_attempts": build_attempts,
            "retry_budget": {
                "schema_build": {"used": build_repairs_used, "maximum": config.max_repairs},
                "post_render": {
                    "used": fidelity_repairs_used,
                    "maximum": config.max_fidelity_repairs,
                },
            },
            "error": last_failure,
        }
        write_json(workspace.root / "run_report.json", report)
        emit_progress(
            progress,
            "failure",
            "pipeline",
            "Pipeline failed without a valid compiled GLB",
            error=last_failure,
        )
        raise PipelineError(last_failure or "no valid artifact was produced")

    artifact_manifest = json.loads(
        (workspace.artifacts_dir / "build_manifest.json").read_text(encoding="utf-8")
    )
    source_matches_artifact = (
        artifact_manifest.get("program_sha256") == sha256(workspace.program_path)
        and artifact_manifest.get("plan_sha256") == sha256(workspace.plan_path)
    )
    if not source_matches_artifact and last_valid_program is not None and last_valid_plan is not None:
        rejected = workspace.root / "trajectories" / f"iter_{max(0, next_iteration - 1):02d}"
        rejected.mkdir(parents=True, exist_ok=True)
        shutil.copy2(workspace.program_path, rejected / "rejected_program.py")
        shutil.copy2(workspace.plan_path, rejected / "rejected_plan.json")
        _write_bytes_atomic(workspace.program_path, last_valid_program)
        _write_bytes_atomic(workspace.plan_path, last_valid_plan)
        comparison = last_valid_comparison
        restoration = (
            "The last failed repair was retained in its trajectory; source was restored "
            "to the last compiled artifact."
        )
        last_failure = f"{last_failure}. {restoration}" if last_failure else restoration

    status = "complete" if comparison and comparison["passed"] else "needs-review"
    report = {
        "schema_version": 1,
        "status": status,
        "workspace": str(workspace.root),
        "backend": config.backend,
        "reconstruction_mode": reconstruction_mode,
        "granularity": granularity,
        "agent_runs": agent_runs,
        "build_attempts": build_attempts,
        "retry_budget": {
            "schema_build": {"used": build_repairs_used, "maximum": config.max_repairs},
            "post_render": {
                "used": fidelity_repairs_used,
                "maximum": config.max_fidelity_repairs,
            },
        },
        "score": comparison["score"] if comparison else None,
        "passed": comparison["passed"] if comparison else False,
        "warning": last_failure,
        "deliverables": {
            "program": "src/program.py",
            "plan": "src/plan.json",
            "glb": "artifacts/model.glb",
            "blend": "artifacts/scene.blend",
            "comparison": "artifacts/comparison.json",
            "build_manifest": "artifacts/build_manifest.json",
            **(
                {"surface_comparison": "artifacts/surface_comparison.json"}
                if get_granularity_profile(granularity).surface_evaluation_enabled
                else {}
            ),
        },
    }
    write_json(workspace.root / "run_report.json", report)
    workspace.update_manifest(
        status=status,
        score=report["score"],
        reconstruction_mode=reconstruction_mode,
        granularity=granularity,
        deliverables=report["deliverables"],
    )
    if status == "complete":
        emit_progress(
            progress,
            "success",
            "pipeline",
            f"Pipeline complete — fidelity {float(report['score']):.4f}",
        )
    else:
        emit_progress(
            progress,
            "warning",
            "pipeline",
            f"Pipeline produced a valid GLB that needs review — fidelity {float(report['score']):.4f}",
        )
    return report
