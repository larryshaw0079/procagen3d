"""GLB-guided agent → Blender source → compiled GLB pipeline."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import struct
import tempfile
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .backends import CLIBackend, create_backend
from .blender import BlenderError, BlenderRuntime, require_success
from .glb_probe import probe_glb
from .granularity import DEFAULT_GRANULARITY, validate_granularity
from .metrics import (
    DetailGateThresholds,
    FidelityGateThresholds,
    MaterialGateThresholds,
    StructuralGateThresholds,
    SurfaceGateThresholds,
    compare_workspace,
    mask_metrics,
)
from .plan_schema import PlanSchemaError, validate_plan_document
from .progress import ProgressReporter, emit_progress, progress_step
from .prompts import (
    assembly_planning_prompt,
    dedicated_material_prompt,
    incremental_part_prompt,
    initial_prompt,
    repair_prompt,
    targeted_repair_prompt,
)
from .quality import (
    DETAIL_QUALITY_SETTINGS,
    MATERIAL_QUALITY_SETTINGS,
    STRUCTURAL_QUALITY_SETTINGS,
    SURFACE_QUALITY_SETTINGS,
    QualityProfile,
    resolve_quality_profile,
)
from .reconstruction import DEFAULT_RECONSTRUCTION_MODE, validate_reconstruction_mode
from .source_guard import SourceGuardError, assert_safe_source
from .stages import (
    geometry_gates_passed,
    material_gate_failures,
    normalized_object_key,
    part_object_names,
    select_repair_target,
    structured_part_order,
    validate_incremental_probe,
    validate_pipeline_mode,
)
from .workspace import Workspace, sha256, write_json


class PipelineError(RuntimeError):
    """A recoverable stage failure that prevents a valid deliverable."""


class _AgentInvocationError(PipelineError):
    """An agent failure carrying its bounded run-report summary."""

    def __init__(self, message: str, *, run: dict[str, Any]):
        super().__init__(message)
        self.run = run


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
    max_initial_agent_retries: int = 1
    surface_fidelity: str | None = None
    detail_richness: str | None = None
    material_fidelity: str | None = None
    structural_coherence: str | None = None
    pipeline_mode: str = "structured"
    max_part_repairs: int = 1
    max_geometry_repairs: int = 1
    max_material_repairs: int = 1
    dedicated_materials: bool = True
    export_urdf: bool = False
    min_structured_parts: int = 2


CANONICAL_VIEWS = ("front", "back", "left", "right", "top", "iso")
_PENDING_PART_FAILURE_MAX_CHARS = 12_000
_RUN_REPORT_RECOVERY_MAX_BYTES = 1_000_000


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
        root / "reference_views" / "diagnostics" / "manifest.json",
        *(
            root / "reference_views" / "diagnostics" / kind / f"{name}.png"
            for kind in ("depth", "normal", "object_id")
            for name in CANONICAL_VIEWS
        ),
    ]
    if not all(not path.is_symlink() and path.is_file() for path in expected):
        return False
    try:
        glb_report = json.loads((root / "glb_probe.json").read_text(encoding="utf-8"))
        scene = json.loads((root / "reference_scene.json").read_text(encoding="utf-8"))
        camera = json.loads((root / "camera_contract.json").read_text(encoding="utf-8"))
        masks = json.loads((root / "reference_views" / "masks.json").read_text(encoding="utf-8"))
        diagnostics = json.loads(
            (root / "reference_views" / "diagnostics" / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        if not all(
            isinstance(value, dict)
            for value in (glb_report, scene, camera, masks, diagnostics)
        ):
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
        diagnostic_views = diagnostics.get("views")
        if diagnostics.get("schema_version") != 1 or not isinstance(
            diagnostic_views, dict
        ):
            return False
        if set(diagnostic_views) != set(CANONICAL_VIEWS):
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
            diagnostic_record = diagnostic_views[name]
            if not isinstance(diagnostic_record, dict):
                return False
            for kind in ("depth", "normal", "object_id"):
                expected_relative = f"diagnostics/{kind}/{name}.png"
                if diagnostic_record.get(kind) != expected_relative:
                    return False
                diagnostic_header = (
                    root / "reference_views" / expected_relative
                ).read_bytes()[:24]
                if (
                    len(diagnostic_header) != 24
                    or diagnostic_header[:8] != b"\x89PNG\r\n\x1a\n"
                    or diagnostic_header[12:16] != b"IHDR"
                    or struct.unpack(">II", diagnostic_header[16:24])
                    != (render_size, render_size)
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
    paths.extend(
        root
        / "evidence"
        / "reference_views"
        / "diagnostics"
        / kind
        / f"{name}.png"
        for kind in ("depth", "normal", "object_id")
        for name in CANONICAL_VIEWS
    )
    if include_candidate:
        paths.extend(root / "artifacts" / "renders" / f"{name}.png" for name in CANONICAL_VIEWS)
        paths.extend(
            root / "artifacts" / "renders" / "diagnostics" / kind / f"{name}.png"
            for kind in ("depth", "normal", "object_id")
            for name in CANONICAL_VIEWS
        )
        paths.extend(
            root / "artifacts" / "surface_residuals" / f"{name}.png"
            for name in CANONICAL_VIEWS
        )
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


def _agent_run_payload(
    result: Any,
    *,
    files_modified: list[str] | None = None,
    salvaged: bool = False,
    error: str | None = None,
) -> dict[str, Any]:
    """Keep report metadata bounded while full provider evidence stays in trajectory files."""

    return {
        "backend": getattr(result, "backend", None),
        "model": getattr(result, "model", None),
        "duration_s": getattr(result, "duration_s", None),
        "usage": getattr(result, "usage", {}) or {},
        "files_modified": files_modified or [],
        "provider_success": bool(getattr(result, "ok", False)),
        "exit_reason": getattr(result, "exit_reason", None),
        "timed_out": bool(getattr(result, "timed_out", False)),
        "returncode": getattr(result, "returncode", None),
        "salvaged": salvaged,
        **({"error": error} if error else {}),
    }


def _retain_bounded_agent_source(staged_src: Path, trajectory: Path) -> None:
    """Retain regular source candidates for review without trusting or promoting them."""

    for source, label, destination in (
        (
            staged_src / "program.py",
            "agent src/program.py",
            trajectory / "rejected_program.py",
        ),
        (
            staged_src / "plan.json",
            "agent src/plan.json",
            trajectory / "rejected_plan.json",
        ),
    ):
        try:
            candidate_value = _source_snapshot(source, label=label)
        except PipelineError:
            continue
        destination.write_bytes(candidate_value)


def _annotated_agent_run(
    value: dict[str, Any],
    *,
    iteration: int,
    phase: str,
) -> dict[str, Any]:
    run = dict(value)
    run.setdefault("iteration", iteration)
    run.setdefault("phase", phase)
    return run


def _failed_agent_run(
    exc: PipelineError,
    *,
    iteration: int,
    phase: str,
) -> dict[str, Any]:
    if isinstance(exc, _AgentInvocationError):
        value = exc.run
    else:
        value = {
            "provider_success": False,
            "salvaged": False,
            "error": str(exc),
        }
    return _annotated_agent_run(value, iteration=iteration, phase=phase)


def _write_bytes_atomic(path: Path, value: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".restore-tmp")
    temporary.write_bytes(value)
    temporary.replace(path)


def _restore_source_pair(workspace: Workspace, program: bytes, plan: bytes) -> None:
    """Atomically restore the accepted program/plan pair after a rejected edit."""

    with tempfile.TemporaryDirectory(
        prefix=".procagen3d-restore-", dir=workspace.root
    ) as directory:
        staged_src = Path(directory) / "src"
        staged_src.mkdir()
        (staged_src / "program.py").write_bytes(program)
        (staged_src / "plan.json").write_bytes(plan)
        _replace_directory(staged_src, workspace.src_dir)


def _comparison_quality(comparison: dict[str, Any], *, attempt: int) -> tuple[Any, ...]:
    """Return a deterministic higher-is-better key for candidate retention."""

    raw_score = comparison.get("score")
    score = (
        float(raw_score)
        if isinstance(raw_score, (int, float))
        and not isinstance(raw_score, bool)
        and math.isfinite(raw_score)
        else float("-inf")
    )
    hard_gates = comparison.get("hard_gates")
    hard_passed = bool(hard_gates.get("passed")) if isinstance(hard_gates, dict) else False
    failures = hard_gates.get("failures") if isinstance(hard_gates, dict) else None
    failure_count = len(failures) if isinstance(failures, list) else 0
    # Non-compensating gate state outranks the aggregate score. An earlier
    # build attempt makes otherwise equal selection stable across platforms.
    return (
        bool(comparison.get("passed")),
        hard_passed,
        -failure_count,
        score,
        -attempt,
    )


def _comparison_failure(comparison: dict[str, Any], *, min_score: float) -> str | None:
    if comparison.get("passed"):
        return None
    hard_gate_failures = comparison.get("hard_gates", {}).get("failures", [])
    failed_names = [
        str(item.get("gate"))
        for item in hard_gate_failures
        if isinstance(item, dict) and item.get("gate")
    ]
    failure_parts: list[str] = []
    if not bool(comparison.get("score_passed", comparison.get("passed"))):
        failure_parts.append(
            f"aggregate score {float(comparison['score']):.4f} below {min_score:.4f}"
        )
    if failed_names:
        failure_parts.append("hard gates failed: " + ", ".join(failed_names))
    return "; ".join(failure_parts) or "fidelity acceptance failed"


def _snapshot_directory(source: Path, destination: Path) -> None:
    """Replace a private best-candidate snapshot with a complete directory copy."""

    staged = destination.parent / f".{destination.name}.stage-{uuid.uuid4().hex}"
    shutil.copytree(source, staged)
    if destination.exists():
        _replace_directory(staged, destination)
    else:
        staged.replace(destination)


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
            provider_failure: str | None = None
            if not result.ok:
                detail = (
                    getattr(result, "error", None)
                    or getattr(result, "stderr", "")[-2000:]
                    or getattr(result, "final_message", "")
                    or getattr(result, "exit_reason", "error")
                )
                provider_failure = (
                    f"{backend_name} agent failed "
                    f"({getattr(result, 'exit_reason', 'error')}): {detail}"
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
                message = (
                    "agent src/ must remain a regular, non-symlink directory "
                    "inside the disposable workspace"
                )
                raise _AgentInvocationError(
                    message,
                    run=_agent_run_payload(result, error=message),
                )
            changed_relative: list[str] = []
            unauthorized: list[str] = []
            for path in getattr(result, "files_modified", ()):
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
                _retain_bounded_agent_source(staged_src, trajectory)
                message = "agent changed files outside src/: " + ", ".join(unauthorized)
                raise _AgentInvocationError(
                    message,
                    run=_agent_run_payload(
                        result,
                        files_modified=changed_relative,
                        error=message,
                    ),
                )
            # Snapshot both required deliverables before any source promotion.
            try:
                program_value = _source_snapshot(
                    staged_program,
                    label="agent src/program.py",
                )
                plan_value = _source_snapshot(staged_plan, label="agent src/plan.json")
                if provider_failure is not None:
                    # A provider timeout/non-success may still leave a complete
                    # source pair. Salvage only after the same host-owned schema
                    # and AST safety checks used before Blender execution.
                    _validate_plan(staged_plan)
                    _guard_program(staged_program)
            except PipelineError as exc:
                _retain_bounded_agent_source(staged_src, trajectory)
                message = str(exc)
                if provider_failure is not None:
                    message = f"{provider_failure}; source salvage rejected: {message}"
                raise _AgentInvocationError(
                    message,
                    run=_agent_run_payload(
                        result,
                        files_modified=changed_relative,
                        error=message,
                    ),
                ) from exc
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
            salvaged = provider_failure is not None
            payload = _agent_run_payload(
                result,
                files_modified=changed_relative,
                salvaged=salvaged,
                error=provider_failure,
            )
            stage.complete(
                (
                    f"{backend_name} timed out/failed after producing valid source; "
                    "salvaged plan.json and program.py"
                    if salvaged
                    else f"{backend_name} produced plan.json and program.py"
                ),
                kind="warning" if salvaged else "success",
                model=getattr(result, "model", None),
                files_modified=changed_relative,
                provider_duration_s=getattr(result, "duration_s", None),
                salvaged=salvaged,
            )
    return payload


def _assembly_runtime_document(
    plan: dict[str, Any], *, part_ids: list[str] | tuple[str, ...] | None = None
) -> dict[str, Any] | None:
    assembly = plan.get("assembly")
    if not isinstance(assembly, dict) or assembly.get("placement") != "host-solved":
        return None
    try:
        from .assembly import solve_assembly_transforms

        solved = solve_assembly_transforms(plan)
    except (ImportError, ValueError) as exc:
        raise PipelineError(f"assembly transform solver rejected the plan: {exc}") from exc
    assembly_order = tuple(assembly.get("part_order", []))
    selected_ids = assembly_order if part_ids is None else tuple(part_ids)
    if selected_ids != assembly_order[: len(selected_ids)]:
        raise PipelineError(
            "assembly runtime part filter must be an assembly-order prefix"
        )
    part_records = []
    for part_id in selected_ids:
        part = next(
            (
                item
                for item in plan.get("parts", [])
                if isinstance(item, dict) and item.get("id") == part_id
            ),
            None,
        )
        if part is None or part_id not in solved:
            raise PipelineError(f"assembly runtime cannot resolve part {part_id!r}")
        part_records.append(
            {
                "id": part_id,
                "object_names": list(part.get("object_names", [])),
                "world_matrix": [list(row) for row in solved[part_id]],
            }
        )
    return {
        "schema_version": 1,
        "placement": "host-solved",
        "parts": part_records,
    }


def _urdf_link_runtime_document(plan: dict[str, Any]) -> dict[str, Any] | None:
    """Build the independent connector-centred link and motion contract.

    Blender scene construction must continue to use part frames from
    :func:`_assembly_runtime_document`.  URDF meshes instead use their incoming
    connector as the link frame so revolute and prismatic motion occurs about
    the declared mechanical interface rather than the modeling origin.
    """

    assembly = plan.get("assembly")
    if not isinstance(assembly, dict) or assembly.get("placement") != "host-solved":
        return None
    try:
        from .urdf import resolve_urdf_link_frames, resolve_urdf_motion_probes

        frames = resolve_urdf_link_frames(plan)
        probes = resolve_urdf_motion_probes(plan)
    except (ImportError, ValueError) as exc:
        raise PipelineError(f"URDF link-frame solver rejected the plan: {exc}") from exc

    part_by_id = {
        str(part.get("id")): part
        for part in plan.get("parts", [])
        if isinstance(part, dict) and isinstance(part.get("id"), str)
    }
    assembly_order = tuple(assembly.get("part_order", []))
    if set(assembly_order) != set(frames) or len(assembly_order) != len(frames):
        raise PipelineError("URDF link frames do not cover the assembly order exactly")

    def rows(matrix: Any) -> list[list[float]]:
        return [[float(value) for value in row] for row in matrix]

    parts = []
    for part_id in assembly_order:
        part = part_by_id.get(part_id)
        frame = frames.get(part_id)
        if part is None or frame is None:
            raise PipelineError(f"URDF link runtime cannot resolve part {part_id!r}")
        parts.append(
            {
                "id": part_id,
                "object_names": list(part.get("object_names", [])),
                "part_world_matrix": rows(frame.part_world),
                "part_from_link_matrix": rows(frame.part_from_link),
                "link_world_matrix": rows(frame.link_world),
                "incoming_mate_id": frame.incoming_mate,
            }
        )

    motion_probes = []
    for probe in probes:
        expected = dict(probe.link_world)
        motion_probes.append(
            {
                "mate_id": probe.mate_id,
                "joint_type": probe.joint_type,
                "assembly_parameter": probe.assembly_parameter,
                "urdf_position": probe.urdf_position,
                "expected_link_world_matrices": {
                    part_id: rows(expected[part_id]) for part_id in assembly_order
                },
            }
        )
    return {
        "schema_version": 2,
        "placement": "urdf-link",
        "parts": parts,
        "motion_probes": motion_probes,
    }


def build_workspace(
    workspace: Workspace,
    runtime: BlenderRuntime,
    *,
    min_score: float = 0.35,
    reconstruction_mode: str = DEFAULT_RECONSTRUCTION_MODE,
    granularity: str = DEFAULT_GRANULARITY,
    quality_profile: QualityProfile | None = None,
    timeout_s: int = 900,
    trajectory_dir: Path | None = None,
    progress: ProgressReporter | None = None,
) -> dict[str, Any]:
    """Compile the current source in a clean directory and compare it."""

    if not 0.0 <= min_score <= 1.0:
        raise ValueError("min_score must be between 0 and 1")
    reconstruction_mode = validate_reconstruction_mode(reconstruction_mode)
    granularity = validate_granularity(granularity)
    quality_profile = quality_profile or resolve_quality_profile(granularity)
    surface_settings = SURFACE_QUALITY_SETTINGS[quality_profile.surface_fidelity]
    if timeout_s <= 0:
        raise ValueError("Blender timeout must be greater than zero")
    reference_snapshot: bytes | None = None
    reference_digest: str | None = None
    if reconstruction_mode == "glb-ref" or surface_settings.enabled:
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
        staged_assembly = clean_root / "assembly_transforms.json"
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
            if validated_plan.get("quality_profile") != quality_profile.as_dict():
                raise PipelineError(
                    "src/plan.json quality_profile must match the resolved build profile: "
                    f"expected {quality_profile.as_dict()!r}, found "
                    f"{validated_plan.get('quality_profile')!r}"
                )
            assembly_runtime = _assembly_runtime_document(validated_plan)
            if assembly_runtime is not None:
                write_json(staged_assembly, assembly_runtime)
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
            if staged_assembly.is_file():
                build_arguments.extend(["--assembly-transforms", staged_assembly])
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
        if surface_settings.enabled:
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
                        str(surface_settings.sample_budget),
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
                    quality_profile=quality_profile.as_dict(),
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
            host_solved_assembly = assembly_runtime is not None
            replay_source_of_truth = ["src/program.py"]
            if reconstruction_mode == "glb-ref":
                replay_source_of_truth.extend(
                    ["inputs/reference.glb", "host-reference-preload-v1"]
                )
            if host_solved_assembly:
                replay_source_of_truth.extend(
                    ["src/plan.json#assembly", "host-assembly-placement-v1"]
                )
            write_json(
                staged_artifacts / "build_manifest.json",
                {
                    "schema_version": 1,
                    "program_sha256": program_digest,
                    "plan_sha256": plan_digest,
                    "model_sha256": model_digest,
                    "blend_sha256": blend_digest,
                    "scene_report_sha256": sha256(scene_report_path),
                    "clean_room": True,
                    "reconstruction_mode": reconstruction_mode,
                    "granularity": granularity,
                    "quality_profile": quality_profile.as_dict(),
                    "program_is_standalone_replay_source": (
                        reconstruction_mode == "procedural"
                        and not host_solved_assembly
                    ),
                    "replay_source_of_truth": replay_source_of_truth,
                    "assembly_placement": (
                        "host-solved" if host_solved_assembly else None
                    ),
                    "assembly_transforms_sha256": (
                        sha256(staged_assembly) if host_solved_assembly else None
                    ),
                    "source_glb_imported_at_build_time": (
                        reconstruction_mode == "glb-ref"
                    ),
                    "reference_glb_sha256": reference_digest,
                    "reference_glb_used_for_surface_evaluation": (
                        surface_settings.enabled
                    ),
                    "reference_contract": (
                        "host-imported-normalized-source; originals removed before export"
                        if reconstruction_mode == "glb-ref"
                        else (
                            "measurement-and-host-surface-evaluation-evidence-only"
                            if surface_settings.enabled
                            else "measurement-evidence-only"
                        )
                    ),
                    "compiled_glb_verified_in_separate_process": True,
                    "surface_distance_evaluation": (
                        {
                            "report": "artifacts/surface_comparison.json",
                            "samples_per_direction": (
                                surface_settings.sample_budget
                            ),
                            "max_mean_distance": (
                                surface_settings.max_mean_distance
                            ),
                            "max_p95_distance": (
                                surface_settings.max_p95_distance
                            ),
                            "max_mean_normal_angle_degrees": (
                                surface_settings.max_mean_normal_angle_degrees
                            ),
                            "min_visible_coverage": (
                                surface_settings.min_visible_coverage
                            ),
                            "surface_area_ratio_range": [
                                surface_settings.min_surface_area_ratio,
                                surface_settings.max_surface_area_ratio,
                            ],
                            "residual_artifacts": "artifacts/surface_residuals",
                        }
                        if surface_settings.enabled
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
            "Scoring color, detail, surface, material, and structural hard gates",
        ) as stage:
            surface_thresholds = None
            if surface_settings.enabled:
                assert surface_settings.max_mean_distance is not None
                assert surface_settings.max_p95_distance is not None
                surface_thresholds = SurfaceGateThresholds(
                    max_mean_surface_distance=surface_settings.max_mean_distance,
                    max_p95_surface_distance=surface_settings.max_p95_distance,
                    max_mean_normal_angle_degrees=(
                        surface_settings.max_mean_normal_angle_degrees
                    ),
                    min_visible_coverage=surface_settings.min_visible_coverage,
                    min_surface_area_ratio=surface_settings.min_surface_area_ratio,
                    max_surface_area_ratio=surface_settings.max_surface_area_ratio,
                )
            material_settings = MATERIAL_QUALITY_SETTINGS[
                quality_profile.material_fidelity
            ]
            detail_settings = DETAIL_QUALITY_SETTINGS[quality_profile.detail_richness]
            structural_settings = STRUCTURAL_QUALITY_SETTINGS[
                quality_profile.structural_coherence
            ]
            comparison = compare_workspace(
                reference_masks=workspace.evidence_dir / "reference_views" / "masks.json",
                candidate_masks=staged_artifacts / "renders" / "masks.json",
                reference_scene=workspace.evidence_dir / "reference_scene.json",
                candidate_scene=staged_artifacts / "scene_report.json",
                output=staged_artifacts / "comparison.json",
                min_score=min_score,
                gate_thresholds=FidelityGateThresholds(
                    min_mean_spatial_rgb_similarity=(
                        material_settings.min_spatial_rgb_similarity
                    ),
                    min_mean_palette_similarity=(
                        material_settings.min_palette_similarity
                    ),
                ),
                surface_comparison=surface_comparison_path,
                surface_gate_thresholds=surface_thresholds,
                reference_probe=workspace.evidence_dir / "glb_probe.json",
                candidate_probe=staged_artifacts / "model_probe.json",
                plan=staged_plan,
                material_gate_thresholds=MaterialGateThresholds(
                    max_default_white_primitive_fraction=(
                        material_settings.max_default_white_primitive_fraction
                    )
                ),
                detail_gate_thresholds=DetailGateThresholds(
                    **asdict(detail_settings)
                ),
                structural_gate_thresholds=StructuralGateThresholds(
                    **asdict(structural_settings)
                ),
            )
            comparison["granularity"] = granularity
            comparison["quality_profile"] = quality_profile.as_dict()
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


def _structured_state_path(workspace: Workspace) -> Path:
    return workspace.root / "structured_state.json"


def _structured_contract_sha256(plan: dict[str, Any]) -> str:
    # Dedicated PBR authoring is the only stage allowed to change the material
    # declarations. Everything else affects geometry, execution, acceptance, or
    # articulation and therefore belongs to the structured contract.
    payload = {
        key: value
        for key, value in plan.items()
        if key not in {"materials", "material_plan"}
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _structured_pbr_sha256(plan: dict[str, Any]) -> str:
    """Bind the material declarations independently from geometry policy."""

    encoded = json.dumps(
        {
            "materials": plan.get("materials"),
            "material_plan": plan.get("material_plan"),
        },
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _structured_acceptance_sha256(plan: dict[str, Any]) -> str:
    """Hash structure and acceptance policy while allowing placement tuning.

    Geometry repair may adjust part-local connector frames and mate fit offsets
    to correct assembly placement. It must not rewrite expected bounds, contact
    tolerances, joint limits, quality settings, or any other acceptance target.
    """

    payload = json.loads(json.dumps(plan, ensure_ascii=False, allow_nan=False))
    payload.pop("materials", None)
    payload.pop("material_plan", None)
    assembly = payload.get("assembly")
    if isinstance(assembly, dict):
        for connector in assembly.get("connectors", []):
            if isinstance(connector, dict):
                connector.pop("frame", None)
        for mate in assembly.get("mates", []):
            if isinstance(mate, dict):
                mate.pop("fit_offset", None)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _structured_placement_changes(
    before: dict[str, Any], after: dict[str, Any]
) -> tuple[str, ...]:
    """Return connector-frame and mate-offset fields changed by one repair."""

    changes: list[str] = []
    before_assembly = before.get("assembly")
    after_assembly = after.get("assembly")
    if not isinstance(before_assembly, dict) or not isinstance(after_assembly, dict):
        return ()
    before_connectors = {
        connector.get("id"): connector
        for connector in before_assembly.get("connectors", [])
        if isinstance(connector, dict) and isinstance(connector.get("id"), str)
    }
    for connector in after_assembly.get("connectors", []):
        if not isinstance(connector, dict) or not isinstance(connector.get("id"), str):
            continue
        connector_id = str(connector["id"])
        prior = before_connectors.get(connector_id)
        if isinstance(prior, dict) and prior.get("frame") != connector.get("frame"):
            changes.append(f"assembly.connectors[{connector_id!r}].frame")
    before_mates = {
        mate.get("id"): mate
        for mate in before_assembly.get("mates", [])
        if isinstance(mate, dict) and isinstance(mate.get("id"), str)
    }
    for mate in after_assembly.get("mates", []):
        if not isinstance(mate, dict) or not isinstance(mate.get("id"), str):
            continue
        mate_id = str(mate["id"])
        prior = before_mates.get(mate_id)
        if isinstance(prior, dict) and prior.get("fit_offset") != mate.get(
            "fit_offset"
        ):
            changes.append(f"assembly.mates[{mate_id!r}].fit_offset")
    return tuple(changes)


def _structured_topology_sha256(plan: dict[str, Any]) -> str:
    parts = []
    for part in plan.get("parts", []):
        if not isinstance(part, dict):
            continue
        attachment = part.get("attachment")
        parts.append(
            {
                "id": part.get("id"),
                "object_names": part.get("object_names"),
                "attachment": (
                    {
                        "parent_id": attachment.get("parent_id"),
                        "type": attachment.get("type"),
                    }
                    if isinstance(attachment, dict)
                    else None
                ),
            }
        )
    assembly = plan.get("assembly")
    assembly_topology: Any = assembly
    if isinstance(assembly, dict):
        assembly_topology = {
            "placement": assembly.get("placement"),
            "part_order": assembly.get("part_order"),
            "connectors": [
                {
                    key: connector.get(key)
                    for key in ("id", "part_id", "interface", "role")
                }
                for connector in assembly.get("connectors", [])
                if isinstance(connector, dict)
            ],
            "mates": [
                {
                    key: mate.get(key)
                    for key in (
                        "id",
                        "type",
                        "parent_connector_id",
                        "child_connector_id",
                    )
                }
                for mate in assembly.get("mates", [])
                if isinstance(mate, dict)
            ],
        }
    articulation = plan.get("articulation")
    articulation_topology: Any = articulation
    if isinstance(articulation, dict):
        articulation_topology = {
            "enabled": articulation.get("enabled"),
            "mechanical": articulation.get("mechanical"),
            "robot_name": articulation.get("robot_name"),
            "joints": [
                {
                    key: joint.get(key)
                    for key in ("name", "parent", "child", "type")
                }
                for joint in articulation.get("joints", [])
                if isinstance(joint, dict)
            ],
        }
    encoded = json.dumps(
        {
            "parts": parts,
            "assembly": assembly_topology,
            "articulation": articulation_topology,
        },
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _structured_plan_bindings(
    workspace: Workspace, plan: dict[str, Any]
) -> dict[str, str]:
    """Return exact source and normalized structured-plan bindings."""

    return {
        "program_sha256": sha256(workspace.program_path),
        "plan_sha256": sha256(workspace.plan_path),
        "contract_sha256": _structured_contract_sha256(plan),
        "pbr_sha256": _structured_pbr_sha256(plan),
        "acceptance_sha256": _structured_acceptance_sha256(plan),
        "topology_sha256": _structured_topology_sha256(plan),
    }


def _read_regular_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PipelineError(f"{label} must be a regular JSON file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"{label} is unreadable: {exc}") from exc
    if not isinstance(value, dict):
        raise PipelineError(f"{label} must contain an object")
    return value


def _probe_signature_for_parts(
    model_path: Path,
    *,
    plan: dict[str, Any],
    completed_part_ids: list[str],
    label: str,
) -> dict[str, Any]:
    if model_path.is_symlink() or not model_path.is_file():
        raise PipelineError(f"{label} model.glb must be a regular file")
    try:
        probe = probe_glb(model_path)
        validation = validate_incremental_probe(
            plan=plan,
            completed_part_ids=completed_part_ids,
            probe=probe,
            previous_signature=None,
        )
    except (OSError, ValueError, KeyError, struct.error) as exc:
        raise PipelineError(f"{label} model.glb is invalid: {exc}") from exc
    return dict(validation["geometry_signature"])


def _validate_structured_resume_artifacts(
    workspace: Workspace,
    *,
    plan: dict[str, Any],
    state: dict[str, Any],
) -> None:
    """Bind resumable state to its checkpoints and currently published GLB."""

    completed = list(state["completed_parts"])
    checkpoints = state["checkpoints"]
    for index, (part_id, entry) in enumerate(zip(completed, checkpoints, strict=True)):
        if not isinstance(entry, dict):
            raise PipelineError("structured checkpoint entries must be objects")
        checkpoint_name = _checkpoint_name(index, part_id)
        expected_relative = f"checkpoints/{checkpoint_name}"
        if entry.get("path") != expected_relative:
            raise PipelineError(
                f"structured checkpoint {part_id!r} has an unexpected path"
            )
        iteration = entry.get("iteration")
        if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
            raise PipelineError(
                f"structured checkpoint {part_id!r} has an invalid iteration"
            )
        checkpoint_dir = workspace.root / "checkpoints" / checkpoint_name
        if checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
            raise PipelineError(
                f"structured checkpoint {part_id!r} directory is unavailable"
            )
        report = _read_regular_json(
            checkpoint_dir / "checkpoint.json",
            label=f"structured checkpoint {part_id!r} report",
        )
        if report.get("schema_version") != 1 or report.get("checkpoint") != checkpoint_name:
            raise PipelineError(
                f"structured checkpoint {part_id!r} report identity is invalid"
            )
        checkpoint_manifest = _read_regular_json(
            checkpoint_dir / "build_manifest.json",
            label=f"structured checkpoint {part_id!r} manifest",
        )
        if (
            checkpoint_manifest.get("schema_version") != 1
            or checkpoint_manifest.get("kind")
            != "incremental-part-checkpoint"
            or checkpoint_manifest.get("checkpoint") != checkpoint_name
            or checkpoint_manifest.get("clean_room") is not True
            or checkpoint_manifest.get("compiled_glb_verified_in_separate_process")
            is not True
            or any(
                checkpoint_manifest.get(field) != report.get(field)
                for field in (
                    "program_sha256",
                    "plan_sha256",
                    "model_sha256",
                    "scene_report_sha256",
                )
            )
        ):
            raise PipelineError(
                f"structured checkpoint {part_id!r} manifest binding is invalid"
            )
        trajectory = workspace.root / "trajectories" / f"iter_{iteration:02d}"
        if trajectory.is_symlink() or not trajectory.is_dir():
            raise PipelineError(
                f"structured checkpoint {part_id!r} trajectory is unavailable"
            )
        for filename, digest_field in (
            ("program.py", "program_sha256"),
            ("plan.json", "plan_sha256"),
        ):
            source_path = trajectory / filename
            digest = report.get(digest_field)
            if (
                source_path.is_symlink()
                or not source_path.is_file()
                or not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                or sha256(source_path) != digest
            ):
                raise PipelineError(
                    f"structured checkpoint {part_id!r} {filename} binding is invalid"
                )
        model_path = checkpoint_dir / "model.glb"
        model_digest = report.get("model_sha256")
        if (
            not isinstance(model_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", model_digest) is None
            or model_path.is_symlink()
            or not model_path.is_file()
            or sha256(model_path) != model_digest
        ):
            raise PipelineError(
                f"structured checkpoint {part_id!r} model binding is invalid"
            )
        trajectory_model = trajectory / "model.glb"
        if (
            trajectory_model.is_symlink()
            or not trajectory_model.is_file()
            or sha256(trajectory_model) != model_digest
        ):
            raise PipelineError(
                f"structured checkpoint {part_id!r} trajectory model binding is invalid"
            )
        scene_report_path = checkpoint_dir / "scene_report.json"
        scene_report_digest = report.get("scene_report_sha256")
        if (
            not isinstance(scene_report_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", scene_report_digest) is None
            or scene_report_path.is_symlink()
            or not scene_report_path.is_file()
            or sha256(scene_report_path) != scene_report_digest
        ):
            raise PipelineError(
                f"structured checkpoint {part_id!r} scene report binding is invalid"
            )
        signature = _probe_signature_for_parts(
            model_path,
            plan=plan,
            completed_part_ids=completed[: index + 1],
            label=f"structured checkpoint {part_id!r}",
        )
        validation = report.get("validation")
        if (
            not isinstance(validation, dict)
            or validation.get("completed_parts") != completed[: index + 1]
            or validation.get("geometry_signature") != signature
        ):
            raise PipelineError(
                f"structured checkpoint {part_id!r} validation binding is invalid"
            )

    if not completed:
        return
    manifest = _read_regular_json(
        workspace.artifacts_dir / "build_manifest.json",
        label="published structured build manifest",
    )
    if (
        manifest.get("schema_version") != 1
        or manifest.get("clean_room") is not True
        or manifest.get("compiled_glb_verified_in_separate_process") is not True
    ):
        raise PipelineError("published structured build manifest policy is invalid")
    if manifest.get("program_sha256") != state.get("program_sha256"):
        raise PipelineError("published structured build has a stale program binding")
    if manifest.get("plan_sha256") != state.get("plan_sha256"):
        raise PipelineError("published structured build has a stale plan binding")
    model_path = workspace.artifacts_dir / "model.glb"
    model_digest = manifest.get("model_sha256")
    if (
        not isinstance(model_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", model_digest) is None
        or model_path.is_symlink()
        or not model_path.is_file()
        or sha256(model_path) != model_digest
    ):
        raise PipelineError("published structured model binding is invalid")
    scene_report_path = workspace.artifacts_dir / "scene_report.json"
    scene_report_digest = manifest.get("scene_report_sha256")
    if (
        not isinstance(scene_report_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", scene_report_digest) is None
        or scene_report_path.is_symlink()
        or not scene_report_path.is_file()
        or sha256(scene_report_path) != scene_report_digest
    ):
        raise PipelineError("published structured scene report binding is invalid")
    signature = _probe_signature_for_parts(
        model_path,
        plan=plan,
        completed_part_ids=completed,
        label="published structured build",
    )
    if signature != state.get("geometry_signature"):
        raise PipelineError(
            "published structured geometry does not match structured_state.json"
        )


def _bounded_pending_part_error(error: str) -> str:
    """Retain the diagnostic tail needed for repair without unbounded state growth."""

    if len(error) <= _PENDING_PART_FAILURE_MAX_CHARS:
        return error
    omitted = len(error) - _PENDING_PART_FAILURE_MAX_CHARS
    marker = f"[... {omitted} earlier diagnostic characters omitted ...]\n"
    return marker + error[-(_PENDING_PART_FAILURE_MAX_CHARS - len(marker)) :]


def _run_report_part_failure(workspace: Workspace, *, part_id: str) -> str | None:
    """Recover one legacy failed-run diagnostic without trusting unrelated reports."""

    path = workspace.root / "run_report.json"
    if path.is_symlink() or not path.is_file():
        return None
    try:
        if path.stat().st_size > _RUN_REPORT_RECOVERY_MAX_BYTES:
            return None
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        not isinstance(report, dict)
        or report.get("status") != "failed"
        or report.get("stage") != "structured-authoring"
        or report.get("pipeline_mode") != "structured"
        or report.get("workspace") != str(workspace.root)
    ):
        return None
    stages = report.get("structured_stages")
    if not isinstance(stages, list):
        return None
    failed_parts = [
        stage
        for stage in stages
        if isinstance(stage, dict)
        and stage.get("phase") in {"part-authoring", "part-repair"}
        and stage.get("passed") is False
    ]
    if not failed_parts:
        return None
    latest = failed_parts[-1]
    error = latest.get("error")
    if latest.get("part_id") != part_id or not isinstance(error, str) or not error:
        return None
    return _bounded_pending_part_error(error)


def _resume_part_failure(
    workspace: Workspace,
    *,
    state: dict[str, Any],
    part_id: str,
) -> str | None:
    """Seed the next repair from state, or one backward-compatible run report."""

    if "pending_part_failure" in state:
        pending = state.get("pending_part_failure")
        if not isinstance(pending, dict):
            raise PipelineError("structured_state.json has an invalid pending part failure")
        error = pending.get("error")
        if pending.get("part_id") != part_id or not isinstance(error, str) or not error:
            raise PipelineError(
                "structured_state.json pending part failure does not match the next part"
            )
        return error
    return _run_report_part_failure(workspace, part_id=part_id)


def _load_structured_state(workspace: Workspace) -> dict[str, Any] | None:
    path = _structured_state_path(workspace)
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise PipelineError("structured_state.json must be a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PipelineError(f"structured_state.json is invalid JSON: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 2:
        raise PipelineError("structured_state.json has an unsupported schema")
    phase = value.get("phase")
    if phase not in {"parts", "geometry", "materials", "final"}:
        raise PipelineError("structured_state.json has an invalid phase")
    completed = value.get("completed_parts")
    signature = value.get("geometry_signature")
    if not isinstance(completed, list) or not isinstance(signature, dict):
        raise PipelineError("structured_state.json lacks part progress")
    if not all(isinstance(part_id, str) and part_id for part_id in completed):
        raise PipelineError("structured_state.json has invalid completed part IDs")
    checkpoints = value.get("checkpoints")
    if not isinstance(checkpoints, list):
        raise PipelineError("structured_state.json lacks checkpoint progress")
    checkpoint_ids = [
        checkpoint.get("part_id") if isinstance(checkpoint, dict) else None
        for checkpoint in checkpoints
    ]
    if checkpoint_ids != completed:
        raise PipelineError(
            "structured_state.json checkpoints do not match completed parts"
        )
    state_program_digest = value.get("program_sha256")
    if not isinstance(state_program_digest, str) or not state_program_digest:
        raise PipelineError("structured_state.json lacks its program_sha256 binding")
    state_plan_digest = value.get("plan_sha256")
    if not isinstance(state_plan_digest, str) or not state_plan_digest:
        raise PipelineError("structured_state.json lacks its plan_sha256 binding")
    contract_digest = value.get("contract_sha256")
    if not isinstance(contract_digest, str) or not contract_digest:
        raise PipelineError("structured_state.json lacks its structured contract binding")
    pbr_digest = value.get("pbr_sha256")
    if not isinstance(pbr_digest, str) or not pbr_digest:
        raise PipelineError("structured_state.json lacks its PBR binding")
    acceptance_digest = value.get("acceptance_sha256")
    if not isinstance(acceptance_digest, str) or not acceptance_digest:
        raise PipelineError("structured_state.json lacks its acceptance-policy binding")
    topology_digest = value.get("topology_sha256")
    if not isinstance(topology_digest, str) or not topology_digest:
        raise PipelineError("structured_state.json lacks its topology binding")
    if (
        not workspace.plan_path.is_file()
        or workspace.plan_path.is_symlink()
        or not workspace.program_path.is_file()
        or workspace.program_path.is_symlink()
    ):
        raise PipelineError(
            "structured_state.json cannot be resumed without regular source files"
        )
    current_program_digest = sha256(workspace.program_path)
    if state_program_digest != current_program_digest:
        raise PipelineError(
            "structured_state.json does not match the current src/program.py; "
            "restore the accepted source before resuming"
        )
    current_plan_digest = sha256(workspace.plan_path)
    if state_plan_digest != current_plan_digest:
        raise PipelineError(
            "structured_state.json does not match the current src/plan.json; "
            "discard or archive the stale structured checkpoints before restarting"
        )
    current_plan = _validate_plan(workspace.plan_path)
    if contract_digest != _structured_contract_sha256(current_plan):
        raise PipelineError(
            "structured_state.json contract binding does not match src/plan.json"
        )
    if pbr_digest != _structured_pbr_sha256(current_plan):
        raise PipelineError(
            "structured_state.json PBR binding does not match src/plan.json"
        )
    if acceptance_digest != _structured_acceptance_sha256(current_plan):
        raise PipelineError(
            "structured_state.json acceptance binding does not match src/plan.json"
        )
    if topology_digest != _structured_topology_sha256(current_plan):
        raise PipelineError(
            "structured_state.json topology binding does not match src/plan.json"
        )
    current_order = structured_part_order(current_plan)
    stored_order = value.get("part_order")
    if not isinstance(stored_order, list) or tuple(stored_order) != current_order:
        raise PipelineError(
            "structured_state.json part order does not match src/plan.json"
        )
    if tuple(completed) != current_order[: len(completed)]:
        raise PipelineError(
            "structured_state.json completed parts are not an assembly-order prefix"
        )
    if "pending_part_failure" in value:
        pending = value.get("pending_part_failure")
        if (
            not isinstance(pending, dict)
            or set(pending) != {"part_id", "error"}
        ):
            raise PipelineError(
                "structured_state.json has an invalid pending part failure"
            )
        pending_part_id = pending.get("part_id")
        pending_error = pending.get("error")
        if (
            phase != "parts"
            or len(completed) >= len(current_order)
            or pending_part_id != current_order[len(completed)]
        ):
            raise PipelineError(
                "structured_state.json pending part failure does not match the next part"
            )
        if (
            not isinstance(pending_error, str)
            or not pending_error
            or len(pending_error) > _PENDING_PART_FAILURE_MAX_CHARS
        ):
            raise PipelineError(
                "structured_state.json pending part failure has an invalid error"
            )
    expected_signature_keys = {
        normalized_object_key(name)
        for name in part_object_names(current_plan, completed)
    }
    if (
        set(signature) != expected_signature_keys
        or not all(isinstance(record, dict) for record in signature.values())
    ):
        raise PipelineError(
            "structured_state.json geometry signature does not cover completed objects"
        )
    if phase != "parts" and tuple(completed) != current_order:
        raise PipelineError(
            f"structured_state.json phase {phase!r} requires all parts to be complete"
        )
    geometry_passed = value.get("geometry_passed")
    materials_status = value.get("materials_status")
    if not isinstance(geometry_passed, bool):
        raise PipelineError("structured_state.json has an invalid geometry status")
    if materials_status not in {
        "pending",
        "applied",
        "skipped",
        "failed",
        "blocked-geometry",
    }:
        raise PipelineError("structured_state.json has an invalid materials status")
    valid_status = {
        "parts": (
            len(completed) < len(current_order)
            and not geometry_passed
            and materials_status == "pending"
        ),
        "geometry": not geometry_passed and materials_status == "pending",
        "materials": geometry_passed and materials_status == "pending",
        "final": (
            (not geometry_passed and materials_status == "blocked-geometry")
            or (
                geometry_passed
                and materials_status in {"applied", "skipped", "failed"}
            )
        ),
    }[phase]
    if not valid_status:
        raise PipelineError(
            "structured_state.json phase, geometry, and materials statuses conflict"
        )
    _validate_structured_resume_artifacts(
        workspace,
        plan=current_plan,
        state=value,
    )
    return value


def _new_structured_state(workspace: Workspace, plan: dict[str, Any]) -> dict[str, Any]:
    # The workspace file is the replay source of truth. Normalize it before
    # hashing so callers cannot accidentally bind state to a pre-normalized
    # in-memory plan that differs semantically from later stage reads.
    plan = _validate_plan(workspace.plan_path)
    state = {
        "schema_version": 2,
        "phase": "parts",
        **_structured_plan_bindings(workspace, plan),
        "part_order": list(structured_part_order(plan)),
        "completed_parts": [],
        "geometry_signature": {},
        "checkpoints": [],
        "geometry_passed": False,
        "materials_status": "pending",
    }
    write_json(_structured_state_path(workspace), state)
    return state


def _validate_explicit_assembly_contract(
    plan: dict[str, Any], *, min_parts: int
) -> None:
    if isinstance(min_parts, bool) or not isinstance(min_parts, int) or min_parts < 1:
        raise ValueError("min_structured_parts must be a positive integer")
    parts = plan.get("parts")
    if not isinstance(parts, list) or len(parts) < min_parts:
        raise PipelineError(
            f"structured planning requires at least {min_parts} semantic parts; "
            f"received {len(parts) if isinstance(parts, list) else 0}"
        )
    object_owners: dict[str, str] = {}
    normalized_owners: dict[str, tuple[str, str]] = {}
    for part in parts:
        if not isinstance(part, dict):
            continue
        part_id = str(part.get("id"))
        for object_name in part.get("object_names", []):
            if not isinstance(object_name, str):
                continue
            previous_owner = object_owners.get(object_name)
            if previous_owner is not None:
                raise PipelineError(
                    f"structured object name {object_name!r} is owned by both "
                    f"{previous_owner!r} and {part_id!r}"
                )
            object_owners[object_name] = part_id
            normalized = normalized_object_key(object_name)
            if not normalized:
                raise PipelineError(
                    f"structured object name {object_name!r} has no stable identity"
                )
            collision = normalized_owners.get(normalized)
            if collision is not None:
                prior_name, prior_owner = collision
                raise PipelineError(
                    "structured object names collide after GLB normalization: "
                    f"{prior_name!r} ({prior_owner!r}) and "
                    f"{object_name!r} ({part_id!r})"
                )
            normalized_owners[normalized] = (object_name, part_id)
    assembly = plan.get("assembly")
    if not isinstance(assembly, dict):
        raise PipelineError("structured planning requires an explicit assembly object")
    if assembly.get("placement") != "host-solved":
        raise PipelineError(
            "structured assembly must set assembly.placement to 'host-solved'"
        )
    connectors = assembly.get("connectors")
    mates = assembly.get("mates")
    if not isinstance(connectors, list) or not isinstance(mates, list):
        raise PipelineError("structured assembly requires connector and mate arrays")
    connector_parts = {
        connector.get("id"): connector.get("part_id")
        for connector in connectors
        if isinstance(connector, dict)
    }
    mated_children = {
        connector_parts.get(mate.get("child_connector_id"))
        for mate in mates
        if isinstance(mate, dict)
    }
    missing_mates = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        attachment = part.get("attachment")
        kind = attachment.get("type") if isinstance(attachment, dict) else None
        if kind != "root" and part.get("id") not in mated_children:
            missing_mates.append(str(part.get("id")))
    if missing_mates:
        raise PipelineError(
            "structured non-root parts require host-solved mates: "
            + ", ".join(sorted(missing_mates))
        )


def _validate_structured_planning_candidate(
    workspace: Workspace,
    *,
    min_parts: int,
) -> dict[str, Any]:
    """Validate source authored by the structured planning stage.

    Runtime plan normalization intentionally keeps old, non-assembly plans readable.
    The structured planner has a stronger contract: it must author the assembly
    explicitly instead of relying on those compatibility defaults.
    """

    try:
        raw_plan = json.loads(
            workspace.plan_path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-standard numeric constant {token!r}")
            ),
        )
    except OSError as exc:
        raise PipelineError(f"src/plan.json is unreadable: {exc}") from exc
    except (json.JSONDecodeError, ValueError) as exc:
        raise PipelineError(f"src/plan.json is invalid JSON: {exc}") from exc
    if not isinstance(raw_plan, dict) or "assembly" not in raw_plan:
        raise PipelineError(
            "structured planning must produce an explicit assembly object"
        )
    normalized_plan = _validate_plan(workspace.plan_path)
    _validate_explicit_assembly_contract(normalized_plan, min_parts=min_parts)
    return normalized_plan


def _preserved_structured_planning_failure(
    workspace: Workspace,
    *,
    min_parts: int,
) -> str | None:
    """Return why an unfinished explicit planning candidate cannot resume.

    A structured state file means part authoring has already begun, so replanning
    would invalidate accepted checkpoints. Old plans without an explicit assembly
    continue through the compatibility path instead of being silently upgraded.
    """

    if os.path.lexists(_structured_state_path(workspace)):
        return None
    try:
        raw_plan = json.loads(workspace.plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw_plan, dict) or "assembly" not in raw_plan:
        return None
    try:
        _validate_structured_planning_candidate(
            workspace,
            min_parts=min_parts,
        )
    except PipelineError as exc:
        return str(exc)
    return None


def _checkpoint_name(index: int, part_id: str) -> str:
    slug = re.sub(r"[^a-z0-9_-]+", "-", part_id.lower()).strip("-_") or "part"
    return f"{index + 1:03d}-{slug}"


def _existing_trajectory(workspace: Workspace, iteration: int) -> Path:
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise PipelineError("trajectory iteration must be a non-negative integer")
    path = workspace.root / "trajectories" / f"iter_{iteration:02d}"
    if path.is_symlink() or not path.is_dir():
        raise PipelineError(f"trajectory iter_{iteration:02d} is unavailable")
    return path


def _build_incremental_checkpoint(
    workspace: Workspace,
    runtime: BlenderRuntime,
    *,
    plan: dict[str, Any],
    completed_part_ids: list[str],
    previous_signature: dict[str, Any],
    reconstruction_mode: str,
    granularity: str,
    timeout_s: int,
    checkpoint_name: str,
    progress: ProgressReporter | None = None,
) -> dict[str, Any]:
    """Build, re-import, render, and validate one incremental part prefix."""

    program_snapshot = _source_snapshot(workspace.program_path, label="src/program.py")
    plan_snapshot = _source_snapshot(workspace.plan_path, label="src/plan.json")
    _guard_program(workspace.program_path)
    reference_snapshot: bytes | None = None
    if reconstruction_mode == "glb-ref":
        reference_snapshot, _ = _verified_glb_snapshot(workspace)

    with tempfile.TemporaryDirectory(
        prefix=".procagen3d-part-", dir=workspace.root
    ) as directory:
        clean_root = Path(directory)
        staged_program = clean_root / "program.py"
        staged_reference = clean_root / "reference.glb"
        staged_assembly = clean_root / "assembly_transforms.json"
        staged_artifacts = clean_root / "artifacts"
        staged_program.write_bytes(program_snapshot)
        if reference_snapshot is not None:
            staged_reference.write_bytes(reference_snapshot)
        assembly_runtime = _assembly_runtime_document(
            plan,
            part_ids=completed_part_ids,
        )
        if assembly_runtime is not None:
            write_json(staged_assembly, assembly_runtime)

        stage_id = f"part-checkpoint-{len(completed_part_ids):03d}"
        with progress_step(
            progress,
            stage_id,
            f"Building incremental checkpoint {checkpoint_name}",
        ) as stage:
            arguments: list[str | Path] = [
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
                arguments.extend(["--reference-glb", staged_reference])
            if staged_assembly.is_file():
                arguments.extend(["--assembly-transforms", staged_assembly])
            result = runtime.run_stage(
                "build_asset", arguments, cwd=clean_root, timeout_s=timeout_s
            )
            staged_artifacts.mkdir(parents=True, exist_ok=True)
            (staged_artifacts / "build.stdout.log").write_text(
                result.stdout, encoding="utf-8"
            )
            (staged_artifacts / "build.stderr.log").write_text(
                result.stderr, encoding="utf-8"
            )
            require_success(result, stage=f"incremental part {checkpoint_name}")
            model_path = staged_artifacts / "model.glb"
            model_probe = probe_glb(model_path)
            if (
                not model_probe.get("self_contained")
                or model_probe.get("reference_readiness") != "pass"
            ):
                raise PipelineError("incremental checkpoint GLB is not self-contained")
            validation = validate_incremental_probe(
                plan=plan,
                completed_part_ids=completed_part_ids,
                probe=model_probe,
                previous_signature=previous_signature,
            )
            compiled = runtime.run_stage(
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
                compiled.stdout, encoding="utf-8"
            )
            (staged_artifacts / "compiled_probe.stderr.log").write_text(
                compiled.stderr, encoding="utf-8"
            )
            require_success(compiled, stage=f"incremental re-import {checkpoint_name}")
            model_probe["path"] = "model.glb"
            write_json(staged_artifacts / "model_probe.json", model_probe)
            scene_report_path = staged_artifacts / "scene_report.json"
            if scene_report_path.is_symlink() or not scene_report_path.is_file():
                raise PipelineError(
                    "incremental compiled probe did not publish scene_report.json"
                )
            checkpoint_report = {
                "schema_version": 1,
                "checkpoint": checkpoint_name,
                "program_sha256": hashlib.sha256(program_snapshot).hexdigest(),
                "plan_sha256": hashlib.sha256(plan_snapshot).hexdigest(),
                "model_sha256": sha256(model_path),
                "scene_report_sha256": sha256(scene_report_path),
                "validation": validation,
                "scene": model_probe.get("scene", {}),
            }
            write_json(staged_artifacts / "checkpoint.json", checkpoint_report)
            write_json(
                staged_artifacts / "build_manifest.json",
                {
                    "schema_version": 1,
                    "kind": "incremental-part-checkpoint",
                    "program_sha256": checkpoint_report["program_sha256"],
                    "plan_sha256": checkpoint_report["plan_sha256"],
                    "model_sha256": checkpoint_report["model_sha256"],
                    "scene_report_sha256": checkpoint_report[
                        "scene_report_sha256"
                    ],
                    "checkpoint": checkpoint_name,
                    "clean_room": True,
                    "compiled_glb_verified_in_separate_process": True,
                },
            )
            checkpoint_target = workspace.root / "checkpoints" / checkpoint_name
            _snapshot_directory(staged_artifacts, checkpoint_target)
            _replace_directory(staged_artifacts, workspace.artifacts_dir)
            stage.complete(
                f"Accepted {len(completed_part_ids)} of {len(plan.get('parts', []))} parts",
                completed_parts=len(completed_part_ids),
                checkpoint=f"checkpoints/{checkpoint_name}",
            )
    return checkpoint_report


def _structured_geometry_signature(
    workspace: Workspace, plan: dict[str, Any]
) -> dict[str, Any]:
    """Validate the full compiled part inventory and return its frozen signature."""

    probe_path = workspace.artifacts_dir / "model_probe.json"
    if probe_path.is_symlink() or not probe_path.is_file():
        raise PipelineError("structured build did not publish a regular model_probe.json")
    try:
        probe = json.loads(probe_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"structured model probe is unreadable: {exc}") from exc
    if not isinstance(probe, dict):
        raise PipelineError("structured model probe must contain an object")
    try:
        validation = validate_incremental_probe(
            plan=plan,
            completed_part_ids=list(structured_part_order(plan)),
            probe=probe,
            previous_signature=None,
        )
    except ValueError as exc:
        raise PipelineError(f"structured model inventory is invalid: {exc}") from exc
    return dict(validation["geometry_signature"])


def _assert_material_scene_identity(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    label: str,
) -> None:
    """Freeze object hierarchy and placement while allowing material seams."""

    if set(before) != set(after):
        raise PipelineError(f"{label} changed the structured object inventory")
    fields = (
        "name",
        "local_matrix",
        "world_matrix",
        "parent_identity",
        "bounds",
    )
    changes: list[str] = []
    for object_key in sorted(before):
        prior = before.get(object_key)
        current = after.get(object_key)
        if not isinstance(prior, dict) or not isinstance(current, dict):
            raise PipelineError(f"{label} has an invalid geometry identity record")
        for field in fields:
            if field not in prior or field not in current:
                raise PipelineError(
                    f"{label} lacks {field!r} for object {object_key!r}"
                )
            if prior[field] != current[field]:
                changes.append(f"{object_key}.{field}")
    if changes:
        summary = ", ".join(changes[:12])
        if len(changes) > 12:
            summary += f", and {len(changes) - 12} more"
        raise PipelineError(f"{label} changed frozen scene identity: {summary}")


def _solve_part_world_transforms(plan: dict[str, Any]) -> dict[str, list[list[float]]]:
    try:
        from .assembly import solve_assembly_transforms

        solved = solve_assembly_transforms(plan)
    except (ImportError, ValueError) as exc:
        raise PipelineError(f"assembly transform solver rejected the plan: {exc}") from exc
    return {
        str(part_id): [[float(value) for value in row] for row in matrix]
        for part_id, matrix in solved.items()
    }


def _run_incremental_authoring(
    workspace: Workspace,
    runtime: BlenderRuntime,
    config: PipelineConfig,
    *,
    reconstruction_mode: str,
    granularity: str,
    quality_profile: QualityProfile,
    user_prompt: str,
    state: dict[str, Any],
    next_iteration: int,
    agent_runs: list[dict[str, Any]],
    structured_stages: list[dict[str, Any]],
    progress: ProgressReporter | None,
) -> tuple[int, int | None, dict[str, Any]]:
    plan = _validate_plan(workspace.plan_path)
    order = structured_part_order(plan)
    contract_digest = _structured_contract_sha256(plan)
    acceptance_digest = _structured_acceptance_sha256(plan)
    if state.get("program_sha256") != sha256(workspace.program_path):
        raise PipelineError("structured program changed before incremental authoring")
    if state.get("contract_sha256") != contract_digest:
        raise PipelineError("structured part contract changed before incremental authoring")
    if state.get("pbr_sha256") != _structured_pbr_sha256(plan):
        raise PipelineError("structured PBR baseline changed before incremental authoring")
    if state.get("acceptance_sha256") != acceptance_digest:
        raise PipelineError("structured acceptance policy changed before incremental authoring")
    if state.get("topology_sha256") != _structured_topology_sha256(plan):
        raise PipelineError("structured topology changed before incremental authoring")
    if tuple(state.get("part_order", ())) != order:
        raise PipelineError("assembly.part_order changed after structured planning")
    completed = list(state.get("completed_parts", []))
    if tuple(completed) != order[: len(completed)]:
        raise PipelineError("structured part progress is not an assembly-order prefix")
    previous_signature = dict(state.get("geometry_signature", {}))
    transforms = _solve_part_world_transforms(plan)
    active_iteration = _matching_source_iteration(workspace)

    for part_index in range(len(completed), len(order)):
        part_id = order[part_index]
        part = next(
            item for item in plan["parts"] if isinstance(item, dict) and item.get("id") == part_id
        )
        failure = _resume_part_failure(
            workspace,
            state=state,
            part_id=part_id,
        )
        accepted = False
        accepted_program = workspace.program_path.read_bytes()
        accepted_plan = workspace.plan_path.read_bytes()
        for attempt in range(config.max_part_repairs + 1):
            iteration = next_iteration
            next_iteration += 1
            repairing = failure is not None or attempt > 0
            phase = "part-repair" if repairing else "part-authoring"
            try:
                run = _invoke_agent(
                    workspace,
                    backend_name=config.backend,
                    prompt=incremental_part_prompt(
                        part=part,
                        assembly=dict(plan["assembly"]),
                        completed_part_ids=completed,
                        part_index=part_index,
                        part_count=len(order),
                        solved_world_transform=transforms.get(part_id),
                        checkpoint_failure=failure,
                    ),
                    iteration=iteration,
                    timeout_s=config.llm_timeout_s,
                    include_candidate=bool(completed),
                    is_repair=repairing,
                    progress=progress,
                )
                agent_runs.append(
                    _annotated_agent_run(run, iteration=iteration, phase=phase)
                )
                if workspace.plan_path.read_bytes() != accepted_plan:
                    raise PipelineError(
                        "incremental part author edited the frozen src/plan.json"
                    )
                updated_plan = _validate_plan(workspace.plan_path)
                if structured_part_order(updated_plan) != order:
                    raise PipelineError("part author changed assembly.part_order")
                if _structured_contract_sha256(updated_plan) != contract_digest:
                    raise PipelineError(
                        "part author changed the frozen parts/assembly/articulation contract"
                    )
                plan = updated_plan
                checkpoint = _build_incremental_checkpoint(
                    workspace,
                    runtime,
                    plan=plan,
                    completed_part_ids=[*completed, part_id],
                    previous_signature=previous_signature,
                    reconstruction_mode=reconstruction_mode,
                    granularity=granularity,
                    timeout_s=config.blender_timeout_s,
                    checkpoint_name=_checkpoint_name(part_index, part_id),
                    progress=progress,
                )
            except (PipelineError, BlenderError, OSError, ValueError) as exc:
                failure = _bounded_pending_part_error(str(exc))
                _restore_source_pair(workspace, accepted_program, accepted_plan)
                state.update(
                    pending_part_failure={"part_id": part_id, "error": failure},
                    **_structured_plan_bindings(workspace, plan),
                )
                write_json(_structured_state_path(workspace), state)
                if isinstance(exc, _AgentInvocationError):
                    agent_runs.append(
                        _failed_agent_run(exc, iteration=iteration, phase=phase)
                    )
                structured_stages.append(
                    {
                        "phase": phase,
                        "part_id": part_id,
                        "iteration": iteration,
                        "attempt": attempt,
                        "passed": False,
                        "error": failure,
                    }
                )
                if attempt >= config.max_part_repairs:
                    raise PipelineError(
                        f"incremental part {part_id!r} failed after {attempt + 1} attempts: {failure}"
                    ) from exc
                continue

            validation = checkpoint["validation"]
            completed.append(part_id)
            previous_signature = dict(validation["geometry_signature"])
            active_iteration = iteration
            trajectory = _existing_trajectory(workspace, iteration)
            checkpoint_model = (
                workspace.root
                / "checkpoints"
                / _checkpoint_name(part_index, part_id)
                / "model.glb"
            )
            if checkpoint_model.is_file():
                shutil.copy2(checkpoint_model, trajectory / "model.glb")
            state.pop("pending_part_failure", None)
            state.update(
                phase="parts" if len(completed) < len(order) else "geometry",
                completed_parts=completed,
                geometry_signature=previous_signature,
                **_structured_plan_bindings(workspace, plan),
            )
            state.setdefault("checkpoints", []).append(
                {
                    "part_id": part_id,
                    "path": f"checkpoints/{_checkpoint_name(part_index, part_id)}",
                    "iteration": iteration,
                }
            )
            write_json(_structured_state_path(workspace), state)
            structured_stages.append(
                {
                    "phase": phase,
                    "part_id": part_id,
                    "iteration": iteration,
                    "attempt": attempt,
                    "passed": True,
                    "checkpoint": f"checkpoints/{_checkpoint_name(part_index, part_id)}",
                }
            )
            accepted = True
            break
        if not accepted:  # pragma: no cover - the exhausted branch raises above
            raise PipelineError(f"incremental part {part_id!r} was not accepted")
    return next_iteration, active_iteration, state


def _run_geometry_acceptance(
    workspace: Workspace,
    runtime: BlenderRuntime,
    config: PipelineConfig,
    *,
    reconstruction_mode: str,
    granularity: str,
    quality_profile: QualityProfile,
    user_prompt: str,
    state: dict[str, Any],
    next_iteration: int,
    active_iteration: int | None,
    agent_runs: list[dict[str, Any]],
    structured_stages: list[dict[str, Any]],
    progress: ProgressReporter | None,
) -> tuple[int, int | None, dict[str, Any], dict[str, Any]]:
    current_plan = _validate_plan(workspace.plan_path)
    _validate_explicit_assembly_contract(
        current_plan,
        min_parts=config.min_structured_parts,
    )
    if state.get("program_sha256") != sha256(workspace.program_path):
        raise PipelineError("geometry stage source does not match the accepted program")
    if state.get("contract_sha256") != _structured_contract_sha256(current_plan):
        raise PipelineError("geometry stage source does not match the accepted contract")
    if state.get("pbr_sha256") != _structured_pbr_sha256(current_plan):
        raise PipelineError("geometry stage source changed frozen PBR declarations")
    if state.get("acceptance_sha256") != _structured_acceptance_sha256(current_plan):
        raise PipelineError("geometry stage source changed the acceptance policy")
    if state.get("topology_sha256") != _structured_topology_sha256(current_plan):
        raise PipelineError("geometry stage source changed the assembly topology")
    comparison: dict[str, Any] | None = None
    accepted_plan = current_plan
    current_signature: dict[str, Any] | None = None
    for attempt in range(config.max_geometry_repairs + 1):
        if comparison is None:
            comparison = build_workspace(
                workspace,
                runtime,
                min_score=config.min_score,
                reconstruction_mode=reconstruction_mode,
                granularity=granularity,
                quality_profile=quality_profile,
                timeout_s=config.blender_timeout_s,
                trajectory_dir=(
                    _existing_trajectory(workspace, active_iteration)
                    if active_iteration is not None
                    else None
                ),
                progress=progress,
            )
            accepted_plan = _validate_plan(workspace.plan_path)
            current_signature = _structured_geometry_signature(
                workspace, accepted_plan
            )
        assert current_signature is not None
        geometry_passed = geometry_gates_passed(comparison)
        structured_stages.append(
            {
                "phase": "geometry-acceptance",
                "attempt": attempt,
                "passed": geometry_passed,
                "score": comparison.get("score"),
            }
        )
        if geometry_passed or attempt >= config.max_geometry_repairs:
            next_phase = "materials" if geometry_passed else "final"
            state.update(
                phase=next_phase,
                geometry_passed=geometry_passed,
                geometry_signature=current_signature,
                materials_status=(
                    state.get("materials_status", "pending")
                    if geometry_passed
                    else "blocked-geometry"
                ),
                **_structured_plan_bindings(workspace, accepted_plan),
            )
            write_json(_structured_state_path(workspace), state)
            break
        target = select_repair_target(comparison, include_materials=False)
        if target is None:
            state.update(
                phase="final",
                geometry_passed=False,
                geometry_signature=current_signature,
                materials_status="blocked-geometry",
                **_structured_plan_bindings(workspace, accepted_plan),
            )
            structured_stages[-1]["passed"] = False
            structured_stages[-1]["error"] = (
                "geometry gates failed without a deterministic repair target"
            )
            write_json(_structured_state_path(workspace), state)
            break
        iteration = next_iteration
        next_iteration += 1
        accepted_program_source = workspace.program_path.read_bytes()
        accepted_plan_source = workspace.plan_path.read_bytes()
        with tempfile.TemporaryDirectory(
            prefix=".procagen3d-geometry-transaction-", dir=workspace.root
        ) as transaction_directory:
            accepted_artifacts = Path(transaction_directory) / "artifacts"
            _snapshot_directory(workspace.artifacts_dir, accepted_artifacts)
            try:
                run = _invoke_agent(
                    workspace,
                    backend_name=config.backend,
                    prompt=targeted_repair_prompt(
                        user_prompt=user_prompt,
                        target=target,
                        iteration=iteration,
                        reconstruction_mode=reconstruction_mode,
                        granularity=granularity,
                        quality_profile=quality_profile,
                        geometry_only=True,
                    ),
                    iteration=iteration,
                    timeout_s=config.llm_timeout_s,
                    include_candidate=True,
                    is_repair=True,
                    progress=progress,
                )
                agent_runs.append(
                    _annotated_agent_run(
                        run, iteration=iteration, phase="geometry-repair"
                    )
                )
                updated_plan = _validate_plan(workspace.plan_path)
                _validate_explicit_assembly_contract(
                    updated_plan,
                    min_parts=config.min_structured_parts,
                )
                if _structured_topology_sha256(updated_plan) != state.get(
                    "topology_sha256"
                ):
                    raise PipelineError(
                        "geometry repair changed frozen part ownership or assembly topology"
                    )
                if _structured_acceptance_sha256(updated_plan) != state.get(
                    "acceptance_sha256"
                ):
                    raise PipelineError(
                        "geometry repair changed frozen dimensions, constraints, or acceptance policy"
                    )
                if _structured_pbr_sha256(updated_plan) != state.get("pbr_sha256"):
                    raise PipelineError(
                        "geometry repair changed PBR declarations reserved for the material pass"
                    )
                placement_changes = _structured_placement_changes(
                    accepted_plan, updated_plan
                )
                if len(placement_changes) > 1:
                    raise PipelineError(
                        "geometry repair changed multiple placement fields: "
                        + ", ".join(placement_changes)
                    )
                candidate_comparison = build_workspace(
                    workspace,
                    runtime,
                    min_score=config.min_score,
                    reconstruction_mode=reconstruction_mode,
                    granularity=granularity,
                    quality_profile=quality_profile,
                    timeout_s=config.blender_timeout_s,
                    trajectory_dir=_existing_trajectory(workspace, iteration),
                    progress=progress,
                )
                candidate_signature = _structured_geometry_signature(
                    workspace, updated_plan
                )
            except (PipelineError, BlenderError, OSError, ValueError) as exc:
                _restore_source_pair(
                    workspace,
                    accepted_program_source,
                    accepted_plan_source,
                )
                _snapshot_directory(accepted_artifacts, workspace.artifacts_dir)
                if isinstance(exc, _AgentInvocationError):
                    agent_runs.append(
                        _failed_agent_run(
                            exc,
                            iteration=iteration,
                            phase="geometry-repair",
                        )
                    )
                structured_stages.append(
                    {
                        "phase": "geometry-repair",
                        "iteration": iteration,
                        "attempt": attempt,
                        "passed": False,
                        "target": target.as_dict(),
                        "error": str(exc),
                        "rolled_back": True,
                    }
                )
                continue
        state.update(
            phase="geometry",
            geometry_signature=candidate_signature,
            **_structured_plan_bindings(workspace, updated_plan),
        )
        write_json(_structured_state_path(workspace), state)
        structured_stages.append(
            {
                "phase": "geometry-repair",
                "iteration": iteration,
                "attempt": attempt,
                "passed": True,
                "target": target.as_dict(),
                "geometry_gates_passed": geometry_gates_passed(
                    candidate_comparison
                ),
            }
        )
        active_iteration = iteration
        accepted_plan = updated_plan
        current_signature = candidate_signature
        comparison = candidate_comparison
    assert comparison is not None
    return next_iteration, active_iteration, state, comparison


def _run_material_stage(
    workspace: Workspace,
    runtime: BlenderRuntime,
    config: PipelineConfig,
    *,
    reconstruction_mode: str,
    granularity: str,
    quality_profile: QualityProfile,
    user_prompt: str,
    state: dict[str, Any],
    next_iteration: int,
    active_iteration: int | None,
    agent_runs: list[dict[str, Any]],
    structured_stages: list[dict[str, Any]],
    progress: ProgressReporter | None,
) -> tuple[int, int | None, dict[str, Any], dict[str, Any] | None, str | None]:
    """Apply PBR materials in isolation and reject geometry-changing edits."""

    if not config.dedicated_materials:
        plan = _validate_plan(workspace.plan_path)
        _validate_explicit_assembly_contract(
            plan,
            min_parts=config.min_structured_parts,
        )
        if state.get("program_sha256") != sha256(workspace.program_path):
            raise PipelineError(
                "material skip source does not match accepted geometry program"
            )
        if state.get("contract_sha256") != _structured_contract_sha256(plan):
            raise PipelineError(
                "material skip source does not match accepted geometry contract"
            )
        if state.get("pbr_sha256") != _structured_pbr_sha256(plan):
            raise PipelineError(
                "material skip source does not match accepted PBR baseline"
            )
        if state.get("acceptance_sha256") != _structured_acceptance_sha256(plan):
            raise PipelineError(
                "material skip source does not match accepted geometry policy"
            )
        if state.get("topology_sha256") != _structured_topology_sha256(plan):
            raise PipelineError(
                "material skip source does not match accepted assembly topology"
            )
        state.update(
            phase="final",
            materials_status="skipped",
            **_structured_plan_bindings(workspace, plan),
        )
        write_json(_structured_state_path(workspace), state)
        return next_iteration, active_iteration, state, None, None

    try:
        from .materials import (
            build_material_pass_context,
            compare_material_pass_geometry,
            material_plan_from_document,
        )
    except ImportError as exc:  # pragma: no cover - packaging guard
        raise PipelineError(f"dedicated material component is unavailable: {exc}") from exc

    plan = _validate_plan(workspace.plan_path)
    _validate_explicit_assembly_contract(
        plan,
        min_parts=config.min_structured_parts,
    )
    if state.get("program_sha256") != sha256(workspace.program_path):
        raise PipelineError("material stage source does not match accepted geometry program")
    if state.get("contract_sha256") != _structured_contract_sha256(plan):
        raise PipelineError("material stage source does not match accepted geometry contract")
    if state.get("pbr_sha256") != _structured_pbr_sha256(plan):
        raise PipelineError("material stage source does not match accepted PBR baseline")
    if state.get("acceptance_sha256") != _structured_acceptance_sha256(plan):
        raise PipelineError("material stage source does not match accepted geometry policy")
    if state.get("topology_sha256") != _structured_topology_sha256(plan):
        raise PipelineError("material stage source does not match accepted assembly topology")
    order = list(structured_part_order(plan))
    if (
        state.get("phase") != "materials"
        or state.get("geometry_passed") is not True
        or state.get("materials_status") != "pending"
        or state.get("completed_parts") != order
    ):
        raise PipelineError("material stage requires accepted, fully assembled geometry")
    _validate_structured_resume_artifacts(
        workspace,
        plan=plan,
        state=state,
    )
    pre_material_scene = _read_regular_json(
        workspace.artifacts_dir / "scene_report.json",
        label="pre-material scene report",
    )
    reference_report = _read_regular_json(
        workspace.evidence_dir / "glb_probe.json",
        label="reference GLB probe",
    )
    part_ids = [str(part["id"]) for part in plan["parts"]]
    existing_material_plan = plan.get("material_plan")
    context = build_material_pass_context(
        existing_material_plan,
        part_ids=part_ids,
        reference_report=reference_report,
        pre_material_geometry_report=pre_material_scene,
    )
    source_program = workspace.program_path.read_bytes()
    source_plan = workspace.plan_path.read_bytes()
    baseline_geometry_signature = dict(state.get("geometry_signature", {}))
    pre_material_artifacts = workspace.root / "checkpoints" / "pre-material"
    _snapshot_directory(workspace.artifacts_dir, pre_material_artifacts)
    failure: str | None = None
    last_guard: dict[str, Any] | None = None

    for attempt in range(config.max_material_repairs + 1):
        if attempt:
            _restore_source_pair(workspace, source_program, source_plan)
            _snapshot_directory(pre_material_artifacts, workspace.artifacts_dir)
        last_guard = None
        iteration = next_iteration
        next_iteration += 1
        phase = "material-authoring" if attempt == 0 else "material-repair"
        try:
            run = _invoke_agent(
                workspace,
                backend_name=config.backend,
                prompt=dedicated_material_prompt(
                    plan=plan,
                    geometry_signature=context,
                    failure=failure,
                ),
                iteration=iteration,
                timeout_s=config.llm_timeout_s,
                include_candidate=True,
                is_repair=attempt > 0,
                progress=progress,
            )
            agent_runs.append(
                _annotated_agent_run(run, iteration=iteration, phase=phase)
            )
            updated_plan = _validate_plan(workspace.plan_path)
            _validate_explicit_assembly_contract(
                updated_plan,
                min_parts=config.min_structured_parts,
            )
            if state.get("contract_sha256") != _structured_contract_sha256(
                updated_plan
            ):
                raise PipelineError(
                    "dedicated material pass changed the frozen structured contract"
                )
            parsed_materials = material_plan_from_document(
                updated_plan.get("material_plan")
            )
            if not parsed_materials.enabled:
                raise PipelineError("dedicated material pass produced an empty material_plan")
            parsed_materials.validate_part_ids(part_ids)
            planned_objects = {
                str(part["id"]): set(part.get("object_names", []))
                for part in updated_plan["parts"]
                if isinstance(part, dict) and isinstance(part.get("id"), str)
            }
            for assignment in parsed_materials.assignments:
                unknown_objects = sorted(
                    set(assignment.object_names)
                    - planned_objects.get(assignment.part_id, set())
                )
                if unknown_objects:
                    raise PipelineError(
                        f"material assignment for part {assignment.part_id!r} references "
                        "unowned objects: " + ", ".join(unknown_objects)
                    )
            assigned_parts = {
                assignment.part_id for assignment in parsed_materials.assignments
            }
            missing_material_parts = sorted(set(part_ids) - assigned_parts)
            if missing_material_parts:
                raise PipelineError(
                    "dedicated material pass left parts unassigned: "
                    + ", ".join(missing_material_parts)
                )
            checkpoint = _build_incremental_checkpoint(
                workspace,
                runtime,
                plan=updated_plan,
                completed_part_ids=list(structured_part_order(updated_plan)),
                # Material slots may repartition glTF primitives and duplicate
                # seam vertices without changing evaluated geometry. Inventory
                # is still exact here; the material-independent triangle guard
                # below owns shape immutability for this phase.
                previous_signature=None,
                reconstruction_mode=reconstruction_mode,
                granularity=granularity,
                timeout_s=config.blender_timeout_s,
                checkpoint_name="materials",
                progress=progress,
            )
            checkpoint_scene = _read_regular_json(
                workspace.artifacts_dir / "scene_report.json",
                label="material checkpoint scene report",
            )
            checkpoint_signature = dict(
                checkpoint["validation"]["geometry_signature"]
            )
            _assert_material_scene_identity(
                baseline_geometry_signature,
                checkpoint_signature,
                label="material checkpoint",
            )
            checkpoint_guard = compare_material_pass_geometry(
                pre_material_scene, checkpoint_scene
            )
            last_guard = checkpoint_guard.as_dict()
            if not checkpoint_guard.passed:
                changed = ", ".join(
                    item.field for item in checkpoint_guard.violations
                )
                raise PipelineError(
                    "material-only geometry guard rejected changes to " + changed
                )
            material_comparison = build_workspace(
                workspace,
                runtime,
                min_score=config.min_score,
                reconstruction_mode=reconstruction_mode,
                granularity=granularity,
                quality_profile=quality_profile,
                timeout_s=config.blender_timeout_s,
                trajectory_dir=_existing_trajectory(workspace, iteration),
                progress=progress,
            )
            final_scene = _read_regular_json(
                workspace.artifacts_dir / "scene_report.json",
                label="final material scene report",
            )
            final_guard = compare_material_pass_geometry(
                pre_material_scene, final_scene
            )
            last_guard = final_guard.as_dict()
            last_guard["checkpoint_guard"] = checkpoint_guard.as_dict()
            if not final_guard.passed:
                changed = ", ".join(item.field for item in final_guard.violations)
                raise PipelineError(
                    "final material geometry guard rejected changes to " + changed
                )
            final_signature = _structured_geometry_signature(
                workspace, updated_plan
            )
            _assert_material_scene_identity(
                baseline_geometry_signature,
                final_signature,
                label="final material build",
            )
            failed_material_gates = material_gate_failures(material_comparison)
            if failed_material_gates:
                selected = failed_material_gates[0]
                raise PipelineError(
                    "dedicated material gate failed: "
                    + str(
                        selected.get("message")
                        or selected.get("gate")
                        or "unknown material gate"
                    )
                )
            write_json(workspace.artifacts_dir / "material_guard.json", last_guard)
            material_checkpoint = workspace.root / "checkpoints" / "materials"
            write_json(material_checkpoint / "material_guard.json", last_guard)
            trajectory = _existing_trajectory(workspace, iteration)
            trajectory_model = trajectory / "model.glb"
            published_model = workspace.artifacts_dir / "model.glb"
            if (
                trajectory_model.is_symlink()
                or not trajectory_model.is_file()
                or published_model.is_symlink()
                or not published_model.is_file()
                or sha256(trajectory_model) != sha256(published_model)
            ):
                raise PipelineError(
                    "final material trajectory GLB does not match the published build"
                )
            active_iteration = iteration
            state.update(
                phase="final",
                materials_status="applied",
                material_guard=last_guard,
                geometry_signature=final_signature,
                **_structured_plan_bindings(workspace, updated_plan),
            )
            write_json(_structured_state_path(workspace), state)
            structured_stages.append(
                {
                    "phase": phase,
                    "iteration": iteration,
                    "attempt": attempt,
                    "passed": True,
                    "checkpoint": "checkpoints/materials",
                    "guard": last_guard,
                    "scene": checkpoint.get("scene", {}),
                    "comparison": {
                        "score": material_comparison.get("score"),
                        "passed": material_comparison.get("passed"),
                        "material_gates_passed": True,
                    },
                }
            )
            return next_iteration, active_iteration, state, last_guard, None
        except (PipelineError, BlenderError, OSError, ValueError) as exc:
            failure = str(exc)
            _restore_source_pair(workspace, source_program, source_plan)
            _snapshot_directory(pre_material_artifacts, workspace.artifacts_dir)
            if isinstance(exc, _AgentInvocationError):
                agent_runs.append(
                    _failed_agent_run(exc, iteration=iteration, phase=phase)
                )
            structured_stages.append(
                {
                    "phase": phase,
                    "iteration": iteration,
                    "attempt": attempt,
                    "passed": False,
                    "error": failure,
                    **({"guard": last_guard} if last_guard is not None else {}),
                }
            )
            if attempt < config.max_material_repairs:
                continue

    _restore_source_pair(workspace, source_program, source_plan)
    _snapshot_directory(pre_material_artifacts, workspace.artifacts_dir)
    state.update(
        phase="final",
        materials_status="failed",
        material_error=failure,
        **_structured_plan_bindings(workspace, plan),
    )
    write_json(_structured_state_path(workspace), state)
    return next_iteration, active_iteration, state, last_guard, failure


def _validated_urdf_link_meshes(
    manifest_path: Path,
    *,
    plan: dict[str, Any],
    model_path: Path,
) -> dict[str, str]:
    """Validate Blender's split manifest and return deterministic URDF URIs."""

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"URDF part manifest is unreadable: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise PipelineError("URDF part manifest has an unsupported schema")
    if manifest.get("frame_convention") != "incoming-connector":
        raise PipelineError(
            "URDF part meshes must be exported in incoming-connector frames so "
            "revolute children rotate about their mate, not the modeling origin"
        )
    if manifest.get("source_sha256") != sha256(model_path):
        raise PipelineError("URDF part manifest does not match artifacts/model.glb")
    records = manifest.get("parts")
    if not isinstance(records, list):
        raise PipelineError("URDF part manifest lacks part records")

    expected = {
        str(part["id"])
        for part in plan.get("parts", [])
        if isinstance(part, dict) and isinstance(part.get("id"), str)
    }
    staged_parts = manifest_path.parent
    seen: set[str] = set()
    link_meshes: dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict):
            raise PipelineError("URDF part manifest contains a non-object record")
        part_id = record.get("part_id")
        filename = record.get("path")
        if not isinstance(part_id, str) or part_id in seen:
            raise PipelineError("URDF part manifest contains a duplicate or invalid part id")
        if filename != f"{part_id}.glb":
            raise PipelineError(
                f"URDF part {part_id!r} did not use its deterministic GLB filename"
            )
        part_path = staged_parts / filename
        if not part_path.is_file() or part_path.is_symlink():
            raise PipelineError(f"URDF part mesh is missing: {filename}")
        if record.get("sha256") != sha256(part_path):
            raise PipelineError(f"URDF part mesh hash mismatch: {filename}")
        try:
            from .urdf import URDFExportError, assert_link_local_mesh

            assert_link_local_mesh(part_path)
        except URDFExportError as exc:
            raise PipelineError(str(exc)) from exc
        seen.add(part_id)
        link_meshes[part_id] = f"urdf_parts/{filename}"
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise PipelineError(
            "URDF part manifest must cover the plan exactly; "
            f"missing={missing}, extra={extra}"
        )
    return dict(sorted(link_meshes.items()))


def _validated_urdf_zero_pose_report(
    report_path: Path,
    *,
    model_path: Path,
    urdf_path: Path,
    assembly_path: Path,
) -> dict[str, Any]:
    """Validate and bind Blender's rest-geometry and motion report."""

    report = _read_regular_json(
        report_path,
        label="URDF zero-pose validation report",
    )
    if report.get("schema_version") != 2 or report.get("status") != "passed":
        raise PipelineError("URDF articulation validation did not report a passing result")
    expected_hashes = {
        "model_sha256": sha256(model_path),
        "urdf_sha256": sha256(urdf_path),
        "assembly_sha256": sha256(assembly_path),
    }
    for field, expected in expected_hashes.items():
        if report.get(field) != expected:
            raise PipelineError(
                f"URDF zero-pose validation report has a stale {field} binding"
            )
    for field in (
        "relative_tolerance",
        "absolute_tolerance",
        "max_bounds_error",
    ):
        value = report.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise PipelineError(
                f"URDF zero-pose validation report has an invalid {field}"
            )
    if float(report["relative_tolerance"]) <= 0.0:
        raise PipelineError(
            "URDF zero-pose validation report must use a positive tolerance"
        )
    if float(report["absolute_tolerance"]) <= 0.0:
        raise PipelineError(
            "URDF zero-pose validation report must use a positive absolute tolerance"
        )
    if float(report["max_bounds_error"]) > float(report["absolute_tolerance"]):
        raise PipelineError("URDF zero-pose validation exceeded its reported tolerance")
    for field in (
        "max_motion_matrix_error",
        "max_motion_translation_error",
        "max_motion_rotation_error_rad",
    ):
        value = report.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise PipelineError(
                f"URDF articulation validation report has an invalid {field}"
            )
    if float(report["max_motion_matrix_error"]) > float(
        report["relative_tolerance"]
    ):
        raise PipelineError("URDF nonzero motion validation exceeded its tolerance")
    count_fields = (
        "part_count",
        "object_count",
        "source_vertex_count",
        "reconstructed_vertex_count",
        "source_triangle_count",
        "reconstructed_triangle_count",
        "vertex_count_changed_object_count",
        "motion_probe_count",
        "movable_joint_count",
    )
    for field in count_fields:
        value = report.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise PipelineError(
                f"URDF zero-pose validation report has an invalid {field}"
            )
    if report["part_count"] <= 0 or report["object_count"] <= 0:
        raise PipelineError("URDF zero-pose validation report has empty coverage")
    if report["motion_probe_count"] <= 0:
        raise PipelineError("URDF articulation validation has no nonzero motion probes")
    if report["motion_probe_count"] != report["movable_joint_count"]:
        raise PipelineError("URDF articulation validation has inconsistent probe coverage")
    if report["source_triangle_count"] != report["reconstructed_triangle_count"]:
        raise PipelineError("URDF zero-pose validation changed the triangle count")
    if report["vertex_count_changed_object_count"] > report["object_count"]:
        raise PipelineError(
            "URDF zero-pose validation has an invalid changed-vertex object count"
        )
    parts = report.get("parts")
    if not isinstance(parts, list) or len(parts) != report.get("part_count"):
        raise PipelineError("URDF zero-pose validation report has invalid part coverage")
    motion_probes = report.get("motion_probes")
    if (
        not isinstance(motion_probes, list)
        or len(motion_probes) != report["motion_probe_count"]
    ):
        raise PipelineError("URDF articulation validation has invalid motion coverage")
    return report


def _export_urdf_artifacts(
    workspace: Workspace,
    runtime: BlenderRuntime,
    *,
    plan: dict[str, Any],
    timeout_s: int,
) -> dict[str, Any]:
    """Validate articulation, split host-solved links, and export the URDF."""

    try:
        from .urdf import export_urdf, render_urdf
    except ImportError as exc:  # pragma: no cover - packaging guard
        raise PipelineError(f"URDF component is unavailable: {exc}") from exc

    model_path = workspace.artifacts_dir / "model.glb"
    # Fail before invoking Blender when the plan has not independently opted
    # into a valid mechanical articulation tree.
    render_urdf(plan, model_path)
    assembly_runtime = _urdf_link_runtime_document(plan)
    link_meshes: dict[str, str] | None = None
    zero_pose_validation: dict[str, Any] | None = None
    if assembly_runtime is not None:
        with tempfile.TemporaryDirectory(
            prefix=".procagen3d-urdf-", dir=workspace.root
        ) as directory:
            staging_root = Path(directory)
            assembly_path = staging_root / "assembly_transforms.json"
            staged_parts = staging_root / "urdf_parts"
            write_json(assembly_path, assembly_runtime)
            result = runtime.run_stage(
                "export_urdf_parts",
                [
                    "--glb",
                    model_path,
                    "--assembly-transforms",
                    assembly_path,
                    "--output-dir",
                    staged_parts,
                ],
                cwd=staging_root,
                timeout_s=timeout_s,
            )
            staged_parts.mkdir(parents=True, exist_ok=True)
            (staged_parts / "export.stdout.log").write_text(
                result.stdout, encoding="utf-8"
            )
            (staged_parts / "export.stderr.log").write_text(
                result.stderr, encoding="utf-8"
            )
            require_success(result, stage="URDF per-link mesh export")
            link_meshes = _validated_urdf_link_meshes(
                staged_parts / "manifest.json",
                plan=plan,
                model_path=model_path,
            )
            # Validate URI coverage before publishing the split meshes.
            rendered = render_urdf(plan, model_path, link_meshes=link_meshes)
            staged_urdf = staging_root / "model.urdf"
            staged_urdf.write_text(rendered.xml, encoding="utf-8")
            validation_path = staged_parts / "zero_pose_validation.json"
            validation_result = runtime.run_stage(
                "validate_urdf_zero_pose",
                [
                    "--model-glb",
                    model_path,
                    "--urdf",
                    staged_urdf,
                    "--assembly-transforms",
                    assembly_path,
                    "--parts-dir",
                    staged_parts,
                    "--out",
                    validation_path,
                ],
                cwd=staging_root,
                timeout_s=timeout_s,
            )
            (staged_parts / "validation.stdout.log").write_text(
                validation_result.stdout, encoding="utf-8"
            )
            (staged_parts / "validation.stderr.log").write_text(
                validation_result.stderr, encoding="utf-8"
            )
            require_success(validation_result, stage="URDF zero-pose validation")
            zero_pose_validation = _validated_urdf_zero_pose_report(
                validation_path,
                model_path=model_path,
                urdf_path=staged_urdf,
                assembly_path=assembly_path,
            )
            _replace_directory(staged_parts, workspace.artifacts_dir / "urdf_parts")

    result = export_urdf(
        plan,
        model_path,
        workspace.artifacts_dir / "model.urdf",
        enabled=True,
        link_meshes=link_meshes,
    )
    report = result.as_dict()
    if link_meshes is not None:
        report["parts_manifest"] = "urdf_parts/manifest.json"
    if zero_pose_validation is not None:
        if result.urdf_sha256 != zero_pose_validation["urdf_sha256"]:
            raise PipelineError(
                "published URDF does not match the zero-pose validated document"
            )
        validation_path = (
            workspace.artifacts_dir / "urdf_parts" / "zero_pose_validation.json"
        )
        report["zero_pose_validation"] = {
            "path": "urdf_parts/zero_pose_validation.json",
            "sha256": sha256(validation_path),
            "part_count": zero_pose_validation["part_count"],
            "object_count": zero_pose_validation["object_count"],
            "source_vertex_count": zero_pose_validation["source_vertex_count"],
            "reconstructed_vertex_count": zero_pose_validation[
                "reconstructed_vertex_count"
            ],
            "source_triangle_count": zero_pose_validation["source_triangle_count"],
            "reconstructed_triangle_count": zero_pose_validation[
                "reconstructed_triangle_count"
            ],
            "vertex_count_changed_object_count": zero_pose_validation[
                "vertex_count_changed_object_count"
            ],
            "max_bounds_error": zero_pose_validation["max_bounds_error"],
            "absolute_tolerance": zero_pose_validation["absolute_tolerance"],
            "motion_probe_count": zero_pose_validation["motion_probe_count"],
            "movable_joint_count": zero_pose_validation["movable_joint_count"],
            "max_motion_matrix_error": zero_pose_validation[
                "max_motion_matrix_error"
            ],
            "max_motion_translation_error": zero_pose_validation[
                "max_motion_translation_error"
            ],
            "max_motion_rotation_error_rad": zero_pose_validation[
                "max_motion_rotation_error_rad"
            ],
        }
    return report


def run_pipeline(
    workspace: Workspace,
    config: PipelineConfig,
    *,
    prepare_only: bool = False,
    force_probe: bool = False,
    progress: ProgressReporter | None = None,
) -> dict[str, Any]:
    pipeline_mode = validate_pipeline_mode(config.pipeline_mode)
    if config.max_initial_agent_retries < 0:
        raise ValueError("max_initial_agent_retries cannot be negative")
    if config.max_repairs < 0:
        raise ValueError("max_repairs cannot be negative")
    if config.max_fidelity_repairs < 1:
        raise ValueError("max_fidelity_repairs must be at least one")
    for name, value in (
        ("max_part_repairs", config.max_part_repairs),
        ("max_geometry_repairs", config.max_geometry_repairs),
        ("max_material_repairs", config.max_material_repairs),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} cannot be negative")
    if (
        isinstance(config.min_structured_parts, bool)
        or not isinstance(config.min_structured_parts, int)
        or config.min_structured_parts < 1
    ):
        raise ValueError("min_structured_parts must be a positive integer")
    reconstruction_mode = validate_reconstruction_mode(config.reconstruction_mode)
    granularity = validate_granularity(config.granularity)
    quality_profile = resolve_quality_profile(
        granularity,
        surface_fidelity=config.surface_fidelity,
        detail_richness=config.detail_richness,
        material_fidelity=config.material_fidelity,
        structural_coherence=config.structural_coherence,
    )
    workspace.update_manifest(
        reconstruction_mode=reconstruction_mode,
        granularity=granularity,
        quality_profile=quality_profile.as_dict(),
        pipeline_mode=pipeline_mode,
        dedicated_materials=config.dedicated_materials,
        export_urdf=config.export_urdf,
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
                "quality_profile": quality_profile.as_dict(),
                "pipeline_mode": pipeline_mode,
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
            "quality_profile": quality_profile.as_dict(),
            "pipeline_mode": pipeline_mode,
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
            quality_profile=quality_profile.as_dict(),
            pipeline_mode=pipeline_mode,
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
    structured_stages: list[dict[str, Any]] = []
    structured_warnings: list[str] = []
    next_iteration = workspace.next_trajectory_iteration()
    active_iteration = _matching_source_iteration(workspace)
    initial_agent_retries_used = 0
    build_repairs_used = 0
    created_structured_plan = False
    source_pair_missing = (
        not workspace.program_path.is_file() or not workspace.plan_path.is_file()
    )
    initial_agent_failure = (
        _preserved_structured_planning_failure(
            workspace,
            min_parts=config.min_structured_parts,
        )
        if pipeline_mode == "structured" and not source_pair_missing
        else None
    )
    started_from_preserved_failure = initial_agent_failure is not None
    if source_pair_missing or started_from_preserved_failure:
        while True:
            iteration = next_iteration
            if started_from_preserved_failure:
                phase = (
                    "initial-repair"
                    if initial_agent_retries_used == 0
                    else "initial-repair-retry"
                )
            else:
                phase = (
                    "initial"
                    if initial_agent_retries_used == 0
                    else "initial-retry"
                )
            structured_retry = (
                pipeline_mode == "structured" and initial_agent_failure is not None
            )
            try:
                run = _invoke_agent(
                    workspace,
                    backend_name=config.backend,
                    prompt=(
                        assembly_planning_prompt(
                            root=workspace.root,
                            image=workspace.image_path,
                            user_prompt=user_prompt,
                            reconstruction_mode=reconstruction_mode,
                            granularity=granularity,
                            quality_profile=quality_profile,
                            export_urdf=config.export_urdf,
                            failure=initial_agent_failure,
                        )
                        if pipeline_mode == "structured"
                        else initial_prompt(
                            root=workspace.root,
                            image=workspace.image_path,
                            user_prompt=user_prompt,
                            reconstruction_mode=reconstruction_mode,
                            granularity=granularity,
                            quality_profile=quality_profile,
                        )
                    ),
                    iteration=iteration,
                    timeout_s=config.llm_timeout_s,
                    include_candidate=structured_retry,
                    is_repair=structured_retry,
                    progress=progress,
                )
                if pipeline_mode == "structured":
                    _validate_structured_planning_candidate(
                        workspace,
                        min_parts=config.min_structured_parts,
                    )
                    created_structured_plan = True
            except PipelineError as exc:
                if pipeline_mode == "structured":
                    initial_agent_failure = str(exc)
                agent_runs.append(
                    _failed_agent_run(exc, iteration=iteration, phase=phase)
                )
                next_iteration += 1
                if initial_agent_retries_used < config.max_initial_agent_retries:
                    initial_agent_retries_used += 1
                    emit_progress(
                        progress,
                        "warning",
                        "initial-agent",
                        "Initial agent failed; retrying in a new preserved trajectory",
                        error=str(exc),
                        retry=initial_agent_retries_used,
                        maximum=config.max_initial_agent_retries,
                    )
                    continue
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
                        "quality_profile": quality_profile.as_dict(),
                        "pipeline_mode": pipeline_mode,
                        "agent_runs": agent_runs,
                        "build_attempts": [],
                        "retry_budget": {
                            "initial_agent": {
                                "used": initial_agent_retries_used,
                                "maximum": config.max_initial_agent_retries,
                            },
                            "schema_build": {
                                "used": 0,
                                "maximum": config.max_repairs,
                            },
                            "post_render": {
                                "used": 0,
                                "maximum": config.max_fidelity_repairs,
                            },
                        },
                        "error": str(exc),
                    },
                )
                raise
            agent_runs.append(
                _annotated_agent_run(run, iteration=iteration, phase=phase)
            )
            next_iteration += 1
            break
        active_iteration = _matching_source_iteration(workspace)

    structured_state: dict[str, Any] | None = None
    if pipeline_mode == "structured":
        try:
            structured_state = _load_structured_state(workspace)
            try:
                raw_plan = json.loads(workspace.plan_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                # Existing malformed/legacy source remains eligible for the
                # compatibility build-repair path. Newly planned structured source
                # was already required to validate above.
                raw_plan = None
            explicit_structured_plan = (
                isinstance(raw_plan, dict) and "assembly" in raw_plan
            )
            if explicit_structured_plan:
                structured_plan = _validate_plan(workspace.plan_path)
                _validate_explicit_assembly_contract(
                    structured_plan,
                    min_parts=config.min_structured_parts,
                )
                if structured_state is None:
                    structured_state = _new_structured_state(
                        workspace,
                        structured_plan,
                    )
            elif created_structured_plan:
                # The initial planning branch already established this invariant;
                # retain a defensive failure if source promotion ever violates it.
                raise PipelineError(
                    "structured planning did not preserve an explicit assembly"
                )

            if structured_state is not None and structured_state.get("phase") == "parts":
                next_iteration, active_iteration, structured_state = (
                    _run_incremental_authoring(
                        workspace,
                        runtime,
                        config,
                        reconstruction_mode=reconstruction_mode,
                        granularity=granularity,
                        quality_profile=quality_profile,
                        user_prompt=user_prompt,
                        state=structured_state,
                        next_iteration=next_iteration,
                        agent_runs=agent_runs,
                        structured_stages=structured_stages,
                        progress=progress,
                    )
                )
            if structured_state is not None and structured_state.get("phase") == "geometry":
                (
                    next_iteration,
                    active_iteration,
                    structured_state,
                    _geometry_comparison,
                ) = _run_geometry_acceptance(
                    workspace,
                    runtime,
                    config,
                    reconstruction_mode=reconstruction_mode,
                    granularity=granularity,
                    quality_profile=quality_profile,
                    user_prompt=user_prompt,
                    state=structured_state,
                    next_iteration=next_iteration,
                    active_iteration=active_iteration,
                    agent_runs=agent_runs,
                    structured_stages=structured_stages,
                    progress=progress,
                )
                if not structured_state.get("geometry_passed"):
                    structured_warnings.append(
                        "Geometry acceptance still has deterministic gate failures; "
                        "the dedicated material pass was not run and the asset remains needs-review"
                    )
            if structured_state is not None and structured_state.get("phase") == "materials":
                (
                    next_iteration,
                    active_iteration,
                    structured_state,
                    _material_guard,
                    material_failure,
                ) = _run_material_stage(
                    workspace,
                    runtime,
                    config,
                    reconstruction_mode=reconstruction_mode,
                    granularity=granularity,
                    quality_profile=quality_profile,
                    user_prompt=user_prompt,
                    state=structured_state,
                    next_iteration=next_iteration,
                    active_iteration=active_iteration,
                    agent_runs=agent_runs,
                    structured_stages=structured_stages,
                    progress=progress,
                )
                if material_failure:
                    structured_warnings.append(
                        "Dedicated material pass was rejected and the pre-material source was restored: "
                        + material_failure
                    )
        except (PipelineError, BlenderError, OSError, ValueError) as exc:
            workspace.update_manifest(status="failed")
            report = {
                "schema_version": 2,
                "status": "failed",
                "workspace": str(workspace.root),
                "backend": config.backend,
                "pipeline_mode": pipeline_mode,
                "reconstruction_mode": reconstruction_mode,
                "granularity": granularity,
                "quality_profile": quality_profile.as_dict(),
                "stage": "structured-authoring",
                "structured_stages": structured_stages,
                "agent_runs": agent_runs,
                "build_attempts": [],
                "error": str(exc),
            }
            write_json(workspace.root / "run_report.json", report)
            raise

    last_failure: str | None = None
    comparison: dict[str, Any] | None = None
    valid_artifact = False
    fidelity_repairs_used = 0
    build_attempt = 0
    build_phase = "initial"
    maximum_builds = (
        1
        if structured_state is not None
        else 1 + config.max_repairs + config.max_fidelity_repairs
    )
    best_snapshot_context = tempfile.TemporaryDirectory(
        prefix=".procagen3d-best-",
        dir=workspace.root,
    )
    best_artifacts = Path(best_snapshot_context.name) / "artifacts"
    best_program: bytes | None = None
    best_plan: bytes | None = None
    best_comparison: dict[str, Any] | None = None
    best_quality: tuple[Any, ...] | None = None
    best_attempt: int | None = None
    best_iteration: int | None = None
    latest_valid_attempt: int | None = None
    improvement_stalled = False
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
                quality_profile=quality_profile,
                timeout_s=config.blender_timeout_s,
                trajectory_dir=(
                    workspace.root / "trajectories" / f"iter_{active_iteration:02d}"
                    if active_iteration is not None
                    else None
                ),
                progress=progress,
            )
            valid_artifact = True
            latest_valid_attempt = build_attempt
            quality = _comparison_quality(comparison, attempt=build_attempt)
            improved = best_quality is None or quality > best_quality
            built_attempt: dict[str, Any] = {
                "attempt": build_attempt,
                "phase": build_phase,
                "built": True,
                "comparison": comparison,
                "improved": improved,
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
            if improved:
                _snapshot_directory(workspace.artifacts_dir, best_artifacts)
                best_program = workspace.program_path.read_bytes()
                best_plan = workspace.plan_path.read_bytes()
                best_comparison = comparison
                best_quality = quality
                best_attempt = build_attempt
                best_iteration = active_iteration
            build_attempts.append(built_attempt)
            last_failure = _comparison_failure(comparison, min_score=config.min_score)
            if last_failure:
                emit_progress(
                    progress,
                    "warning",
                    "build-attempt",
                    last_failure,
                )
            if comparison.get("passed"):
                last_failure = None
                break
            if build_phase == "post-render-repair" and not improved:
                improvement_stalled = True
                emit_progress(
                    progress,
                    "warning",
                    "build-attempt",
                    "Post-render repair did not improve the retained candidate; stopping early",
                    best_attempt=best_attempt,
                    stalled_attempt=build_attempt,
                )
                break
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
                "no unscoped fallback repair is allowed after structured staging"
                if structured_state is not None
                else (
                    "preparing a schema/build repair"
                    if build_repairs_used < config.max_repairs
                    else "no schema/build repairs remain"
                )
            )
            emit_progress(
                progress,
                "warning",
                "build-attempt",
                f"Build attempt {build_attempt + 1} failed; {next_action}",
                error=last_failure,
            )
            if structured_state is not None:
                # Structured source already passed clean builds at each stage.
                # Do not bypass those contracts with an unscoped fallback edit.
                break
            if build_repairs_used >= config.max_repairs:
                break
            repair_iteration = next_iteration
            next_iteration += 1
            build_repairs_used += 1
            try:
                run = _invoke_agent(
                    workspace,
                    backend_name=config.backend,
                    prompt=repair_prompt(
                        root=workspace.root,
                        user_prompt=user_prompt,
                        failure=last_failure,
                        comparison=comparison,
                        iteration=repair_iteration,
                        reconstruction_mode=reconstruction_mode,
                        granularity=granularity,
                        quality_profile=quality_profile,
                    ),
                    iteration=repair_iteration,
                    timeout_s=config.llm_timeout_s,
                    include_candidate=valid_artifact,
                    is_repair=True,
                    progress=progress,
                )
            except PipelineError as repair_exc:
                last_failure = str(repair_exc)
                agent_runs.append(
                    _failed_agent_run(
                        repair_exc,
                        iteration=repair_iteration,
                        phase="schema-build-repair",
                    )
                )
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
            agent_runs.append(
                _annotated_agent_run(
                    run,
                    iteration=repair_iteration,
                    phase="schema-build-repair",
                )
            )
            build_attempt += 1
            build_phase = "schema-build-repair"
            active_iteration = _matching_source_iteration(workspace)
            continue

        # Successful candidates enter an adaptive fidelity loop. Stop on pass,
        # on the first non-improving valid repair, or when the independent
        # post-render budget is exhausted.
        if structured_state is not None:
            # Geometry and material repairs have their own bounded, guarded
            # transactions. A generic edit here could bypass part freezing or
            # the material-only geometry fingerprint.
            break
        if fidelity_repairs_used >= config.max_fidelity_repairs:
            break
        repair_iteration = next_iteration
        next_iteration += 1
        fidelity_repairs_used += 1
        repair_target = (
            select_repair_target(comparison, include_materials=True)
            if comparison is not None
            else None
        )
        try:
            run = _invoke_agent(
                workspace,
                backend_name=config.backend,
                prompt=(
                    targeted_repair_prompt(
                        user_prompt=user_prompt,
                        target=repair_target,
                        iteration=repair_iteration,
                        reconstruction_mode=reconstruction_mode,
                        granularity=granularity,
                        quality_profile=quality_profile,
                    )
                    if repair_target is not None
                    else repair_prompt(
                        root=workspace.root,
                        user_prompt=user_prompt,
                        failure=last_failure,
                        comparison=comparison,
                        iteration=repair_iteration,
                        reconstruction_mode=reconstruction_mode,
                        granularity=granularity,
                        quality_profile=quality_profile,
                    )
                ),
                iteration=repair_iteration,
                timeout_s=config.llm_timeout_s,
                include_candidate=True,
                is_repair=True,
                progress=progress,
            )
        except PipelineError as exc:
            last_failure = str(exc)
            agent_runs.append(
                _failed_agent_run(
                    exc,
                    iteration=repair_iteration,
                    phase="post-render-repair",
                )
            )
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
        agent_runs.append(
            _annotated_agent_run(
                run,
                iteration=repair_iteration,
                phase="post-render-repair",
            )
        )
        build_attempt += 1
        build_phase = "post-render-repair"
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
            "quality_profile": quality_profile.as_dict(),
            "agent_runs": agent_runs,
            "build_attempts": build_attempts,
            "retry_budget": {
                "initial_agent": {
                    "used": initial_agent_retries_used,
                    "maximum": config.max_initial_agent_retries,
                },
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
        best_snapshot_context.cleanup()
        raise PipelineError(last_failure or "no valid artifact was produced")

    if (
        best_program is None
        or best_plan is None
        or best_comparison is None
        or best_attempt is None
        or not best_artifacts.is_dir()
    ):
        best_snapshot_context.cleanup()
        raise PipelineError("valid builds completed without a retainable best candidate")

    current_source_is_best = (
        workspace.program_path.is_file()
        and workspace.plan_path.is_file()
        and workspace.program_path.read_bytes() == best_program
        and workspace.plan_path.read_bytes() == best_plan
    )
    candidate_restored = latest_valid_attempt != best_attempt or not current_source_is_best
    if not current_source_is_best:
        rejected = workspace.root / "trajectories" / f"iter_{max(0, next_iteration - 1):02d}"
        rejected.mkdir(parents=True, exist_ok=True)
        if workspace.program_path.is_file():
            shutil.copy2(workspace.program_path, rejected / "rejected_program.py")
        if workspace.plan_path.is_file():
            shutil.copy2(workspace.plan_path, rejected / "rejected_plan.json")
        _restore_source_pair(workspace, best_program, best_plan)

    # Always publish from the private snapshot so the selected artifacts and
    # source are one deterministic candidate, never whichever build ran last.
    _replace_directory(best_artifacts, workspace.artifacts_dir)
    comparison = best_comparison
    for attempt_record in build_attempts:
        attempt_record["selected"] = bool(
            attempt_record.get("built")
            and attempt_record.get("attempt") == best_attempt
        )

    artifact_manifest = json.loads(
        (workspace.artifacts_dir / "build_manifest.json").read_text(encoding="utf-8")
    )
    source_matches_artifact = (
        artifact_manifest.get("program_sha256") == sha256(workspace.program_path)
        and artifact_manifest.get("plan_sha256") == sha256(workspace.plan_path)
    )
    best_snapshot_context.cleanup()
    if not source_matches_artifact:
        raise PipelineError("selected best-candidate artifacts do not match restored source")

    if structured_state is not None:
        # Final bounded repairs can update plan metadata after the structured
        # phases. Bind the terminal state to the selected source, never to a
        # rejected trajectory.
        terminal_plan = _validate_plan(workspace.plan_path)
        if structured_state.get("topology_sha256") != _structured_topology_sha256(
            terminal_plan
        ):
            raise PipelineError(
                "selected structured candidate changed frozen assembly topology"
            )
        if structured_state.get("acceptance_sha256") != _structured_acceptance_sha256(
            terminal_plan
        ):
            raise PipelineError(
                "selected structured candidate changed frozen acceptance policy"
            )
        if structured_state.get("contract_sha256") != _structured_contract_sha256(
            terminal_plan
        ):
            raise PipelineError(
                "selected structured candidate changed the accepted geometry contract"
            )
        structured_state.update(
            **_structured_plan_bindings(workspace, terminal_plan),
        )
        write_json(_structured_state_path(workspace), structured_state)

    if (
        structured_state is not None
        and isinstance(structured_state.get("material_guard"), dict)
    ):
        write_json(
            workspace.artifacts_dir / "material_guard.json",
            structured_state["material_guard"],
        )

    urdf_report: dict[str, Any] | None = None
    urdf_failure: str | None = None
    if config.export_urdf:
        try:
            urdf_report = _export_urdf_artifacts(
                workspace,
                runtime,
                plan=_validate_plan(workspace.plan_path),
                timeout_s=config.blender_timeout_s,
            )
            write_json(workspace.artifacts_dir / "urdf_report.json", urdf_report)
        except (BlenderError, PipelineError, OSError, ValueError) as exc:
            urdf_failure = str(exc)
            urdf_report = {
                "status": "failed",
                "enabled": True,
                "error": urdf_failure,
            }
            write_json(workspace.artifacts_dir / "urdf_report.json", urdf_report)

    warning_parts: list[str] = list(structured_warnings)
    acceptance_warning = _comparison_failure(comparison, min_score=config.min_score)
    if acceptance_warning:
        warning_parts.append(acceptance_warning)
    if improvement_stalled:
        warning_parts.append(
            f"Post-render improvement stalled at build attempt {build_attempt}; "
            f"best build attempt {best_attempt} was retained"
        )
    if candidate_restored:
        warning_parts.append(
            "The last failed or non-improving repair was retained in its trajectory; "
            "source was restored to the best compiled artifact"
        )
    if last_failure and last_failure not in warning_parts:
        warning_parts.append(last_failure)
    if urdf_failure:
        warning_parts.append("URDF export failed: " + urdf_failure)
    last_failure = ". ".join(warning_parts) or None

    structured_stage_failed = bool(
        structured_state is not None
        and (
            structured_state.get("geometry_passed") is not True
            or structured_state.get("materials_status")
            not in {"applied", "skipped"}
        )
    )
    status = (
        "complete"
        if comparison
        and comparison["passed"]
        and not urdf_failure
        and not structured_stage_failed
        else "needs-review"
    )
    deliverables = {
        "program": "src/program.py",
        "plan": "src/plan.json",
        "glb": "artifacts/model.glb",
        "blend": "artifacts/scene.blend",
        "comparison": "artifacts/comparison.json",
        "build_manifest": "artifacts/build_manifest.json",
        **(
            {
                "surface_comparison": "artifacts/surface_comparison.json",
                "surface_residuals": "artifacts/surface_residuals/manifest.json",
            }
            if SURFACE_QUALITY_SETTINGS[quality_profile.surface_fidelity].enabled
            else {}
        ),
        **(
            {"material_guard": "artifacts/material_guard.json"}
            if (workspace.artifacts_dir / "material_guard.json").is_file()
            else {}
        ),
        **(
            {
                "urdf": "artifacts/model.urdf",
                "urdf_report": "artifacts/urdf_report.json",
                **(
                    {"urdf_parts": "artifacts/urdf_parts/manifest.json"}
                    if (workspace.artifacts_dir / "urdf_parts" / "manifest.json").is_file()
                    else {}
                ),
            }
            if (workspace.artifacts_dir / "model.urdf").is_file()
            else (
                {"urdf_report": "artifacts/urdf_report.json"}
                if urdf_report is not None
                else {}
            )
        ),
    }
    report = {
        "schema_version": 2,
        "status": status,
        "workspace": str(workspace.root),
        "backend": config.backend,
        "pipeline_mode": pipeline_mode,
        "reconstruction_mode": reconstruction_mode,
        "granularity": granularity,
        "quality_profile": quality_profile.as_dict(),
        "structured_stages": structured_stages,
        "structured_state": structured_state,
        "urdf": urdf_report,
        "agent_runs": agent_runs,
        "build_attempts": build_attempts,
        "retry_budget": {
            "initial_agent": {
                "used": initial_agent_retries_used,
                "maximum": config.max_initial_agent_retries,
            },
            "schema_build": {
                "used": build_repairs_used,
                "maximum": 0 if structured_state is not None else config.max_repairs,
            },
            "post_render": {
                "used": fidelity_repairs_used,
                "maximum": (
                    0 if structured_state is not None else config.max_fidelity_repairs
                ),
            },
            **(
                {
                    "part": {"maximum_per_part": config.max_part_repairs},
                    "geometry": {"maximum": config.max_geometry_repairs},
                    "materials": {"maximum": config.max_material_repairs},
                }
                if structured_state is not None
                else {}
            ),
        },
        "best_candidate": {
            "build_attempt": best_attempt,
            "trajectory_iteration": best_iteration,
            "score": comparison["score"] if comparison else None,
            "passed": comparison["passed"] if comparison else False,
            "restored": candidate_restored,
        },
        "score": comparison["score"] if comparison else None,
        "passed": comparison["passed"] if comparison else False,
        "warning": last_failure,
        "deliverables": deliverables,
    }
    write_json(workspace.root / "run_report.json", report)
    workspace.update_manifest(
        status=status,
        score=report["score"],
        reconstruction_mode=reconstruction_mode,
        granularity=granularity,
        quality_profile=quality_profile.as_dict(),
        pipeline_mode=pipeline_mode,
        dedicated_materials=config.dedicated_materials,
        export_urdf=config.export_urdf,
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
