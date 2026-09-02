from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import procagen3d.pipeline as pipeline
from procagen3d.blender import BlenderError
from procagen3d.workspace import Workspace, sha256, write_json


def _frame(origin: list[float]) -> dict[str, object]:
    return {
        "origin": origin,
        "x_axis": [1.0, 0.0, 0.0],
        "y_axis": [0.0, 1.0, 0.0],
        "z_axis": [0.0, 0.0, 1.0],
    }


def _connector(connector_id: str, part_id: str, origin: list[float]) -> dict[str, object]:
    return {
        "id": connector_id,
        "part_id": part_id,
        "interface": "cylindrical",
        "role": "neutral",
        "frame": _frame(origin),
        "nominal_dimensions": {"diameter": 0.1},
    }


def _plan() -> dict[str, object]:
    return {
        "subject": "zero pose fixture",
        "subject_kind": "object",
        "parts": [
            {"id": "base", "object_names": ["Base"]},
            {"id": "arm", "object_names": ["Arm"]},
        ],
        "assembly": {
            "version": 1,
            "placement": "host-solved",
            "part_order": ["base", "arm"],
            "connectors": [
                _connector("base_hinge", "base", [1.0, 0.0, 0.0]),
                _connector("arm_hinge", "arm", [0.0, 0.0, 0.0]),
            ],
            "mates": [
                {
                    "id": "hinge",
                    "type": "revolute",
                    "parent_connector_id": "base_hinge",
                    "child_connector_id": "arm_hinge",
                    "fit": "none",
                    "clearance": 0.0,
                    "fit_offset": [0.0, 0.0, 0.0],
                    "nominal_dimensions": {"diameter": 0.1},
                    "rest": 0.0,
                    "limits": {"lower": -1.0, "upper": 1.0},
                }
            ],
        },
        "articulation": {
            "enabled": True,
            "mechanical": True,
            "joints": [],
        },
    }


def _result(*, ok: bool = True, stderr: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        ok=ok,
        stdout="validated" if ok else "",
        stderr=stderr,
        timed_out=False,
        returncode=0 if ok else 1,
    )


class _Runtime:
    def __init__(
        self,
        model: Path,
        *,
        fail_validation: bool = False,
        triangle_mismatch: bool = False,
    ) -> None:
        self.model = model
        self.fail_validation = fail_validation
        self.triangle_mismatch = triangle_mismatch
        self.stages: list[str] = []

    def run_stage(self, stage, arguments, *, cwd, timeout_s):
        self.stages.append(stage)
        if stage == "export_urdf_parts":
            output = Path(arguments[arguments.index("--output-dir") + 1])
            output.mkdir(parents=True)
            records = []
            for part_id in ("base", "arm"):
                path = output / f"{part_id}.glb"
                path.write_bytes(f"mesh:{part_id}".encode())
                records.append(
                    {
                        "part_id": part_id,
                        "path": path.name,
                        "sha256": sha256(path),
                    }
                )
            write_json(
                output / "manifest.json",
                {
                    "schema_version": 1,
                    "source_sha256": sha256(self.model),
                    "parts": records,
                },
            )
            return _result()

        assert stage == "validate_urdf_zero_pose"
        if self.fail_validation:
            return _result(ok=False, stderr="zero-pose mismatch")
        urdf = Path(arguments[arguments.index("--urdf") + 1])
        assembly = Path(
            arguments[arguments.index("--assembly-transforms") + 1]
        )
        output = Path(arguments[arguments.index("--out") + 1])
        assert 'rpy="1.57079632679 0 0"' in urdf.read_text(encoding="utf-8")
        write_json(
            output,
            {
                "schema_version": 1,
                "status": "passed",
                "model_sha256": sha256(self.model),
                "urdf_sha256": sha256(urdf),
                "assembly_sha256": sha256(assembly),
                "part_count": 2,
                "object_count": 2,
                "source_vertex_count": 100,
                "reconstructed_vertex_count": 104,
                "source_triangle_count": 50,
                "reconstructed_triangle_count": (
                    51 if self.triangle_mismatch else 50
                ),
                "vertex_count_changed_object_count": 1,
                "relative_tolerance": 1.0e-5,
                "absolute_tolerance": 2.0e-5,
                "max_bounds_error": 1.0e-8,
                "visual_rpy": [1.5707963267948966, 0.0, 0.0],
                "parts": [{"part_id": "base"}, {"part_id": "arm"}],
            },
        )
        return _result()


def _workspace(tmp_path: Path) -> tuple[Workspace, Path]:
    workspace = Workspace(tmp_path / "workspace")
    workspace.artifacts_dir.mkdir(parents=True)
    model = workspace.artifacts_dir / "model.glb"
    model.write_bytes(b"compiled model")
    return workspace, model


def test_urdf_zero_pose_gate_runs_before_split_meshes_are_published(
    tmp_path: Path,
) -> None:
    workspace, model = _workspace(tmp_path)
    runtime = _Runtime(model)

    report = pipeline._export_urdf_artifacts(
        workspace,
        runtime,  # type: ignore[arg-type]
        plan=_plan(),
        timeout_s=30,
    )

    assert runtime.stages == ["export_urdf_parts", "validate_urdf_zero_pose"]
    validation = report["zero_pose_validation"]
    assert validation["path"] == "urdf_parts/zero_pose_validation.json"
    assert validation["part_count"] == 2
    assert validation["object_count"] == 2
    assert validation["source_vertex_count"] == 100
    assert validation["reconstructed_vertex_count"] == 104
    assert validation["source_triangle_count"] == 50
    assert validation["reconstructed_triangle_count"] == 50
    assert validation["vertex_count_changed_object_count"] == 1
    published = workspace.artifacts_dir / validation["path"]
    assert published.is_file()
    assert validation["sha256"] == sha256(published)


def test_failed_zero_pose_gate_preserves_preexisting_published_parts(
    tmp_path: Path,
) -> None:
    workspace, model = _workspace(tmp_path)
    published = workspace.artifacts_dir / "urdf_parts"
    published.mkdir()
    sentinel = published / "keep.txt"
    sentinel.write_text("previous valid parts", encoding="utf-8")
    runtime = _Runtime(model, fail_validation=True)

    with pytest.raises(BlenderError, match="zero-pose mismatch"):
        pipeline._export_urdf_artifacts(
            workspace,
            runtime,  # type: ignore[arg-type]
            plan=_plan(),
            timeout_s=30,
        )

    assert runtime.stages == ["export_urdf_parts", "validate_urdf_zero_pose"]
    assert sentinel.read_text(encoding="utf-8") == "previous valid parts"


def test_zero_pose_report_allows_seam_vertices_but_rejects_triangle_changes(
    tmp_path: Path,
) -> None:
    workspace, model = _workspace(tmp_path)
    runtime = _Runtime(model, triangle_mismatch=True)

    with pytest.raises(pipeline.PipelineError, match="changed the triangle count"):
        pipeline._export_urdf_artifacts(
            workspace,
            runtime,  # type: ignore[arg-type]
            plan=_plan(),
            timeout_s=30,
        )

    assert not (workspace.artifacts_dir / "urdf_parts").exists()
