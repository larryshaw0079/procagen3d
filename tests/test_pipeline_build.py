from __future__ import annotations

import base64
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

import procagen3d.pipeline as pipeline
from procagen3d.workspace import Workspace, write_json


def _workspace(tmp_path: Path) -> Workspace:
    image = tmp_path / "reference.png"
    glb = tmp_path / "reference.glb"
    image.write_bytes(b"png")
    glb.write_bytes(b"glb")
    workspace = Workspace.create(
        base=tmp_path / "outputs",
        slug="fixture",
        image=image,
        glb=glb,
        prompt="",
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
            "construction_strategy": "one primitive",
            "identity_features": [],
            "limitations": [],
        },
    )
    workspace.program_path.write_text("def build():\n    pass\n", encoding="utf-8")
    return workspace


def _write_evidence(root: Path, *, size: int = 128) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "reference_views").mkdir()
    write_json(
        root / "glb_probe.json",
        {"self_contained": True, "reference_readiness": "pass"},
    )
    write_json(
        root / "reference_scene.json",
        {
            "geometry_object_count": 1,
            "bounds": {
                "min": [0.0, 0.0, 0.0],
                "max": [1.0, 1.0, 1.0],
                "dimensions": [1.0, 1.0, 1.0],
                "center": [0.5, 0.5, 0.5],
            },
        },
    )
    write_json(
        root / "camera_contract.json",
        {
            "resolution": [size, size],
            "views": [
                {
                    "name": name,
                    "location": [0.0, -4.5, 1.0],
                    "target": [0.0, 0.0, 1.0],
                    "ortho_scale": 2.5,
                }
                for name in pipeline.CANONICAL_VIEWS
            ],
        },
    )
    write_json(
        root / "reference_views" / "masks.json",
        {
            "schema_version": 2,
            "views": {
                name: {
                    "width": size,
                    "height": size,
                    "foreground_pixels": 0,
                    "encoding": "base64-msb-packbits",
                    "data": base64.b64encode(bytes((size * size + 7) // 8)).decode("ascii"),
                    "rgb_encoding": "base64-rgb8",
                    "rgb_data": base64.b64encode(bytes(size * size * 3)).decode("ascii"),
                }
                for name in pipeline.CANONICAL_VIEWS
            },
        },
    )
    for name in pipeline.CANONICAL_VIEWS:
        header = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", size, size)
        (root / "reference_views" / f"{name}.png").write_bytes(header)


def test_complete_evidence_requires_v2_masks_at_the_requested_resolution(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    _write_evidence(evidence)
    assert pipeline._complete_evidence(evidence, render_size=128)

    masks_path = evidence / "reference_views" / "masks.json"
    masks = json.loads(masks_path.read_text(encoding="utf-8"))
    masks["schema_version"] = 1
    write_json(masks_path, masks)
    assert not pipeline._complete_evidence(evidence, render_size=128)

    masks["schema_version"] = 2
    masks["views"]["iso"]["width"] = 64
    write_json(masks_path, masks)
    assert not pipeline._complete_evidence(evidence, render_size=128)


def test_character_plan_requires_typed_anatomy_analysis(tmp_path: Path) -> None:
    plan = {
        "subject": "wanderer",
        "subject_kind": "character",
        "coordinate_frame": {"up": "+Z"},
        "dimensions": [1.0, 0.5, 2.0],
        "parts": [{"name": "Body"}],
        "materials": [],
        "construction_strategy": "semantic primitives",
        "identity_features": [],
        "limitations": [],
    }
    path = tmp_path / "plan.json"
    write_json(path, plan)
    with pytest.raises(pipeline.PipelineError, match="character_analysis object"):
        pipeline._validate_plan(path)

    plan["subject_kind"] = ["character"]
    write_json(path, plan)
    with pytest.raises(pipeline.PipelineError, match="subject_kind"):
        pipeline._validate_plan(path)
    plan["subject_kind"] = "character"

    plan["character_analysis"] = {
        "pose": "upright with the staff held on character-left",
        "proportions": {"head_to_body": 0.3},
        "facial_landmarks": ["two large eyes"],
        "hair_or_headwear": ["hood"],
        "clothing_layers": ["cloak"],
        "held_props": ["staff"],
        "left_right_asymmetry": ["staff on character-left"],
        "inferred_features": ["back of cloak"],
    }
    write_json(path, plan)
    pipeline._validate_plan(path)


def test_complete_evidence_requires_canonical_camera_order(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    _write_evidence(evidence)
    contract_path = evidence / "camera_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["views"].reverse()
    write_json(contract_path, contract)

    assert not pipeline._complete_evidence(evidence, render_size=128)


def test_complete_evidence_treats_corruption_as_a_cache_miss(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    _write_evidence(evidence)
    camera_path = evidence / "camera_contract.json"
    original_camera = camera_path.read_bytes()
    camera_path.write_text("[]", encoding="utf-8")
    assert not pipeline._complete_evidence(evidence, render_size=128)

    camera_path.write_bytes(original_camera)
    front = evidence / "reference_views" / "front.png"
    front.write_bytes(b"")
    assert not pipeline._complete_evidence(evidence, render_size=128)

    _write_evidence(tmp_path / "fresh-evidence")
    masks_path = tmp_path / "fresh-evidence" / "reference_views" / "masks.json"
    masks = json.loads(masks_path.read_text(encoding="utf-8"))
    masks["views"]["front"]["rgb_data"] = "not-base64"
    write_json(masks_path, masks)
    assert not pipeline._complete_evidence(tmp_path / "fresh-evidence", render_size=128)


def test_prepare_reference_promotes_portable_provenance(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = _workspace(tmp_path)

    class FakeRuntime:
        def run_stage(self, stage, arguments, *, cwd, timeout_s):
            assert stage == "reference_probe"
            evidence = Path(arguments[arguments.index("--evidence-dir") + 1])
            probe = json.loads((evidence / "glb_probe.json").read_text(encoding="utf-8"))
            _write_evidence(evidence, size=128)
            write_json(evidence / "glb_probe.json", probe)
            scene = json.loads((evidence / "reference_scene.json").read_text(encoding="utf-8"))
            scene["source"] = str(workspace.glb_path)
            write_json(evidence / "reference_scene.json", scene)
            return SimpleNamespace(ok=True, stdout="ok", stderr="")

    monkeypatch.setattr(
        pipeline,
        "probe_glb",
        lambda path: {
            "path": str(path),
            "self_contained": True,
            "reference_readiness": "pass",
        },
    )

    result = pipeline.prepare_reference(
        workspace,
        FakeRuntime(),
        render_size=128,
        timeout_s=10,
        force=True,
    )

    probe = json.loads((workspace.evidence_dir / "glb_probe.json").read_text(encoding="utf-8"))
    scene = json.loads(
        (workspace.evidence_dir / "reference_scene.json").read_text(encoding="utf-8")
    )
    assert result["path"] == "inputs/reference.glb"
    assert probe["path"] == "inputs/reference.glb"
    assert scene["source"] == "inputs/reference.glb"


def test_build_reports_keep_workspace_relative_provenance(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = _workspace(tmp_path)
    workspace.evidence_dir.mkdir(exist_ok=True)
    (workspace.evidence_dir / "camera_contract.json").write_text("{}", encoding="utf-8")

    class FakeRuntime:
        def __init__(self):
            self.stages = []

        def run_stage(self, stage, arguments, *, cwd, timeout_s):
            self.stages.append(stage)
            artifacts = Path(arguments[arguments.index("--artifacts-dir") + 1])
            artifacts.mkdir(parents=True, exist_ok=True)
            if stage == "build_asset":
                (artifacts / "model.glb").write_bytes(b"compiled glb")
                (artifacts / "scene.blend").write_bytes(b"blend")
            elif stage == "compiled_probe":
                write_json(
                    artifacts / "scene_report.json",
                    {
                        "bounds": {
                            "min": [0.0, 0.0, 0.0],
                            "max": [1.0, 1.0, 1.0],
                            "dimensions": [1.0, 1.0, 1.0],
                            "center": [0.5, 0.5, 0.5],
                        },
                    },
                )
            else:  # pragma: no cover - makes unexpected stages explicit
                raise AssertionError(stage)
            return SimpleNamespace(ok=True, stdout="ok", stderr="")

    monkeypatch.setattr(
        pipeline,
        "probe_glb",
        lambda path: {
            "path": str(path),
            "self_contained": True,
            "reference_readiness": "pass",
        },
    )

    def fake_compare(**kwargs):
        report = {"score": 1.0, "passed": True}
        write_json(kwargs["output"], report)
        return report

    monkeypatch.setattr(pipeline, "compare_workspace", fake_compare)

    runtime = FakeRuntime()
    events = []
    result = pipeline.build_workspace(
        workspace,
        runtime,
        min_score=0.0,
        timeout_s=10,
        progress=events.append,
    )

    assert result["passed"] is True
    assert runtime.stages == ["build_asset", "compiled_probe"]
    assert [event.stage for event in events if event.kind == "start"] == [
        "source-validation",
        "source-guard",
        "blender-build",
        "export-validation",
        "compiled-probe",
        "artifact-provenance",
        "fidelity",
    ]
    model_probe = json.loads(
        (workspace.artifacts_dir / "model_probe.json").read_text(encoding="utf-8")
    )
    scene_report = json.loads(
        (workspace.artifacts_dir / "scene_report.json").read_text(encoding="utf-8")
    )
    build_manifest = json.loads(
        (workspace.artifacts_dir / "build_manifest.json").read_text(encoding="utf-8")
    )
    assert model_probe["path"] == "artifacts/model.glb"
    assert scene_report["program"] == "src/program.py"
    assert build_manifest["compiled_glb_verified_in_separate_process"] is True


def test_failed_staged_build_preserves_previous_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = _workspace(tmp_path)
    workspace.evidence_dir.mkdir(exist_ok=True)
    (workspace.evidence_dir / "camera_contract.json").write_text("{}", encoding="utf-8")
    workspace.artifacts_dir.mkdir(exist_ok=True)
    previous = workspace.artifacts_dir / "previous-valid.glb"
    previous.write_bytes(b"keep me")

    class FakeRuntime:
        def run_stage(self, stage, arguments, *, cwd, timeout_s):
            artifacts = Path(arguments[arguments.index("--artifacts-dir") + 1])
            artifacts.mkdir(parents=True)
            (artifacts / "model.glb").write_bytes(b"invalid new glb")
            (artifacts / "scene.blend").write_bytes(b"new blend")
            write_json(artifacts / "scene_report.json", {"program": str(cwd / "program.py")})
            return SimpleNamespace(ok=True, stdout="ok", stderr="")

    monkeypatch.setattr(
        pipeline,
        "probe_glb",
        lambda path: {
            "path": str(path),
            "self_contained": False,
            "reference_readiness": "fail",
        },
    )

    with pytest.raises(pipeline.PipelineError, match="not a self-contained drawable scene"):
        pipeline.build_workspace(workspace, FakeRuntime(), min_score=0.0, timeout_s=10)

    assert previous.read_bytes() == b"keep me"
    assert not (workspace.artifacts_dir / "model.glb").exists()


def test_source_change_during_build_rejects_staged_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = _workspace(tmp_path)
    workspace.evidence_dir.mkdir(exist_ok=True)
    (workspace.evidence_dir / "camera_contract.json").write_text("{}", encoding="utf-8")
    workspace.artifacts_dir.mkdir(exist_ok=True)
    previous = workspace.artifacts_dir / "previous-valid.glb"
    previous.write_bytes(b"keep me")

    class MutatingRuntime:
        def run_stage(self, stage, arguments, *, cwd, timeout_s):
            artifacts = Path(arguments[arguments.index("--artifacts-dir") + 1])
            artifacts.mkdir(parents=True, exist_ok=True)
            if stage == "build_asset":
                (artifacts / "model.glb").write_bytes(b"compiled glb")
                (artifacts / "scene.blend").write_bytes(b"blend")
                workspace.program_path.write_text(
                    "def build():\n    changed_during_build = True\n",
                    encoding="utf-8",
                )
            elif stage == "compiled_probe":
                write_json(
                    artifacts / "scene_report.json",
                    {
                        "bounds": {
                            "min": [0.0, 0.0, 0.0],
                            "max": [1.0, 1.0, 1.0],
                            "dimensions": [1.0, 1.0, 1.0],
                            "center": [0.5, 0.5, 0.5],
                        },
                    },
                )
            return SimpleNamespace(ok=True, stdout="ok", stderr="")

    monkeypatch.setattr(
        pipeline,
        "probe_glb",
        lambda path: {
            "path": str(path),
            "self_contained": True,
            "reference_readiness": "pass",
        },
    )

    def fake_compare(**kwargs):
        report = {"score": 1.0, "passed": True}
        write_json(kwargs["output"], report)
        return report

    monkeypatch.setattr(pipeline, "compare_workspace", fake_compare)

    with pytest.raises(pipeline.PipelineError, match="changed during the build"):
        pipeline.build_workspace(workspace, MutatingRuntime(), min_score=0.0, timeout_s=10)

    assert previous.read_bytes() == b"keep me"
    assert not (workspace.artifacts_dir / "model.glb").exists()
