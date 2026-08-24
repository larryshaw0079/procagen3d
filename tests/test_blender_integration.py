from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from procagen3d.blender import BlenderRuntime, require_success
from procagen3d.pipeline import CANONICAL_VIEWS
from procagen3d.source_guard import assert_safe_source
from procagen3d.workspace import write_json


pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.environ.get("PROCAGEN3D_RUN_BLENDER_TESTS") != "1",
    reason="set PROCAGEN3D_RUN_BLENDER_TESTS=1 to launch headless Blender",
)
def test_surface_comparison_is_deterministic_and_bidirectional(tmp_path: Path) -> None:
    runtime = BlenderRuntime.discover()
    reference_artifacts = tmp_path / "reference-artifacts"
    candidate_artifacts = tmp_path / "candidate-artifacts"
    reference_program = tmp_path / "reference.py"
    candidate_program = tmp_path / "candidate.py"
    reference_program.write_text(
        """
import bpy

def build():
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.5))
""",
        encoding="utf-8",
    )
    # The surface stage normalizes the one-unit reference to two units. The
    # candidate is already in the pipeline's normalized coordinate frame.
    candidate_program.write_text(
        """
import bpy

def build():
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 1.0))
""",
        encoding="utf-8",
    )
    for program, artifacts in (
        (reference_program, reference_artifacts),
        (candidate_program, candidate_artifacts),
    ):
        result = runtime.run_stage(
            "build_asset",
            ["--program", program, "--artifacts-dir", artifacts],
            cwd=tmp_path,
            timeout_s=180,
        )
        require_success(result, stage="surface-comparison fixture build")

    first = tmp_path / "surface-first.json"
    second = tmp_path / "surface-second.json"
    arguments = [
        "--reference-glb",
        reference_artifacts / "model.glb",
        "--candidate-glb",
        candidate_artifacts / "model.glb",
        "--samples",
        "512",
    ]
    for output in (first, second):
        result = runtime.run_stage(
            "surface_compare",
            [*arguments, "--output", output],
            cwd=tmp_path,
            timeout_s=180,
        )
        require_success(result, stage="integration surface comparison")

    assert first.read_bytes() == second.read_bytes()
    report = json.loads(first.read_text(encoding="utf-8"))
    assert report["candidate_to_reference"]["samples"] == 512
    assert report["reference_to_candidate"]["samples"] == 512
    assert report["symmetric"]["mean"] == pytest.approx(0.0, abs=1.0e-6)
    assert report["symmetric"]["p95"] == pytest.approx(0.0, abs=1.0e-6)


@pytest.mark.skipif(
    os.environ.get("PROCAGEN3D_RUN_BLENDER_TESTS") != "1",
    reason="set PROCAGEN3D_RUN_BLENDER_TESTS=1 to launch headless Blender",
)
def test_compiled_glb_probe_cannot_inherit_generated_render_handler(tmp_path: Path) -> None:
    runtime = BlenderRuntime.discover()
    artifacts = tmp_path / "artifacts"
    program = tmp_path / "program.py"
    source = '''
import bpy

def poison_render(_scene):
    bpy.ops.mesh.primitive_cube_add(size=20.0, location=(0.0, 0.0, 10.0))

def build():
    bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, location=(0.0, 0.0, 1.0))
    bpy.context.object.name = "OnlyExportedSphere"
    bpy.app.handlers.render_pre.append(poison_render)
'''
    assert_safe_source(source)
    program.write_text(source, encoding="utf-8")

    build = runtime.run_stage(
        "build_asset",
        ["--program", program, "--artifacts-dir", artifacts],
        cwd=tmp_path,
        timeout_s=180,
    )
    require_success(build, stage="integration source build")

    target = [0.0, 0.0, 1.0]
    directions = {
        "front": [0.0, -4.5, 1.0],
        "back": [0.0, 4.5, 1.0],
        "left": [-4.5, 0.0, 1.0],
        "right": [4.5, 0.0, 1.0],
        "top": [0.0, 0.0, 5.5],
        "iso": [2.7, -2.7, 3.3],
    }
    contract = tmp_path / "camera_contract.json"
    write_json(
        contract,
        {
            "projection": "ORTHO",
            "resolution": [64, 64],
            "views": [
                {
                    "name": name,
                    "location": directions[name],
                    "target": target,
                    "ortho_scale": 2.5,
                }
                for name in CANONICAL_VIEWS
            ],
        },
    )

    compiled = runtime.run_stage(
        "compiled_probe",
        [
            "--glb",
            artifacts / "model.glb",
            "--artifacts-dir",
            artifacts,
            "--camera-contract",
            contract,
        ],
        cwd=tmp_path,
        timeout_s=180,
    )
    require_success(compiled, stage="integration compiled GLB probe")

    report = json.loads((artifacts / "scene_report.json").read_text(encoding="utf-8"))
    assert report["geometry_object_count"] == 1
    assert report["mesh_count"] == 1
    assert max(report["bounds"]["dimensions"]) <= 2.01
    assert {path.stem for path in (artifacts / "renders").glob("*.png")} == set(
        CANONICAL_VIEWS
    )


@pytest.mark.skipif(
    os.environ.get("PROCAGEN3D_RUN_BLENDER_TESTS") != "1",
    reason="set PROCAGEN3D_RUN_BLENDER_TESTS=1 to launch headless Blender",
)
def test_custom_mesh_world_transform_survives_parented_glb_round_trip(
    tmp_path: Path,
) -> None:
    runtime = BlenderRuntime.discover()
    artifacts = tmp_path / "artifacts"
    program = tmp_path / "program.py"
    source = '''
import bpy
from mathutils import Matrix

def parent_keep_world(obj, parent):
    world = Matrix.LocRotScale(
        obj.location.copy(),
        obj.rotation_euler.to_quaternion(),
        obj.scale.copy(),
    )
    obj.parent = parent
    bpy.context.view_layer.update()
    obj.matrix_world = world

def build():
    collection = bpy.context.scene.collection

    root = bpy.data.objects.new("TranslatedRoot", None)
    collection.objects.link(root)
    root.location = (3.0, -2.0, 1.0)

    branch = bpy.data.objects.new("TranslatedBranch", None)
    collection.objects.link(branch)
    branch.location = (7.0, -5.0, 4.0)
    parent_keep_world(branch, root)

    mesh = bpy.data.meshes.new("CustomMeshData")
    mesh.from_pydata(
        [
            (-0.5, -0.5, -0.5),
            (0.5, -0.5, -0.5),
            (0.5, 0.5, -0.5),
            (-0.5, 0.5, -0.5),
            (-0.5, -0.5, 0.5),
            (0.5, -0.5, 0.5),
            (0.5, 0.5, 0.5),
            (-0.5, 0.5, 0.5),
        ],
        [],
        [
            (0, 3, 2, 1),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (1, 2, 6, 5),
            (2, 3, 7, 6),
            (3, 0, 4, 7),
        ],
    )
    mesh.update()
    obj = bpy.data.objects.new("ParentedCustomMesh", mesh)
    collection.objects.link(obj)
    obj.location = (11.0, -7.0, 9.0)
    obj.rotation_euler = (0.2, -0.3, 0.4)
    obj.scale = (1.5, 0.75, 2.0)
    parent_keep_world(obj, branch)
'''
    assert_safe_source(source)
    program.write_text(source, encoding="utf-8")

    build = runtime.run_stage(
        "build_asset",
        ["--program", program, "--artifacts-dir", artifacts],
        cwd=tmp_path,
        timeout_s=180,
    )
    require_success(build, stage="integration parented custom-mesh build")

    target = [11.0, -7.0, 9.0]
    directions = {
        "front": [11.0, -11.5, 9.0],
        "back": [11.0, -2.5, 9.0],
        "left": [6.5, -7.0, 9.0],
        "right": [15.5, -7.0, 9.0],
        "top": [11.0, -7.0, 13.5],
        "iso": [13.7, -9.7, 11.3],
    }
    contract = tmp_path / "camera_contract.json"
    write_json(
        contract,
        {
            "projection": "ORTHO",
            "resolution": [32, 32],
            "views": [
                {
                    "name": name,
                    "location": directions[name],
                    "target": target,
                    "ortho_scale": 5.0,
                }
                for name in CANONICAL_VIEWS
            ],
        },
    )

    compiled = runtime.run_stage(
        "compiled_probe",
        [
            "--glb",
            artifacts / "model.glb",
            "--artifacts-dir",
            artifacts,
            "--camera-contract",
            contract,
        ],
        cwd=tmp_path,
        timeout_s=180,
    )
    require_success(compiled, stage="integration parented custom-mesh GLB probe")

    report = json.loads((artifacts / "scene_report.json").read_text(encoding="utf-8"))
    assert report["geometry_object_count"] == 1
    imported = next(
        item for item in report["objects"] if item["name"] == "ParentedCustomMesh"
    )
    imported_center = [
        (minimum + maximum) * 0.5
        for minimum, maximum in zip(
            imported["bounds"]["min"], imported["bounds"]["max"], strict=True
        )
    ]
    assert imported_center == pytest.approx(target, abs=1.0e-5)
    # A unit cube with the authored XYZ rotation and non-uniform scale has
    # these world-axis-aligned spans.  Checking all three catches translation,
    # rotation, or scale being dropped/baked under the wrong parent.
    assert imported["bounds"]["dimensions"] == pytest.approx(
        [2.0254857292, 1.8094640576, 2.4582140829], abs=1.0e-5
    )


@pytest.mark.skipif(
    os.environ.get("PROCAGEN3D_RUN_BLENDER_TESTS") != "1",
    reason="set PROCAGEN3D_RUN_BLENDER_TESTS=1 to launch headless Blender",
)
def test_reference_derived_build_exports_only_new_candidate_objects(tmp_path: Path) -> None:
    runtime = BlenderRuntime.discover()
    reference_artifacts = tmp_path / "reference-artifacts"
    reference_program = tmp_path / "reference_program.py"
    reference_program.write_text(
        '''
import bpy

def build():
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.5))
    bpy.context.object.name = "HostOwnedReferenceCube"
''',
        encoding="utf-8",
    )
    reference_build = runtime.run_stage(
        "build_asset",
        ["--program", reference_program, "--artifacts-dir", reference_artifacts],
        cwd=tmp_path,
        timeout_s=180,
    )
    require_success(reference_build, stage="integration reference fixture build")

    derived_artifacts = tmp_path / "derived-artifacts"
    derived_program = tmp_path / "derived_program.py"
    derived_source = '''
import bpy

def build():
    reference = bpy.data.collections.get("PROCAGEN3D_REFERENCE")
    if reference is None:
        raise RuntimeError("host reference collection is missing")
    source = next(obj for obj in reference.all_objects if obj.type == "MESH")
    world = source.matrix_world.copy()
    candidate = source.copy()
    candidate.data = source.data.copy()
    candidate.name = "DerivedCandidateCube"
    bpy.context.scene.collection.objects.link(candidate)
    candidate.parent = None
    candidate.matrix_world = world
'''
    assert_safe_source(derived_source)
    derived_program.write_text(derived_source, encoding="utf-8")
    derived_build = runtime.run_stage(
        "build_asset",
        [
            "--program",
            derived_program,
            "--artifacts-dir",
            derived_artifacts,
            "--mode",
            "glb-ref",
            "--reference-glb",
            reference_artifacts / "model.glb",
        ],
        cwd=tmp_path,
        timeout_s=180,
    )
    require_success(derived_build, stage="integration glb-ref build")

    contract = tmp_path / "derived_camera_contract.json"
    write_json(
        contract,
        {
            "projection": "ORTHO",
            "resolution": [32, 32],
            "views": [
                {
                    "name": name,
                    "location": {
                        "front": [0.0, -4.5, 1.0],
                        "back": [0.0, 4.5, 1.0],
                        "left": [-4.5, 0.0, 1.0],
                        "right": [4.5, 0.0, 1.0],
                        "top": [0.0, 0.0, 5.5],
                        "iso": [2.7, -2.7, 3.3],
                    }[name],
                    "target": [0.0, 0.0, 1.0],
                    "ortho_scale": 2.5,
                }
                for name in CANONICAL_VIEWS
            ],
        },
    )
    compiled = runtime.run_stage(
        "compiled_probe",
        [
            "--glb",
            derived_artifacts / "model.glb",
            "--artifacts-dir",
            derived_artifacts,
            "--camera-contract",
            contract,
        ],
        cwd=tmp_path,
        timeout_s=180,
    )
    require_success(compiled, stage="integration glb-ref GLB probe")

    report = json.loads((derived_artifacts / "scene_report.json").read_text(encoding="utf-8"))
    assert report["geometry_object_count"] == 1
    assert [item["name"] for item in report["objects"]] == ["DerivedCandidateCube"]
    assert report["bounds"]["dimensions"] == pytest.approx([2.0, 2.0, 2.0], abs=1.0e-5)
