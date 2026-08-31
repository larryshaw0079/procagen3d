from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from procagen3d.blender import BlenderRuntime, require_success
from procagen3d.glb_probe import probe_glb
from procagen3d.materials import compare_material_pass_geometry
from procagen3d.pipeline import CANONICAL_VIEWS
from procagen3d.workspace import write_json


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("PROCAGEN3D_RUN_BLENDER_TESTS") != "1",
        reason="set PROCAGEN3D_RUN_BLENDER_TESTS=1 to launch headless Blender",
    ),
]


def _cube_program(*, partition_faces_by_material: bool) -> str:
    partition = ""
    if partition_faces_by_material:
        partition = """
    accent = make_material("Accent", (0.1, 0.25, 0.9, 1.0))
    cube.data.materials.append(accent)
    for polygon in cube.data.polygons:
        polygon.material_index = polygon.index % 2
"""
    return f'''
import bpy

def make_material(name, color):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    material.diffuse_color = color
    principled = material.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = color
    principled.inputs["Metallic"].default_value = 0.0
    principled.inputs["Roughness"].default_value = 0.5
    return material

def build():
    vertices = [
        (-0.5, -0.5, 0.0),
        (0.5, -0.5, 0.0),
        (0.5, 0.5, 0.0),
        (-0.5, 0.5, 0.0),
        (-0.5, -0.5, 1.0),
        (0.5, -0.5, 1.0),
        (0.5, 0.5, 1.0),
        (-0.5, 0.5, 1.0),
    ]
    faces = [
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]
    mesh = bpy.data.meshes.new("MaterialGuardCubeMesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    cube = bpy.data.objects.new("MaterialGuardCube", mesh)
    bpy.context.scene.collection.objects.link(cube)
    neutral = make_material("Neutral", (0.65, 0.65, 0.65, 1.0))
    cube.data.materials.append(neutral)
{partition}
'''


def _camera_contract(path: Path) -> Path:
    target = [0.0, 0.0, 0.5]
    locations = {
        "front": [0.0, -4.5, 0.5],
        "back": [0.0, 4.5, 0.5],
        "left": [-4.5, 0.0, 0.5],
        "right": [4.5, 0.0, 0.5],
        "top": [0.0, 0.0, 5.0],
        "iso": [2.7, -2.7, 3.2],
    }
    write_json(
        path,
        {
            "projection": "ORTHO",
            "resolution": [32, 32],
            "views": [
                {
                    "name": name,
                    "location": locations[name],
                    "target": target,
                    "ortho_scale": 2.0,
                }
                for name in CANONICAL_VIEWS
            ],
        },
    )
    return path


def _primitive_count(report: dict[str, Any]) -> int:
    return sum(
        len(mesh.get("primitives", []))
        for mesh in report.get("meshes", [])
        if isinstance(mesh, dict)
    )


def test_material_geometry_fingerprint_ignores_glb_material_primitive_splits(
    tmp_path: Path,
) -> None:
    runtime = BlenderRuntime.discover()
    contract = _camera_contract(tmp_path / "camera_contract.json")
    reports: dict[str, dict[str, Any]] = {}
    primitive_counts: dict[str, int] = {}

    for label, partitioned in (("single", False), ("partitioned", True)):
        program = tmp_path / f"{label}.py"
        build_artifacts = tmp_path / f"{label}-build"
        probe_artifacts = tmp_path / f"{label}-probe"
        program.write_text(
            _cube_program(partition_faces_by_material=partitioned),
            encoding="utf-8",
        )
        built = runtime.run_stage(
            "build_asset",
            ["--program", program, "--artifacts-dir", build_artifacts],
            cwd=tmp_path,
            timeout_s=180,
        )
        require_success(built, stage=f"{label} material-partition fixture build")

        model = build_artifacts / "model.glb"
        primitive_counts[label] = _primitive_count(probe_glb(model))
        compiled = runtime.run_stage(
            "compiled_probe",
            [
                "--glb",
                model,
                "--artifacts-dir",
                probe_artifacts,
                "--camera-contract",
                contract,
            ],
            cwd=tmp_path,
            timeout_s=180,
        )
        require_success(compiled, stage=f"{label} material-partition compiled probe")
        reports[label] = json.loads(
            (probe_artifacts / "scene_report.json").read_text(encoding="utf-8")
        )

    assert primitive_counts == {"single": 1, "partitioned": 2}
    single_fingerprint = reports["single"]["material_geometry_fingerprint"]
    partitioned_fingerprint = reports["partitioned"][
        "material_geometry_fingerprint"
    ]
    for fingerprint in (single_fingerprint, partitioned_fingerprint):
        assert fingerprint["schema_version"] == 1
        assert (
            fingerprint["algorithm"]
            == "oriented-world-triangle-multiset-sha256-v1"
        )
        assert len(fingerprint["digest"]) == 64
        assert fingerprint["objects"][0]["name"] == "MaterialGuardCube"
        assert fingerprint["objects"][0]["triangles"] == 12
    assert single_fingerprint["digest"] == partitioned_fingerprint["digest"]

    guard = compare_material_pass_geometry(
        reports["single"], reports["partitioned"]
    )
    assert guard.passed is True
    assert guard.violations == ()
    assert guard.before_digest == single_fingerprint["digest"]
    assert guard.after_digest == partitioned_fingerprint["digest"]
