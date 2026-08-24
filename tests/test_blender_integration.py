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
