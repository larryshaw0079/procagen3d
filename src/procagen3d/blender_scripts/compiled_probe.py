"""Validate and render an exported GLB in a fresh factory-startup process."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bpy

from common import (
    geometry_report,
    render_views,
    reset_scene,
    validate_drawable_scene,
    write_json,
)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glb", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--camera-contract", type=Path, required=True)
    raw = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(raw)


def main():
    args = arguments()
    reset_scene()
    if "FINISHED" not in bpy.ops.import_scene.gltf(filepath=str(args.glb)):
        raise RuntimeError("compiled GLB validation import did not finish")
    bpy.context.scene.frame_set(0)
    for obj in bpy.context.scene.objects:
        if obj.type == "ARMATURE":
            obj.data.pose_position = "REST"
    validate_drawable_scene()

    report = geometry_report(include_components=False)
    report["artifact"] = "model.glb"
    report["pose_policy"] = "compiled GLB, frame-0, armatures-in-rest-position"
    write_json(args.artifacts_dir / "scene_report.json", report)

    contract = json.loads(args.camera_contract.read_text(encoding="utf-8"))
    render_views(args.artifacts_dir / "renders", contract)
    print("PROCAGEN3D_COMPILED_GLB_READY")


if __name__ == "__main__":
    main()
