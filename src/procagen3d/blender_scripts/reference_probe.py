"""Import and measure a reference GLB in an isolated Blender process."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bpy

from common import (
    camera_contract,
    geometry_objects,
    geometry_report,
    normalize_objects,
    render_views,
    reset_scene,
    world_vertices,
    write_json,
)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glb", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=256)
    raw = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(raw)


def main():
    args = arguments()
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    reset_scene()
    before = set(bpy.context.scene.objects)
    if "FINISHED" not in bpy.ops.import_scene.gltf(filepath=str(args.glb)):
        raise RuntimeError("reference GLB import did not finish")
    imported = [obj for obj in bpy.context.scene.objects if obj not in before]
    bpy.context.scene.frame_set(0)
    for obj in imported:
        if obj.type == "ARMATURE":
            obj.data.pose_position = "REST"
    bpy.context.view_layer.update()
    objects = geometry_objects()
    if not objects:
        raise RuntimeError("reference GLB imported without geometry objects")
    normalization = normalize_objects(objects, imported_objects=imported)
    report = geometry_report(include_components=True)
    report["normalization"] = normalization
    report["pose_policy"] = "frame-0, armatures-in-rest-position"
    report["source"] = str(args.glb)
    contract = camera_contract(args.size, points=world_vertices())
    write_json(args.evidence_dir / "camera_contract.json", contract)
    masks = render_views(args.evidence_dir / "reference_views", contract)
    report["canonical_evidence"] = {
        "renders": "evidence/reference_views",
        "masks": "evidence/reference_views/masks.json",
        "diagnostics": "evidence/reference_views/" + masks["diagnostics"]["manifest"],
    }
    write_json(args.evidence_dir / "reference_scene.json", report)
    print("PROCAGEN3D_REFERENCE_READY")


if __name__ == "__main__":
    main()
