"""Execute a guarded build() program, then save, export, report, and render it."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bpy

from common import reset_scene, validate_drawable_scene


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    raw = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(raw)


def export_objects(geometry):
    selected = set(geometry)
    for obj in tuple(geometry):
        parent = obj.parent
        while parent is not None:
            selected.add(parent)
            parent = parent.parent
        for modifier in obj.modifiers:
            dependency = getattr(modifier, "object", None)
            if dependency is not None:
                selected.add(dependency)
    return selected


def main():
    args = arguments()
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    reset_scene()
    namespace = runpy.run_path(str(args.program), run_name="__procagen3d_program__")
    build = namespace.get("build")
    if not callable(build):
        raise RuntimeError("program.py must define callable build()")
    build()
    objects = validate_drawable_scene()

    blend_path = args.artifacts_dir / "scene.blend"
    if "FINISHED" not in bpy.ops.wm.save_as_mainfile(filepath=str(blend_path)) or not blend_path.is_file():
        raise RuntimeError("Blender scene save did not finish")
    bpy.ops.object.select_all(action="DESELECT")
    selected = export_objects(objects)
    for obj in selected:
        obj.select_set(True)
    model_path = args.artifacts_dir / "model.glb"
    export_arguments = dict(
        filepath=str(model_path),
        export_format="GLB",
        use_selection=True,
        export_apply=True,
        export_yup=True,
    )
    properties = bpy.ops.export_scene.gltf.get_rna_type().properties.keys()
    if "use_renderable" in properties:
        export_arguments["use_renderable"] = True
    if "FINISHED" not in bpy.ops.export_scene.gltf(**export_arguments) or not model_path.is_file():
        raise RuntimeError("GLB export did not finish")
    print("PROCAGEN3D_BUILD_READY")


if __name__ == "__main__":
    main()
