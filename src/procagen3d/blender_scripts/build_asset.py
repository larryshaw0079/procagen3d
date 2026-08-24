"""Execute a guarded build() program, then save, export, report, and render it."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bpy

from common import geometry_objects, normalize_objects, reset_scene, validate_drawable_scene


REFERENCE_COLLECTION = "PROCAGEN3D_REFERENCE"


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("procedural", "glb-ref"),
        default="procedural",
    )
    parser.add_argument("--reference-glb", type=Path)
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


def import_reference(path):
    """Import normalized evidence and return the exact objects owned by the host."""

    if path is None or not path.is_file():
        raise RuntimeError("glb-ref mode requires a verified --reference-glb")
    before = set(bpy.context.scene.objects)
    if "FINISHED" not in bpy.ops.import_scene.gltf(filepath=str(path)):
        raise RuntimeError("reference GLB import did not finish")
    imported = [obj for obj in bpy.context.scene.objects if obj not in before]
    bpy.context.scene.frame_set(0)
    for obj in imported:
        if obj.type == "ARMATURE":
            obj.data.pose_position = "REST"
    bpy.context.view_layer.update()
    reference_geometry = [obj for obj in geometry_objects() if obj in imported]
    if not reference_geometry:
        raise RuntimeError("reference GLB imported without geometry objects")
    normalize_objects(reference_geometry, imported_objects=imported)
    bpy.context.view_layer.update()

    reference_objects = set(bpy.context.scene.objects)
    collection = bpy.data.collections.new(REFERENCE_COLLECTION)
    bpy.context.scene.collection.children.link(collection)
    for obj in reference_objects:
        for owner in tuple(obj.users_collection):
            owner.objects.unlink(obj)
        collection.objects.link(obj)
    bpy.context.scene["procagen3d_reconstruction_mode"] = "glb-ref"
    bpy.context.scene["procagen3d_reference_collection"] = REFERENCE_COLLECTION
    return reference_objects


def remove_reference(reference_objects, candidates):
    """Detach generated candidates, reject live source dependencies, then purge evidence."""

    candidate_set = set(candidates)
    reference_collection = bpy.data.collections.get(REFERENCE_COLLECTION)
    reference_collections = set()
    pending_collections = [reference_collection] if reference_collection is not None else []
    while pending_collections:
        collection = pending_collections.pop()
        if collection in reference_collections:
            continue
        reference_collections.add(collection)
        pending_collections.extend(collection.children)
    for obj in tuple(bpy.context.scene.objects):
        if obj in reference_objects:
            continue
        if not any(owner not in reference_collections for owner in obj.users_collection):
            bpy.context.scene.collection.objects.link(obj)
        if obj.parent in reference_objects:
            world = obj.matrix_world.copy()
            obj.parent = None
            obj.matrix_world = world
        for modifier in obj.modifiers:
            dependency = getattr(modifier, "object", None)
            if dependency in reference_objects:
                raise RuntimeError(
                    f"generated object {obj.name!r} still depends on reference object "
                    f"{dependency.name!r}; apply or replace that modifier in build()"
                )
    bpy.context.view_layer.update()
    for obj in tuple(reference_objects):
        if obj.name in bpy.data.objects:
            bpy.data.objects.remove(obj, do_unlink=True)
    if reference_collection is not None:
        bpy.data.collections.remove(reference_collection)
    bpy.context.view_layer.update()
    return [obj for obj in candidates if obj in candidate_set and obj.name in bpy.data.objects]


def main():
    args = arguments()
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    reset_scene()
    reference_objects = set()
    if args.mode == "glb-ref":
        reference_objects = import_reference(args.reference_glb)
    elif args.reference_glb is not None:
        raise RuntimeError("procedural mode must not receive --reference-glb")
    bpy.context.scene["procagen3d_reconstruction_mode"] = args.mode
    namespace = runpy.run_path(str(args.program), run_name="__procagen3d_program__")
    build = namespace.get("build")
    if not callable(build):
        raise RuntimeError("program.py must define callable build()")
    build()
    bpy.context.view_layer.update()
    objects = [obj for obj in geometry_objects() if obj not in reference_objects]
    if args.mode == "glb-ref" and not objects:
        raise RuntimeError(
            "glb-ref build() must create candidate geometry; "
            "the host reference collection is evidence, not the output"
        )
    if reference_objects:
        objects = remove_reference(reference_objects, objects)
    objects = validate_drawable_scene(objects)

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
