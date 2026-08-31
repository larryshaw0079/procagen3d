"""Split a compiled host-solved assembly into link-local GLB meshes for URDF."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bpy
from mathutils import Matrix

from common import geometry_objects, reset_scene, write_json


SAFE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glb", type=Path, required=True)
    parser.add_argument("--assembly-transforms", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    raw = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(raw)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_part(part, source_by_name, output_dir):
    part_id = part.get("id")
    names = part.get("object_names")
    rows = part.get("world_matrix")
    if not isinstance(part_id, str) or not SAFE_NAME.fullmatch(part_id):
        raise RuntimeError(f"part id {part_id!r} is not a safe URDF link name")
    if not isinstance(names, list) or not names:
        raise RuntimeError(f"part {part_id!r} has no object names")
    try:
        inverse_world = Matrix(rows).inverted_safe()
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"part {part_id!r} has an invalid world matrix") from exc

    sources = []
    for name in names:
        source = source_by_name.get(name)
        if source is None:
            raise RuntimeError(f"part {part_id!r} cannot find compiled object {name!r}")
        sources.append((name, source))

    duplicates = []
    renamed_sources = []
    output = output_dir / f"{part_id}.glb"
    try:
        # Free the exact planned names before creating export copies. Blender
        # otherwise silently appends .001 because the imported compiled source
        # objects still occupy those names.
        for index, (name, source) in enumerate(sources):
            source.name = f"__PROCAGEN3D_SOURCE_{part_id}_{index}"
            renamed_sources.append((source, name))
            duplicate = source.copy()
            duplicate.data = source.data.copy()
            duplicate.animation_data_clear()
            duplicate.parent = None
            duplicate.matrix_world = inverse_world @ source.matrix_world
            duplicate.name = name
            bpy.context.scene.collection.objects.link(duplicate)
            duplicates.append(duplicate)
        bpy.context.view_layer.update()

        bpy.ops.object.select_all(action="DESELECT")
        for duplicate in duplicates:
            duplicate.select_set(True)
        export_arguments = {
            "filepath": str(output),
            "export_format": "GLB",
            "use_selection": True,
            "export_apply": True,
            "export_yup": True,
        }
        properties = bpy.ops.export_scene.gltf.get_rna_type().properties.keys()
        if "use_renderable" in properties:
            export_arguments["use_renderable"] = True
        if (
            "FINISHED" not in bpy.ops.export_scene.gltf(**export_arguments)
            or not output.is_file()
        ):
            raise RuntimeError(f"URDF part export did not finish for {part_id!r}")
    finally:
        for duplicate in duplicates:
            mesh = duplicate.data
            bpy.data.objects.remove(duplicate, do_unlink=True)
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)
        for source, name in renamed_sources:
            source.name = name
        bpy.context.view_layer.update()
    return {
        "part_id": part_id,
        "path": output.name,
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
        "object_names": list(names),
    }


def main():
    args = arguments()
    if not args.glb.is_file() or not args.assembly_transforms.is_file():
        raise RuntimeError("URDF split inputs must be regular files")
    document = json.loads(args.assembly_transforms.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1 or document.get("placement") != "host-solved":
        raise RuntimeError("URDF split requires a host-solved assembly document")
    parts = document.get("parts")
    if not isinstance(parts, list) or not parts:
        raise RuntimeError("URDF split assembly has no parts")

    reset_scene()
    if "FINISHED" not in bpy.ops.import_scene.gltf(filepath=str(args.glb)):
        raise RuntimeError("compiled GLB import did not finish")
    bpy.context.view_layer.update()
    sources = geometry_objects()
    source_by_name = {obj.name: obj for obj in sources}
    if len(source_by_name) != len(sources):
        raise RuntimeError("compiled GLB object names are not unique")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = [export_part(part, source_by_name, args.output_dir) for part in parts]
    write_json(
        args.output_dir / "manifest.json",
        {
            "schema_version": 1,
            "source": args.glb.name,
            "source_sha256": sha256(args.glb),
            "parts": records,
        },
    )
    print("PROCAGEN3D_URDF_PARTS_READY")


if __name__ == "__main__":
    main()
