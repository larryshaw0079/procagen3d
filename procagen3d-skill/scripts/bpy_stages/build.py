"""Execute a ProcAgen3D program, export GLB, save the blend, render views."""

import json
import time
import traceback
from pathlib import Path

import bpy

from .render import reference_camera_contract, render_views
from .runtime import FAIL, OK, WARN, finish
from .scene import collect_scene_graph


BANNED_PATTERNS = [
    "bpy.ops.render",
    "export_scene",
    "wm.save",
    "wm.open_mainfile",
    "urllib",
    "requests",
    "subprocess",
    "os.system",
]


def stage_build(args):
    program_path = Path(args.program)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    src = program_path.read_text()

    warnings = []
    for pat in BANNED_PATTERNS:
        if pat in src:
            warnings.append(f"program contains banned call '{pat}' "
                            "(programs must only build geometry)")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    diag = {"build_ok": False, "program": program_path.name, "warnings": warnings}
    try:
        ns = {"__name__": "procagen3d_program", "__file__": str(program_path)}
        exec(compile(src, program_path.name, "exec"), ns)
        if "build" not in ns or not callable(ns["build"]):
            raise RuntimeError(
                "program defines no build() entry point "
                "(doctrine: def build() must exist and construct the scene)")
        ns["build"]()
        bpy.context.view_layer.update()
    except Exception:
        tb = traceback.format_exc()
        diag["error"] = tb
        (out_dir / "diagnostics.json").write_text(json.dumps(diag, indent=2))
        print("PROCAGEN3D_BUILD_ERROR")
        print(tb)
        finish(1)

    # Invalid/non-finite camera metadata would make export_extras emit
    # non-compliant JSON. Keep the geometry build usable, warn, and omit only
    # the invalid optional contract from derivative artifacts.
    reference_camera_contract(bpy.context.scene, discard_invalid=True)

    graph = collect_scene_graph()
    if graph["totals"]["meshes"] == 0:
        diag["error"] = "build() produced no mesh objects"
        (out_dir / "diagnostics.json").write_text(json.dumps(diag, indent=2))
        print("PROCAGEN3D_BUILD_ERROR")
        print(diag["error"])
        finish(1)

    (out_dir / "scene_graph.json").write_text(json.dumps(graph, indent=2))

    # Semantic names must survive into the GLB: mesh datablocks inherit the
    # object's name (otherwise the GLB carries 'Cube.001' mesh names).
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            obj.data.name = obj.name

    # Export GLB before cameras/lights are added for rendering.
    bpy.ops.export_scene.gltf(
        filepath=str(out_dir / "model.glb"),
        export_format="GLB",
        export_extras=True,
        export_apply=True,
        use_renderable=True,
    )
    bpy.ops.wm.save_as_mainfile(filepath=str(out_dir / "scene.blend"))

    if not args.no_render:
        render_views(out_dir, args.size, args.engine, args.form_diagnostics)

    diag["build_ok"] = True
    diag["stats"] = graph["totals"]
    diag["elapsed_s"] = round(time.time() - t0, 1)
    (out_dir / "diagnostics.json").write_text(json.dumps(diag, indent=2))
    for w in warnings:
        print(f"{WARN}:PROGRAM] {w}")
    print(f"{OK} build complete: {graph['totals']['meshes']} meshes, "
          f"{graph['totals']['triangles']} tris, "
          f"{len(graph['joints'])} joints, {diag['elapsed_s']}s")
    finish(0)


def stage_render(args):
    out_dir = Path(args.out)
    blend = out_dir / "scene.blend"
    if not blend.exists():
        print(f"{FAIL}:NO_SCENE] {blend} not found (run build first)")
        finish(1)
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    render_views(out_dir, args.size, args.engine, args.form_diagnostics)
    finish(0)
