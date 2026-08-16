"""Shared bpy runtime helpers and grep-able status prefixes."""

import sys

import bpy

OK = "[PROCAGEN3D:OK]"
WARN = "[PROCAGEN3D:WARN"
FAIL = "[PROCAGEN3D:FAIL"


def finish(code):
    print(f"PROCAGEN3D_EXIT:{code}")
    sys.stdout.flush()
    sys.exit(code)


def script_args():
    argv = sys.argv
    return argv[argv.index("--") + 1 :] if "--" in argv else []


def depsgraph():
    bpy.context.view_layer.update()
    return bpy.context.evaluated_depsgraph_get()


def mesh_objects():
    # Hidden Boolean cutters and construction helpers are not part of the
    # judged/export-facing asset. They must not inflate totals, form envelopes,
    # collision sets, or the camera framing used for proof renders.
    return [o for o in bpy.context.scene.objects
            if o.type == "MESH" and not o.hide_render]
