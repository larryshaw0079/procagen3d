"""ProcAgen3D Blender-side stages.

Not meant to be run directly — invoked by scripts/procagen3d.py as:

    blender --background --factory-startup --python-exit-code 1 \
        --python blender_stages.py -- <stage> [args...]

Stages:
    build   Execute a ProcAgen3D program, dump scene_graph.json, export GLB,
            save scene.blend, render canonical views + contact sheet.
    render  Re-render canonical views from an existing scene.blend.
    fit     Render a registered reference view and score image-fit gates.
    joints  Validate articulation (pivot placement, axis, limits, sweep
            collisions, rest-pose restore) against an existing scene.blend.

Exit codes are reported both as process exit code and as a stdout sentinel
line ``PROCAGEN3D_EXIT:<code>`` (0 = ok, 1 = failure) so the driver is robust to
Blender's own exit-code quirks.
"""

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if sys.path[:1] != [str(_SCRIPT_DIR)]:
    sys.path.insert(0, str(_SCRIPT_DIR))

from bpy_stages.build import stage_build, stage_render
from bpy_stages.fit import stage_fit
from bpy_stages.joints import stage_joints
from bpy_stages.runtime import script_args


def main():
    parser = argparse.ArgumentParser(prog="blender_stages")
    sub = parser.add_subparsers(dest="stage", required=True)

    p_build = sub.add_parser("build")
    p_build.add_argument("--program", required=True)
    p_build.add_argument("--out", required=True)
    p_build.add_argument("--size", type=int, default=512)
    p_build.add_argument("--engine", default="workbench",
                         choices=["workbench", "eevee", "cycles"])
    p_build.add_argument("--no-render", action="store_true")
    p_build.add_argument("--form-diagnostics", action="store_true")

    p_render = sub.add_parser("render")
    p_render.add_argument("--out", required=True)
    p_render.add_argument("--size", type=int, default=512)
    p_render.add_argument("--engine", default="workbench",
                          choices=["workbench", "eevee", "cycles"])
    p_render.add_argument("--form-diagnostics", action="store_true")

    p_fit = sub.add_parser("fit")
    p_fit.add_argument("--out", required=True)
    p_fit.add_argument("--spec", required=True)
    p_fit.add_argument("--engine", default="workbench",
                       choices=["workbench", "eevee", "cycles"])

    p_joints = sub.add_parser("joints")
    p_joints.add_argument("--out", required=True)
    p_joints.add_argument("--strict", action="store_true")

    args = parser.parse_args(script_args())
    {"build": stage_build, "render": stage_render, "fit": stage_fit,
     "joints": stage_joints}[args.stage](args)


if __name__ == "__main__":
    main()
