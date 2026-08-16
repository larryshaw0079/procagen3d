#!/usr/bin/env python3
"""ProcAgen3D driver — code-native 3D asset generation (arXiv:2607.22738).

Pure Python 3.10+ stdlib. Blender-side stages are dispatched to
blender_stages.py running under Blender's bundled Python; everything else
(check/score/guard/edit-gates) runs under plain python3.

Subcommands:
    lint       <program.py>              source safety and runtime-import gate
    build      <program.py> --out DIR    build, export GLB, render views
    render     <dir>                     re-render canonical views
    fit        <dir> --spec FILE         registered image-fit gates
    check      <dir>                     deterministic scene-graph gates
    joints     <dir>                     validate articulation (Blender)
    score      <dir> --spec FILE         measure constraints against spec
    guard      <old.py> <new.py>         doctrine guard for repair iterations
    edit-gates <base_dir> <edited_dir> --target PATTERN
                                         deterministic local-edit gates

Exit code 0 = pass, 1 = at least one failure (read the printed reasons).
"""

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if sys.path[:1] != [str(_SCRIPT_DIR)]:
    sys.path.insert(0, str(_SCRIPT_DIR))

from harness.blender import cmd_build, cmd_fit, cmd_joints, cmd_render
from harness.check import cmd_check
from harness.edit_gates import cmd_edit_gates
from harness.guard import cmd_guard
from harness.program_source import cmd_lint
from harness.score import cmd_score


def main():
    parser = argparse.ArgumentParser(
        prog="procagen3d", description="ProcAgen3D code-native 3D asset pipeline")
    parser.add_argument("--blender", help="path to blender executable")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("lint", help="validate program source without Blender")
    p.add_argument("program")
    p.set_defaults(func=cmd_lint)

    p = sub.add_parser("build", help="build program, export GLB, render views")
    p.add_argument("program")
    p.add_argument("--out", required=True)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--engine", default="workbench",
                   choices=["workbench", "eevee", "cycles"])
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--form-diagnostics", action="store_true",
                   help="also render a neutral clay six-view form sheet")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("render", help="re-render canonical views")
    p.add_argument("dir")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--engine", default="workbench",
                   choices=["workbench", "eevee", "cycles"])
    p.add_argument("--form-diagnostics", action="store_true",
                   help="also render a neutral clay six-view form sheet")
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("fit", help="render and score a registered reference fit")
    p.add_argument("dir")
    p.add_argument("--spec", required=True,
                   help="fit_spec.json (copied into the asset directory)")
    p.add_argument("--engine", default="workbench",
                   choices=["workbench", "eevee", "cycles"])
    p.set_defaults(func=cmd_fit)

    p = sub.add_parser("check", help="deterministic scene-graph gates")
    p.add_argument("dir")
    p.add_argument("--tier", choices=["quick", "standard", "showcase"],
                   default="standard",
                   help="detail-floor tier (references/detail.md)")
    p.add_argument("--form", choices=["auto", "rectilinear", "curved", "mixed"],
                   default="auto",
                   help="primary-form profile (references/complex-forms.md)")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("joints", help="validate articulation")
    p.add_argument("dir")
    p.add_argument("--strict", action="store_true",
                   help="include the joint's parent part in sweep collisions")
    p.set_defaults(func=cmd_joints)

    p = sub.add_parser("score", help="measure spec constraints against the build")
    p.add_argument("dir")
    p.add_argument("--spec", required=True)
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("guard", help="doctrine guard between repair iterations")
    p.add_argument("old")
    p.add_argument("new")
    p.add_argument("--allow-shrink", action="store_true")
    p.add_argument("--allow-drop", action="append")
    p.set_defaults(func=cmd_guard)

    p = sub.add_parser("edit-gates", help="deterministic local-edit gates")
    p.add_argument("base")
    p.add_argument("edited")
    p.add_argument("--target", required=True)
    p.add_argument("--mode", default="auto", choices=["auto", "modify", "add"])
    p.add_argument("--tol", type=float, default=1e-4)
    p.set_defaults(func=cmd_edit_gates)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
