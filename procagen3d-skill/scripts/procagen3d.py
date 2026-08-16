#!/usr/bin/env python3
"""ProcAgen3D driver — code-native 3D asset generation (arXiv:2607.22738).

Pure Python 3.10+ stdlib. Blender-side stages are dispatched to
blender_stages.py running under Blender's bundled Python; everything else
(check/score/guard/edit-gates) runs under plain python3.

Subcommands:
    next                                      exact next workflow step/command
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

Exit codes: 0 = pass, 1 = pipeline failure, 2 = invalid workflow state/order,
3 = hard repair stop. Read the printed `[PROCAGEN3D:*]` reasons.
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
from harness.tags import fail
from harness.workflow import (
    WorkflowStateError,
    authorize_pipeline_command,
    cmd_next,
    record_pipeline_result,
)


def main():
    parser = argparse.ArgumentParser(
        prog="procagen3d", description="ProcAgen3D code-native 3D asset pipeline")
    parser.add_argument("--blender", help="path to blender executable")
    parser.add_argument(
        "--state",
        help="workflow state path (default: .procagen3d/state.json)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("next", help="initialize or report the exact next workflow step")
    action = p.add_mutually_exclusive_group()
    action.add_argument("--init", action="store_true",
                        help="create a new state file (refuses overwrite)")
    action.add_argument("--done", metavar="STEP",
                        help="complete the current manual step")
    action.add_argument("--repair", action="store_true",
                        help="start one bounded full-program repair")
    p.add_argument("--state", dest="next_state",
                   help="state path; accepted here for `next` convenience")
    p.add_argument("--out", help="asset output directory (required with --init)")
    p.add_argument("--program", help="authoring program path; defaults to <out>/<slug>.py")
    p.add_argument("--tier", choices=["quick", "standard", "showcase"],
                   default="standard")
    p.add_argument("--form", choices=["rectilinear", "curved", "mixed"],
                   default="rectilinear")
    p.add_argument("--reference", action="append", default=[],
                   help="preserved reference path; repeat for multiple views")
    p.add_argument("--spec", help="constraint spec path; adds the score stage")
    p.add_argument("--joints", action="store_true",
                   help="declare articulation; adds the joints stage")
    p.add_argument("--max-repairs", type=int, default=3)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--engine", default="workbench",
                   choices=["workbench", "eevee", "cycles"])
    p.add_argument("--evidence", action="append", default=[],
                   help="evidence file for --done/--repair; repeat as needed")
    p.add_argument("--note", default="", help="manual-step verdict")
    p.add_argument("--reason", default="", help="repair reason and preserve intent")
    p.add_argument(
        "--allow-shrink",
        action="store_true",
        help="permit a declared representation rewrite in the repair guard",
    )
    p.add_argument(
        "--allow-drop",
        action="append",
        default=[],
        help="name pattern the repair guard may drop; repeat as needed",
    )
    p.add_argument("--json", action="store_true", help="emit status as JSON")
    p.set_defaults(
        func=cmd_next,
        cli_path=str(Path(__file__).resolve()),
        python=sys.executable,
    )

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
    if args.cmd == "next":
        sys.exit(args.func(args))

    try:
        state_context = authorize_pipeline_command(args)
    except (OSError, WorkflowStateError) as exc:
        fail("STATE", str(exc))
        sys.exit(2)

    try:
        result = args.func(args)
    except SystemExit as exc:
        if exc.code is None:
            result = 0
        elif isinstance(exc.code, int):
            result = exc.code
        else:
            print(exc.code, file=sys.stderr)
            result = 1
    try:
        record_pipeline_result(state_context, result)
    except (OSError, WorkflowStateError) as exc:
        fail("STATE", str(exc))
        if result == 0:
            result = 2
    sys.exit(result)


if __name__ == "__main__":
    main()
