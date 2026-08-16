"""Locate Blender and dispatch build/render/fit/joints stages."""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from .program_source import (
    ProgramSourceError,
    freeze_program_source,
    lint_program_source,
    report_source_issues,
)
from .tags import fail

SCRIPT_DIR = Path(__file__).resolve().parent.parent


def find_blender(explicit=None):
    candidates = [explicit, os.environ.get("PROCAGEN3D_BLENDER"), shutil.which("blender")]
    for cand in candidates:
        if cand and Path(cand).exists():
            return str(cand)
    for hit in sorted(Path.home().glob(".cache/procagen3d/*/blender")):
        if hit.is_file():
            return str(hit)
    sys.exit("ProcAgen3D: Blender not found. Set PROCAGEN3D_BLENDER=/path/to/blender, "
             "put blender on PATH, or install under ~/.cache/procagen3d/.")


def run_blender(stage_args, blender=None):
    cmd = [
        find_blender(blender), "--background", "--factory-startup",
        "--python-exit-code", "1",
        "--python", str(SCRIPT_DIR / "blender_stages.py"), "--", *stage_args,
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)
    code = proc.returncode
    for line in proc.stdout.splitlines():
        m = re.match(r"^PROCAGEN3D_EXIT:(\d+)$", line.strip())
        if m:
            code = int(m.group(1))
        elif line.strip() and not line.startswith(
                ("Blender ", "Read prefs", "Time:", "Saved:", "Info:",
                 "Fra:", "Read blend:", "WARN (gpu", "INFO ",
                 "Color management")) and "| INFO" not in line:
            print(line)
    return code


def cmd_build(args):
    program = Path(args.program)
    if not program.is_file():
        fail("PROGRAM_NOT_FOUND", str(program))
        return 1
    source = program.read_text(encoding="utf-8")
    source_issues = lint_program_source(source, program)
    if source_issues:
        return report_source_issues(source_issues, program)
    try:
        kept_source = freeze_program_source(source, program)
    except ProgramSourceError as exc:
        fail("PROGRAM_RUNTIME", str(exc))
        return 1
    frozen_issues = lint_program_source(kept_source, program)
    if frozen_issues:
        return report_source_issues(frozen_issues, program)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kept = out / "program.py"
    if program.resolve() != kept.resolve() or kept_source != source:
        kept.write_text(kept_source, encoding="utf-8")
    stage = ["build", "--program", str(kept), "--out", str(out),
             "--size", str(args.size), "--engine", args.engine]
    if args.no_render:
        stage.append("--no-render")
    if args.form_diagnostics:
        stage.append("--form-diagnostics")
    return run_blender(stage, args.blender)


def cmd_render(args):
    stage = ["render", "--out", args.dir, "--size", str(args.size),
             "--engine", args.engine]
    if args.form_diagnostics:
        stage.append("--form-diagnostics")
    return run_blender(stage, args.blender)


def cmd_fit(args):
    source = Path(args.spec)
    if not source.is_file():
        sys.exit(f"ProcAgen3D: fit spec not found: {source}")
    out = Path(args.dir)
    if not (out / "scene.blend").is_file():
        sys.exit(f"ProcAgen3D: {out / 'scene.blend'} not found (run build first)")
    kept = out / "fit_spec.json"
    if source.resolve() != kept.resolve():
        shutil.copyfile(source, kept)
    stage = ["fit", "--out", str(out), "--spec", str(kept),
             "--engine", args.engine]
    return run_blender(stage, args.blender)


def cmd_joints(args):
    stage = ["joints", "--out", args.dir]
    if args.strict:
        stage.append("--strict")
    return run_blender(stage, args.blender)
