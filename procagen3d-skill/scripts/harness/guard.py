"""Doctrine guard between full-program repair iterations."""

import difflib
import re
from pathlib import Path

from .tags import OK, fail, warn


PART_FUNC_RE = re.compile(r"^def\s+(build_[A-Za-z0-9_]+)\s*\(", re.MULTILINE)


def cmd_guard(args):
    old = Path(args.old).read_text()
    new = Path(args.new).read_text()
    failures = 0

    if "def build(" not in new:
        failures += 1
        fail("ENTRY_POINT_LOST", "corrected program dropped def build()")

    if len(new) < 0.85 * len(old) and not args.allow_shrink:
        failures += 1
        fail("SOURCE_SHRINK", f"program shrank {len(old)} -> {len(new)} bytes "
             f"(> 15%) — a repair must not gut the program; pass "
             f"--allow-shrink only if this is an intended rewrite")

    dropped = (set(PART_FUNC_RE.findall(old)) - set(PART_FUNC_RE.findall(new))
               - set(args.allow_drop or []))
    if dropped:
        failures += 1
        fail("DROPPED_PART_FUNCS", f"{sorted(dropped)} vanished — repairs must "
             f"preserve parts; pass --allow-drop NAME if a rename/merge is intended")

    old_joints = old.count("add_joint(") - old.count("def add_joint(")
    new_joints = new.count("add_joint(") - new.count("def add_joint(")
    if new_joints < old_joints:
        warn("FEWER_JOINTS", f"joint declarations {old_joints} -> {new_joints}")

    diff = list(difflib.unified_diff(old.splitlines(), new.splitlines(), n=0))
    changed = sum(1 for l in diff if l[:1] in "+-" and l[:3] not in ("+++", "---"))
    if failures == 0:
        print(f"{OK} guard passed ({changed} changed lines)")
    return 1 if failures else 0
