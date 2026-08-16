"""Regression tests for source safety and runtime freezing."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = SKILL_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from harness.program_source import (  # noqa: E402
    RUNTIME_BEGIN,
    RUNTIME_PATH,
    ProgramSourceError,
    freeze_program_source,
    lint_program_source,
)


class TransformApplyLintTests(unittest.TestCase):
    def test_rejects_omitted_transform_flags(self):
        cases = (
            "bpy.ops.object.transform_apply(scale=True)\n",
            "bpy.ops.object.transform_apply()\n",
            "bpy.ops.object.transform_apply(location=False, scale=True)\n",
        )
        for source in cases:
            with self.subTest(source=source):
                issues = lint_program_source(source)
                self.assertEqual([issue.code for issue in issues], [
                    "TRANSFORM_APPLY_FLAGS",
                ])

    def test_accepts_all_explicit_transform_flags(self):
        source = """
bpy.ops.object.transform_apply(
    location=apply_location,
    rotation=apply_rotation,
    scale=apply_scale,
)
"""
        self.assertEqual(lint_program_source(source), [])

    def test_rejects_positional_and_expanded_arguments(self):
        source = """
bpy.ops.object.transform_apply(
    False,
    location=False,
    rotation=False,
    scale=True,
    **options,
)
"""
        issues = lint_program_source(source)
        self.assertEqual(len(issues), 1)
        self.assertIn("positional arguments", issues[0].message)
        self.assertIn("**kwargs", issues[0].message)

    def test_reports_syntax_errors(self):
        issues = lint_program_source("def broken(:\n", "broken.py")
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "PROGRAM_SYNTAX")
        self.assertEqual(issues[0].line, 1)


class RuntimeFreezeTests(unittest.TestCase):
    def test_legacy_program_is_byte_for_byte_unchanged(self):
        source = "import bpy\n\ndef build():\n    return None\n"
        self.assertEqual(freeze_program_source(source), source)

    def test_runtime_is_frozen_after_future_imports_and_aliases_work(self):
        source = '''#!/usr/bin/env blender
# coding: utf-8
"""Authored asset."""
from __future__ import annotations

from procagen3d_runtime import (
    PROCAGEN3D_RUNTIME_VERSION,
    box as make_box,
)

VERSION = PROCAGEN3D_RUNTIME_VERSION
BUILDER = make_box
'''
        frozen = freeze_program_source(source, "asset.py")

        self.assertIn(RUNTIME_BEGIN, frozen)
        self.assertIn('PROCAGEN3D_RUNTIME_VERSION = "1.0.0"', frozen)
        self.assertIn("make_box = box", frozen)
        self.assertNotIn("\nfrom procagen3d_runtime import (", frozen)
        self.assertLess(
            frozen.index("from __future__ import annotations"),
            frozen.index(RUNTIME_BEGIN),
        )
        compile(frozen, "asset.py", "exec")
        self.assertEqual(freeze_program_source(frozen), frozen)

    def test_rejects_unknown_export(self):
        with self.assertRaisesRegex(ProgramSourceError, "exports no name"):
            freeze_program_source(
                "from procagen3d_runtime import imaginary_helper\n",
                "asset.py",
            )

    def test_rejects_wildcard_import(self):
        with self.assertRaisesRegex(ProgramSourceError, "wildcard"):
            freeze_program_source(
                "from procagen3d_runtime import *\n",
                "asset.py",
            )

    def test_rejects_non_module_scope_runtime_import(self):
        source = """
def build():
    from procagen3d_runtime import box
"""
        with self.assertRaisesRegex(ProgramSourceError, "module scope"):
            freeze_program_source(source, "asset.py")

    def test_rejects_plain_runtime_import(self):
        with self.assertRaisesRegex(ProgramSourceError, "module scope"):
            freeze_program_source("import procagen3d_runtime\n", "asset.py")

    def test_rejects_runtime_submodule_import(self):
        with self.assertRaisesRegex(ProgramSourceError, "module scope"):
            freeze_program_source(
                "from procagen3d_runtime.geometry import box\n",
                "asset.py",
            )

    def test_rejects_import_added_to_already_vendored_program(self):
        frozen = freeze_program_source(
            "from procagen3d_runtime import box\n",
            "asset.py",
        )
        with self.assertRaisesRegex(ProgramSourceError, "already contains"):
            freeze_program_source(
                frozen + "\nfrom procagen3d_runtime import ellipsoid\n",
                "asset.py",
            )

    def test_rejects_malformed_vendor_markers(self):
        with self.assertRaisesRegex(ProgramSourceError, "malformed"):
            freeze_program_source(RUNTIME_BEGIN + "\n", "asset.py")

    def test_checked_in_runtime_compiles_and_passes_source_lint(self):
        source = RUNTIME_PATH.read_text(encoding="utf-8")
        compile(source, str(RUNTIME_PATH), "exec")
        self.assertEqual(lint_program_source(source, RUNTIME_PATH), [])


if __name__ == "__main__":
    unittest.main()
