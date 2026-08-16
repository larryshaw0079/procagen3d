"""Driver tests that do not require Blender."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = SKILL_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from harness import blender as blender_driver  # noqa: E402
from harness.program_source import RUNTIME_BEGIN  # noqa: E402


def build_args(program: Path, out: Path) -> SimpleNamespace:
    return SimpleNamespace(
        program=str(program),
        out=str(out),
        size=64,
        engine="workbench",
        no_render=True,
        form_diagnostics=False,
        blender=None,
    )


class BuildSourceGateTests(unittest.TestCase):
    def test_unsafe_source_fails_before_output_or_blender(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            program = root / "unsafe.py"
            out = root / "out"
            program.write_text(
                "import bpy\n"
                "bpy.ops.object.transform_apply(scale=True)\n",
                encoding="utf-8",
            )
            with patch.object(blender_driver, "run_blender") as run:
                self.assertEqual(blender_driver.cmd_build(build_args(program, out)), 1)
            run.assert_not_called()
            self.assertFalse(out.exists())

    def test_legacy_source_is_retained_unchanged(self):
        source = "import bpy\n\ndef build():\n    return None\n"
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            program = root / "legacy.py"
            out = root / "out"
            program.write_text(source, encoding="utf-8")
            with patch.object(blender_driver, "run_blender", return_value=0) as run:
                self.assertEqual(blender_driver.cmd_build(build_args(program, out)), 0)

            self.assertEqual((out / "program.py").read_text(encoding="utf-8"), source)
            stage_args = run.call_args.args[0]
            self.assertEqual(stage_args[0], "build")
            self.assertEqual(Path(stage_args[2]), out / "program.py")

    def test_runtime_import_is_frozen_before_blender(self):
        source = """
from procagen3d_runtime import box

def build():
    return box
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            program = root / "authored.py"
            out = root / "out"
            program.write_text(source, encoding="utf-8")
            with patch.object(blender_driver, "run_blender", return_value=0) as run:
                self.assertEqual(blender_driver.cmd_build(build_args(program, out)), 0)

            retained = (out / "program.py").read_text(encoding="utf-8")
            self.assertIn(RUNTIME_BEGIN, retained)
            self.assertNotIn("\nfrom procagen3d_runtime import box", retained)
            self.assertEqual(Path(run.call_args.args[0][2]), out / "program.py")


if __name__ == "__main__":
    unittest.main()
