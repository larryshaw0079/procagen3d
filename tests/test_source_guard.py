from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from procagen3d.source_guard import (
    SourceGuardError,
    assert_safe_source,
    validate_file,
    validate_source,
)


def codes(source: str) -> set[str]:
    return {item.code for item in validate_source(source)}


class SourceGuardTests(unittest.TestCase):
    def test_accepts_standalone_blender_program(self) -> None:
        source = '''
from __future__ import annotations
import bpy
import math
import random as rng
from mathutils import Vector
import bmesh

def helper(radius: float) -> float:
    return math.tau * radius

def build():
    rng.seed(7)
    bpy.ops.mesh.primitive_cube_add(location=(0.0, 0.0, helper(0.1)))
    obj = bpy.context.object
    obj.name = "Body"
    obj.location += Vector((0.0, 0.0, 0.0))
    bpy.context.scene.collection.objects.link(bpy.data.objects.new("Marker", None))
'''
        self.assertEqual(validate_source(source), ())

    def test_requires_one_top_level_build(self) -> None:
        self.assertIn("missing-build", codes("import bpy\n"))
        self.assertIn(
            "multiple-build",
            codes("def build():\n    pass\ndef build():\n    pass\n"),
        )
        conditional = "if True:\n    def build():\n        pass\n"
        self.assertTrue({"missing-build", "conditional-build"} <= codes(conditional))

    def test_build_must_be_plain_synchronous_and_argument_free(self) -> None:
        cases = (
            "async def build():\n    pass\n",
            "def build(value=1):\n    pass\n",
            "def build(*args):\n    pass\n",
            "@staticmethod\ndef build():\n    pass\n",
            "def build():\n    yield 1\n",
        )
        for source in cases:
            with self.subTest(source=source):
                self.assertIn("invalid-build", codes(source))

    def test_nested_generator_does_not_make_build_a_generator(self) -> None:
        source = '''
def build():
    def values():
        yield 1
    list(values())
'''
        self.assertEqual(validate_source(source), ())

    def test_rejects_module_rebinding_of_build(self) -> None:
        for suffix in ("build = 3\n", "if True:\n    build = lambda: None\n", "del build\n"):
            source = "def build():\n    pass\n" + suffix
            with self.subTest(suffix=suffix):
                self.assertIn("build-rebound", codes(source))

    def test_reports_syntax_error_without_executing(self) -> None:
        violations = validate_source("def build(:\n    pass\n", filename="generated.py")
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].code, "syntax-error")
        self.assertEqual(violations[0].line, 1)

    def test_allows_only_explicit_safe_imports(self) -> None:
        safe = '''
import bpy as blender
from bpy import data
import math
from math import sin
import random
from mathutils.geometry import tessellate_polygon
def build():
    pass
'''
        self.assertEqual(validate_source(safe), ())

        for module in ("os", "pathlib", "subprocess", "socket", "urllib.request", "requests"):
            source = f"import {module}\ndef build():\n    pass\n"
            with self.subTest(module=module):
                self.assertIn("forbidden-import", codes(source))

    def test_rejects_relative_and_wildcard_imports(self) -> None:
        relative = "from .helper import shape\ndef build():\n    pass\n"
        wildcard = "from math import *\ndef build():\n    pass\n"
        self.assertIn("forbidden-import", codes(relative))
        self.assertIn("wildcard-import", codes(wildcard))

    def test_rejects_file_and_dynamic_execution_builtins(self) -> None:
        for expression in (
            'open("mesh.dat", "rb")',
            'eval("1 + 1")',
            'exec("pass")',
            'compile("1", "x", "eval")',
            '__import__("os")',
        ):
            source = f"def build():\n    {expression}\n"
            with self.subTest(expression=expression):
                self.assertIn("forbidden-builtin", codes(source))

    def test_rejects_runtime_glb_import_and_export_with_aliases(self) -> None:
        cases = (
            '''
import bpy
def build():
    bpy.ops.import_scene.gltf(filepath="source.glb")
''',
            '''
import bpy as b
def build():
    b.ops.export_scene.gltf(filepath="result.glb")
''',
            '''
from bpy import ops as operations
def build():
    operations.wm.obj_import(filepath="source.obj")
''',
            '''
import bpy
def build():
    operators = bpy.ops
    operators.import_scene.gltf(filepath="source.glb")
''',
        )
        for source in cases:
            with self.subTest(source=source):
                self.assertIn("forbidden-blender-io", codes(source))

    def test_rejects_file_backed_blender_data_and_library_linking(self) -> None:
        cases = (
            'bpy.data.images.load("texture.png")',
            'bpy.data.images["Texture"].save()',
            'bpy.data.texts.load("payload.py")',
            'bpy.data.texts["Payload"].as_module()',
            'bpy.data.libraries.load("library.blend")',
            'bpy.data.libraries.write("library.blend", set())',
            'bpy.ops.wm.append(filename="Body")',
            'bpy.ops.wm.link(filename="Body")',
            'bpy.ops.wm.save_as_mainfile(filepath="scene.blend")',
            'bpy.ops.render.render(write_still=True)',
        )
        for expression in cases:
            source = f"import bpy\ndef build():\n    {expression}\n"
            with self.subTest(expression=expression):
                self.assertIn("forbidden-blender-io", codes(source))

    def test_rejects_obvious_getattr_bypass(self) -> None:
        source = '''
import bpy
def build():
    loader = getattr(bpy.data, "libraries").load
    loader("asset.blend")
'''
        self.assertIn("forbidden-blender-io", codes(source))

    def test_rejects_dynamic_introspection_bypasses(self) -> None:
        cases = (
            '''
def build():
    (lambda: None).__globals__["__builtins__"]["open"]("secret")
''',
            '''
import bpy
def build():
    getattr(getattr(getattr(bpy, "ops"), "import_scene"), "gltf")()
''',
        )
        for source in cases:
            with self.subTest(source=source):
                found = codes(source)
                self.assertTrue(
                    "forbidden-introspection" in found or "forbidden-builtin" in found
                )

    def test_scene_construction_must_be_inside_build(self) -> None:
        direct = '''
import bpy
def build():
    pass
bpy.ops.mesh.primitive_cube_add()
'''
        double_build = '''
def build():
    pass
build()
'''
        self.assertIn("module-side-effect", codes(direct))
        self.assertIn("module-side-effect", codes(double_build))

        guarded = '''
def build():
    pass
if __name__ == "__main__":
    build()
'''
        self.assertEqual(validate_source(guarded), ())

    def test_module_assignments_cannot_mutate_blender_state(self) -> None:
        source = '''
import bpy
DEFAULT_COLOR = (0.2, 0.3, 0.4, 1.0)
def build():
    pass
bpy.context.scene.render.engine = "BLENDER_EEVEE_NEXT"
'''
        self.assertIn("module-side-effect", codes(source))

        subscript = '''
values = [1]
def build():
    pass
values[0] = 2
'''
        self.assertIn("module-side-effect", codes(subscript))

    def test_definitions_cannot_execute_blender_code_before_build(self) -> None:
        default_call = '''
import bpy
def helper(value=bpy.ops.mesh.primitive_cube_add()):
    return value
def build():
    pass
'''
        class_body = '''
import bpy
class HiddenBuild:
    bpy.ops.mesh.primitive_cube_add()
def build():
    pass
'''
        decorated = '''
def decorate(function):
    return function
@decorate
def helper():
    pass
def build():
    pass
'''
        for source in (default_call, class_body, decorated):
            with self.subTest(source=source):
                self.assertIn("module-side-effect", codes(source))

    def test_rejects_external_urls_and_model_literals_but_not_docstrings(self) -> None:
        source = '''
"""Explains that reference.glb is measurement-only."""
def build():
    source_model = "reference.glb"
    endpoint = "https://example.invalid/model"
'''
        found = codes(source)
        self.assertIn("runtime-model-reference", found)
        self.assertIn("external-reference", found)

        docstring_only = '''
"""The offline reference was reference.glb; this module never loads it."""
def build():
    pass
'''
        self.assertEqual(validate_source(docstring_only), ())

    def test_assertion_api_includes_locations(self) -> None:
        with self.assertRaises(SourceGuardError) as caught:
            assert_safe_source(
                'def build():\n    open("x")\n', filename="candidate.py"
            )
        self.assertIn("candidate.py:2", str(caught.exception))
        self.assertEqual(caught.exception.violations[0].code, "forbidden-builtin")

    def test_validate_file_reads_utf8_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "program.py"
            path.write_text('def build():\n    name = "café"\n', encoding="utf-8")
            self.assertEqual(validate_file(path), ())


if __name__ == "__main__":
    unittest.main()
