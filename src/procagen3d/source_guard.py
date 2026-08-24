"""Static safety checks for generated Blender programs.

The generated program is an asset source file, not a general-purpose agent
script.  It may construct a scene with Blender's Python API, but it must not
read another asset at runtime, write/export files, start processes, or reach
the network.  This module deliberately uses a small import allow-list and an
AST inspection pass; it never imports or executes the candidate source.

This is a guardrail, not a Python sandbox.  The compiler still runs accepted
programs in an isolated Blender process with its own timeout and output
directory.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# ``mathutils`` and ``bmesh`` are Blender-bundled modules commonly needed to
# create geometry.  They have no asset-loading or process APIs of their own.
_ALLOWED_IMPORT_ROOTS = frozenset(
    {"__future__", "bmesh", "bpy", "math", "mathutils", "random"}
)
_ALLOWED_FUTURE_IMPORTS = frozenset({"annotations"})

# Referencing these names is enough to reject the source.  Checking loads, not
# merely calls, also catches simple indirection such as ``reader = open``.
_FORBIDDEN_BUILTINS = frozenset(
    {
        "__builtins__",
        "__import__",
        "breakpoint",
        "compile",
        "delattr",
        "dir",
        "eval",
        "exec",
        "exit",
        "getattr",
        "globals",
        "input",
        "locals",
        "open",
        "quit",
        "setattr",
        "vars",
    }
)

_FORBIDDEN_BPY_EXACT = frozenset(
    {
        "bpy.data.libraries.load",
        "bpy.data.libraries.write",
        "bpy.ops.render.render",
        "bpy.ops.script.python_file_run",
        "bpy.ops.text.open",
        "bpy.ops.text.reload",
        "bpy.ops.text.run_script",
        "bpy.ops.wm.append",
        "bpy.ops.wm.link",
        "bpy.ops.wm.open_mainfile",
        "bpy.ops.wm.path_open",
        "bpy.ops.wm.read_factory_settings",
        "bpy.ops.wm.read_homefile",
        "bpy.ops.wm.recover_auto_save",
        "bpy.ops.wm.recover_last_session",
        "bpy.ops.wm.revert_mainfile",
        "bpy.ops.wm.save_as_mainfile",
        "bpy.ops.wm.save_homefile",
        "bpy.ops.wm.save_mainfile",
        "bpy.ops.wm.url_open",
        "bpy.utils.execfile",
        "bpy.utils.load_scripts",
    }
)

# These data APIs bypass Python's builtin ``open`` if left unchecked.  Entries
# cover both collection methods (``images.load``) and methods reached through a
# subscripted datablock (``images['Texture'].save``).
_FILE_BACKED_BPY_METHODS = {
    "cache_files": frozenset({"load", "reload"}),
    "fonts": frozenset({"load", "pack", "unpack"}),
    "images": frozenset({"load", "pack", "reload", "save", "save_render", "unpack"}),
    "libraries": frozenset({"load", "write"}),
    "movieclips": frozenset({"load", "reload"}),
    "sounds": frozenset({"load", "pack", "unpack"}),
    "texts": frozenset({"as_module", "load"}),
}

# Operators under these categories are explicitly about opening or writing an
# external resource.  Normal scene linking such as
# ``collection.objects.link(obj)`` is intentionally not included.
_FILE_BPY_OPERATOR_CATEGORIES = frozenset(
    {
        "asset",
        "cachefile",
        "clip",
        "export_anim",
        "export_mesh",
        "export_scene",
        "extensions",
        "file",
        "font",
        "image",
        "import_anim",
        "import_curve",
        "import_mesh",
        "import_scene",
        "preferences",
        "sound",
    }
)


@dataclass(frozen=True)
class SourceViolation:
    """One deterministic rejection reason in a candidate program."""

    code: str
    message: str
    line: int | None = None
    column: int | None = None

    def format(self, filename: str = "<program>") -> str:
        location = filename
        if self.line is not None:
            location += f":{self.line}"
            if self.column is not None:
                location += f":{self.column + 1}"
        return f"{location}: {self.code}: {self.message}"


class SourceGuardError(ValueError):
    """Raised when :func:`assert_safe_source` rejects generated source."""

    def __init__(self, violations: Iterable[SourceViolation], *, filename: str = "<program>"):
        self.violations = tuple(violations)
        self.filename = filename
        detail = "\n".join(item.format(filename) for item in self.violations)
        super().__init__(f"generated Blender source was rejected:\n{detail}")


def _violation(code: str, message: str, node: ast.AST | None = None) -> SourceViolation:
    return SourceViolation(
        code=code,
        message=message,
        line=getattr(node, "lineno", None),
        column=getattr(node, "col_offset", None),
    )


def _bound_import_name(alias: ast.alias) -> str:
    return alias.asname or alias.name.split(".", 1)[0]


def _assigned_names(target: ast.AST) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for item in target.elts:
            names.update(_assigned_names(item))
        return names
    if isinstance(target, ast.Starred):
        return _assigned_names(target.value)
    return set()


def _module_scope_statements(
    statements: Iterable[ast.stmt], *, conditional: bool = False
) -> Iterable[tuple[ast.stmt, bool]]:
    """Yield statements that execute in module scope, including branches.

    Function and class bodies create their own scopes and are not traversed.
    Control-flow bodies do not, so a definition or assignment to ``build`` in
    one of them can conditionally replace the public entry point.
    """

    for statement in statements:
        yield statement, conditional
        child_groups: list[list[ast.stmt]] = []
        if isinstance(statement, (ast.If, ast.For, ast.AsyncFor, ast.While)):
            child_groups.extend((statement.body, statement.orelse))
        elif isinstance(statement, (ast.With, ast.AsyncWith)):
            child_groups.append(statement.body)
        elif isinstance(statement, (ast.Try, ast.TryStar)):
            child_groups.extend((statement.body, statement.orelse, statement.finalbody))
            child_groups.extend(handler.body for handler in statement.handlers)
        elif isinstance(statement, ast.Match):
            child_groups.extend(case.body for case in statement.cases)
        for group in child_groups:
            yield from _module_scope_statements(group, conditional=True)


def _is_docstring_node(node: ast.AST, parent_body: list[ast.stmt]) -> bool:
    if not parent_body or parent_body[0] is not node or not isinstance(node, ast.Expr):
        return False
    return isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)


def _docstring_constants(tree: ast.AST) -> set[int]:
    constants: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and _is_docstring_node(body[0], body):
            constants.add(id(body[0].value))
    return constants


def _validate_build_entrypoint(tree: ast.Module) -> list[SourceViolation]:
    violations: list[SourceViolation] = []
    direct = [
        statement
        for statement in tree.body
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
        and statement.name == "build"
    ]
    if not direct:
        violations.append(
            _violation(
                "missing-build",
                "define exactly one top-level `def build():` entry point",
            )
        )
    elif len(direct) > 1:
        violations.append(
            _violation(
                "multiple-build",
                "more than one top-level build() definition makes the entry point ambiguous",
                direct[1],
            )
        )

    if len(direct) == 1:
        build = direct[0]
        if isinstance(build, ast.AsyncFunctionDef):
            violations.append(
                _violation("invalid-build", "build() must be synchronous", build)
            )
        if build.decorator_list:
            violations.append(
                _violation(
                    "invalid-build",
                    "build() must not be decorated or replaced at definition time",
                    build.decorator_list[0],
                )
            )
        args = build.args
        if (
            args.posonlyargs
            or args.args
            or args.vararg is not None
            or args.kwonlyargs
            or args.kwarg is not None
        ):
            violations.append(
                _violation(
                    "invalid-build",
                    "build() must accept no arguments",
                    build,
                )
            )

        class YieldFinder(ast.NodeVisitor):
            found: ast.AST | None = None

            def visit_Yield(self, node: ast.Yield) -> None:  # noqa: N802 - ast API
                self.found = self.found or node

            def visit_YieldFrom(self, node: ast.YieldFrom) -> None:  # noqa: N802 - ast API
                self.found = self.found or node

            # A nested generator does not turn build() itself into a generator.
            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
                return

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
                return

            def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
                return

        finder = YieldFinder()
        for statement in build.body:
            finder.visit(statement)
        if finder.found is not None:
            violations.append(
                _violation(
                    "invalid-build",
                    "build() must execute immediately, not return a generator",
                    finder.found,
                )
            )

    direct_ids = {id(node) for node in direct}
    for statement, conditional in _module_scope_statements(tree.body):
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if statement.name == "build" and id(statement) not in direct_ids:
                violations.append(
                    _violation(
                        "conditional-build",
                        "build() must be defined unconditionally at module scope",
                        statement,
                    )
                )
            continue
        rebound = False
        if isinstance(statement, ast.ClassDef):
            rebound = statement.name == "build"
        elif isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets: list[ast.AST]
            if isinstance(statement, ast.Assign):
                targets = list(statement.targets)
            else:
                targets = [statement.target]
            rebound = any("build" in _assigned_names(target) for target in targets)
        elif isinstance(statement, (ast.For, ast.AsyncFor)):
            rebound = "build" in _assigned_names(statement.target)
        elif isinstance(statement, (ast.With, ast.AsyncWith)):
            rebound = any(
                item.optional_vars is not None
                and "build" in _assigned_names(item.optional_vars)
                for item in statement.items
            )
        elif isinstance(statement, ast.Import):
            rebound = any(_bound_import_name(alias) == "build" for alias in statement.names)
        elif isinstance(statement, ast.ImportFrom):
            rebound = any((alias.asname or alias.name) == "build" for alias in statement.names)
        elif isinstance(statement, ast.Delete):
            rebound = any("build" in _assigned_names(target) for target in statement.targets)
        if rebound:
            qualifier = "conditionally " if conditional else ""
            violations.append(
                _violation(
                    "build-rebound",
                    f"module code {qualifier}rebinds or deletes build",
                    statement,
                )
            )
    return violations


def _is_main_build_guard(statement: ast.stmt) -> bool:
    if not isinstance(statement, ast.If) or statement.orelse:
        return False
    test = statement.test
    if not (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    ):
        return False
    return len(statement.body) == 1 and isinstance(statement.body[0], ast.Expr) and isinstance(
        statement.body[0].value, ast.Call
    ) and isinstance(statement.body[0].value.func, ast.Name) and statement.body[0].value.func.id == "build" and not (
        statement.body[0].value.args or statement.body[0].value.keywords
    )


def _validate_module_execution(tree: ast.Module) -> list[SourceViolation]:
    """Keep scene construction behind build(), not at module import time."""

    def plain_binding(target: ast.AST) -> bool:
        if isinstance(target, ast.Name):
            return True
        if isinstance(target, (ast.Tuple, ast.List)):
            return all(plain_binding(item) for item in target.elts)
        if isinstance(target, ast.Starred):
            return plain_binding(target.value)
        return False

    violations: list[SourceViolation] = []
    for index, statement in enumerate(tree.body):
        if isinstance(statement, (ast.Import, ast.ImportFrom, ast.Pass)):
            continue
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            definition_expressions: list[ast.AST] = [
                *statement.decorator_list,
                *statement.args.defaults,
                *(value for value in statement.args.kw_defaults if value is not None),
            ]
            if statement.returns is not None:
                definition_expressions.append(statement.returns)
            definition_expressions.extend(
                argument.annotation
                for argument in (
                    *statement.args.posonlyargs,
                    *statement.args.args,
                    *statement.args.kwonlyargs,
                )
                if argument.annotation is not None
            )
            if statement.args.vararg is not None and statement.args.vararg.annotation is not None:
                definition_expressions.append(statement.args.vararg.annotation)
            if statement.args.kwarg is not None and statement.args.kwarg.annotation is not None:
                definition_expressions.append(statement.args.kwarg.annotation)
            if statement.decorator_list or any(
                isinstance(node, (ast.Call, ast.NamedExpr))
                for expression in definition_expressions
                for node in ast.walk(expression)
            ):
                violations.append(
                    _violation(
                        "module-side-effect",
                        "decorators, called defaults, and executable annotations are not allowed",
                        statement,
                    )
                )
            continue
        if isinstance(statement, ast.ClassDef):
            violations.append(
                _violation(
                    "module-side-effect",
                    "module-level classes execute a body at import time; use helper functions",
                    statement,
                )
            )
            continue
        if (
            index == 0
            and isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        ):
            continue
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            value = statement.value
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            if all(plain_binding(target) for target in targets) and (
                value is None or not any(isinstance(node, ast.Call) for node in ast.walk(value))
            ):
                continue
        if _is_main_build_guard(statement):
            continue
        violations.append(
            _violation(
                "module-side-effect",
                "module-level execution is not allowed; construct the scene inside build()",
                statement,
            )
        )
    return violations


class _SafetyVisitor(ast.NodeVisitor):
    def __init__(self, *, docstrings: set[int]):
        self.violations: list[SourceViolation] = []
        self.aliases: dict[str, str] = {}
        self.docstrings = docstrings

    def add(self, code: str, message: str, node: ast.AST) -> None:
        self.violations.append(_violation(code, message, node))

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802 - ast API
        for alias in node.names:
            root = alias.name.split(".", 1)[0]
            if root not in _ALLOWED_IMPORT_ROOTS:
                self.add(
                    "forbidden-import",
                    f"module {alias.name!r} is not allowed in a standalone Blender asset",
                    node,
                )
            self.aliases[_bound_import_name(alias)] = alias.name

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802 - ast API
        module = node.module or ""
        root = module.split(".", 1)[0]
        if node.level:
            self.add(
                "forbidden-import",
                "relative imports make the generated program depend on another source file",
                node,
            )
        elif root not in _ALLOWED_IMPORT_ROOTS:
            self.add(
                "forbidden-import",
                f"module {module!r} is not allowed in a standalone Blender asset",
                node,
            )
        if module == "__future__":
            unsupported = [
                alias.name for alias in node.names if alias.name not in _ALLOWED_FUTURE_IMPORTS
            ]
            if unsupported:
                self.add(
                    "forbidden-import",
                    f"unsupported __future__ import(s): {', '.join(unsupported)}",
                    node,
                )
        for alias in node.names:
            if alias.name == "*":
                self.add(
                    "wildcard-import",
                    "wildcard imports hide which runtime capabilities the program uses",
                    node,
                )
                continue
            local = alias.asname or alias.name
            self.aliases[local] = f"{module}.{alias.name}" if module else alias.name

    def _qualified_name(self, node: ast.AST) -> str | None:
        parts: list[str] = []
        current = node
        while isinstance(current, (ast.Attribute, ast.Subscript)):
            if isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            else:
                # The concrete key does not affect which API is reached.  By
                # skipping it, ``bpy.data.images['x'].save`` resolves to
                # ``bpy.data.images.save`` and remains statically visible.
                current = current.value
        if not isinstance(current, ast.Name):
            return None
        base = self.aliases.get(current.id, current.id)
        return ".".join((base, *reversed(parts))) if parts else base

    @staticmethod
    def _dangerous_bpy_access(name: str) -> str | None:
        if (
            name in _FORBIDDEN_BPY_EXACT
            or name == "bpy.data.libraries"
            or name.startswith("bpy.data.libraries.")
        ):
            return "Blender library loading/writing is not allowed"

        parts = name.split(".")
        if len(parts) >= 4 and parts[:2] == ["bpy", "data"]:
            collection = parts[2]
            method = parts[3]
            if method in _FILE_BACKED_BPY_METHODS.get(collection, ()):
                return f"bpy.data.{collection}.{method} performs runtime file access"

        if len(parts) >= 3 and parts[:2] == ["bpy", "ops"]:
            category = parts[2]
            operation = parts[3] if len(parts) >= 4 else ""
            if category in _FILE_BPY_OPERATOR_CATEGORIES:
                return f"bpy.ops.{category} accesses an external resource"
            if category.startswith(("import_", "export_")):
                return f"bpy.ops.{category} imports or exports an asset"
            if "_import" in operation or "_export" in operation:
                return f"Blender operator {category}.{operation} imports or exports an asset"
        return None

    def _record_assignment_alias(self, target: ast.AST, value: ast.AST) -> None:
        name = self._qualified_name(value)
        if name is None:
            return
        for local in _assigned_names(target):
            self.aliases[local] = name

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802 - ast API
        for target in node.targets:
            self._record_assignment_alias(target, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: N802 - ast API
        if node.value is not None:
            self._record_assignment_alias(node.target, node.value)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802 - ast API
        if node.attr.startswith("__"):
            self.add(
                "forbidden-introspection",
                f"dunder attribute {node.attr!r} can escape the standalone source contract",
                node,
            )
            return
        name = self._qualified_name(node)
        reason = self._dangerous_bpy_access(name) if name is not None else None
        if reason is not None:
            self.add("forbidden-blender-io", f"{name}: {reason}", node)
            # The shorter child chain is part of the same access.  Avoid noisy
            # duplicate diagnostics such as one for both ``libraries`` and
            # ``libraries.load``.
            return
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802 - ast API
        if (
            isinstance(node.ctx, ast.Load)
            and node.id.startswith("__")
            and node.id != "__name__"
        ):
            self.add(
                "forbidden-introspection",
                f"dunder name {node.id!r} is not allowed",
                node,
            )
        if isinstance(node.ctx, ast.Load) and node.id in _FORBIDDEN_BUILTINS:
            self.add(
                "forbidden-builtin",
                f"builtin {node.id!r} can access files or execute arbitrary code",
                node,
            )

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802 - ast API
        # Catch straightforward getattr indirection without banning legitimate
        # dynamic property access throughout the Blender API.
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            base = self._qualified_name(node.args[0])
            if base is not None:
                combined = f"{base}.{node.args[1].value}"
                reason = self._dangerous_bpy_access(combined)
                if reason is not None:
                    self.add(
                        "forbidden-blender-io",
                        f"dynamic access to {combined}: {reason}",
                        node,
                    )
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:  # noqa: N802 - ast API
        if id(node) in self.docstrings or not isinstance(node.value, str):
            return
        value = node.value.strip().lower()
        if value.startswith(("http://", "https://", "ftp://", "file://")):
            self.add(
                "external-reference",
                "network and file URLs are not allowed in a standalone asset",
                node,
            )
        # A literal source-model path is never needed by a procedural program;
        # the GLB belongs only to the application's evidence workspace.
        clean = value.split("?", 1)[0].split("#", 1)[0]
        if clean in {"__builtins__", "__class__", "__globals__", "__subclasses__"}:
            self.add(
                "forbidden-introspection",
                f"introspection key {clean!r} is not allowed",
                node,
            )
        if clean.endswith((".glb", ".gltf")):
            self.add(
                "runtime-model-reference",
                "the generated program must not refer to a runtime GLB/glTF asset",
                node,
            )


def _deduplicate(violations: Iterable[SourceViolation]) -> tuple[SourceViolation, ...]:
    unique: dict[tuple[str, str, int | None, int | None], SourceViolation] = {}
    for item in violations:
        key = (item.code, item.message, item.line, item.column)
        unique.setdefault(key, item)
    return tuple(
        sorted(
            unique.values(),
            key=lambda item: (
                item.line if item.line is not None else -1,
                item.column if item.column is not None else -1,
                item.code,
                item.message,
            ),
        )
    )


def validate_source(
    source: str, *, filename: str = "<program>"
) -> tuple[SourceViolation, ...]:
    """Return every static source violation without executing *source*.

    An empty tuple means the source passed this guard.  The ``filename`` is
    used for syntax diagnostics and later error formatting only.
    """

    try:
        tree = ast.parse(source, filename=filename, mode="exec")
    except SyntaxError as exc:
        return (
            SourceViolation(
                code="syntax-error",
                message=exc.msg,
                line=exc.lineno,
                column=(exc.offset - 1) if exc.offset is not None else None,
            ),
        )

    violations = _validate_build_entrypoint(tree)
    violations.extend(_validate_module_execution(tree))
    visitor = _SafetyVisitor(docstrings=_docstring_constants(tree))
    visitor.visit(tree)
    violations.extend(visitor.violations)
    return _deduplicate(violations)


def assert_safe_source(source: str, *, filename: str = "<program>") -> None:
    """Raise :class:`SourceGuardError` unless *source* passes the guard."""

    violations = validate_source(source, filename=filename)
    if violations:
        raise SourceGuardError(violations, filename=filename)


def validate_file(path: Path) -> tuple[SourceViolation, ...]:
    """Read and validate one UTF-8 generated program."""

    path = path.expanduser().resolve()
    return validate_source(path.read_text(encoding="utf-8"), filename=str(path))


def assert_safe_file(path: Path) -> None:
    """Read *path* and raise unless it passes the source guard."""

    path = path.expanduser().resolve()
    assert_safe_source(path.read_text(encoding="utf-8"), filename=str(path))
