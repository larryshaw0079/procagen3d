"""Static program checks and canonical-runtime vendoring.

This module deliberately stays in the stdlib-only side of the harness.  It
validates generated source before Blender starts and turns the small authoring
import ``from procagen3d_runtime import ...`` into a self-contained delivered
``program.py``.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path

from .tags import OK, fail


RUNTIME_MODULE = "procagen3d_runtime"
RUNTIME_PATH = Path(__file__).resolve().parents[2] / "runtime" / f"{RUNTIME_MODULE}.py"
RUNTIME_BEGIN = "# === ProcAgen3D canonical modeling runtime (vendored) ==="
RUNTIME_END = "# === End ProcAgen3D canonical modeling runtime ==="
TRANSFORM_FLAGS = ("location", "rotation", "scale")


@dataclass(frozen=True)
class SourceIssue:
    """One deterministic source failure with stable machine-readable fields."""

    code: str
    message: str
    line: int = 0
    column: int = 0

    def display(self, filename: str | Path) -> str:
        where = str(filename)
        if self.line:
            where += f":{self.line}:{self.column + 1}"
        return f"{where}: {self.message}"


class ProgramSourceError(ValueError):
    """Raised when a program cannot be safely frozen for delivery."""


def _parse(source: str, filename: str | Path) -> tuple[ast.Module | None, list[SourceIssue]]:
    try:
        return ast.parse(source, filename=str(filename)), []
    except SyntaxError as exc:
        return None, [SourceIssue(
            "PROGRAM_SYNTAX",
            exc.msg,
            int(exc.lineno or 0),
            max(0, int(exc.offset or 1) - 1),
        )]


def _dotted_name(node: ast.AST) -> str | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def lint_program_source(source: str, filename: str | Path = "<program>") -> list[SourceIssue]:
    """Reject Blender calls whose omitted defaults can move object origins.

    Blender's omitted operator defaults apply all three transform channels.
    Every direct ``bpy.ops.object.transform_apply`` call therefore has to
    state all three dimensions explicitly. Calls through the tested canonical
    ``apply_transform`` wrapper do not need further inspection.
    """

    tree, issues = _parse(source, filename)
    if tree is None:
        return issues

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _dotted_name(node.func) != "bpy.ops.object.transform_apply":
            continue

        problems = []
        if node.args:
            problems.append("positional arguments are not allowed")

        keywords: dict[str, ast.AST] = {}
        for keyword in node.keywords:
            if keyword.arg is None:
                problems.append("**kwargs is not allowed")
                continue
            if keyword.arg in keywords:
                problems.append(f"duplicate keyword {keyword.arg!r}")
            keywords[keyword.arg] = keyword.value

        missing = [name for name in TRANSFORM_FLAGS if name not in keywords]
        if missing:
            problems.append("missing explicit " + ", ".join(missing))

        if problems:
            issues.append(SourceIssue(
                "TRANSFORM_APPLY_FLAGS",
                "unsafe bpy.ops.object.transform_apply call: "
                + "; ".join(problems)
                + ". Use apply_transform(obj, ...) from procagen3d_runtime or "
                  "spell location=, rotation=, and scale= explicitly.",
                node.lineno,
                node.col_offset,
            ))
    return issues


def lint_program_path(path: str | Path) -> list[SourceIssue]:
    path = Path(path)
    return lint_program_source(path.read_text(encoding="utf-8"), path)


def report_source_issues(issues: list[SourceIssue], filename: str | Path) -> int:
    for issue in issues:
        fail(issue.code, issue.display(filename))
    return 1 if issues else 0


def cmd_lint(args) -> int:
    path = Path(args.program)
    if not path.is_file():
        fail("PROGRAM_NOT_FOUND", str(path))
        return 1
    issues = lint_program_path(path)
    if issues:
        return report_source_issues(issues, path)
    try:
        frozen = freeze_program_path(path)
    except ProgramSourceError as exc:
        fail("PROGRAM_RUNTIME", str(exc))
        return 1
    frozen_issues = lint_program_source(frozen, path)
    if frozen_issues:
        return report_source_issues(frozen_issues, path)
    mode = "canonical runtime vendoring ready" if frozen != path.read_text(
        encoding="utf-8") else "self-contained"
    print(f"{OK} source lint passed ({mode})")
    return 0


def _runtime_exports(runtime_source: str) -> set[str]:
    tree = ast.parse(runtime_source, filename=str(RUNTIME_PATH))
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == "__all__"
                   for target in targets):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError) as exc:
            raise ProgramSourceError("canonical runtime has an invalid __all__") from exc
        if not isinstance(value, (list, tuple)) or not all(
                isinstance(name, str) for name in value):
            raise ProgramSourceError("canonical runtime __all__ must contain strings")
        return set(value)
    raise ProgramSourceError("canonical runtime does not define __all__")


def _insertion_line(tree: ast.Module, source_lines: list[str]) -> int:
    """Return a safe line boundary after shebang/docstring/future imports."""

    line = 0
    if source_lines and source_lines[0].startswith("#!"):
        line = 1
    for index in range(min(2, len(source_lines))):
        stripped = source_lines[index].lstrip()
        if "coding" in stripped and stripped.startswith("#"):
            line = max(line, index + 1)

    body = list(tree.body)
    cursor = 0
    if body and isinstance(body[0], ast.Expr) and isinstance(
            body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        line = max(line, int(body[0].end_lineno or body[0].lineno))
        cursor = 1
    while cursor < len(body):
        node = body[cursor]
        if not isinstance(node, ast.ImportFrom) or node.module != "__future__":
            break
        line = max(line, int(node.end_lineno or node.lineno))
        cursor += 1
    return line


def freeze_program_source(
    source: str,
    filename: str | Path = "<program>",
    runtime_source: str | None = None,
) -> str:
    """Vendor an explicit canonical-runtime import into a standalone program.

    Programs without a runtime import are returned byte-for-byte unchanged, so
    existing self-contained assets remain fully backward compatible.
    """

    tree, issues = _parse(source, filename)
    if tree is None:
        raise ProgramSourceError(issues[0].display(filename))

    begin_count = source.count(RUNTIME_BEGIN)
    end_count = source.count(RUNTIME_END)
    if begin_count or end_count:
        if begin_count != 1 or end_count != 1 or source.index(
                RUNTIME_BEGIN) >= source.index(RUNTIME_END):
            raise ProgramSourceError(
                f"{filename}: malformed canonical-runtime vendor markers")
        lingering = [
            node for node in ast.walk(tree)
            if ((isinstance(node, ast.Import)
                 and any(alias.name == RUNTIME_MODULE for alias in node.names))
                or (isinstance(node, ast.ImportFrom)
                    and node.module == RUNTIME_MODULE))
        ]
        if lingering:
            node = lingering[0]
            raise ProgramSourceError(
                f"{filename}:{node.lineno}: program already contains the "
                "vendored runtime; use its helpers directly instead of "
                f"importing {RUNTIME_MODULE} again")
        return source

    imports = [
        node for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == RUNTIME_MODULE
        and node.level == 0
    ]
    unsupported = [
        node for node in ast.walk(tree)
        if ((isinstance(node, ast.Import)
             and any(alias.name == RUNTIME_MODULE for alias in node.names))
            or (isinstance(node, ast.ImportFrom)
                and (node.module == RUNTIME_MODULE
                     or (node.module or "").startswith(RUNTIME_MODULE + "."))
                and node not in imports))
    ]
    if unsupported:
        node = unsupported[0]
        raise ProgramSourceError(
            f"{filename}:{node.lineno}: import the runtime only at module scope "
            f"with `from {RUNTIME_MODULE} import ...`")
    if not imports:
        return source

    if runtime_source is None:
        runtime_source = RUNTIME_PATH.read_text(encoding="utf-8")
    exports = _runtime_exports(runtime_source)
    aliases = []
    for node in imports:
        for alias in node.names:
            if alias.name == "*":
                raise ProgramSourceError(
                    f"{filename}:{node.lineno}: wildcard canonical-runtime "
                    "imports are not allowed; import each helper explicitly")
            if alias.name not in exports:
                raise ProgramSourceError(
                    f"{filename}:{node.lineno}: canonical runtime exports no "
                    f"name {alias.name!r}")
            if alias.asname and alias.asname != alias.name:
                aliases.append(f"{alias.asname} = {alias.name}")

    lines = source.splitlines(keepends=True)
    insertion_line = _insertion_line(tree, lines)
    for node in imports:
        for index in range(node.lineno - 1, int(node.end_lineno or node.lineno)):
            lines[index] = "\n" if lines[index].endswith("\n") else ""

    digest = hashlib.sha256(runtime_source.encode("utf-8")).hexdigest()[:12]
    block = (
        f"\n{RUNTIME_BEGIN}\n"
        f"# source: {RUNTIME_MODULE}.py sha256:{digest}\n"
        f"{runtime_source.rstrip()}\n"
        + (("\n" + "\n".join(aliases) + "\n") if aliases else "")
        + f"{RUNTIME_END}\n\n"
    )
    return "".join(lines[:insertion_line]) + block + "".join(lines[insertion_line:])


def freeze_program_path(path: str | Path) -> str:
    path = Path(path)
    return freeze_program_source(path.read_text(encoding="utf-8"), path)
