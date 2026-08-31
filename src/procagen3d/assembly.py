"""Deterministic part-assembly validation and connector-frame transforms.

The assembly document deliberately contains only JSON-compatible values.  It is
safe to author in ``plan.json`` and independent of Blender: the host can validate
the graph and solve a child's world transform before generated ``bpy`` code runs.

Connector frames are expressed in their owning part's local coordinates.  Their
axes form a right-handed orthonormal basis.  A mate is solved as::

    child_world = parent_world @ parent_frame @ fit_offset @ joint_delta
                  @ inverse(child_frame)

The joint axis is the connector frame's local +Z axis.  Revolute parameters are
radians, prismatic parameters use normalized Blender units, and spherical
parameters are XYZ Euler angles in radians.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


ASSEMBLY_VERSION = 1
MATE_TYPES = ("rigid", "revolute", "prismatic", "spherical")
FIT_TYPES = ("none", "clearance", "transition", "interference")
CONNECTOR_INTERFACES = (
    "generic",
    "planar",
    "cylindrical",
    "spherical",
    "bolt-pattern",
    "peg-socket",
    "seat-face",
    "flange",
    "tab-slot",
    "press-fit",
    "lip-rabbet",
    "snap-tab",
    "key",
    "custom",
)
CONNECTOR_ROLES = ("neutral", "male", "female")

Vector3 = tuple[float, float, float]
Matrix4 = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]

_IDENTITY: Matrix4 = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)
_FRAME_TOLERANCE = 1.0e-5


class AssemblyError(ValueError):
    """Raised when an assembly graph or transform request is invalid."""


@dataclass(frozen=True)
class AssemblyValidationIssue:
    """One stable, machine-readable assembly validation failure."""

    path: tuple[str | int, ...]
    keyword: str
    message: str

    @property
    def location(self) -> str:
        value = "$"
        for component in self.path:
            if isinstance(component, int):
                value += f"[{component}]"
            elif component.isidentifier():
                value += f".{component}"
            else:
                value += f"[{component!r}]"
        return value

    def __str__(self) -> str:
        return f"{self.location}: {self.message}"


@dataclass(frozen=True)
class AssemblyValidationReport:
    """Validation result plus the deterministic dependency order when available."""

    issues: tuple[AssemblyValidationIssue, ...]
    ordered_part_ids: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.issues

    def raise_for_errors(self) -> None:
        if not self.issues:
            return
        details = "\n".join(f"- {issue}" for issue in self.issues)
        raise AssemblyError(
            f"assembly is invalid ({len(self.issues)} "
            f"error{'s' if len(self.issues) != 1 else ''}):\n{details}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "ordered_part_ids": list(self.ordered_part_ids),
            "issues": [
                {
                    "path": issue.location,
                    "keyword": issue.keyword,
                    "message": issue.message,
                }
                for issue in self.issues
            ],
        }


def identity_matrix() -> Matrix4:
    """Return the immutable 4x4 identity matrix used by the solver."""

    return _IDENTITY


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _vector3(value: Any) -> Vector3 | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    if not all(_finite_number(component) for component in value):
        return None
    return (float(value[0]), float(value[1]), float(value[2]))


def _dot(left: Vector3, right: Vector3) -> float:
    return sum(a * b for a, b in zip(left, right))


def _cross(left: Vector3, right: Vector3) -> Vector3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _length(vector: Vector3) -> float:
    return math.sqrt(_dot(vector, vector))


def _frame_problem(frame: Any) -> str | None:
    if not isinstance(frame, Mapping):
        return "must be an object"
    origin = _vector3(frame.get("origin"))
    axes = tuple(
        _vector3(frame.get(name)) for name in ("x_axis", "y_axis", "z_axis")
    )
    if origin is None:
        return "origin must contain three finite numbers"
    if any(axis is None for axis in axes):
        return "x_axis, y_axis, and z_axis must each contain three finite numbers"
    x_axis, y_axis, z_axis = axes
    assert x_axis is not None and y_axis is not None and z_axis is not None
    if any(abs(_length(axis) - 1.0) > _FRAME_TOLERANCE for axis in axes):
        return "axes must have unit length"
    if any(
        abs(_dot(left, right)) > _FRAME_TOLERANCE
        for left, right in (
            (x_axis, y_axis),
            (x_axis, z_axis),
            (y_axis, z_axis),
        )
    ):
        return "axes must be mutually orthogonal"
    handedness = _dot(_cross(x_axis, y_axis), z_axis)
    if abs(handedness - 1.0) > _FRAME_TOLERANCE:
        return "axes must form a right-handed basis (x_axis cross y_axis = z_axis)"
    return None


def frame_matrix(frame: Mapping[str, Any]) -> Matrix4:
    """Convert a validated connector frame into a row-major 4x4 matrix."""

    problem = _frame_problem(frame)
    if problem is not None:
        raise AssemblyError(f"invalid connector frame: {problem}")
    origin = _vector3(frame["origin"])
    x_axis = _vector3(frame["x_axis"])
    y_axis = _vector3(frame["y_axis"])
    z_axis = _vector3(frame["z_axis"])
    assert origin is not None and x_axis is not None
    assert y_axis is not None and z_axis is not None
    return (
        (x_axis[0], y_axis[0], z_axis[0], origin[0]),
        (x_axis[1], y_axis[1], z_axis[1], origin[1]),
        (x_axis[2], y_axis[2], z_axis[2], origin[2]),
        (0.0, 0.0, 0.0, 1.0),
    )


def _matrix4(value: Any, *, name: str) -> Matrix4:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise AssemblyError(f"{name} must be a 4x4 matrix")
    rows: list[tuple[float, float, float, float]] = []
    for row in value:
        if not isinstance(row, (list, tuple)) or len(row) != 4:
            raise AssemblyError(f"{name} must be a 4x4 matrix")
        if not all(_finite_number(component) for component in row):
            raise AssemblyError(f"{name} must contain only finite numbers")
        rows.append(tuple(float(component) for component in row))
    if any(
        abs(actual - expected) > _FRAME_TOLERANCE
        for actual, expected in zip(rows[3], (0, 0, 0, 1))
    ):
        raise AssemblyError(f"{name} must be an affine matrix with final row [0, 0, 0, 1]")
    return tuple(rows)  # type: ignore[return-value]


def multiply_matrices(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> Matrix4:
    """Multiply two finite affine 4x4 matrices without third-party libraries."""

    a = _matrix4(left, name="left matrix")
    b = _matrix4(right, name="right matrix")
    return tuple(
        tuple(sum(a[row][inner] * b[inner][column] for inner in range(4)) for column in range(4))
        for row in range(4)
    )  # type: ignore[return-value]


def invert_rigid_transform(matrix: Sequence[Sequence[float]]) -> Matrix4:
    """Invert a right-handed rigid transform after checking its rotation basis."""

    value = _matrix4(matrix, name="rigid transform")
    columns: tuple[Vector3, Vector3, Vector3] = tuple(
        tuple(value[row][column] for row in range(3)) for column in range(3)
    )  # type: ignore[assignment]
    frame = {
        "origin": [value[row][3] for row in range(3)],
        "x_axis": list(columns[0]),
        "y_axis": list(columns[1]),
        "z_axis": list(columns[2]),
    }
    problem = _frame_problem(frame)
    if problem is not None:
        raise AssemblyError(f"rigid transform rotation is invalid: {problem}")
    rotation_transpose = tuple(
        tuple(value[column][row] for column in range(3)) for row in range(3)
    )
    translation = tuple(value[row][3] for row in range(3))
    inverse_translation = tuple(
        -sum(rotation_transpose[row][column] * translation[column] for column in range(3))
        for row in range(3)
    )
    return (
        (*rotation_transpose[0], inverse_translation[0]),
        (*rotation_transpose[1], inverse_translation[1]),
        (*rotation_transpose[2], inverse_translation[2]),
        (0.0, 0.0, 0.0, 1.0),
    )


def transform_point(matrix: Sequence[Sequence[float]], point: Sequence[float]) -> Vector3:
    """Apply an affine transform to one 3D point."""

    value = _matrix4(matrix, name="transform")
    vector = _vector3(point)
    if vector is None:
        raise AssemblyError("point must contain three finite numbers")
    return tuple(
        sum(value[row][column] * vector[column] for column in range(3))
        + value[row][3]
        for row in range(3)
    )  # type: ignore[return-value]


def _translation(vector: Vector3) -> Matrix4:
    return (
        (1.0, 0.0, 0.0, vector[0]),
        (0.0, 1.0, 0.0, vector[1]),
        (0.0, 0.0, 1.0, vector[2]),
        (0.0, 0.0, 0.0, 1.0),
    )


def _rotation_x(angle: float) -> Matrix4:
    cosine, sine = math.cos(angle), math.sin(angle)
    return (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, cosine, -sine, 0.0),
        (0.0, sine, cosine, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def _rotation_y(angle: float) -> Matrix4:
    cosine, sine = math.cos(angle), math.sin(angle)
    return (
        (cosine, 0.0, sine, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (-sine, 0.0, cosine, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def _rotation_z(angle: float) -> Matrix4:
    cosine, sine = math.cos(angle), math.sin(angle)
    return (
        (cosine, -sine, 0.0, 0.0),
        (sine, cosine, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def _parameter_for_mate(mate: Mapping[str, Any], parameter: Any) -> float | Vector3:
    mate_type = mate.get("type")
    if mate_type == "rigid":
        if parameter is not None:
            raise AssemblyError("a rigid mate does not accept a joint parameter")
        return 0.0
    resolved = mate.get("rest") if parameter is None else parameter
    if mate_type in {"revolute", "prismatic"}:
        if not _finite_number(resolved):
            raise AssemblyError(f"{mate_type} joint parameter must be a finite number")
        return float(resolved)
    if mate_type == "spherical":
        vector = _vector3(resolved)
        if vector is None:
            raise AssemblyError("spherical joint parameter must contain three finite XYZ angles")
        return vector
    raise AssemblyError(f"unsupported mate type {mate_type!r}")


def _parameter_within_limits(mate: Mapping[str, Any], parameter: float | Vector3) -> bool:
    limits = mate.get("limits")
    if not isinstance(limits, Mapping):
        return True
    lower = limits.get("lower")
    upper = limits.get("upper")
    if isinstance(parameter, float):
        return (
            _finite_number(lower)
            and _finite_number(upper)
            and float(lower) <= parameter <= float(upper)
        )
    low_vector = _vector3(lower)
    high_vector = _vector3(upper)
    return (
        low_vector is not None
        and high_vector is not None
        and all(
            low <= value <= high
            for low, value, high in zip(low_vector, parameter, high_vector)
        )
    )


def _joint_delta(mate: Mapping[str, Any], parameter: float | Vector3) -> Matrix4:
    mate_type = mate["type"]
    if mate_type == "rigid":
        return _IDENTITY
    if not _parameter_within_limits(mate, parameter):
        raise AssemblyError("joint parameter is outside the mate limits")
    if mate_type == "revolute":
        assert isinstance(parameter, float)
        return _rotation_z(parameter)
    if mate_type == "prismatic":
        assert isinstance(parameter, float)
        return _translation((0.0, 0.0, parameter))
    assert mate_type == "spherical" and isinstance(parameter, tuple)
    # Intrinsic XYZ Euler rotation: Rx is applied first, followed by Ry and Rz.
    return multiply_matrices(
        multiply_matrices(_rotation_z(parameter[2]), _rotation_y(parameter[1])),
        _rotation_x(parameter[0]),
    )


def solve_mate_transform(
    parent_world: Sequence[Sequence[float]],
    parent_frame: Mapping[str, Any],
    child_frame: Mapping[str, Any],
    mate: Mapping[str, Any],
    *,
    joint_parameter: Any = None,
) -> Matrix4:
    """Solve one child's world transform from paired connector frames.

    ``joint_parameter`` overrides the mate's rest value.  Revolute values are
    radians, prismatic values are normalized Blender units, and spherical
    values are three XYZ Euler angles in radians.
    """

    parent = _matrix4(parent_world, name="parent world transform")
    parent_connector = frame_matrix(parent_frame)
    child_connector_inverse = invert_rigid_transform(frame_matrix(child_frame))
    fit_offset = _vector3(mate.get("fit_offset", (0.0, 0.0, 0.0)))
    if fit_offset is None:
        raise AssemblyError("mate fit_offset must contain three finite numbers")
    parameter = _parameter_for_mate(mate, joint_parameter)
    return multiply_matrices(
        multiply_matrices(
            multiply_matrices(
                multiply_matrices(parent, parent_connector),
                _translation(fit_offset),
            ),
            _joint_delta(mate, parameter),
        ),
        child_connector_inverse,
    )


def solve_child_transform(
    plan: Mapping[str, Any],
    mate_id: str,
    *,
    parent_world: Sequence[Sequence[float]] | None = None,
    joint_parameter: Any = None,
) -> Matrix4:
    """Resolve connectors by ID and solve one mate from a normalized plan."""

    report = validate_assembly(plan)
    report.raise_for_errors()
    assembly = plan.get("assembly")
    assert isinstance(assembly, Mapping)
    mates = assembly.get("mates")
    connectors = assembly.get("connectors")
    assert isinstance(mates, list) and isinstance(connectors, list)
    matching = [mate for mate in mates if isinstance(mate, Mapping) and mate.get("id") == mate_id]
    if not matching:
        raise AssemblyError(f"unknown mate id {mate_id!r}")
    mate = matching[0]
    by_id = {
        connector["id"]: connector
        for connector in connectors
        if isinstance(connector, Mapping) and isinstance(connector.get("id"), str)
    }
    parent_connector = by_id[mate["parent_connector_id"]]
    child_connector = by_id[mate["child_connector_id"]]
    return solve_mate_transform(
        _IDENTITY if parent_world is None else parent_world,
        parent_connector["frame"],
        child_connector["frame"],
        mate,
        joint_parameter=joint_parameter,
    )


def _matrices_close(
    left: Sequence[Sequence[float]],
    right: Sequence[Sequence[float]],
    *,
    tolerance: float = 1.0e-6,
) -> bool:
    a = _matrix4(left, name="left matrix")
    b = _matrix4(right, name="right matrix")
    return all(
        abs(a[row][column] - b[row][column]) <= tolerance
        for row in range(4)
        for column in range(4)
    )


def solve_assembly_transforms(
    plan: Mapping[str, Any],
    joint_values: Mapping[str, Any] | None = None,
    *,
    root_transform: Sequence[Sequence[float]] | None = None,
) -> dict[str, Matrix4]:
    """Solve world transforms for the complete ordered part graph.

    ``joint_values`` is keyed by mate ID.  The unique root receives
    ``root_transform`` (identity by default).  Multiple mates may constrain the
    same child, but they must agree on one transform.  A legacy part without a
    connector mate receives the root transform, preserving plans whose generated
    geometry is already authored directly in one shared coordinate frame.
    """

    report = validate_assembly(plan)
    report.raise_for_errors()
    assembly = plan.get("assembly")
    parts_value = plan.get("parts")
    if not isinstance(assembly, Mapping) or not isinstance(parts_value, list):
        raise AssemblyError("plan must contain normalized parts and assembly objects")
    connectors_value = assembly.get("connectors")
    mates_value = assembly.get("mates")
    assert isinstance(connectors_value, list) and isinstance(mates_value, list)
    connectors = {
        connector["id"]: connector
        for connector in connectors_value
        if isinstance(connector, Mapping) and isinstance(connector.get("id"), str)
    }
    incoming: dict[str, list[Mapping[str, Any]]] = {}
    parent_for_mate: dict[str, str] = {}
    for mate in mates_value:
        if not isinstance(mate, Mapping):
            continue
        parent_connector = connectors[mate["parent_connector_id"]]
        child_connector = connectors[mate["child_connector_id"]]
        child_part = child_connector["part_id"]
        incoming.setdefault(child_part, []).append(mate)
        parent_for_mate[mate["id"]] = parent_connector["part_id"]

    values = joint_values or {}
    known_mate_ids = {
        mate.get("id") for mate in mates_value if isinstance(mate, Mapping)
    }
    unknown_joint_ids = sorted(set(values) - known_mate_ids)
    if unknown_joint_ids:
        raise AssemblyError(
            "joint_values contains unknown mate ids: "
            + ", ".join(repr(identifier) for identifier in unknown_joint_ids)
        )
    root = _matrix4(
        _IDENTITY if root_transform is None else root_transform,
        name="root transform",
    )
    transforms: dict[str, Matrix4] = {}
    part_by_id = {
        part.get("id"): part
        for part in parts_value
        if isinstance(part, Mapping) and isinstance(part.get("id"), str)
    }
    for part_id in report.ordered_part_ids:
        part = part_by_id[part_id]
        attachment = part.get("attachment")
        is_root = isinstance(attachment, Mapping) and attachment.get("type") == "root"
        mates = sorted(incoming.get(part_id, ()), key=lambda mate: str(mate.get("id")))
        if is_root:
            if mates:
                raise AssemblyError(f"root part {part_id!r} must not have an incoming mate")
            transforms[part_id] = root
            continue
        if not mates:
            # Compatibility path: old generated programs author every object's
            # geometry in the shared normalized world frame.
            transforms[part_id] = root
            continue
        candidates: list[tuple[str, Matrix4]] = []
        for mate in mates:
            mate_id = mate["id"]
            parent_part = parent_for_mate[mate_id]
            if parent_part not in transforms:
                raise AssemblyError(
                    f"mate {mate_id!r} depends on unsolved parent part {parent_part!r}"
                )
            parent_connector = connectors[mate["parent_connector_id"]]
            child_connector = connectors[mate["child_connector_id"]]
            candidates.append(
                (
                    mate_id,
                    solve_mate_transform(
                        transforms[parent_part],
                        parent_connector["frame"],
                        child_connector["frame"],
                        mate,
                        joint_parameter=values.get(mate_id),
                    ),
                )
            )
        selected_id, selected = candidates[0]
        for candidate_id, candidate in candidates[1:]:
            if not _matrices_close(selected, candidate):
                raise AssemblyError(
                    f"mates {selected_id!r} and {candidate_id!r} over-constrain "
                    f"part {part_id!r} with different transforms"
                )
        transforms[part_id] = selected
    return transforms


def _stable_topological_order(
    part_ids: Sequence[str],
    edges: Iterable[tuple[str, str]],
    preference: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    known = set(part_ids)
    unique_edges = {
        (parent, child)
        for parent, child in edges
        if parent in known and child in known and parent != child
    }
    children: dict[str, set[str]] = {identifier: set() for identifier in part_ids}
    indegree = {identifier: 0 for identifier in part_ids}
    for parent, child in unique_edges:
        if child not in children[parent]:
            children[parent].add(child)
            indegree[child] += 1
    priority = {identifier: index for index, identifier in enumerate(preference)}
    fallback = {identifier: index for index, identifier in enumerate(part_ids)}

    def rank(identifier: str) -> tuple[int, int, str]:
        return (
            priority.get(identifier, len(priority) + fallback.get(identifier, 0)),
            fallback.get(identifier, len(fallback)),
            identifier,
        )

    ready = sorted((identifier for identifier, degree in indegree.items() if degree == 0), key=rank)
    ordered: list[str] = []
    while ready:
        current = ready.pop(0)
        ordered.append(current)
        for child in sorted(children[current], key=rank):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
                ready.sort(key=rank)
    cyclic = tuple(
        sorted(
            (identifier for identifier, degree in indegree.items() if degree > 0),
            key=rank,
        )
    )
    return tuple(ordered), cyclic


def _graph_edges(
    parts: Sequence[Mapping[str, Any]], assembly: Mapping[str, Any]
) -> tuple[tuple[str, str], ...]:
    edges: list[tuple[str, str]] = []
    for part in parts:
        identifier = part.get("id")
        attachment = part.get("attachment")
        if not isinstance(identifier, str) or not isinstance(attachment, Mapping):
            continue
        parent = attachment.get("parent_id")
        if isinstance(parent, str) and parent != "__root__":
            edges.append((parent, identifier))
    connectors = assembly.get("connectors")
    mates = assembly.get("mates")
    if isinstance(connectors, list) and isinstance(mates, list):
        connector_parts = {
            connector.get("id"): connector.get("part_id")
            for connector in connectors
            if isinstance(connector, Mapping)
            and isinstance(connector.get("id"), str)
            and isinstance(connector.get("part_id"), str)
        }
        for mate in mates:
            if not isinstance(mate, Mapping):
                continue
            parent = connector_parts.get(mate.get("parent_connector_id"))
            child = connector_parts.get(mate.get("child_connector_id"))
            if isinstance(parent, str) and isinstance(child, str):
                edges.append((parent, child))
    return tuple(edges)


def _legacy_order(parts: Sequence[Mapping[str, Any]]) -> list[str]:
    part_ids = [part.get("id") for part in parts]
    valid_ids = [identifier for identifier in part_ids if isinstance(identifier, str)]
    edges: list[tuple[str, str]] = []
    for part in parts:
        identifier = part.get("id")
        attachment = part.get("attachment")
        if not isinstance(identifier, str) or not isinstance(attachment, Mapping):
            continue
        parent = attachment.get("parent_id")
        if isinstance(parent, str) and parent != "__root__":
            edges.append((parent, identifier))
    ordered, cyclic = _stable_topological_order(valid_ids, edges, valid_ids)
    return list(ordered if not cyclic and len(ordered) == len(valid_ids) else valid_ids)


def normalize_assembly(parts: Any, value: Any) -> Any:
    """Return a compatibility-normalized assembly without mutating input values.

    Legacy plans synthesize a dependency-respecting ``part_order`` and empty
    connector/mate lists.  Explicit assembly documents retain their authored
    order and receive only non-semantic defaults.
    """

    if not isinstance(parts, list) or not all(isinstance(part, Mapping) for part in parts):
        return value
    if value is None:
        return {
            "version": ASSEMBLY_VERSION,
            "part_order": _legacy_order(parts),
            "connectors": [],
            "mates": [],
        }
    if not isinstance(value, Mapping):
        return value
    assembly = dict(value)
    assembly.setdefault("version", ASSEMBLY_VERSION)
    assembly.setdefault("part_order", _legacy_order(parts))
    raw_connectors = assembly.get("connectors", [])
    if isinstance(raw_connectors, list):
        connectors: list[Any] = []
        for raw in raw_connectors:
            if not isinstance(raw, Mapping):
                connectors.append(raw)
                continue
            connector = dict(raw)
            connector.setdefault("interface", "generic")
            connector.setdefault("role", "neutral")
            connector.setdefault("nominal_dimensions", {})
            connectors.append(connector)
        assembly["connectors"] = connectors
    else:
        assembly["connectors"] = raw_connectors
    raw_mates = assembly.get("mates", [])
    if isinstance(raw_mates, list):
        mates: list[Any] = []
        for raw in raw_mates:
            if not isinstance(raw, Mapping):
                mates.append(raw)
                continue
            mate = dict(raw)
            mate.setdefault("fit", "none")
            mate.setdefault("clearance", 0.0)
            mate.setdefault("fit_offset", [0.0, 0.0, 0.0])
            mate.setdefault("nominal_dimensions", {})
            if mate.get("type") in {"revolute", "prismatic"}:
                mate.setdefault("rest", 0.0)
            elif mate.get("type") == "spherical":
                mate.setdefault("rest", [0.0, 0.0, 0.0])
            mates.append(mate)
        assembly["mates"] = mates
    else:
        assembly["mates"] = raw_mates
    return assembly


def _dimension_problem(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return "must be an object of positive finite dimensions"
    for name, dimension in value.items():
        if not isinstance(name, str) or not name.strip():
            return "dimension names must be non-empty strings"
        if not _finite_number(dimension) or float(dimension) <= 0:
            return f"dimension {name!r} must be a positive finite number"
    return None


def _joint_constraints_problem(mate: Mapping[str, Any]) -> str | None:
    mate_type = mate.get("type")
    rest = mate.get("rest")
    limits = mate.get("limits")
    if mate_type == "rigid":
        if rest is not None or limits is not None:
            return "rigid mates must not declare rest or limits"
        return None
    if mate_type in {"revolute", "prismatic"}:
        if not _finite_number(rest):
            return f"{mate_type} mates require a finite scalar rest value"
        if limits is None:
            return None
        if not isinstance(limits, Mapping):
            return "limits must be an object"
        lower, upper = limits.get("lower"), limits.get("upper")
        if not _finite_number(lower) or not _finite_number(upper):
            return f"{mate_type} limits must contain finite scalar lower and upper values"
        if float(lower) > float(upper):
            return "joint limit lower must not exceed upper"
        if not float(lower) <= float(rest) <= float(upper):
            return "joint rest value must lie within its limits"
        return None
    if mate_type == "spherical":
        rest_vector = _vector3(rest)
        if rest_vector is None:
            return "spherical mates require a three-angle rest value"
        if limits is None:
            return None
        if not isinstance(limits, Mapping):
            return "limits must be an object"
        lower = _vector3(limits.get("lower"))
        upper = _vector3(limits.get("upper"))
        if lower is None or upper is None:
            return "spherical limits must contain three-angle lower and upper values"
        if any(low > high for low, high in zip(lower, upper)):
            return "each spherical lower limit must not exceed its upper limit"
        if any(not low <= value <= high for low, value, high in zip(lower, rest_vector, upper)):
            return "each spherical rest angle must lie within its limits"
        return None
    return None


def validate_assembly(plan: Mapping[str, Any]) -> AssemblyValidationReport:
    """Validate structure, connector frames, mates, and declared build order.

    This semantic validator assumes basic JSON-shape validation may also run,
    but never raises on malformed input.  Every independent issue is collected.
    """

    issues: list[AssemblyValidationIssue] = []
    parts_value = plan.get("parts")
    assembly = plan.get("assembly")
    if not isinstance(parts_value, list) or not all(
        isinstance(part, Mapping) for part in parts_value
    ):
        return AssemblyValidationReport((), ())
    parts: list[Mapping[str, Any]] = list(parts_value)
    part_ids = [part.get("id") for part in parts]
    valid_part_ids = [identifier for identifier in part_ids if isinstance(identifier, str)]
    known_parts = set(valid_part_ids)
    if not isinstance(assembly, Mapping):
        return AssemblyValidationReport((), tuple(valid_part_ids))

    part_order = assembly.get("part_order")
    declared_order = (
        [item for item in part_order if isinstance(item, str)]
        if isinstance(part_order, list)
        else []
    )
    if isinstance(part_order, list):
        seen: set[str] = set()
        for index, identifier in enumerate(part_order):
            if not isinstance(identifier, str):
                continue
            if identifier in seen:
                issues.append(
                    AssemblyValidationIssue(
                        ("assembly", "part_order", index),
                        "uniquePartOrder",
                        f"part id {identifier!r} may appear only once in part_order",
                    )
                )
            seen.add(identifier)
            if identifier not in known_parts:
                issues.append(
                    AssemblyValidationIssue(
                        ("assembly", "part_order", index),
                        "knownPart",
                        "must reference a declared part id",
                    )
                )
        missing = [identifier for identifier in valid_part_ids if identifier not in seen]
        if missing:
            issues.append(
                AssemblyValidationIssue(
                    ("assembly", "part_order"),
                    "completePartOrder",
                    "must contain every declared part id exactly once; missing "
                    + ", ".join(repr(identifier) for identifier in missing),
                )
            )

    connectors_value = assembly.get("connectors")
    connectors = connectors_value if isinstance(connectors_value, list) else []
    connector_by_id: dict[str, Mapping[str, Any]] = {}
    for index, connector in enumerate(connectors):
        if not isinstance(connector, Mapping):
            continue
        identifier = connector.get("id")
        if isinstance(identifier, str):
            if identifier in connector_by_id:
                issues.append(
                    AssemblyValidationIssue(
                        ("assembly", "connectors", index, "id"),
                        "uniqueConnectorId",
                        f"connector id {identifier!r} must be unique",
                    )
                )
            else:
                connector_by_id[identifier] = connector
        part_id = connector.get("part_id")
        if isinstance(part_id, str) and part_id not in known_parts:
            issues.append(
                AssemblyValidationIssue(
                    ("assembly", "connectors", index, "part_id"),
                    "knownPart",
                    "must reference a declared part id",
                )
            )
        problem = _frame_problem(connector.get("frame"))
        if problem is not None:
            issues.append(
                AssemblyValidationIssue(
                    ("assembly", "connectors", index, "frame"),
                    "orthonormalFrame",
                    problem,
                )
            )
        dimensions_problem = _dimension_problem(connector.get("nominal_dimensions"))
        if dimensions_problem is not None:
            issues.append(
                AssemblyValidationIssue(
                    ("assembly", "connectors", index, "nominal_dimensions"),
                    "positiveDimensions",
                    dimensions_problem,
                )
            )

    mates_value = assembly.get("mates")
    mates = mates_value if isinstance(mates_value, list) else []
    mate_ids: set[str] = set()
    used_parent_connectors: set[str] = set()
    used_child_connectors: set[str] = set()
    for index, mate in enumerate(mates):
        if not isinstance(mate, Mapping):
            continue
        identifier = mate.get("id")
        if isinstance(identifier, str):
            if identifier in mate_ids:
                issues.append(
                    AssemblyValidationIssue(
                        ("assembly", "mates", index, "id"),
                        "uniqueMateId",
                        f"mate id {identifier!r} must be unique",
                    )
                )
            mate_ids.add(identifier)
        parent_id = mate.get("parent_connector_id")
        child_id = mate.get("child_connector_id")
        parent_connector = connector_by_id.get(parent_id) if isinstance(parent_id, str) else None
        child_connector = connector_by_id.get(child_id) if isinstance(child_id, str) else None
        if isinstance(parent_id, str):
            if parent_connector is None:
                issues.append(
                    AssemblyValidationIssue(
                        ("assembly", "mates", index, "parent_connector_id"),
                        "knownConnector",
                        "must reference a declared connector id",
                    )
                )
            if parent_id in used_parent_connectors or parent_id in used_child_connectors:
                issues.append(
                    AssemblyValidationIssue(
                        ("assembly", "mates", index, "parent_connector_id"),
                        "singleMatePerConnector",
                        "a connector may participate in only one mate",
                    )
                )
            used_parent_connectors.add(parent_id)
        if isinstance(child_id, str):
            if child_connector is None:
                issues.append(
                    AssemblyValidationIssue(
                        ("assembly", "mates", index, "child_connector_id"),
                        "knownConnector",
                        "must reference a declared connector id",
                    )
                )
            if child_id in used_parent_connectors or child_id in used_child_connectors:
                issues.append(
                    AssemblyValidationIssue(
                        ("assembly", "mates", index, "child_connector_id"),
                        "singleMatePerConnector",
                        "a connector may participate in only one mate",
                    )
                )
            used_child_connectors.add(child_id)
        if parent_id == child_id and isinstance(parent_id, str):
            issues.append(
                AssemblyValidationIssue(
                    ("assembly", "mates", index, "child_connector_id"),
                    "distinctConnectorPair",
                    "parent and child connectors must be different",
                )
            )

        if parent_connector is not None and child_connector is not None:
            parent_part = parent_connector.get("part_id")
            child_part = child_connector.get("part_id")
            if parent_part == child_part:
                issues.append(
                    AssemblyValidationIssue(
                        ("assembly", "mates", index),
                        "distinctPartPair",
                        "a mate must connect two different parts",
                    )
                )
            if isinstance(parent_part, str) and isinstance(child_part, str):
                matching_child = next(
                    (part for part in parts if part.get("id") == child_part), None
                )
                attachment = matching_child.get("attachment") if matching_child else None
                if isinstance(attachment, Mapping):
                    attachment_parent = attachment.get("parent_id")
                    if attachment_parent != parent_part:
                        issues.append(
                            AssemblyValidationIssue(
                                ("assembly", "mates", index, "child_connector_id"),
                                "attachmentMateAgreement",
                                "mate parent part must match the child part attachment.parent_id",
                            )
                        )
                    if mate.get("type") != "rigid" and attachment.get("type") != "articulated":
                        issues.append(
                            AssemblyValidationIssue(
                                ("assembly", "mates", index, "type"),
                                "articulatedAttachment",
                                "kinematic mates require the child attachment type 'articulated'",
                            )
                        )

            parent_interface = parent_connector.get("interface")
            child_interface = child_connector.get("interface")
            if (
                isinstance(parent_interface, str)
                and isinstance(child_interface, str)
                and parent_interface not in {"generic", "custom"}
                and child_interface not in {"generic", "custom"}
                and parent_interface != child_interface
            ):
                issues.append(
                    AssemblyValidationIssue(
                        ("assembly", "mates", index),
                        "compatibleInterface",
                        "paired connectors must use the same interface family",
                    )
                )
            parent_role = parent_connector.get("role")
            child_role = child_connector.get("role")
            if parent_role == child_role and parent_role in {"male", "female"}:
                issues.append(
                    AssemblyValidationIssue(
                        ("assembly", "mates", index),
                        "compatibleConnectorRole",
                        "paired gendered connectors must use complementary roles",
                    )
                )

        fit_offset = _vector3(mate.get("fit_offset"))
        if fit_offset is None:
            issues.append(
                AssemblyValidationIssue(
                    ("assembly", "mates", index, "fit_offset"),
                    "finiteVector",
                    "must contain three finite numbers",
                )
            )
        dimensions_problem = _dimension_problem(mate.get("nominal_dimensions"))
        if dimensions_problem is not None:
            issues.append(
                AssemblyValidationIssue(
                    ("assembly", "mates", index, "nominal_dimensions"),
                    "positiveDimensions",
                    dimensions_problem,
                )
            )
        constraints_problem = _joint_constraints_problem(mate)
        if constraints_problem is not None:
            issues.append(
                AssemblyValidationIssue(
                    ("assembly", "mates", index),
                    "jointConstraints",
                    constraints_problem,
                )
            )

    all_edges = list(_graph_edges(parts, assembly))
    preference = declared_order or valid_part_ids
    computed_order, cyclic = _stable_topological_order(valid_part_ids, all_edges, preference)
    if cyclic:
        issues.append(
            AssemblyValidationIssue(
                ("assembly", "mates"),
                "acyclicAssembly",
                "attachment and mate dependencies must not form a cycle; involved parts: "
                + ", ".join(repr(identifier) for identifier in cyclic),
            )
        )
    if isinstance(part_order, list) and not cyclic:
        positions = {identifier: index for index, identifier in enumerate(declared_order)}
        for parent, child in sorted(set(all_edges)):
            if parent in positions and child in positions and positions[parent] >= positions[child]:
                issues.append(
                    AssemblyValidationIssue(
                        ("assembly", "part_order", positions[child]),
                        "topologicalPartOrder",
                        f"part {child!r} must appear after dependency {parent!r}",
                    )
                )

    ordered_issues = tuple(
        sorted(
            issues,
            key=lambda issue: (
                tuple(str(component) for component in issue.path),
                issue.keyword,
                issue.message,
            ),
        )
    )
    return AssemblyValidationReport(ordered_issues, computed_order)


def topological_part_order(plan: Mapping[str, Any]) -> tuple[str, ...]:
    """Return a stable dependency order or raise with the full validation report."""

    report = validate_assembly(plan)
    report.raise_for_errors()
    return report.ordered_part_ids
