"""Deterministic, opt-in URDF export for explicitly mechanical plans.

This module renders a validated kinematic tree without invoking Blender.  The
pipeline may provide one link-local GLB per part; direct callers may omit that
mapping and receive the conservative combined-model-on-root fallback.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence
from xml.sax.saxutils import escape

from .assembly import (
    AssemblyError,
    Matrix4,
    frame_matrix,
    identity_matrix,
    invert_rigid_transform,
    multiply_matrices,
    normalize_assembly,
    solve_assembly_transforms,
)


_URDF_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
_SUPPORTED_KINDS = frozenset({"object", "scene", "hybrid"})
_SUPPORTED_JOINTS = frozenset(
    {"fixed", "revolute", "continuous", "prismatic", "spherical"}
)
_MOVABLE_JOINTS = frozenset({"revolute", "continuous", "prismatic", "spherical"})
_EPSILON = 1.0e-12
_GLTF_MESH_SUFFIXES = frozenset({".glb", ".gltf"})
# Blender exports glTF in Y-up coordinates.  URDF visual frames are Z-up, so
# applying Rx(+pi/2) maps Blender-exported (x, z, -y) mesh coordinates back to
# the same (x, y, z) link frame used by the assembly solver and joint origins.
_BLENDER_GLTF_TO_URDF_VISUAL_RPY = (math.pi / 2.0, 0.0, 0.0)
_VISUAL_KINEMATIC_WARNING = (
    "URDF output is visual/kinematic only; collision geometry, inertial properties, "
    "transmissions, and actuators are not exported"
)


class URDFExportError(ValueError):
    """Raised when a plan cannot be represented safely as a URDF tree."""


@dataclass(frozen=True)
class URDFExportReport:
    """Machine-readable result of an optional URDF export."""

    status: str
    enabled: bool
    output_path: Path | None
    model_path: Path
    model_uri: str | None
    robot_name: str | None
    root_link: str | None
    link_names: tuple[str, ...]
    joint_names: tuple[str, ...]
    bytes_written: int
    urdf_sha256: str | None
    model_sha256: str | None
    warnings: tuple[str, ...]
    link_meshes: tuple[tuple[str, str], ...] = ()

    @property
    def link_count(self) -> int:
        return len(self.link_names)

    @property
    def joint_count(self) -> int:
        return len(self.joint_names)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "status": self.status,
            "enabled": self.enabled,
            "output_path": str(self.output_path) if self.output_path else None,
            "model_path": str(self.model_path),
            "model_uri": self.model_uri,
            "robot_name": self.robot_name,
            "root_link": self.root_link,
            "link_names": list(self.link_names),
            "joint_names": list(self.joint_names),
            "link_meshes": dict(self.link_meshes),
            "link_count": self.link_count,
            "joint_count": self.joint_count,
            "bytes_written": self.bytes_written,
            "urdf_sha256": self.urdf_sha256,
            "model_sha256": self.model_sha256,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class URDFDocument:
    """Rendered URDF together with the normalized tree metadata."""

    xml: str
    robot_name: str
    model_path: Path
    model_uri: str
    root_link: str
    link_names: tuple[str, ...]
    joint_names: tuple[str, ...]
    warnings: tuple[str, ...]
    link_meshes: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class _Joint:
    name: str
    kind: str
    source_kind: str
    parent: str
    child: str
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]
    axis: tuple[float, float, float] | None
    lower: float | None
    upper: float | None
    effort: float | None
    velocity: float | None


@dataclass(frozen=True)
class URDFLinkFrame:
    """Connector-centred frame used by one host-solved URDF link.

    ``part_from_link`` maps link-local coordinates into the generated part's
    local coordinates.  ``link_world`` is therefore ``part_world @
    part_from_link`` at the assembly rest pose.
    """

    part_world: Matrix4
    part_from_link: Matrix4
    link_world: Matrix4
    incoming_mate: str | None


@dataclass(frozen=True)
class URDFMotionProbe:
    """One deterministic nonzero joint sample and its expected link poses."""

    mate_id: str
    joint_type: str
    assembly_parameter: float
    urdf_position: float
    link_world: tuple[tuple[str, Matrix4], ...]


@dataclass(frozen=True)
class _ResolvedAssembly:
    plan: Mapping[str, Any]
    connectors: Mapping[str, Mapping[str, Any]]
    mates: tuple[Mapping[str, Any], ...]
    link_frames: Mapping[str, URDFLinkFrame]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _xml_attr(value: str) -> str:
    return escape(value, {'"': "&quot;", "'": "&apos;"})


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise URDFExportError(f"{label} must be an object")
    return value


def _require_sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise URDFExportError(f"{label} must be an array")
    return value


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise URDFExportError(f"{label} must be a non-empty string")
    if any(ord(character) < 32 for character in value):
        raise URDFExportError(f"{label} must not contain control characters")
    return value


def _require_name(value: Any, label: str) -> str:
    name = _require_text(value, label)
    if not _URDF_NAME.fullmatch(name):
        raise URDFExportError(
            f"{label} {name!r} is not a safe URDF name; use letters, digits, "
            "underscore, dot, or hyphen, and begin with a letter or underscore"
        )
    return name


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise URDFExportError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise URDFExportError(f"{label} must be a finite number")
    return result


def _vector3(value: Any, label: str) -> tuple[float, float, float]:
    items = _require_sequence(value, label)
    if len(items) != 3:
        raise URDFExportError(f"{label} must contain exactly three numbers")
    return (
        _number(items[0], f"{label}[0]"),
        _number(items[1], f"{label}[1]"),
        _number(items[2], f"{label}[2]"),
    )


def _unit_axis(value: Any, label: str) -> tuple[float, float, float]:
    axis = _vector3(value, label)
    length = math.sqrt(sum(component * component for component in axis))
    if length <= _EPSILON:
        raise URDFExportError(f"{label} must be non-zero")
    return tuple(component / length for component in axis)  # type: ignore[return-value]


def _fmt(value: float) -> str:
    if abs(value) < _EPSILON:
        value = 0.0
    return format(value, ".12g")


def _fmt_vector(value: tuple[float, float, float]) -> str:
    return " ".join(_fmt(component) for component in value)


def _visual_mesh_rpy(mesh_uri: str) -> tuple[float, float, float]:
    """Return the URDF visual-frame correction for a mesh URI."""

    # Format detection belongs to the URI path; query strings and fragments do
    # not change the underlying mesh format.  Matching is case-insensitive.
    uri_path = mesh_uri.split("#", 1)[0].split("?", 1)[0]
    if PurePosixPath(uri_path).suffix.lower() in _GLTF_MESH_SUFFIXES:
        return _BLENDER_GLTF_TO_URDF_VISUAL_RPY
    return (0.0, 0.0, 0.0)


def _default_robot_name(plan: Mapping[str, Any]) -> str:
    subject = str(plan.get("subject", "procagen3d_robot")).strip()
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", subject).strip("_.-")
    if not normalized:
        normalized = "procagen3d_robot"
    if not re.match(r"[A-Za-z_]", normalized):
        normalized = f"robot_{normalized}"
    return normalized


def _model_file(model_glb: str | os.PathLike[str]) -> Path:
    path = Path(model_glb).expanduser().resolve()
    if not path.is_file():
        raise URDFExportError(f"model GLB does not exist: {path}")
    if path.suffix.lower() != ".glb":
        raise URDFExportError(f"model path must end in .glb: {path}")
    return path


def _model_uri(model_path: Path, package_model_path: str | os.PathLike[str] | None) -> str:
    if package_model_path is None:
        return model_path.name
    uri = _require_text(os.fspath(package_model_path), "package_model_path").strip()
    if uri.endswith("/"):
        uri = f"{uri}{model_path.name}"
    return uri.replace("\\", "/")


def _link_mesh_uri(value: str | os.PathLike[str], label: str) -> str:
    """Normalize one declared link mesh path without depending on the CWD."""

    try:
        raw_uri = os.fspath(value)
    except TypeError as exc:
        raise URDFExportError(f"{label} must be a string or path") from exc
    uri = _require_text(raw_uri, label).strip().replace("\\", "/")
    scheme = re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", uri)
    if scheme:
        path_portion = uri[scheme.end() :]
        if any(part == ".." for part in PurePosixPath(path_portion).parts):
            raise URDFExportError(f"{label} URI must not traverse parent directories")
        return uri
    if uri.startswith(("/", "~/")):
        path = Path(uri).expanduser().resolve()
        if not path.is_file():
            raise URDFExportError(f"{label} local mesh does not exist: {path}")
        return path.as_uri()
    path = PurePosixPath(uri)
    if not path.parts or any(part == ".." for part in path.parts):
        raise URDFExportError(
            f"{label} must be a URI, an existing absolute path, or a safe relative path"
        )
    normalized = path.as_posix()
    if normalized in {"", "."}:
        raise URDFExportError(f"{label} must identify a mesh file")
    return normalized


def _normalize_link_meshes(
    value: Mapping[str, str | os.PathLike[str]] | None,
    links: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    if not isinstance(value, Mapping):
        raise URDFExportError("link_meshes must be an object mapping link ids to mesh paths")
    normalized: dict[str, str] = {}
    for raw_link, raw_uri in value.items():
        link = _require_name(raw_link, "link_meshes key")
        normalized[link] = _link_mesh_uri(raw_uri, f"link_meshes[{link!r}]")
    expected = set(links)
    provided = set(normalized)
    missing = sorted(expected - provided)
    extra = sorted(provided - expected)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing links {missing!r}")
        if extra:
            details.append(f"unknown links {extra!r}")
        raise URDFExportError(
            "link_meshes must cover every plan link exactly: " + "; ".join(details)
        )
    return tuple(sorted(normalized.items()))


def _links(plan: Mapping[str, Any]) -> tuple[tuple[str, ...], dict[str, Mapping[str, Any]]]:
    parts = _require_sequence(plan.get("parts"), "plan.parts")
    if not parts:
        raise URDFExportError("plan.parts must declare at least one link")
    ordered: list[str] = []
    by_id: dict[str, Mapping[str, Any]] = {}
    for index, raw_part in enumerate(parts):
        part = _require_mapping(raw_part, f"plan.parts[{index}]")
        part_id = _require_name(part.get("id"), f"plan.parts[{index}].id")
        if part_id in by_id:
            raise URDFExportError(f"duplicate link name {part_id!r} in plan.parts")
        by_id[part_id] = part
        ordered.append(part_id)
    return tuple(ordered), by_id


def _frame_to_origin(frame_value: Any, label: str) -> dict[str, list[float]]:
    frame = _require_mapping(frame_value, label)
    origin = _vector3(frame.get("origin"), f"{label}.origin")
    x_axis = _unit_axis(frame.get("x_axis"), f"{label}.x_axis")
    y_axis = _unit_axis(frame.get("y_axis"), f"{label}.y_axis")
    z_axis = _unit_axis(frame.get("z_axis"), f"{label}.z_axis")

    tolerance = 1.0e-5
    pairs = ((x_axis, y_axis), (x_axis, z_axis), (y_axis, z_axis))
    if any(abs(sum(a * b for a, b in zip(left, right))) > tolerance for left, right in pairs):
        raise URDFExportError(f"{label} axes must be mutually orthogonal")
    cross = (
        x_axis[1] * y_axis[2] - x_axis[2] * y_axis[1],
        x_axis[2] * y_axis[0] - x_axis[0] * y_axis[2],
        x_axis[0] * y_axis[1] - x_axis[1] * y_axis[0],
    )
    if math.sqrt(sum((a - b) ** 2 for a, b in zip(cross, z_axis))) > tolerance:
        raise URDFExportError(f"{label} axes must form a right-handed frame")

    # Axes are the columns of R.  Decompose R = Rz(yaw) Ry(pitch) Rx(roll).
    sy = math.hypot(x_axis[0], x_axis[1])
    if sy > 1.0e-9:
        roll = math.atan2(y_axis[2], z_axis[2])
        pitch = math.atan2(-x_axis[2], sy)
        yaw = math.atan2(x_axis[1], x_axis[0])
    else:
        roll = math.atan2(-z_axis[1], y_axis[1])
        pitch = math.atan2(-x_axis[2], sy)
        yaw = 0.0
    return {
        "xyz": list(origin),
        "rpy": [roll, pitch, yaw],
        "axis": list(z_axis),
    }


def _transform_to_origin(matrix: Sequence[Sequence[float]], label: str) -> dict[str, list[float]]:
    if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
        raise URDFExportError(f"{label} must be a 4x4 transform")
    frame = {
        "origin": [matrix[row][3] for row in range(3)],
        "x_axis": [matrix[row][0] for row in range(3)],
        "y_axis": [matrix[row][1] for row in range(3)],
        "z_axis": [matrix[row][2] for row in range(3)],
    }
    return _frame_to_origin(frame, label)


def _limits_relative_to_rest(mate: Mapping[str, Any]) -> Any:
    limits = mate.get("limits", mate.get("limit"))
    if not isinstance(limits, Mapping):
        return limits
    adjusted = dict(limits)
    rest = mate.get("rest", 0.0)
    if (
        mate.get("type") in {"revolute", "prismatic"}
        and isinstance(rest, (int, float))
        and not isinstance(rest, bool)
    ):
        for bound in ("lower", "upper"):
            value = adjusted.get(bound)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                adjusted[bound] = float(value) - float(rest)
    return adjusted


def _resolve_assembly(
    plan: Mapping[str, Any],
    parts_by_id: Mapping[str, Mapping[str, Any]],
) -> _ResolvedAssembly:
    """Resolve a connector graph into connector-centred URDF link frames."""

    assembly_value = plan.get("assembly")
    if not isinstance(assembly_value, Mapping):
        raise URDFExportError("plan.assembly must be an object")
    assembly = normalize_assembly(list(parts_by_id.values()), assembly_value)
    if not isinstance(assembly, Mapping):
        raise URDFExportError("plan.assembly could not be normalized")
    connector_values: list[Any] = []
    raw_connectors = assembly.get("connectors", [])
    connector_values.extend(_require_sequence(raw_connectors, "plan.assembly.connectors"))
    for part_id, part in parts_by_id.items():
        if "connectors" not in part:
            continue
        for connector_value in _require_sequence(
            part["connectors"], f"plan.parts[{part_id!r}].connectors"
        ):
            connector = dict(
                _require_mapping(
                    connector_value, f"plan.parts[{part_id!r}].connectors[]"
                )
            )
            connector.setdefault("part_id", part_id)
            connector_values.append(connector)

    solver_assembly = normalize_assembly(
        list(parts_by_id.values()),
        {**assembly, "connectors": connector_values},
    )
    if not isinstance(solver_assembly, Mapping):  # pragma: no cover - defensive
        raise URDFExportError("assembly connectors could not be normalized")
    assembly = solver_assembly
    normalized_plan = dict(plan)
    normalized_plan["assembly"] = assembly
    try:
        solved = solve_assembly_transforms(normalized_plan)
    except AssemblyError as exc:
        raise URDFExportError(f"assembly mates cannot be solved for URDF: {exc}") from exc
    normalized_connectors = assembly.get("connectors", [])
    connector_values = list(
        _require_sequence(normalized_connectors, "plan.assembly.connectors")
    )

    connectors: dict[str, Mapping[str, Any]] = {}
    for index, connector_value in enumerate(connector_values):
        connector = _require_mapping(
            connector_value, f"plan.assembly.connectors[{index}]"
        )
        connector_id = _require_text(
            connector.get("id"), f"plan.assembly.connectors[{index}].id"
        )
        if connector_id in connectors:
            raise URDFExportError(f"duplicate assembly connector id {connector_id!r}")
        part_id = _require_name(
            connector.get("part_id"),
            f"plan.assembly.connectors[{index}].part_id",
        )
        if part_id not in parts_by_id:
            raise URDFExportError(
                f"assembly connector {connector_id!r} references unknown part {part_id!r}"
            )
        connectors[connector_id] = connector

    raw_mates = _require_sequence(assembly.get("mates", []), "plan.assembly.mates")
    mates: list[Mapping[str, Any]] = []
    incoming: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for index, mate_value in enumerate(raw_mates):
        mate = _require_mapping(mate_value, f"plan.assembly.mates[{index}]")
        mate_id = _require_name(
            mate.get("id", f"mate_{index}"), f"plan.assembly.mates[{index}].id"
        )
        parent_connector_id = _require_text(
            mate.get("parent_connector_id"),
            f"plan.assembly.mates[{index}].parent_connector_id",
        )
        child_connector_id = _require_text(
            mate.get("child_connector_id"),
            f"plan.assembly.mates[{index}].child_connector_id",
        )
        try:
            parent_connector = connectors[parent_connector_id]
            child_connector = connectors[child_connector_id]
        except KeyError as exc:
            raise URDFExportError(
                f"assembly mate {index} references unknown connector {exc.args[0]!r}"
            ) from exc
        parent_part = str(parent_connector["part_id"])
        child_part = str(child_connector["part_id"])
        if child_part in incoming:
            prior = incoming[child_part][0]
            raise URDFExportError(
                f"link {child_part!r} has more than one incoming assembly mate "
                f"({prior!r} and {mate_id!r}); URDF requires one parent joint"
            )
        incoming[child_part] = (mate_id, child_connector)
        mates.append(mate)

    link_frames: dict[str, URDFLinkFrame] = {}
    for part_id in parts_by_id:
        try:
            part_world = solved[part_id]
        except KeyError as exc:
            raise URDFExportError(
                f"assembly has no solved transform for link {part_id!r}"
            ) from exc
        incoming_value = incoming.get(part_id)
        if incoming_value is None:
            part_from_link = identity_matrix()
            incoming_mate = None
        else:
            incoming_mate, child_connector = incoming_value
            try:
                part_from_link = frame_matrix(child_connector["frame"])
            except (AssemblyError, KeyError) as exc:
                raise URDFExportError(
                    f"incoming connector for link {part_id!r} has an invalid frame: {exc}"
                ) from exc
        link_frames[part_id] = URDFLinkFrame(
            part_world=part_world,
            part_from_link=part_from_link,
            link_world=multiply_matrices(part_world, part_from_link),
            incoming_mate=incoming_mate,
        )

    return _ResolvedAssembly(
        plan=normalized_plan,
        connectors=connectors,
        mates=tuple(mates),
        link_frames=link_frames,
    )


def resolve_urdf_link_frames(
    plan: Mapping[str, Any],
) -> dict[str, URDFLinkFrame]:
    """Return the trusted connector-centred frame for every assembly link."""

    _, parts_by_id = _links(plan)
    return dict(_resolve_assembly(plan, parts_by_id).link_frames)


def _probe_parameter(mate: Mapping[str, Any], label: str) -> tuple[float, float]:
    rest = _number(mate.get("rest", 0.0), f"{label}.rest")
    limits = _require_mapping(mate.get("limits"), f"{label}.limits")
    lower = _number(limits.get("lower"), f"{label}.limits.lower")
    upper = _number(limits.get("upper"), f"{label}.limits.upper")
    endpoint = max((lower, upper), key=lambda value: (abs(value - rest), value))
    parameter = rest + 0.5 * (endpoint - rest)
    delta = parameter - rest
    if abs(delta) <= _EPSILON:
        raise URDFExportError(
            f"{label} has no nonzero in-limit displacement from its rest value"
        )
    return parameter, delta


def resolve_urdf_motion_probes(
    plan: Mapping[str, Any],
) -> tuple[URDFMotionProbe, ...]:
    """Sample each movable mate and solve its expected nonzero link poses."""

    _, parts_by_id = _links(plan)
    resolved = _resolve_assembly(plan, parts_by_id)
    probes: list[URDFMotionProbe] = []
    for index, mate in enumerate(resolved.mates):
        joint_type = str(mate.get("type", ""))
        if joint_type not in {"revolute", "prismatic"}:
            continue
        mate_id = _require_name(
            mate.get("id", f"mate_{index}"), f"plan.assembly.mates[{index}].id"
        )
        parameter, delta = _probe_parameter(
            mate, f"plan.assembly.mates[{index}]"
        )
        try:
            posed_parts = solve_assembly_transforms(
                resolved.plan, joint_values={mate_id: parameter}
            )
        except AssemblyError as exc:
            raise URDFExportError(
                f"assembly motion probe for mate {mate_id!r} failed: {exc}"
            ) from exc
        probes.append(
            URDFMotionProbe(
                mate_id=mate_id,
                joint_type=joint_type,
                assembly_parameter=parameter,
                urdf_position=delta,
                link_world=tuple(
                    (
                        part_id,
                        multiply_matrices(
                            posed_parts[part_id], frame.part_from_link
                        ),
                    )
                    for part_id, frame in resolved.link_frames.items()
                ),
            )
        )
    return tuple(probes)


def _assembly_joints(
    plan: Mapping[str, Any],
    parts_by_id: Mapping[str, Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    assembly_value = plan.get("assembly")
    if not isinstance(assembly_value, Mapping):
        return []
    resolved = _resolve_assembly(plan, parts_by_id)
    joints: list[Mapping[str, Any]] = []
    for index, mate in enumerate(resolved.mates):
        parent_connector_id = str(mate["parent_connector_id"])
        child_connector_id = str(mate["child_connector_id"])
        parent_connector = resolved.connectors[parent_connector_id]
        child_connector = resolved.connectors[child_connector_id]
        parent_part = str(parent_connector["part_id"])
        child_part = str(child_connector["part_id"])
        try:
            parent_inverse = invert_rigid_transform(
                resolved.link_frames[parent_part].link_world
            )
            relative = multiply_matrices(
                parent_inverse, resolved.link_frames[child_part].link_world
            )
        except (AssemblyError, KeyError) as exc:
            raise URDFExportError(
                f"assembly mate {index} has no solved parent-link to child-link transform"
            ) from exc
        origin = _transform_to_origin(
            relative,
            f"assembly mate {index} solved link transform",
        )
        joint: dict[str, Any] = {
            "name": mate.get("id", f"mate_{index}"),
            "parent": parent_part,
            "child": child_part,
            "type": mate.get("type"),
            "origin": {"xyz": origin["xyz"], "rpy": origin["rpy"]},
            # A non-root link frame is its incoming child connector, so the
            # assembly motion axis and the URDF motion axis are both local +Z.
            "axis": [0.0, 0.0, 1.0],
        }
        if "limits" in mate:
            joint["limit"] = _limits_relative_to_rest(mate)
        elif "limit" in mate:
            joint["limit"] = _limits_relative_to_rest(mate)
        joints.append(joint)
    return joints


def _part_assembly_joints(parts_by_id: Mapping[str, Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    joints: list[Mapping[str, Any]] = []
    for part_id, part in parts_by_id.items():
        raw = part.get("assembly")
        if not isinstance(raw, Mapping):
            continue
        candidates: Sequence[Any]
        if "joints" in raw:
            candidates = _require_sequence(raw["joints"], f"part {part_id!r}.assembly.joints")
        elif "joint" in raw:
            candidates = (raw["joint"],)
        elif "type" in raw:
            candidates = (raw,)
        else:
            continue
        for candidate in candidates:
            joint = dict(_require_mapping(candidate, f"part {part_id!r} assembly joint"))
            joint.setdefault("child", part_id)
            joints.append(joint)
    return joints


def _raw_joints(
    plan: Mapping[str, Any],
    articulation: Mapping[str, Any],
    parts_by_id: Mapping[str, Mapping[str, Any]],
) -> Sequence[Any]:
    assembly_value = plan.get("assembly")
    host_solved = (
        isinstance(assembly_value, Mapping)
        and assembly_value.get("placement") == "host-solved"
    )
    candidates = articulation.get("joints")
    if candidates is not None:
        values = _require_sequence(candidates, "plan.articulation.joints")
        if values:
            if host_solved:
                raise URDFExportError(
                    "host-solved assemblies must omit articulation.joints; URDF joints "
                    "are derived from the same solved connector mates as the link meshes"
                )
            return values
    candidates = plan.get("joints")
    if candidates is not None:
        values = _require_sequence(candidates, "plan.joints")
        if values:
            if host_solved:
                raise URDFExportError(
                    "host-solved assemblies must omit top-level joints; URDF joints are "
                    "derived from the same solved connector mates as the link meshes"
                )
            return values
    assembly = _assembly_joints(plan, parts_by_id)
    if assembly:
        return assembly
    return _part_assembly_joints(parts_by_id)


def _first(mapping: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _joint_origin(raw: Mapping[str, Any], label: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    origin_value = raw.get("origin")
    if origin_value is None and ("origin_xyz" in raw or "origin_rpy" in raw):
        origin_value = {"xyz": raw.get("origin_xyz"), "rpy": raw.get("origin_rpy")}
    origin = _require_mapping(origin_value, f"{label}.origin")
    xyz_value = _first(origin, ("xyz", "position"))
    rpy_value = _first(origin, ("rpy", "rotation_rpy"))
    if xyz_value is None or rpy_value is None:
        raise URDFExportError(f"{label}.origin must declare both xyz and rpy")
    return _vector3(xyz_value, f"{label}.origin.xyz"), _vector3(
        rpy_value, f"{label}.origin.rpy"
    )


def _limits(
    raw: Mapping[str, Any],
    kind: str,
    label: str,
    warnings: list[str],
) -> tuple[float | None, float | None, float | None, float | None]:
    if kind == "fixed":
        return None, None, None, None
    limit_value = _first(raw, ("limit", "limits"))
    if limit_value is None:
        if kind in {"revolute", "prismatic"}:
            raise URDFExportError(f"{label}.limit is required for a {kind} joint")
        limit: Mapping[str, Any] = {}
    else:
        limit = _require_mapping(limit_value, f"{label}.limit")

    lower: float | None = None
    upper: float | None = None
    if kind in {"revolute", "prismatic"}:
        if "lower" not in limit or "upper" not in limit:
            raise URDFExportError(
                f"{label}.limit must declare finite lower and upper values for a {kind} joint"
            )
        lower = _number(limit["lower"], f"{label}.limit.lower")
        upper = _number(limit["upper"], f"{label}.limit.upper")
        if lower >= upper:
            raise URDFExportError(f"{label}.limit.lower must be less than upper")

    effort_value = limit.get("effort")
    velocity_value = limit.get("velocity")
    if effort_value is None:
        effort = 1.0
        warnings.append(f"{label}: defaulted missing effort limit to 1")
    else:
        effort = _number(effort_value, f"{label}.limit.effort")
    if velocity_value is None:
        velocity = 1.0
        warnings.append(f"{label}: defaulted missing velocity limit to 1")
    else:
        velocity = _number(velocity_value, f"{label}.limit.velocity")
    if effort <= 0:
        raise URDFExportError(f"{label}.limit.effort must be greater than zero")
    if velocity <= 0:
        raise URDFExportError(f"{label}.limit.velocity must be greater than zero")
    return lower, upper, effort, velocity


def _normalize_joints(
    raw_joints: Sequence[Any],
    links: tuple[str, ...],
    *,
    spherical_policy: str,
) -> tuple[tuple[_Joint, ...], tuple[str, ...]]:
    if spherical_policy not in {"reject", "fixed"}:
        raise URDFExportError("spherical_policy must be 'reject' or 'fixed'")
    if not raw_joints:
        raise URDFExportError("articulation must declare at least one joint")

    link_set = set(links)
    joints: list[_Joint] = []
    warnings: list[str] = []
    names: set[str] = set()
    movable_declared = False
    for index, raw_value in enumerate(raw_joints):
        raw = _require_mapping(raw_value, f"joint[{index}]")
        label = f"joint[{index}]"
        name = _require_name(raw.get("name", raw.get("id")), f"{label}.name")
        if name in names:
            raise URDFExportError(f"duplicate joint name {name!r}")
        names.add(name)
        parent = _require_name(
            _first(raw, ("parent", "parent_id", "parent_link", "parent_part_id")),
            f"{label}.parent",
        )
        child = _require_name(
            _first(raw, ("child", "child_id", "child_link", "child_part_id")),
            f"{label}.child",
        )
        if parent not in link_set:
            raise URDFExportError(f"joint {name!r} references unknown parent link {parent!r}")
        if child not in link_set:
            raise URDFExportError(f"joint {name!r} references unknown child link {child!r}")
        if parent == child:
            raise URDFExportError(f"joint {name!r} cannot connect link {parent!r} to itself")

        source_kind = _require_text(raw.get("type"), f"{label}.type").lower()
        if source_kind == "rigid":
            source_kind = "fixed"
        if source_kind not in _SUPPORTED_JOINTS:
            rendered = ", ".join(sorted(_SUPPORTED_JOINTS - {"spherical"}))
            raise URDFExportError(
                f"joint {name!r} has unsupported type {source_kind!r}; use {rendered}, "
                "or spherical with an explicit spherical_policy"
            )
        movable_declared = movable_declared or source_kind in _MOVABLE_JOINTS
        kind = source_kind
        if source_kind == "spherical":
            if spherical_policy == "reject":
                raise URDFExportError(
                    f"joint {name!r} is spherical, which standard URDF cannot represent; "
                    "use spherical_policy='fixed' for a conservative fixed-joint export"
                )
            kind = "fixed"
            warnings.append(f"joint {name!r}: degraded spherical joint to fixed")

        xyz, rpy = _joint_origin(raw, label)
        axis: tuple[float, float, float] | None = None
        if kind != "fixed":
            axis_value = _first(raw, ("axis", "axis_xyz"))
            if axis_value is None:
                raise URDFExportError(f"{label}.axis is required for a {kind} joint")
            original_axis = _vector3(axis_value, f"{label}.axis")
            axis = _unit_axis(axis_value, f"{label}.axis")
            original_length = math.sqrt(sum(value * value for value in original_axis))
            if abs(original_length - 1.0) > 1.0e-6:
                warnings.append(f"joint {name!r}: normalized its non-unit axis")
        lower, upper, effort, velocity = _limits(raw, kind, label, warnings)
        joints.append(
            _Joint(
                name=name,
                kind=kind,
                source_kind=source_kind,
                parent=parent,
                child=child,
                xyz=xyz,
                rpy=rpy,
                axis=axis,
                lower=lower,
                upper=upper,
                effort=effort,
                velocity=velocity,
            )
        )
    if not movable_declared:
        raise URDFExportError(
            "articulation must declare at least one revolute, continuous, prismatic, or spherical joint"
        )
    return tuple(joints), tuple(warnings)


def _validate_tree(links: tuple[str, ...], joints: tuple[_Joint, ...]) -> str:
    if len(joints) != len(links) - 1:
        raise URDFExportError(
            f"URDF articulation must be one connected tree: {len(links)} links require "
            f"{len(links) - 1} joints, but {len(joints)} were declared"
        )
    incoming: dict[str, str] = {}
    children: dict[str, list[str]] = {link: [] for link in links}
    for joint in joints:
        if joint.child in incoming:
            raise URDFExportError(
                f"link {joint.child!r} has more than one parent joint "
                f"({incoming[joint.child]!r} and {joint.name!r})"
            )
        incoming[joint.child] = joint.name
        children[joint.parent].append(joint.child)
    roots = [link for link in links if link not in incoming]
    if len(roots) != 1:
        raise URDFExportError(
            f"URDF articulation must have exactly one root link; found {roots!r}"
        )

    root = roots[0]
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(link: str) -> None:
        if link in visiting:
            raise URDFExportError(f"articulation joint graph contains a cycle at {link!r}")
        if link in visited:
            return
        visiting.add(link)
        for child in children[link]:
            visit(child)
        visiting.remove(link)
        visited.add(link)

    visit(root)
    if len(visited) != len(links):
        missing = sorted(set(links) - visited)
        raise URDFExportError(
            f"articulation is disconnected or cyclic; unreachable links: {missing!r}"
        )
    return root


def _joint_xml(joint: _Joint) -> list[str]:
    lines = [f'  <joint name="{_xml_attr(joint.name)}" type="{joint.kind}">']
    lines.append(f'    <parent link="{_xml_attr(joint.parent)}"/>')
    lines.append(f'    <child link="{_xml_attr(joint.child)}"/>')
    lines.append(
        f'    <origin xyz="{_fmt_vector(joint.xyz)}" rpy="{_fmt_vector(joint.rpy)}"/>'
    )
    if joint.axis is not None:
        lines.append(f'    <axis xyz="{_fmt_vector(joint.axis)}"/>')
    if joint.kind != "fixed":
        attributes: list[str] = []
        if joint.kind in {"revolute", "prismatic"}:
            assert joint.lower is not None and joint.upper is not None
            attributes.extend(
                [f'lower="{_fmt(joint.lower)}"', f'upper="{_fmt(joint.upper)}"']
            )
        assert joint.effort is not None and joint.velocity is not None
        attributes.extend(
            [f'effort="{_fmt(joint.effort)}"', f'velocity="{_fmt(joint.velocity)}"']
        )
        lines.append(f"    <limit {' '.join(attributes)}/>")
    lines.append("  </joint>")
    return lines


def render_urdf(
    plan: Mapping[str, Any],
    model_glb: str | os.PathLike[str],
    *,
    package_model_path: str | os.PathLike[str] | None = None,
    link_meshes: Mapping[str, str | os.PathLike[str]] | None = None,
    spherical_policy: str = "reject",
) -> URDFDocument:
    """Validate ``plan`` and render deterministic URDF XML.

    Both ``articulation.enabled`` and ``articulation.mechanical`` must be true.
    Host-solved plans derive joints from ``assembly.mates`` and connector frames;
    explicit joint arrays are rejected there to prevent a second kinematic truth.
    Legacy plans may use articulation/top-level joint arrays or per-part
    ``assembly.joint(s)`` aliases.
    """

    plan = _require_mapping(plan, "plan")
    subject_kind = _require_text(plan.get("subject_kind"), "plan.subject_kind")
    if subject_kind not in _SUPPORTED_KINDS:
        supported = ", ".join(sorted(_SUPPORTED_KINDS))
        raise URDFExportError(
            f"URDF export is only available for explicit mechanical {supported} plans; "
            f"got subject_kind={subject_kind!r}"
        )
    articulation = _require_mapping(plan.get("articulation"), "plan.articulation")
    if articulation.get("enabled") is not True:
        raise URDFExportError("plan.articulation.enabled must be true for URDF export")
    mechanical = articulation.get("mechanical", plan.get("mechanical"))
    if mechanical is not True:
        raise URDFExportError(
            "plan.articulation.mechanical must be true; mechanical subjects are never inferred"
        )

    model_path = _model_file(model_glb)
    uri = _model_uri(model_path, package_model_path)
    links, parts_by_id = _links(plan)
    normalized_link_meshes = _normalize_link_meshes(link_meshes, links)
    link_meshes_by_id = dict(normalized_link_meshes)
    raw_joints = _raw_joints(plan, articulation, parts_by_id)
    joints, warnings = _normalize_joints(
        raw_joints, links, spherical_policy=spherical_policy
    )
    root = _validate_tree(links, joints)
    robot_name_value = articulation.get("robot_name")
    robot_name = (
        _require_name(robot_name_value, "plan.articulation.robot_name")
        if robot_name_value is not None
        else _default_robot_name(plan)
    )

    lines = ['<?xml version="1.0"?>', f'<robot name="{_xml_attr(robot_name)}">']
    for link_name in sorted(links):
        link_mesh_uri = link_meshes_by_id.get(link_name)
        combined_model = link_mesh_uri is None and link_name == root
        if combined_model:
            link_mesh_uri = uri
        if link_mesh_uri is not None:
            # The combined fallback is known to be the validated model GLB even
            # when callers give it a package URI without a recognizable suffix.
            mesh_format_uri = model_path.name if combined_model else link_mesh_uri
            visual_rpy = _visual_mesh_rpy(mesh_format_uri)
            visual_name = (
                "procagen3d_link_model"
                if normalized_link_meshes
                else "procagen3d_combined_model"
            )
            lines.extend(
                [
                    f'  <link name="{_xml_attr(link_name)}">',
                    f'    <visual name="{visual_name}">',
                    f'      <origin xyz="0 0 0" rpy="{_fmt_vector(visual_rpy)}"/>',
                    "      <geometry>",
                    f'        <mesh filename="{_xml_attr(link_mesh_uri)}"/>',
                    "      </geometry>",
                    "    </visual>",
                    "  </link>",
                ]
            )
        else:
            lines.append(f'  <link name="{_xml_attr(link_name)}"/>')
    for joint in sorted(joints, key=lambda item: item.name):
        lines.extend(_joint_xml(joint))
    lines.append("</robot>")
    xml = "\n".join(lines) + "\n"
    render_warnings = list(warnings)
    if not normalized_link_meshes:
        render_warnings.append(
            "model.glb is a combined asset and is referenced by the root link only; "
            "child links have no segmented visual or collision meshes"
        )
    render_warnings.append(_VISUAL_KINEMATIC_WARNING)
    return URDFDocument(
        xml=xml,
        robot_name=robot_name,
        model_path=model_path,
        model_uri=uri,
        root_link=root,
        link_names=tuple(sorted(links)),
        joint_names=tuple(sorted(joint.name for joint in joints)),
        link_meshes=normalized_link_meshes,
        warnings=tuple(render_warnings),
    )


def plan_to_urdf(
    plan: Mapping[str, Any],
    model_glb: str | os.PathLike[str],
    *,
    package_model_path: str | os.PathLike[str] | None = None,
    link_meshes: Mapping[str, str | os.PathLike[str]] | None = None,
    spherical_policy: str = "reject",
) -> str:
    """Return URDF XML without writing it to disk."""

    return render_urdf(
        plan,
        model_glb,
        package_model_path=package_model_path,
        link_meshes=link_meshes,
        spherical_policy=spherical_policy,
    ).xml


def export_urdf(
    plan: Mapping[str, Any],
    model_glb: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    enabled: bool = False,
    package_model_path: str | os.PathLike[str] | None = None,
    link_meshes: Mapping[str, str | os.PathLike[str]] | None = None,
    spherical_policy: str = "reject",
) -> URDFExportReport:
    """Atomically export a URDF when explicitly enabled.

    With the default ``enabled=False`` this function performs no validation and
    no filesystem writes, returning a ``skipped`` report.  Enabling the call is
    not sufficient by itself: the plan must independently opt in through its
    explicit ``articulation`` declaration.
    """

    raw_model_path = Path(model_glb).expanduser().resolve()
    if not enabled:
        return URDFExportReport(
            status="skipped",
            enabled=False,
            output_path=None,
            model_path=raw_model_path,
            model_uri=None,
            robot_name=None,
            root_link=None,
            link_names=(),
            joint_names=(),
            link_meshes=(),
            bytes_written=0,
            urdf_sha256=None,
            model_sha256=None,
            warnings=("URDF export was not enabled",),
        )

    document = render_urdf(
        plan,
        model_glb,
        package_model_path=package_model_path,
        link_meshes=link_meshes,
        spherical_policy=spherical_policy,
    )
    destination = Path(output_path).expanduser().resolve()
    if destination.suffix.lower() != ".urdf":
        raise URDFExportError(f"output path must end in .urdf: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = document.xml.encode("utf-8")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, 0o644)
        os.replace(temporary_path, destination)
    except BaseException:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise

    return URDFExportReport(
        status="exported",
        enabled=True,
        output_path=destination,
        model_path=document.model_path,
        model_uri=document.model_uri,
        robot_name=document.robot_name,
        root_link=document.root_link,
        link_names=document.link_names,
        joint_names=document.joint_names,
        link_meshes=document.link_meshes,
        bytes_written=len(payload),
        urdf_sha256=_sha256_bytes(payload),
        model_sha256=_sha256_file(document.model_path),
        warnings=document.warnings,
    )
