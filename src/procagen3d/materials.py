"""Typed material planning and material-only change protection.

The reconstruction pipeline deliberately separates geometry authoring from a
dedicated PBR extraction and assignment pass.  This module contains the pure
Python contract for that pass; Blender-specific material creation remains in
the generated program or a host-side worker.

Material assignments have one unambiguous precedence rule: a subpart rule
overrides the whole-part rule for the same part.  Geometry reports captured
before and after the pass can be compared with :func:`compare_material_pass_geometry`
to ensure that a supposedly cosmetic edit did not alter the asset.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


MATERIAL_PLAN_SCHEMA_VERSION = 1
ALPHA_MODES = ("OPAQUE", "MASK", "BLEND")


class MaterialPlanError(ValueError):
    """Raised when a dedicated material plan is malformed or ambiguous."""


class MaterialGeometryChangeError(RuntimeError):
    """Raised when a material-only pass changed protected geometry."""

    def __init__(self, result: MaterialChangeGuardResult) -> None:
        self.result = result
        fields = ", ".join(violation.field for violation in result.violations)
        super().__init__(f"material-only pass changed protected geometry: {fields}")


def _text(value: Any, *, label: str, identifier: bool = False) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MaterialPlanError(f"{label} must be a non-empty string")
    if value != value.strip():
        raise MaterialPlanError(f"{label} must not have leading or trailing whitespace")
    if identifier and any(character.isspace() for character in value):
        raise MaterialPlanError(f"{label} must not contain whitespace")
    return value


def _number(
    value: Any,
    *,
    label: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MaterialPlanError(f"{label} must be a finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise MaterialPlanError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise MaterialPlanError(f"{label} must be a finite number")
    if minimum is not None and result < minimum:
        raise MaterialPlanError(f"{label} must be at least {minimum:g}")
    if maximum is not None and result > maximum:
        raise MaterialPlanError(f"{label} must be at most {maximum:g}")
    return result


def _unit(value: Any, *, label: str) -> float:
    return _number(value, label=label, minimum=0.0, maximum=1.0)


def _vector(
    value: Any,
    *,
    length: int,
    label: str,
    unit_interval: bool,
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise MaterialPlanError(f"{label} must be an array of {length} numbers")
    if len(value) != length:
        raise MaterialPlanError(f"{label} must contain exactly {length} numbers")
    if unit_interval:
        return tuple(_unit(item, label=f"{label}[{index}]") for index, item in enumerate(value))
    return tuple(_number(item, label=f"{label}[{index}]") for index, item in enumerate(value))


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MaterialPlanError(f"{label} must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise MaterialPlanError(f"{label} keys must be strings")
    return value


def _array(value: Any, *, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise MaterialPlanError(f"{label} must be a JSON array")
    return value


def _unknown_keys(
    value: Mapping[str, Any], *, allowed: frozenset[str], label: str
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise MaterialPlanError(f"{label} has unknown fields: {', '.join(unknown)}")


@dataclass(frozen=True, slots=True)
class PBRMaterial:
    """One compact, glTF-compatible Principled/PBR material definition.

    ``base_color_rgba`` carries the alpha factor.  ``alpha_mode`` and
    ``alpha_cutoff`` are optional because most assets are fully opaque.
    """

    material_id: str
    base_color_rgba: tuple[float, float, float, float]
    metallic: float
    roughness: float
    name: str | None = None
    emissive_rgb: tuple[float, float, float] | None = None
    alpha_mode: str | None = None
    alpha_cutoff: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "material_id",
            _text(self.material_id, label="material_id", identifier=True),
        )
        rgba = _vector(
            self.base_color_rgba,
            length=4,
            label=f"material {self.material_id!r} base_color_rgba",
            unit_interval=True,
        )
        object.__setattr__(self, "base_color_rgba", rgba)
        object.__setattr__(
            self,
            "metallic",
            _unit(self.metallic, label=f"material {self.material_id!r} metallic"),
        )
        object.__setattr__(
            self,
            "roughness",
            _unit(self.roughness, label=f"material {self.material_id!r} roughness"),
        )
        if self.name is not None:
            object.__setattr__(
                self,
                "name",
                _text(self.name, label=f"material {self.material_id!r} name"),
            )
        if self.emissive_rgb is not None:
            object.__setattr__(
                self,
                "emissive_rgb",
                _vector(
                    self.emissive_rgb,
                    length=3,
                    label=f"material {self.material_id!r} emissive_rgb",
                    unit_interval=True,
                ),
            )
        if self.alpha_mode is not None:
            if not isinstance(self.alpha_mode, str):
                raise MaterialPlanError(
                    f"material {self.material_id!r} alpha_mode must be one of: "
                    + ", ".join(ALPHA_MODES)
                )
            mode = self.alpha_mode.upper()
            if mode not in ALPHA_MODES:
                raise MaterialPlanError(
                    f"material {self.material_id!r} alpha_mode must be one of: "
                    + ", ".join(ALPHA_MODES)
                )
            object.__setattr__(self, "alpha_mode", mode)
        if self.alpha_cutoff is not None:
            if self.alpha_mode != "MASK":
                raise MaterialPlanError(
                    f"material {self.material_id!r} alpha_cutoff requires alpha_mode MASK"
                )
            object.__setattr__(
                self,
                "alpha_cutoff",
                _unit(
                    self.alpha_cutoff,
                    label=f"material {self.material_id!r} alpha_cutoff",
                ),
            )

    @property
    def effective_name(self) -> str:
        return self.name or self.material_id

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.material_id,
            "base_color_rgba": list(self.base_color_rgba),
            "metallic": self.metallic,
            "roughness": self.roughness,
        }
        if self.name is not None:
            result["name"] = self.name
        if self.emissive_rgb is not None:
            result["emissive_rgb"] = list(self.emissive_rgb)
        if self.alpha_mode is not None:
            result["alpha_mode"] = self.alpha_mode
        if self.alpha_cutoff is not None:
            result["alpha_cutoff"] = self.alpha_cutoff
        return result

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> PBRMaterial:
        value = _mapping(document, label="material")
        _unknown_keys(
            value,
            allowed=frozenset(
                {
                    "id",
                    "base_color_rgba",
                    "metallic",
                    "roughness",
                    "name",
                    "emissive_rgb",
                    "alpha_mode",
                    "alpha_cutoff",
                }
            ),
            label="material",
        )
        required = ("id", "base_color_rgba", "metallic", "roughness")
        missing = [key for key in required if key not in value]
        if missing:
            raise MaterialPlanError(
                "material is missing required fields: " + ", ".join(missing)
            )
        return cls(
            material_id=value["id"],
            base_color_rgba=value["base_color_rgba"],
            metallic=value["metallic"],
            roughness=value["roughness"],
            name=value.get("name"),
            emissive_rgb=value.get("emissive_rgb"),
            alpha_mode=value.get("alpha_mode"),
            alpha_cutoff=value.get("alpha_cutoff"),
        )


@dataclass(frozen=True, slots=True)
class MaterialAssignment:
    """Assign a library material to a whole part or to one semantic subpart.

    A whole-part assignment has no ``subpart_id`` and is the default.  A
    subpart assignment must supply either exact Blender ``object_names`` or a
    short visual/region ``selector`` for the material-pass agent.  Selectors
    are descriptions, never executable expressions.
    """

    part_id: str
    material_id: str
    subpart_id: str | None = None
    object_names: tuple[str, ...] = ()
    selector: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "part_id",
            _text(self.part_id, label="assignment part_id", identifier=True),
        )
        object.__setattr__(
            self,
            "material_id",
            _text(self.material_id, label="assignment material_id", identifier=True),
        )
        if self.subpart_id is not None:
            object.__setattr__(
                self,
                "subpart_id",
                _text(
                    self.subpart_id,
                    label="assignment subpart_id",
                    identifier=True,
                ),
            )
        if isinstance(self.object_names, (str, bytes)) or not isinstance(
            self.object_names, Sequence
        ):
            raise MaterialPlanError("assignment object_names must be an array")
        names = tuple(
            _text(name, label="assignment object name") for name in self.object_names
        )
        if len(names) != len(set(names)):
            raise MaterialPlanError("assignment object_names must be unique")
        object.__setattr__(self, "object_names", names)
        if self.selector is not None:
            object.__setattr__(
                self,
                "selector",
                _text(self.selector, label="assignment selector"),
            )
        if self.subpart_id is None and self.selector is not None:
            raise MaterialPlanError(
                "whole-part assignments cannot define a subpart selector"
            )
        if self.subpart_id is not None and not self.object_names and self.selector is None:
            raise MaterialPlanError(
                "subpart assignments require object_names or a selector"
            )

    @property
    def target_key(self) -> tuple[str, str | None]:
        return self.part_id, self.subpart_id

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "part_id": self.part_id,
            "material_id": self.material_id,
        }
        if self.subpart_id is not None:
            result["subpart_id"] = self.subpart_id
        if self.object_names:
            result["object_names"] = list(self.object_names)
        if self.selector is not None:
            result["selector"] = self.selector
        return result

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> MaterialAssignment:
        value = _mapping(document, label="material assignment")
        _unknown_keys(
            value,
            allowed=frozenset(
                {"part_id", "material_id", "subpart_id", "object_names", "selector"}
            ),
            label="material assignment",
        )
        missing = [key for key in ("part_id", "material_id") if key not in value]
        if missing:
            raise MaterialPlanError(
                "material assignment is missing required fields: " + ", ".join(missing)
            )
        object_names = value.get("object_names", ())
        _array(object_names, label="material assignment object_names")
        return cls(
            part_id=value["part_id"],
            material_id=value["material_id"],
            subpart_id=value.get("subpart_id"),
            object_names=tuple(object_names),
            selector=value.get("selector"),
        )


@dataclass(frozen=True, slots=True)
class MaterialPlan:
    """A compact material library plus deterministic part assignment rules."""

    materials: tuple[PBRMaterial, ...] = ()
    assignments: tuple[MaterialAssignment, ...] = ()
    schema_version: int = MATERIAL_PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != MATERIAL_PLAN_SCHEMA_VERSION
        ):
            raise MaterialPlanError(
                f"schema_version must be {MATERIAL_PLAN_SCHEMA_VERSION}"
            )
        if isinstance(self.materials, (str, bytes)) or not isinstance(
            self.materials, Sequence
        ):
            raise MaterialPlanError("materials must be an array")
        if isinstance(self.assignments, (str, bytes)) or not isinstance(
            self.assignments, Sequence
        ):
            raise MaterialPlanError("assignments must be an array")
        materials = tuple(self.materials)
        assignments = tuple(self.assignments)
        if any(not isinstance(item, PBRMaterial) for item in materials):
            raise MaterialPlanError("materials must contain PBRMaterial values")
        if any(not isinstance(item, MaterialAssignment) for item in assignments):
            raise MaterialPlanError(
                "assignments must contain MaterialAssignment values"
            )
        object.__setattr__(self, "materials", materials)
        object.__setattr__(self, "assignments", assignments)

        material_ids = [material.material_id for material in materials]
        if len(material_ids) != len(set(material_ids)):
            raise MaterialPlanError("material IDs must be unique")
        material_id_set = set(material_ids)
        targets: set[tuple[str, str | None]] = set()
        object_targets: dict[tuple[str, str], tuple[str, str | None]] = {}
        for assignment in assignments:
            if assignment.material_id not in material_id_set:
                raise MaterialPlanError(
                    f"assignment references unknown material {assignment.material_id!r}"
                )
            if assignment.target_key in targets:
                part_id, subpart_id = assignment.target_key
                target = part_id if subpart_id is None else f"{part_id}/{subpart_id}"
                raise MaterialPlanError(f"duplicate material assignment target {target!r}")
            targets.add(assignment.target_key)
            for object_name in assignment.object_names:
                key = assignment.part_id, object_name
                previous = object_targets.get(key)
                if (
                    previous is not None
                    and previous[1] is not None
                    and assignment.subpart_id is not None
                ):
                    raise MaterialPlanError(
                        f"object {object_name!r} is assigned by more than one subpart "
                        f"rule for part {assignment.part_id!r}"
                    )
                if previous is None or assignment.subpart_id is not None:
                    object_targets[key] = assignment.target_key

    @property
    def enabled(self) -> bool:
        return bool(self.materials or self.assignments)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "materials": [material.as_dict() for material in self.materials],
            "assignments": [assignment.as_dict() for assignment in self.assignments],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize with stable keys and a trailing newline when pretty printed."""

        result = json.dumps(
            self.as_dict(),
            ensure_ascii=False,
            indent=indent,
            separators=(",", ":") if indent is None else None,
            sort_keys=True,
        )
        return result + ("\n" if indent is not None else "")

    def validate_part_ids(self, part_ids: Iterable[str]) -> None:
        """Require every assignment to reference a known part graph node."""

        known = {
            _text(part_id, label="known part ID", identifier=True)
            for part_id in part_ids
        }
        unknown = sorted(
            {assignment.part_id for assignment in self.assignments} - known
        )
        if unknown:
            raise MaterialPlanError(
                "assignments reference unknown parts: " + ", ".join(unknown)
            )

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> MaterialPlan:
        value = _mapping(document, label="material plan")
        _unknown_keys(
            value,
            allowed=frozenset({"schema_version", "materials", "assignments"}),
            label="material plan",
        )
        materials_value = value.get("materials", ())
        assignments_value = value.get("assignments", ())
        materials = _array(materials_value, label="material plan materials")
        assignments = _array(assignments_value, label="material plan assignments")
        return cls(
            schema_version=value.get(
                "schema_version", MATERIAL_PLAN_SCHEMA_VERSION
            ),
            materials=tuple(PBRMaterial.from_dict(item) for item in materials),
            assignments=tuple(
                MaterialAssignment.from_dict(item) for item in assignments
            ),
        )

    @classmethod
    def from_json(cls, payload: str) -> MaterialPlan:
        if not isinstance(payload, str):
            raise MaterialPlanError("material plan JSON must be a string")
        try:
            document = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise MaterialPlanError(f"material plan is invalid JSON: {exc}") from exc
        return material_plan_from_document(document)


def material_plan_from_document(document: Any) -> MaterialPlan:
    """Parse a material plan while accepting legacy empty material values.

    Earlier reconstruction plans used ``"materials": []`` without a dedicated
    plan object.  ``None``, an empty array, and an empty object therefore all
    mean that the optional pass is disabled.  Non-empty legacy arrays are
    rejected because their assignment semantics are ambiguous.
    """

    if isinstance(document, MaterialPlan):
        return document
    if document is None:
        return MaterialPlan()
    if isinstance(document, list):
        if document:
            raise MaterialPlanError(
                "non-empty legacy material arrays must be migrated to a material plan object"
            )
        return MaterialPlan()
    if isinstance(document, Mapping):
        return MaterialPlan.from_dict(document)
    raise MaterialPlanError("material plan must be an object, empty array, or null")


def _optional_int(value: Any, *, label: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise MaterialPlanError(f"{label} must be a non-negative integer")
    return value


def _optional_bounds(
    value: Any, *, label: str
) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
    if value is None:
        return None
    bounds = _mapping(value, label=label)
    if "min" not in bounds or "max" not in bounds:
        raise MaterialPlanError(f"{label} must contain min and max")
    minimum = _vector(bounds["min"], length=3, label=f"{label}.min", unit_interval=False)
    maximum = _vector(bounds["max"], length=3, label=f"{label}.max", unit_interval=False)
    if any(low > high for low, high in zip(minimum, maximum)):
        raise MaterialPlanError(f"{label} minimum cannot exceed maximum")
    return minimum, maximum


def _canonical_geometry_value(value: Any, *, float_digits: int = 9) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_geometry_value(item, float_digits=float_digits)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [
            _canonical_geometry_value(item, float_digits=float_digits) for item in value
        ]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise MaterialPlanError("geometry report contains a non-finite number")
        return round(value, float_digits)
    raise MaterialPlanError(
        f"geometry report contains unsupported value {type(value).__name__}"
    )


@dataclass(frozen=True, slots=True)
class GeometryObjectSignature:
    name: str
    object_type: str
    vertices: int
    triangles: int
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.object_type,
            "vertices": self.vertices,
            "triangles": self.triangles,
            "bounds": {"min": list(self.bounds_min), "max": list(self.bounds_max)},
        }


@dataclass(frozen=True, slots=True)
class GeometrySnapshot:
    """Material-independent fingerprint of an evaluated Blender scene report."""

    object_count: int
    mesh_count: int
    vertex_count: int
    triangle_count: int
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    objects: tuple[GeometryObjectSignature, ...]
    geometry_digest: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "object_count": self.object_count,
            "mesh_count": self.mesh_count,
            "vertex_count": self.vertex_count,
            "triangle_count": self.triangle_count,
            "bounds": {"min": list(self.bounds_min), "max": list(self.bounds_max)},
            "objects": [item.as_dict() for item in self.objects],
            "geometry_digest": self.geometry_digest,
        }


_OBJECT_GEOMETRY_FIELDS = frozenset(
    {
        "name",
        "type",
        "vertices",
        "triangles",
        "bounds",
        "structure",
        "matrix_world",
        "transform",
        "geometry_digest",
        "mesh_digest",
    }
)
_TOP_LEVEL_GEOMETRY_FIELDS = frozenset(
    {
        "coordinate_system",
        "bounds",
        "geometry_object_count",
        "mesh_count",
        "objects",
        "cross_sections_x",
        "cross_sections_y",
        "cross_sections_z",
        "structure",
        "welded_components",
        "geometry_digest",
    }
)


def geometry_snapshot_from_report(report: Mapping[str, Any]) -> GeometrySnapshot:
    """Create a strict, deterministic fingerprint from ``geometry_report`` JSON.

    Material names and all render/artifact metadata are intentionally excluded.
    Object topology diagnostics, scene structure, cross-sections, and bounds are
    included when present, so a count-preserving geometry edit still fails the
    guard.
    """

    value = _mapping(report, label="geometry report")
    raw_objects = _array(value.get("objects"), label="geometry report objects")
    objects: list[GeometryObjectSignature] = []
    canonical_objects: list[dict[str, Any]] = []
    for index, raw_object in enumerate(raw_objects):
        item = _mapping(raw_object, label=f"geometry report object {index}")
        try:
            name = _text(item["name"], label=f"geometry report object {index} name")
            object_type = _text(
                item["type"], label=f"geometry report object {name!r} type"
            )
        except KeyError as exc:
            raise MaterialPlanError(
                f"geometry report object {index} is missing {exc.args[0]}"
            ) from exc
        vertices = _optional_int(
            item.get("vertices"), label=f"geometry report object {name!r} vertices"
        )
        triangles = _optional_int(
            item.get("triangles"), label=f"geometry report object {name!r} triangles"
        )
        if vertices is None or triangles is None:
            raise MaterialPlanError(
                f"geometry report object {name!r} requires vertices and triangles"
            )
        bounds = _optional_bounds(
            item.get("bounds"), label=f"geometry report object {name!r} bounds"
        )
        if bounds is None:
            raise MaterialPlanError(f"geometry report object {name!r} requires bounds")
        objects.append(
            GeometryObjectSignature(
                name=name,
                object_type=object_type,
                vertices=vertices,
                triangles=triangles,
                bounds_min=bounds[0],
                bounds_max=bounds[1],
            )
        )
        canonical_objects.append(
            {
                key: item[key]
                for key in sorted(_OBJECT_GEOMETRY_FIELDS & set(item))
            }
        )
    objects.sort(key=lambda item: item.name)
    canonical_objects.sort(key=lambda item: str(item.get("name", "")))
    names = [item.name for item in objects]
    if len(names) != len(set(names)):
        raise MaterialPlanError("geometry report object names must be unique")

    object_count = _optional_int(
        value.get("geometry_object_count"), label="geometry_object_count"
    )
    mesh_count = _optional_int(value.get("mesh_count"), label="mesh_count")
    if object_count is None or mesh_count is None:
        raise MaterialPlanError(
            "geometry report requires geometry_object_count and mesh_count"
        )
    if object_count != len(objects):
        raise MaterialPlanError(
            "geometry_object_count does not match the objects array"
        )
    scene_bounds = _optional_bounds(value.get("bounds"), label="geometry report bounds")
    if scene_bounds is None:
        raise MaterialPlanError("geometry report requires bounds")

    geometry_payload = {
        key: value[key]
        for key in sorted(_TOP_LEVEL_GEOMETRY_FIELDS & set(value))
        if key != "objects"
    }
    geometry_payload["objects"] = canonical_objects
    canonical = _canonical_geometry_value(geometry_payload)
    encoded = json.dumps(
        canonical, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return GeometrySnapshot(
        object_count=object_count,
        mesh_count=mesh_count,
        vertex_count=sum(item.vertices for item in objects),
        triangle_count=sum(item.triangles for item in objects),
        bounds_min=scene_bounds[0],
        bounds_max=scene_bounds[1],
        objects=tuple(objects),
        geometry_digest=hashlib.sha256(encoded).hexdigest(),
    )


@dataclass(frozen=True, slots=True)
class MaterialGuardViolation:
    field: str
    before: Any
    after: Any

    def as_dict(self) -> dict[str, Any]:
        return {"field": self.field, "before": self.before, "after": self.after}


@dataclass(frozen=True, slots=True)
class MaterialChangeGuardResult:
    passed: bool
    violations: tuple[MaterialGuardViolation, ...]
    before_digest: str
    after_digest: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "violations": [item.as_dict() for item in self.violations],
            "before_digest": self.before_digest,
            "after_digest": self.after_digest,
        }


def _material_geometry_fingerprint(
    report: Mapping[str, Any],
) -> tuple[str, str] | None:
    value = report.get("material_geometry_fingerprint")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise MaterialPlanError("material_geometry_fingerprint must be an object")
    algorithm = value.get("algorithm")
    digest = value.get("digest")
    if (
        value.get("schema_version") != 1
        or algorithm != "oriented-world-triangle-multiset-sha256-v1"
        or not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        or not isinstance(value.get("objects"), list)
    ):
        raise MaterialPlanError("material_geometry_fingerprint is malformed")
    return algorithm, digest


def compare_material_pass_geometry(
    before: GeometrySnapshot | Mapping[str, Any],
    after: GeometrySnapshot | Mapping[str, Any],
) -> MaterialChangeGuardResult:
    """Return all forbidden geometry changes made by a material-only pass."""

    before_snapshot = (
        before if isinstance(before, GeometrySnapshot) else geometry_snapshot_from_report(before)
    )
    after_snapshot = (
        after if isinstance(after, GeometrySnapshot) else geometry_snapshot_from_report(after)
    )
    before_fingerprint = (
        _material_geometry_fingerprint(before)
        if isinstance(before, Mapping)
        else None
    )
    after_fingerprint = (
        _material_geometry_fingerprint(after)
        if isinstance(after, Mapping)
        else None
    )
    violations: list[MaterialGuardViolation] = []

    def compare(field: str, first: Any, second: Any) -> None:
        if first != second:
            violations.append(MaterialGuardViolation(field, first, second))

    compare("object_count", before_snapshot.object_count, after_snapshot.object_count)
    compare("mesh_count", before_snapshot.mesh_count, after_snapshot.mesh_count)
    compare(
        "triangle_count", before_snapshot.triangle_count, after_snapshot.triangle_count
    )
    compare("bounds.min", before_snapshot.bounds_min, after_snapshot.bounds_min)
    compare("bounds.max", before_snapshot.bounds_max, after_snapshot.bounds_max)
    before_objects = {item.name: item for item in before_snapshot.objects}
    after_objects = {item.name: item for item in after_snapshot.objects}
    compare("object_names", tuple(sorted(before_objects)), tuple(sorted(after_objects)))
    for name in sorted(before_objects.keys() & after_objects.keys()):
        first = before_objects[name]
        second = after_objects[name]
        compare(f"objects.{name}.type", first.object_type, second.object_type)
        compare(f"objects.{name}.triangles", first.triangles, second.triangles)
        compare(f"objects.{name}.bounds.min", first.bounds_min, second.bounds_min)
        compare(f"objects.{name}.bounds.max", first.bounds_max, second.bounds_max)
    if before_fingerprint is None and after_fingerprint is None:
        # Compatibility for hand-authored/older reports.  New compiled reports
        # use the triangle fingerprint so material-boundary vertex duplication
        # does not look like a geometry edit.
        compare("vertex_count", before_snapshot.vertex_count, after_snapshot.vertex_count)
        for name in sorted(before_objects.keys() & after_objects.keys()):
            compare(
                f"objects.{name}.vertices",
                before_objects[name].vertices,
                after_objects[name].vertices,
            )
        compare(
            "geometry_digest",
            before_snapshot.geometry_digest,
            after_snapshot.geometry_digest,
        )
        before_digest = before_snapshot.geometry_digest
        after_digest = after_snapshot.geometry_digest
    else:
        compare(
            "material_geometry_fingerprint",
            before_fingerprint,
            after_fingerprint,
        )
        before_digest = (
            before_fingerprint[1]
            if before_fingerprint is not None
            else before_snapshot.geometry_digest
        )
        after_digest = (
            after_fingerprint[1]
            if after_fingerprint is not None
            else after_snapshot.geometry_digest
        )
    return MaterialChangeGuardResult(
        passed=not violations,
        violations=tuple(violations),
        before_digest=before_digest,
        after_digest=after_digest,
    )


def enforce_material_only_change(
    before: GeometrySnapshot | Mapping[str, Any],
    after: GeometrySnapshot | Mapping[str, Any],
) -> MaterialChangeGuardResult:
    """Raise when a material-only pass changes geometry; otherwise return its audit."""

    result = compare_material_pass_geometry(before, after)
    if not result.passed:
        raise MaterialGeometryChangeError(result)
    return result


def summarize_material_evidence(
    report: Mapping[str, Any] | None,
    *,
    max_materials: int = 12,
) -> dict[str, Any]:
    """Return bounded, deterministic PBR evidence from a GLB or scene report."""

    if type(max_materials) is not int or max_materials <= 0:
        raise MaterialPlanError("max_materials must be a positive integer")
    if report is None:
        report = {}
    value = _mapping(report, label="material evidence report")
    raw_materials = value.get("materials", ())
    materials = _array(raw_materials, label="material evidence materials")
    records: list[dict[str, Any]] = []
    for sequence_index, raw_material in enumerate(materials):
        if not isinstance(raw_material, Mapping):
            continue
        index = raw_material.get("index", sequence_index)
        if type(index) is not int or index < 0:
            index = sequence_index
        usage = raw_material.get("primitive_usage_count", 0)
        if type(usage) is not int or usage < 0:
            usage = 0
        record = {
            "index": index,
            "name": raw_material.get("name"),
            "base_color_rgba": raw_material.get("base_color_factor"),
            "base_color_texture": raw_material.get("base_color_texture"),
            "metallic": raw_material.get("metallic_factor"),
            "roughness": raw_material.get("roughness_factor"),
            "alpha_mode": raw_material.get("alpha_mode"),
            "primitive_usage_count": usage,
            "default_white_risk": bool(raw_material.get("default_white_risk", False)),
        }
        records.append(record)
    records.sort(
        key=lambda item: (-item["primitive_usage_count"], item["index"], str(item["name"]))
    )

    # Blender scene reports expose only material-slot names.  Retain these as
    # weaker evidence when the richer GLB material records are unavailable.
    slot_names: set[str] = set()
    raw_objects = value.get("objects", ())
    if isinstance(raw_objects, Sequence) and not isinstance(raw_objects, (str, bytes)):
        for raw_object in raw_objects:
            if not isinstance(raw_object, Mapping):
                continue
            slots = raw_object.get("materials", ())
            if not isinstance(slots, Sequence) or isinstance(slots, (str, bytes)):
                continue
            slot_names.update(name for name in slots if isinstance(name, str) and name)

    diagnostics_value = value.get("material_diagnostics")
    diagnostics = diagnostics_value if isinstance(diagnostics_value, Mapping) else {}
    material_count = len(records) if records else len(slot_names)
    used_material_count = diagnostics.get("used_material_count")
    if type(used_material_count) is not int or used_material_count < 0:
        used_material_count = sum(item["primitive_usage_count"] > 0 for item in records)
        if not records:
            used_material_count = len(slot_names)
    return {
        "material_count": material_count,
        "used_material_count": used_material_count,
        "default_white_risk": bool(diagnostics.get("default_white_risk", False)),
        "materials": records[:max_materials],
        "materials_truncated": max(0, len(records) - max_materials),
        "material_slot_names": sorted(slot_names)[:max_materials],
        "material_slot_names_truncated": max(0, len(slot_names) - max_materials),
    }


def build_material_pass_context(
    plan: MaterialPlan | Mapping[str, Any] | list[Any] | None,
    *,
    part_ids: Iterable[str] = (),
    reference_report: Mapping[str, Any] | None = None,
    pre_material_geometry_report: Mapping[str, Any] | None = None,
    max_evidence_materials: int = 12,
) -> dict[str, Any]:
    """Build compact JSON-ready inputs for a dedicated material agent pass."""

    parsed = material_plan_from_document(plan)
    normalized_part_ids = tuple(
        sorted(
            {
                _text(part_id, label="material context part ID", identifier=True)
                for part_id in part_ids
            }
        )
    )
    if normalized_part_ids:
        parsed.validate_part_ids(normalized_part_ids)
    snapshot = (
        geometry_snapshot_from_report(pre_material_geometry_report).as_dict()
        if pre_material_geometry_report is not None
        else None
    )
    return {
        "stage": "dedicated-pbr-material-pass",
        "enabled": parsed.enabled,
        "invariants": [
            "Change only material definitions, material slots, and face assignments.",
            "Do not create, delete, rename, transform, or replace geometry objects.",
            "Do not change vertices, triangles, modifiers, shape keys, or armatures.",
            "Subpart assignments override the whole-part material for that part.",
        ],
        "part_ids": list(normalized_part_ids),
        "material_plan": parsed.as_dict(),
        "reference_material_evidence": summarize_material_evidence(
            reference_report, max_materials=max_evidence_materials
        ),
        "pre_material_geometry": snapshot,
    }


__all__ = [
    "ALPHA_MODES",
    "MATERIAL_PLAN_SCHEMA_VERSION",
    "GeometryObjectSignature",
    "GeometrySnapshot",
    "MaterialAssignment",
    "MaterialChangeGuardResult",
    "MaterialGeometryChangeError",
    "MaterialGuardViolation",
    "MaterialPlan",
    "MaterialPlanError",
    "PBRMaterial",
    "build_material_pass_context",
    "compare_material_pass_geometry",
    "enforce_material_only_change",
    "geometry_snapshot_from_report",
    "material_plan_from_document",
    "summarize_material_evidence",
]
