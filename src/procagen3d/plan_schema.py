"""Authoritative JSON Schema and validation helpers for reconstruction plans."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

from .granularity import DEFAULT_GRANULARITY, GRANULARITY_LEVELS
from .quality import (
    DETAIL_RICHNESS_LEVELS,
    MATERIAL_FIDELITY_LEVELS,
    STRUCTURAL_COHERENCE_LEVELS,
    SURFACE_FIDELITY_LEVELS,
    resolve_quality_profile,
)
from .reconstruction import DEFAULT_RECONSTRUCTION_MODE, RECONSTRUCTION_MODES


_NON_EMPTY_STRING: dict[str, Any] = {
    "type": "string",
    "minLength": 1,
    "pattern": r"\S",
}

_CHARACTER_LIST_FIELDS = (
    "facial_landmarks",
    "hair_or_headwear",
    "clothing_layers",
    "held_props",
    "left_right_asymmetry",
    "inferred_features",
)

_VECTOR3: dict[str, Any] = {
    "type": "array",
    "minItems": 3,
    "maxItems": 3,
    "items": {"type": "number"},
}

_BOUNDS3: dict[str, Any] = {
    "type": "object",
    "required": ["min", "max"],
    "properties": {
        "min": _VECTOR3,
        "max": _VECTOR3,
    },
    "additionalProperties": False,
}

_ATTACHMENT_TYPES = (
    "root",
    "fused",
    "surface-contact",
    "embedded",
    "articulated",
    "intentional-gap",
)

_ATTACHMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "parent_id",
        "type",
        "contact_region",
        "max_gap",
        "max_penetration",
        "min_contact_area",
    ],
    "properties": {
        "parent_id": _NON_EMPTY_STRING,
        "type": {"type": "string", "enum": list(_ATTACHMENT_TYPES)},
        "contact_region": _BOUNDS3,
        "max_gap": {"type": "number", "minimum": 0},
        "max_penetration": {"type": "number", "minimum": 0},
        "min_contact_area": {"type": "number", "minimum": 0},
    },
    "additionalProperties": False,
}

_PART_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "id",
        "name",
        "shape_family",
        "approximate_bounds",
        "visual_role",
        "object_names",
        "attachment",
    ],
    "properties": {
        "id": _NON_EMPTY_STRING,
        "name": _NON_EMPTY_STRING,
        "shape_family": _NON_EMPTY_STRING,
        "approximate_bounds": _BOUNDS3,
        "visual_role": _NON_EMPTY_STRING,
        "object_names": {
            "type": "array",
            "minItems": 1,
            "items": _NON_EMPTY_STRING,
        },
        "attachment": _ATTACHMENT_SCHEMA,
    },
    # Agents may retain measurements or construction notes alongside the
    # executable fields above.
    "additionalProperties": True,
}


PLAN_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://procagen3d.local/schemas/plan.schema.json",
    "title": "ProcAgen3D reconstruction plan",
    "description": (
        "The semantic and geometric plan implemented by src/program.py. "
        "Unknown metadata is permitted so agents can record useful measurements."
    ),
    "type": "object",
    "required": [
        "subject",
        "subject_kind",
        "coordinate_frame",
        "dimensions",
        "parts",
        "materials",
        "construction_strategy",
        "identity_features",
        "limitations",
    ],
    "properties": {
        "subject": {
            **_NON_EMPTY_STRING,
            "description": "A concise identification of the reconstructed subject.",
        },
        "subject_kind": {
            "type": "string",
            "enum": ["object", "character", "hybrid", "scene"],
        },
        "reconstruction_mode": {
            "type": "string",
            "enum": list(RECONSTRUCTION_MODES),
            "default": DEFAULT_RECONSTRUCTION_MODE,
            "description": (
                "procedural synthesizes compact editable geometry; glb-ref "
                "may derive geometry from the supplied reference under the host contract."
            ),
        },
        "granularity": {
            "type": "string",
            "enum": list(GRANULARITY_LEVELS),
            "default": DEFAULT_GRANULARITY,
            "description": (
                "Geometric authoring and validation detail. Fine and surface levels "
                "replace primitive-only fitting with surface-conforming procedural meshes "
                "and enable bidirectional 3D surface-distance gates."
            ),
        },
        "quality_profile": {
            "type": "object",
            "required": [
                "surface_fidelity",
                "detail_richness",
                "material_fidelity",
                "structural_coherence",
            ],
            "properties": {
                "surface_fidelity": {
                    "type": "string",
                    "enum": list(SURFACE_FIDELITY_LEVELS),
                },
                "detail_richness": {
                    "type": "string",
                    "enum": list(DETAIL_RICHNESS_LEVELS),
                },
                "material_fidelity": {
                    "type": "string",
                    "enum": list(MATERIAL_FIDELITY_LEVELS),
                },
                "structural_coherence": {
                    "type": "string",
                    "enum": list(STRUCTURAL_COHERENCE_LEVELS),
                },
            },
            "additionalProperties": False,
            "description": (
                "Independent acceptance requirements. Missing legacy values are "
                "derived from granularity, while newly authored plans must state them."
            ),
        },
        "coordinate_frame": {
            "type": "object",
            "description": (
                "Axis, handedness, origin, ground-plane, and facing conventions. "
                "The build contract uses X width, Y depth, Z up, and ground at Z=0."
            ),
        },
        "dimensions": {
            "type": "array",
            "description": "Overall [width, depth, height] in normalized Blender units.",
            "minItems": 3,
            "maxItems": 3,
            "items": {"type": "number", "exclusiveMinimum": 0},
        },
        "parts": {
            "type": "array",
            "minItems": 1,
            "description": (
                "Semantic parts. Each object should describe its name, shape family, "
                "approximate bounds, parent or attachment, and visual role."
            ),
            "items": _PART_SCHEMA,
        },
        "materials": {"type": "array"},
        "construction_strategy": {
            **_NON_EMPTY_STRING,
            "description": "How program.py will construct and organize the asset.",
        },
        "identity_features": {"type": "array"},
        "limitations": {"type": "array"},
        "character_analysis": {
            "type": "object",
            "required": ["pose", "proportions", *_CHARACTER_LIST_FIELDS],
            "properties": {
                "pose": {**_NON_EMPTY_STRING},
                "proportions": {"type": "object"},
                **{field: {"type": "array"} for field in _CHARACTER_LIST_FIELDS},
            },
        },
    },
    "allOf": [
        {
            "if": {
                "type": "object",
                "required": ["subject_kind"],
                "properties": {
                    "subject_kind": {"enum": ["character", "hybrid"]},
                },
            },
            "then": {"required": ["character_analysis"]},
        }
    ],
}


def plan_schema_text() -> str:
    """Return the authoritative schema as deterministic, prompt-ready JSON."""

    return json.dumps(PLAN_SCHEMA, indent=2, sort_keys=True, ensure_ascii=False)


@dataclass(frozen=True)
class PlanSchemaViolation:
    """One validation failure with a stable JSON-path-like location."""

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


class PlanSchemaError(ValueError):
    """Raised after collecting every violation in a plan document."""

    def __init__(self, violations: Sequence[PlanSchemaViolation]):
        self.violations = tuple(violations)
        count = len(self.violations)
        details = "\n".join(f"- {violation}" for violation in self.violations)
        super().__init__(f"plan violates the JSON Schema ({count} error{'s' if count != 1 else ''}):\n{details}")


def _counted(value: int, singular: str) -> str:
    return f"{value} {singular if value == 1 else singular + 's'}"


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "number":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    raise ValueError(f"unsupported JSON Schema type {expected!r}")


def _type_description(expected: str | Sequence[str]) -> str:
    if isinstance(expected, str):
        return expected
    return " or ".join(expected)


def _iter_violations(
    value: Any,
    schema: Mapping[str, Any],
    *,
    path: tuple[str | int, ...] = (),
) -> Iterator[PlanSchemaViolation]:
    expected_type = schema.get("type")
    if expected_type is not None:
        expected_types = [expected_type] if isinstance(expected_type, str) else expected_type
        if not any(_matches_type(value, item) for item in expected_types):
            yield PlanSchemaViolation(
                path,
                "type",
                f"must be {_type_description(expected_type)}",
            )
            if "enum" in schema and value not in schema["enum"]:
                yield PlanSchemaViolation(
                    path,
                    "enum",
                    "must be one of " + ", ".join(repr(item) for item in schema["enum"]),
                )
            return

    if "enum" in schema and value not in schema["enum"]:
        yield PlanSchemaViolation(
            path,
            "enum",
            "must be one of " + ", ".join(repr(item) for item in schema["enum"]),
        )

    if isinstance(value, Mapping):
        required = schema.get("required", ())
        for name in required:
            if name not in value:
                child_type = schema.get("properties", {}).get(name, {}).get("type")
                if not path and child_type is None:
                    child_type = PLAN_SCHEMA["properties"].get(name, {}).get("type")
                message = (
                    f"{name} object is required"
                    if child_type == "object"
                    else f"required property {name!r} is missing"
                )
                yield PlanSchemaViolation(
                    (*path, name),
                    "required",
                    message,
                )

        properties = schema.get("properties", {})
        for name, child_schema in properties.items():
            if name in value:
                yield from _iter_violations(value[name], child_schema, path=(*path, name))

        additional = schema.get("additionalProperties", True)
        if additional is False:
            for name in value.keys() - properties.keys():
                yield PlanSchemaViolation(
                    (*path, name),
                    "additionalProperties",
                    "is not an allowed property",
                )
        elif isinstance(additional, Mapping):
            for name in value.keys() - properties.keys():
                yield from _iter_violations(value[name], additional, path=(*path, name))

        minimum_properties = schema.get("minProperties")
        if minimum_properties is not None and len(value) < minimum_properties:
            yield PlanSchemaViolation(
                path,
                "minProperties",
                f"must contain at least {_counted(minimum_properties, 'property')}",
            )

    if isinstance(value, list):
        minimum_items = schema.get("minItems")
        maximum_items = schema.get("maxItems")
        if minimum_items is not None and len(value) < minimum_items:
            yield PlanSchemaViolation(
                path,
                "minItems",
                f"must contain at least {_counted(minimum_items, 'item')}",
            )
        if maximum_items is not None and len(value) > maximum_items:
            yield PlanSchemaViolation(
                path,
                "maxItems",
                f"must contain at most {_counted(maximum_items, 'item')}",
            )
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                yield from _iter_violations(item, item_schema, path=(*path, index))

    if isinstance(value, str):
        minimum_length = schema.get("minLength")
        if minimum_length is not None and len(value) < minimum_length:
            yield PlanSchemaViolation(
                path,
                "minLength",
                f"must contain at least {_counted(minimum_length, 'character')}",
            )
        pattern = schema.get("pattern")
        if pattern is not None and re.search(pattern, value) is None:
            yield PlanSchemaViolation(
                path,
                "pattern",
                "must contain at least one non-whitespace character",
            )

    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        minimum = schema.get("minimum")
        if minimum is not None and value < minimum:
            yield PlanSchemaViolation(
                path,
                "minimum",
                f"must be at least {minimum}",
            )
        maximum = schema.get("maximum")
        if maximum is not None and value > maximum:
            yield PlanSchemaViolation(
                path,
                "maximum",
                f"must be at most {maximum}",
            )
        exclusive_minimum = schema.get("exclusiveMinimum")
        if exclusive_minimum is not None and value <= exclusive_minimum:
            yield PlanSchemaViolation(
                path,
                "exclusiveMinimum",
                f"must be greater than {exclusive_minimum}",
            )

    for condition in schema.get("allOf", ()):
        if_schema = condition.get("if")
        then_schema = condition.get("then")
        else_schema = condition.get("else")
        if if_schema is None:
            yield from _iter_violations(value, condition, path=path)
            continue
        condition_matches = not tuple(_iter_violations(value, if_schema, path=path))
        selected = then_schema if condition_matches else else_schema
        if selected is not None:
            yield from _iter_violations(value, selected, path=path)


def plan_schema_violations(value: Any) -> tuple[PlanSchemaViolation, ...]:
    """Return every schema violation in deterministic path/keyword order."""

    violations = tuple(_iter_violations(value, PLAN_SCHEMA))
    return tuple(
        sorted(
            violations,
            key=lambda item: (
                tuple(str(component) for component in item.path),
                item.keyword,
                item.message,
            ),
        )
    )


def _part_id(value: str, *, fallback: str) -> str:
    identifier = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return identifier or fallback


def _number_vector(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    result: list[float] = []
    for item in value:
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(item)
        ):
            return None
        result.append(float(item))
    return result


def _normalized_bounds(value: Any, *, dimensions: Any) -> dict[str, list[float]]:
    minimum: list[float] | None = None
    maximum: list[float] | None = None
    if isinstance(value, Mapping):
        minimum = _number_vector(value.get("min"))
        maximum = _number_vector(value.get("max"))
    elif isinstance(value, (list, tuple)):
        if len(value) == 2:
            minimum = _number_vector(value[0])
            maximum = _number_vector(value[1])
        elif len(value) == 6:
            minimum = _number_vector(value[:3])
            maximum = _number_vector(value[3:])
    if minimum is not None and maximum is not None:
        return {"min": minimum, "max": maximum}

    size = _number_vector(dimensions) or [1.0, 1.0, 1.0]
    return {
        "min": [-size[0] * 0.5, -size[1] * 0.5, 0.0],
        "max": [size[0] * 0.5, size[1] * 0.5, size[2]],
    }


def _contact_region(
    child: Mapping[str, list[float]],
    parent: Mapping[str, list[float]],
) -> dict[str, list[float]]:
    low = [max(child["min"][axis], parent["min"][axis]) for axis in range(3)]
    high = [min(child["max"][axis], parent["max"][axis]) for axis in range(3)]
    for axis in range(3):
        if low[axis] > high[axis]:
            midpoint = (low[axis] + high[axis]) * 0.5
            low[axis] = midpoint
            high[axis] = midpoint
    return {"min": low, "max": high}


def _normalize_parts(value: Any, *, dimensions: Any) -> Any:
    if not isinstance(value, list):
        return value
    normalized: list[Any] = []
    used_ids: set[str] = set()
    root_id = "asset_root"
    root_bounds: dict[str, list[float]] | None = None
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            normalized.append(raw)
            continue
        part = dict(raw)
        name_value = part.get("name", part.get("semantic_name", f"part_{index + 1}"))
        name = name_value if isinstance(name_value, str) and name_value.strip() else f"part_{index + 1}"
        if "id" in part:
            # Typed plans use IDs as foreign keys from attachments and generated
            # Blender-object metadata. Preserve an explicitly supplied ID exactly;
            # schema/semantic validation must reject bad or duplicate IDs rather
            # than silently changing the graph it describes.
            identifier = part["id"]
        else:
            identifier = _part_id(
                name if isinstance(name, str) else f"part_{index + 1}",
                fallback=f"part_{index + 1}",
            )
            base_identifier = identifier
            suffix = 2
            while identifier in used_ids:
                identifier = f"{base_identifier}_{suffix}"
                suffix += 1
        if isinstance(identifier, str):
            used_ids.add(identifier)
        if index == 0:
            root_id = identifier if isinstance(identifier, str) else "asset_root"

        raw_bounds = part.get("approximate_bounds", part.get("approx_bounds"))
        bounds = _normalized_bounds(raw_bounds, dimensions=dimensions)
        if index == 0:
            root_bounds = bounds
        assert root_bounds is not None

        object_names = part.get("object_names")
        if not (
            isinstance(object_names, list)
            and object_names
            and all(isinstance(item, str) and item.strip() for item in object_names)
        ):
            object_names = [name]

        attachment = part.get("attachment")
        if not isinstance(attachment, Mapping):
            attachment = {}
        attachment = dict(attachment)
        if index == 0:
            attachment.setdefault("parent_id", "__root__")
            attachment.setdefault("type", "root")
            region = bounds
        else:
            attachment.setdefault("parent_id", root_id)
            attachment.setdefault("type", "surface-contact")
            region = _contact_region(bounds, root_bounds)
        attachment.setdefault("contact_region", region)
        attachment.setdefault("max_gap", 0.02)
        attachment.setdefault("max_penetration", 0.02)
        attachment.setdefault("min_contact_area", 0.0)

        part.update(
            id=identifier,
            name=name,
            shape_family=(
                part.get("shape_family")
                if isinstance(part.get("shape_family"), str)
                and str(part.get("shape_family")).strip()
                else "unspecified"
            ),
            approximate_bounds=bounds,
            visual_role=(
                part.get("visual_role")
                if isinstance(part.get("visual_role"), str)
                and str(part.get("visual_role")).strip()
                else "unspecified"
            ),
            object_names=list(object_names),
            attachment=attachment,
        )
        normalized.append(part)
    return normalized


def _semantic_plan_violations(value: Mapping[str, Any]) -> tuple[PlanSchemaViolation, ...]:
    violations: list[PlanSchemaViolation] = []
    parts = value.get("parts")
    if not isinstance(parts, list) or not all(isinstance(item, Mapping) for item in parts):
        return ()
    ids = [
        item.get("id") if isinstance(item.get("id"), str) else None
        for item in parts
    ]
    known = {identifier for identifier in ids if identifier is not None}
    parents: dict[str, str] = {}
    root_indexes: list[int] = []
    for index, part in enumerate(parts):
        identifier = ids[index]
        if identifier is not None and ids.count(identifier) > 1:
            violations.append(
                PlanSchemaViolation(
                    ("parts", index, "id"),
                    "uniquePartId",
                    f"part id {identifier!r} must be unique",
                )
            )
        bounds = part.get("approximate_bounds")
        if isinstance(bounds, Mapping):
            minimum = bounds.get("min")
            maximum = bounds.get("max")
            if isinstance(minimum, list) and isinstance(maximum, list):
                for axis, (low, high) in enumerate(zip(minimum, maximum)):
                    if (
                        isinstance(low, (int, float))
                        and isinstance(high, (int, float))
                        and low > high
                    ):
                        violations.append(
                            PlanSchemaViolation(
                                ("parts", index, "approximate_bounds", "min", axis),
                                "orderedBounds",
                                "must not exceed the matching maximum",
                            )
                        )
        attachment = part.get("attachment")
        if not isinstance(attachment, Mapping):
            continue
        parent_id = attachment.get("parent_id")
        kind = attachment.get("type")
        if kind == "root":
            root_indexes.append(index)
            if parent_id != "__root__":
                violations.append(
                    PlanSchemaViolation(
                        ("parts", index, "attachment", "parent_id"),
                        "rootParent",
                        "a root attachment must use '__root__'",
                    )
                )
        elif kind in _ATTACHMENT_TYPES and isinstance(parent_id, str):
            if parent_id == identifier:
                violations.append(
                    PlanSchemaViolation(
                        ("parts", index, "attachment", "parent_id"),
                        "selfParent",
                        "a part cannot attach to itself",
                    )
                )
            elif parent_id not in known:
                violations.append(
                    PlanSchemaViolation(
                        ("parts", index, "attachment", "parent_id"),
                        "knownParent",
                        "must reference another declared part id",
                    )
                )
            elif identifier is not None:
                parents[identifier] = parent_id

        contact_region = attachment.get("contact_region")
        if isinstance(contact_region, Mapping):
            minimum = contact_region.get("min")
            maximum = contact_region.get("max")
            if isinstance(minimum, list) and isinstance(maximum, list):
                for axis, (low, high) in enumerate(zip(minimum, maximum)):
                    if (
                        isinstance(low, (int, float))
                        and isinstance(high, (int, float))
                        and low > high
                    ):
                        violations.append(
                            PlanSchemaViolation(
                                (
                                    "parts",
                                    index,
                                    "attachment",
                                    "contact_region",
                                    "min",
                                    axis,
                                ),
                                "orderedBounds",
                                "must not exceed the matching maximum",
                            )
                        )

    if len(root_indexes) != 1:
        violations.append(
            PlanSchemaViolation(
                ("parts",),
                "singleRoot",
                "attachment graph must contain exactly one root part",
            )
        )

    cyclic_ids: set[str] = set()
    for start in parents:
        path: list[str] = []
        positions: dict[str, int] = {}
        current = start
        while current in parents and current not in positions:
            positions[current] = len(path)
            path.append(current)
            current = parents[current]
        if current in positions:
            cyclic_ids.update(path[positions[current] :])
    for index, identifier in enumerate(ids):
        if identifier in cyclic_ids:
            violations.append(
                PlanSchemaViolation(
                    ("parts", index, "attachment", "parent_id"),
                    "acyclicAttachment",
                    "attachment parent references must not form a cycle",
                )
            )
    return tuple(violations)


def validate_plan_document(value: Any) -> dict[str, Any]:
    """Validate and return a normalized, executable plan.

    Old plans remain readable: mode, granularity, independent quality axes,
    part identifiers, bounds, and typed attachment constraints receive
    deterministic compatibility defaults. Newly authored plans should include
    every field explicitly.
    """

    if not isinstance(value, Mapping):
        normalized: Any = value
    else:
        normalized = dict(value)
        normalized.setdefault("reconstruction_mode", DEFAULT_RECONSTRUCTION_MODE)
        normalized.setdefault("granularity", DEFAULT_GRANULARITY)
        granularity = normalized.get("granularity")
        preset_granularity = (
            granularity if granularity in GRANULARITY_LEVELS else DEFAULT_GRANULARITY
        )
        profile = resolve_quality_profile(preset_granularity).as_dict()
        supplied_profile = normalized.get("quality_profile")
        if isinstance(supplied_profile, Mapping):
            profile.update(supplied_profile)
        elif supplied_profile is not None:
            profile = supplied_profile
        normalized["quality_profile"] = profile
        normalized["parts"] = _normalize_parts(
            normalized.get("parts"), dimensions=normalized.get("dimensions")
        )

    violations = list(plan_schema_violations(normalized))
    if isinstance(normalized, Mapping):
        violations.extend(_semantic_plan_violations(normalized))
    if violations:
        raise PlanSchemaError(
            sorted(
                violations,
                key=lambda item: (
                    tuple(str(component) for component in item.path),
                    item.keyword,
                    item.message,
                ),
            )
        )
    assert isinstance(normalized, dict)
    return normalized
