"""Authoritative JSON Schema and validation helpers for reconstruction plans."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

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
            "items": {"type": "object"},
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


def validate_plan_document(value: Any) -> dict[str, Any]:
    """Validate and return a shallow normalized plan.

    ``reconstruction_mode`` was added after the original plan contract. Old plans
    remain valid and normalize to ``procedural``; newly authored plans should
    include the field explicitly.
    """

    violations = plan_schema_violations(value)
    if violations:
        raise PlanSchemaError(violations)
    normalized = dict(value)
    normalized.setdefault("reconstruction_mode", DEFAULT_RECONSTRUCTION_MODE)
    return normalized
