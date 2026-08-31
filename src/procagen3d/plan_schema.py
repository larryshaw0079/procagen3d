"""Authoritative JSON Schema and validation helpers for reconstruction plans."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

from .assembly import (
    ASSEMBLY_VERSION,
    CONNECTOR_INTERFACES,
    CONNECTOR_ROLES,
    FIT_TYPES,
    MATE_TYPES,
    normalize_assembly,
    validate_assembly,
)
from .granularity import DEFAULT_GRANULARITY, GRANULARITY_LEVELS
from .materials import MaterialPlanError, material_plan_from_document
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

_CONNECTOR_FRAME_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["origin", "x_axis", "y_axis", "z_axis"],
    "properties": {
        "origin": _VECTOR3,
        "x_axis": _VECTOR3,
        "y_axis": _VECTOR3,
        "z_axis": _VECTOR3,
    },
    "additionalProperties": False,
    "description": (
        "A right-handed orthonormal connector frame in owning-part local coordinates."
    ),
}

_NOMINAL_DIMENSIONS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": {"type": "number", "exclusiveMinimum": 0},
}

_CONNECTOR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "id",
        "part_id",
        "interface",
        "role",
        "frame",
        "nominal_dimensions",
    ],
    "properties": {
        "id": _NON_EMPTY_STRING,
        "part_id": _NON_EMPTY_STRING,
        "interface": {"type": "string", "enum": list(CONNECTOR_INTERFACES)},
        "role": {"type": "string", "enum": list(CONNECTOR_ROLES)},
        "frame": _CONNECTOR_FRAME_SCHEMA,
        "nominal_dimensions": _NOMINAL_DIMENSIONS_SCHEMA,
    },
    "additionalProperties": False,
}

_JOINT_LIMIT_VALUE_SCHEMA: dict[str, Any] = {
    "type": ["number", "array"],
    "minItems": 3,
    "maxItems": 3,
    "items": {"type": "number"},
}

_JOINT_LIMITS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["lower", "upper"],
    "properties": {
        "lower": _JOINT_LIMIT_VALUE_SCHEMA,
        "upper": _JOINT_LIMIT_VALUE_SCHEMA,
    },
    "additionalProperties": False,
}

_SCALAR_JOINT_LIMITS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["lower", "upper"],
    "properties": {
        "lower": {"type": "number"},
        "upper": {"type": "number"},
    },
    "additionalProperties": False,
}

_SPHERICAL_JOINT_LIMITS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["lower", "upper"],
    "properties": {
        "lower": _VECTOR3,
        "upper": _VECTOR3,
    },
    "additionalProperties": False,
}

_MATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "id",
        "type",
        "parent_connector_id",
        "child_connector_id",
        "fit",
        "clearance",
        "fit_offset",
        "nominal_dimensions",
    ],
    "properties": {
        "id": _NON_EMPTY_STRING,
        "type": {
            "type": "string",
            "enum": list(MATE_TYPES),
            "description": (
                "Rigid mates omit rest and limits. Revolute and prismatic mates use a "
                "scalar rest value and optional scalar limits. Spherical mates use a "
                "three-angle rest vector and optional three-vector limits."
            ),
        },
        "parent_connector_id": _NON_EMPTY_STRING,
        "child_connector_id": _NON_EMPTY_STRING,
        "fit": {"type": "string", "enum": list(FIT_TYPES)},
        "clearance": {"type": "number", "minimum": 0},
        "fit_offset": _VECTOR3,
        "nominal_dimensions": _NOMINAL_DIMENSIONS_SCHEMA,
        "rest": {
            "type": ["number", "array"],
            "minItems": 3,
            "maxItems": 3,
            "items": {"type": "number"},
            "description": (
                "Joint rest state. Omit for rigid mates; use a scalar for revolute or "
                "prismatic mates and a three-angle vector for spherical mates."
            ),
        },
        "limits": {
            **_JOINT_LIMITS_SCHEMA,
            "description": (
                "Optional lower and upper joint limits matching the rest-state shape. "
                "Rigid mates must omit this property."
            ),
        },
    },
    "additionalProperties": False,
    "allOf": [
        {
            "if": {
                "required": ["type"],
                "properties": {"type": {"enum": ["rigid"]}},
            },
            "then": {
                "properties": {
                    "rest": {
                        "not": {},
                        "description": "Rigid mates must omit rest entirely.",
                    },
                    "limits": {
                        "not": {},
                        "description": "Rigid mates must omit limits entirely.",
                    },
                }
            },
        },
        {
            "if": {
                "required": ["type"],
                "properties": {
                    "type": {"enum": ["revolute", "prismatic"]},
                },
            },
            "then": {
                "required": ["rest"],
                "properties": {
                    "rest": {"type": "number"},
                    "limits": _SCALAR_JOINT_LIMITS_SCHEMA,
                },
            },
        },
        {
            "if": {
                "required": ["type"],
                "properties": {"type": {"enum": ["spherical"]}},
            },
            "then": {
                "required": ["rest"],
                "properties": {
                    "rest": _VECTOR3,
                    "limits": _SPHERICAL_JOINT_LIMITS_SCHEMA,
                },
            },
        },
    ],
}

_ASSEMBLY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["version", "part_order", "connectors", "mates"],
    "properties": {
        "version": {"type": "integer", "enum": [ASSEMBLY_VERSION]},
        "placement": {
            "type": "string",
            "enum": ["host-solved", "authored"],
            "description": (
                "host-solved keeps each part in local coordinates and applies connector "
                "transforms in the trusted Blender build stage"
            ),
        },
        "part_order": {
            "type": "array",
            "minItems": 1,
            "items": _NON_EMPTY_STRING,
        },
        "connectors": {"type": "array", "items": _CONNECTOR_SCHEMA},
        "mates": {"type": "array", "items": _MATE_SCHEMA},
    },
    "additionalProperties": False,
    "description": (
        "Ordered part graph and host-solved connector mates. The parts array remains "
        "the source of semantic and acceptance attachment data."
    ),
}

_COLOR4: dict[str, Any] = {
    "type": "array",
    "minItems": 4,
    "maxItems": 4,
    "items": {"type": "number", "minimum": 0, "maximum": 1},
}

_COLOR3_UNIT: dict[str, Any] = {
    "type": "array",
    "minItems": 3,
    "maxItems": 3,
    "items": {"type": "number", "minimum": 0, "maximum": 1},
}

_PBR_MATERIAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["id", "base_color_rgba", "metallic", "roughness"],
    "properties": {
        "id": _NON_EMPTY_STRING,
        "name": _NON_EMPTY_STRING,
        "base_color_rgba": _COLOR4,
        "metallic": {"type": "number", "minimum": 0, "maximum": 1},
        "roughness": {"type": "number", "minimum": 0, "maximum": 1},
        "emissive_rgb": _COLOR3_UNIT,
        "alpha_mode": {"type": "string", "enum": ["OPAQUE", "MASK", "BLEND"]},
        "alpha_cutoff": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "additionalProperties": False,
}

_MATERIAL_ASSIGNMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["part_id", "material_id"],
    "properties": {
        "part_id": _NON_EMPTY_STRING,
        "material_id": _NON_EMPTY_STRING,
        "subpart_id": _NON_EMPTY_STRING,
        "object_names": {"type": "array", "items": _NON_EMPTY_STRING},
        "selector": _NON_EMPTY_STRING,
    },
    "additionalProperties": False,
}

_MATERIAL_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["schema_version", "materials", "assignments"],
    "properties": {
        "schema_version": {"type": "integer", "enum": [1]},
        "materials": {"type": "array", "items": _PBR_MATERIAL_SCHEMA},
        "assignments": {"type": "array", "items": _MATERIAL_ASSIGNMENT_SCHEMA},
    },
    "additionalProperties": False,
    "description": "Dedicated glTF-safe PBR library and semantic assignment rules.",
}

_ARTICULATION_LIMIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "lower": {"type": "number"},
        "upper": {"type": "number"},
        "effort": {"type": "number", "exclusiveMinimum": 0},
        "velocity": {"type": "number", "exclusiveMinimum": 0},
    },
    "additionalProperties": False,
}

_ARTICULATION_JOINT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["name", "parent", "child", "type", "origin"],
    "properties": {
        "name": _NON_EMPTY_STRING,
        "parent": _NON_EMPTY_STRING,
        "child": _NON_EMPTY_STRING,
        "type": {
            "type": "string",
            "enum": ["fixed", "rigid", "revolute", "continuous", "prismatic", "spherical"],
        },
        "origin": {
            "type": "object",
            "required": ["xyz", "rpy"],
            "properties": {"xyz": _VECTOR3, "rpy": _VECTOR3},
            "additionalProperties": False,
        },
        "axis": _VECTOR3,
        "limit": _ARTICULATION_LIMIT_SCHEMA,
    },
    "additionalProperties": False,
}

_ARTICULATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["enabled", "mechanical"],
    "properties": {
        "enabled": {"type": "boolean"},
        "mechanical": {"type": "boolean"},
        "robot_name": _NON_EMPTY_STRING,
        "joints": {"type": "array", "items": _ARTICULATION_JOINT_SCHEMA},
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
        "parent_id": {
            **_NON_EMPTY_STRING,
            "description": (
                "Declared parent part ID. A root attachment must use the reserved "
                "sentinel '__root__'; non-root attachments name another declared part."
            ),
        },
        "type": {
            "type": "string",
            "enum": list(_ATTACHMENT_TYPES),
            "description": (
                "Only the single root part uses type 'root'; its parent_id is '__root__'."
            ),
        },
        "contact_region": _BOUNDS3,
        "max_gap": {"type": "number", "minimum": 0},
        "max_penetration": {"type": "number", "minimum": 0},
        "min_contact_area": {"type": "number", "minimum": 0},
    },
    "additionalProperties": False,
    "allOf": [
        {
            "if": {
                "required": ["type"],
                "properties": {"type": {"enum": ["root"]}},
            },
            "then": {
                "properties": {
                    "parent_id": {
                        "enum": ["__root__"],
                        "description": (
                            "The reserved root attachment parent sentinel is '__root__'."
                        ),
                    }
                }
            },
            "else": {
                "properties": {
                    "parent_id": {
                        "not": {"enum": ["__root__"]},
                        "description": (
                            "Non-root attachments must name another declared part, not "
                            "the reserved '__root__' sentinel."
                        ),
                    }
                }
            },
        }
    ],
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
        "assembly": _ASSEMBLY_SCHEMA,
        "materials": {"type": "array"},
        "material_plan": _MATERIAL_PLAN_SCHEMA,
        "articulation": _ARTICULATION_SCHEMA,
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
    """One schema or semantic-contract failure with a stable JSON path."""

    path: tuple[str | int, ...]
    keyword: str
    message: str
    contract: str = "schema"

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
        return f"[{self.contract}] {self.location}: {self.message}"


class PlanSchemaError(ValueError):
    """Raised after collecting every schema and semantic plan violation."""

    def __init__(self, violations: Sequence[PlanSchemaViolation]):
        self.violations = tuple(violations)
        count = len(self.violations)
        schema_count = sum(violation.contract == "schema" for violation in self.violations)
        semantic_count = sum(
            violation.contract == "semantic" for violation in self.violations
        )
        if schema_count and semantic_count:
            contract = "the JSON Schema and semantic contract"
            breakdown = f": {schema_count} schema, {semantic_count} semantic"
        elif semantic_count:
            contract = "the semantic contract"
            breakdown = ""
        else:
            contract = "the JSON Schema"
            breakdown = ""
        details = "\n".join(f"- {violation}" for violation in self.violations)
        super().__init__(
            f"plan violates {contract} "
            f"({count} error{'s' if count != 1 else ''}{breakdown}):\n"
            f"{details}"
        )


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
    forbidden = schema.get("not")
    if isinstance(forbidden, Mapping) and not tuple(
        _iter_violations(value, forbidden, path=path)
    ):
        yield PlanSchemaViolation(path, "not", "is not allowed in this context")

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
                    contract="semantic",
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
                                contract="semantic",
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
                        contract="semantic",
                    )
                )
        elif kind in _ATTACHMENT_TYPES and isinstance(parent_id, str):
            if parent_id == identifier:
                violations.append(
                    PlanSchemaViolation(
                        ("parts", index, "attachment", "parent_id"),
                        "selfParent",
                        "a part cannot attach to itself",
                        contract="semantic",
                    )
                )
            elif parent_id not in known:
                violations.append(
                    PlanSchemaViolation(
                        ("parts", index, "attachment", "parent_id"),
                        "knownParent",
                        "must reference another declared part id",
                        contract="semantic",
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
                                contract="semantic",
                            )
                        )

    if len(root_indexes) != 1:
        violations.append(
            PlanSchemaViolation(
                ("parts",),
                "singleRoot",
                "attachment graph must contain exactly one root part",
                contract="semantic",
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
                    contract="semantic",
                )
            )
    for issue in validate_assembly(value).issues:
        violations.append(
            PlanSchemaViolation(
                issue.path,
                issue.keyword,
                issue.message,
                contract="semantic",
            )
        )
    if "material_plan" in value:
        try:
            material_plan = material_plan_from_document(value.get("material_plan"))
            material_plan.validate_part_ids(
                identifier for identifier in ids if identifier is not None
            )
        except MaterialPlanError as exc:
            violations.append(
                PlanSchemaViolation(
                    ("material_plan",),
                    "materialPlan",
                    str(exc),
                    contract="semantic",
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
        normalized["assembly"] = normalize_assembly(
            normalized.get("parts"), normalized.get("assembly")
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
