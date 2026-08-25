"""Deterministic comparison of canonical reference and generated renders."""

from __future__ import annotations

import base64
import binascii
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .plan_schema import validate_plan_document
from .workspace import write_json


CANONICAL_VIEW_NAMES = ("front", "back", "left", "right", "top", "iso")
_MASK_ENCODING = "base64-msb-packbits"
_RGB_ENCODING = "base64-rgb8"
_SCORE_WEIGHTS = {
    "silhouette_iou": 0.50,
    "area_similarity": 0.10,
    "spatial_rgb_similarity": 0.15,
    "dimension_similarity": 0.15,
    "center_similarity": 0.10,
}


@dataclass(frozen=True)
class FidelityGateThresholds:
    """Non-compensating fidelity requirements applied after aggregate scoring.

    The pipeline normalizes the reference's longest dimension to two units, so
    the distance thresholds are expressed in those normalized scene units.
    Callers may pass a different instance to :func:`compare_workspace` when a
    reconstruction profile needs stricter or more permissive gates.
    """

    min_mean_silhouette_iou: float = 0.40
    min_view_silhouette_iou: float = 0.30
    min_view_area_similarity: float = 0.40
    max_center_distance: float = 0.35
    max_ground_offset: float = 0.10
    min_mean_spatial_rgb_similarity: float = 0.30
    min_mean_palette_similarity: float = 0.35

    def __post_init__(self) -> None:
        probability_fields = (
            "min_mean_silhouette_iou",
            "min_view_silhouette_iou",
            "min_view_area_similarity",
            "min_mean_spatial_rgb_similarity",
            "min_mean_palette_similarity",
        )
        distance_fields = ("max_center_distance", "max_ground_offset")
        for name in probability_fields:
            _validate_gate_threshold(name, getattr(self, name), maximum=1.0)
        for name in distance_fields:
            _validate_gate_threshold(name, getattr(self, name))


@dataclass(frozen=True)
class SurfaceGateThresholds:
    """Non-compensating limits for normalized bidirectional surface distance.

    Surface comparison is optional. These defaults match the least strict
    surface-enabled reconstruction profile; callers can supply tighter limits
    without changing the legacy render-and-bounds score.
    """

    max_mean_surface_distance: float = 0.035
    max_p95_surface_distance: float = 0.080
    max_mean_normal_angle_degrees: float | None = None
    min_visible_coverage: float | None = None
    min_surface_area_ratio: float | None = None
    max_surface_area_ratio: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "max_mean_surface_distance",
            "max_p95_surface_distance",
        ):
            _validate_gate_threshold(name, getattr(self, name))
        if self.max_mean_normal_angle_degrees is not None:
            value = _validate_gate_threshold(
                "max_mean_normal_angle_degrees",
                self.max_mean_normal_angle_degrees,
            )
            if value > 180.0:
                raise ValueError(
                    "max_mean_normal_angle_degrees must be finite and between 0 and 180"
                )
        for name in (
            "min_visible_coverage",
            "min_surface_area_ratio",
            "max_surface_area_ratio",
        ):
            value = getattr(self, name)
            if value is not None:
                _validate_gate_threshold(
                    name,
                    value,
                    maximum=1.0 if name == "min_visible_coverage" else None,
                )
        if (
            self.min_surface_area_ratio is not None
            and self.max_surface_area_ratio is not None
            and self.min_surface_area_ratio > self.max_surface_area_ratio
        ):
            raise ValueError("minimum surface-area ratio cannot exceed maximum")


@dataclass(frozen=True)
class MaterialGateThresholds:
    """Export/material requirements that cannot be offset by geometry scores."""

    max_default_white_primitive_fraction: float = 0.25

    def __post_init__(self) -> None:
        _validate_gate_threshold(
            "max_default_white_primitive_fraction",
            self.max_default_white_primitive_fraction,
            maximum=1.0,
        )


@dataclass(frozen=True)
class DetailGateThresholds:
    """Minimum geometric and semantic richness relative to the reference/plan."""

    min_triangle_ratio: float = 0.10
    min_semantic_part_coverage: float = 0.65

    def __post_init__(self) -> None:
        _validate_gate_threshold("min_triangle_ratio", self.min_triangle_ratio, maximum=1.0)
        _validate_gate_threshold(
            "min_semantic_part_coverage",
            self.min_semantic_part_coverage,
            maximum=1.0,
        )


@dataclass(frozen=True)
class StructuralGateThresholds:
    """Topology, contact, and attachment limits for a coherent deliverable."""

    max_loose_component_fraction: float = 0.08
    max_boundary_edge_fraction: float = 0.03
    max_non_manifold_edge_fraction: float = 0.03
    max_inverted_normal_fraction: float = 0.01
    max_degenerate_triangle_fraction: float = 0.002
    max_low_quality_triangle_fraction: float = 0.08
    max_unjoined_attachment_fraction: float = 0.05
    max_intersection_fraction: float = 0.08

    def __post_init__(self) -> None:
        for name in (
            "max_loose_component_fraction",
            "max_boundary_edge_fraction",
            "max_non_manifold_edge_fraction",
            "max_inverted_normal_fraction",
            "max_degenerate_triangle_fraction",
            "max_low_quality_triangle_fraction",
            "max_unjoined_attachment_fraction",
            "max_intersection_fraction",
        ):
            _validate_gate_threshold(name, getattr(self, name), maximum=1.0)


def _validate_gate_threshold(
    name: str,
    value: Any,
    *,
    maximum: float | None = None,
) -> float:
    qualifier = " between 0 and 1" if maximum == 1.0 else " and non-negative"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite{qualifier}")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite{qualifier}") from exc
    if not math.isfinite(result) or result < 0.0 or (
        maximum is not None and result > maximum
    ):
        raise ValueError(f"{name} must be finite{qualifier}")
    return result


def _surface_positive_integer(value: Any, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"surface comparison {label} must be a positive integer")
    return value


def _surface_distance(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"surface comparison {label} must be a finite non-negative number"
        )
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(
            f"surface comparison {label} must be a finite non-negative number"
        ) from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(
            f"surface comparison {label} must be a finite non-negative number"
        )
    return result


def _surface_nonnegative_integer(value: Any, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(
            f"surface comparison {label} must be a non-negative integer"
        )
    return value


def _surface_identity(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"surface comparison {label} must be a JSON object")
    return {
        "object": _surface_string(value.get("object"), label=f"{label}.object"),
        "object_index": _surface_nonnegative_integer(
            value.get("object_index"), label=f"{label}.object_index"
        ),
        "polygon_index": _surface_nonnegative_integer(
            value.get("polygon_index"), label=f"{label}.polygon_index"
        ),
        "triangle_index_in_object": _surface_nonnegative_integer(
            value.get("triangle_index_in_object"),
            label=f"{label}.triangle_index_in_object",
        ),
        "surface_triangle_index": _surface_nonnegative_integer(
            value.get("surface_triangle_index"),
            label=f"{label}.surface_triangle_index",
        ),
    }


def _surface_statistics(
    value: Any,
    *,
    label: str,
    include_samples: bool,
    require_worst_samples: bool,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"surface comparison {label} must be a JSON object")

    statistics = {
        name: _surface_distance(value.get(name), label=f"{label}.{name}")
        for name in ("mean", "rms", "p95", "max")
    }
    tolerance = max(1.0e-12, statistics["max"] * 1.0e-12)
    if statistics["mean"] > statistics["max"] + tolerance:
        raise ValueError(
            f"surface comparison {label}.mean cannot exceed {label}.max"
        )
    if statistics["rms"] + tolerance < statistics["mean"]:
        raise ValueError(
            f"surface comparison {label}.rms cannot be less than {label}.mean"
        )
    if statistics["rms"] > statistics["max"] + tolerance:
        raise ValueError(
            f"surface comparison {label}.rms cannot exceed {label}.max"
        )
    if statistics["p95"] > statistics["max"] + tolerance:
        raise ValueError(
            f"surface comparison {label}.p95 cannot exceed {label}.max"
        )

    result: dict[str, Any] = dict(statistics)
    if include_samples:
        result = {
            "samples": _surface_positive_integer(
                value.get("samples"), label=f"{label}.samples"
            ),
            "source_surface_area": _surface_distance(
                value.get("source_surface_area"),
                label=f"{label}.source_surface_area",
            ),
            **result,
        }
        if result["source_surface_area"] <= 0.0:
            raise ValueError(
                f"surface comparison {label}.source_surface_area must be positive"
            )
    if require_worst_samples:
        worst_samples = value.get("worst_samples")
        if not isinstance(worst_samples, list):
            raise ValueError(
                f"surface comparison {label}.worst_samples must be an array"
            )
        normalized_samples = []
        for index, sample in enumerate(worst_samples):
            sample_label = f"{label}.worst_samples[{index}]"
            if not isinstance(sample, dict):
                raise ValueError(
                    f"surface comparison {sample_label} must be a JSON object"
                )
            sample_index = sample.get("sample_index")
            if (
                type(sample_index) is not int
                or sample_index < 0
                or sample_index >= result["samples"]
            ):
                raise ValueError(
                    f"surface comparison {sample_label}.sample_index is out of range"
                )
            distance = _surface_distance(
                sample.get("distance"), label=f"{sample_label}.distance"
            )
            tolerance = max(1.0e-12, result["max"] * 1.0e-12)
            if distance > result["max"] + tolerance:
                raise ValueError(
                    f"surface comparison {sample_label}.distance cannot exceed {label}.max"
                )
            normalized_sample = {
                "sample_index": sample_index,
                "distance": distance,
                "source": _surface_vector(
                    sample.get("source"), label=f"{sample_label}.source"
                ),
                "nearest": _surface_vector(
                    sample.get("nearest"), label=f"{sample_label}.nearest"
                ),
            }
            for identity_name in ("source_identity", "target_identity"):
                if identity_name in sample:
                    normalized_sample[identity_name] = _surface_identity(
                        sample[identity_name],
                        label=f"{sample_label}.{identity_name}",
                    )
            for vector_name in ("source_normal", "target_normal"):
                if vector_name in sample:
                    normalized_sample[vector_name] = _surface_vector(
                        sample[vector_name], label=f"{sample_label}.{vector_name}"
                    )
            for distance_name in (
                "normal_angle_degrees",
                "unoriented_normal_angle_degrees",
                "point_to_plane_distance",
                "normal_aware_distance",
            ):
                if distance_name in sample:
                    normalized_sample[distance_name] = _surface_distance(
                        sample[distance_name],
                        label=f"{sample_label}.{distance_name}",
                    )
            for finite_name in ("normal_cosine", "signed_target_plane_offset"):
                if finite_name in sample:
                    normalized_sample[finite_name] = _surface_finite_number(
                        sample[finite_name], label=f"{sample_label}.{finite_name}"
                    )
            visible_from = sample.get("visible_from")
            if visible_from is not None:
                if not isinstance(visible_from, list) or not all(
                    isinstance(name, str) and name.strip() for name in visible_from
                ):
                    raise ValueError(
                        f"surface comparison {sample_label}.visible_from must be "
                        "an array of non-empty strings"
                    )
                normalized_sample["visible_from"] = list(visible_from)
            normalized_samples.append(normalized_sample)
        result["worst_samples"] = normalized_samples
    return result


def _surface_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"surface comparison {label} must be a non-empty string")
    return value


def _surface_finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"surface comparison {label} must be a finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"surface comparison {label} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"surface comparison {label} must be a finite number")
    return result


def _surface_vector(value: Any, *, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"surface comparison {label} must be a 3-element array")
    return [
        _surface_finite_number(item, label=f"{label}[{index}]")
        for index, item in enumerate(value)
    ]


def _surface_sampling(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("surface comparison sampling must be a JSON object")
    worst_sample_limit = value.get("worst_sample_limit_per_direction")
    if type(worst_sample_limit) is not int or worst_sample_limit < 0:
        raise ValueError(
            "surface comparison sampling.worst_sample_limit_per_direction "
            "must be a non-negative integer"
        )
    result = {
        "strategy": _surface_string(value.get("strategy"), label="sampling.strategy"),
        "requested_samples_per_direction": _surface_positive_integer(
            value.get("requested_samples_per_direction"),
            label="sampling.requested_samples_per_direction",
        ),
        "percentile_method": _surface_string(
            value.get("percentile_method"), label="sampling.percentile_method"
        ),
        "worst_sample_limit_per_direction": worst_sample_limit,
    }
    if "visibility_sample_budget_per_direction" in value:
        result["visibility_sample_budget_per_direction"] = _surface_positive_integer(
            value["visibility_sample_budget_per_direction"],
            label="sampling.visibility_sample_budget_per_direction",
        )
    return result


def _surface_normalization(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(
            "surface comparison reference_normalization must be a JSON object"
        )
    scale = _surface_distance(
        value.get("scale"), label="reference_normalization.scale"
    )
    longest_dimension = _surface_distance(
        value.get("longest_dimension"),
        label="reference_normalization.longest_dimension",
    )
    if scale <= 0.0 or longest_dimension <= 0.0:
        raise ValueError(
            "surface comparison reference normalization scale and longest_dimension "
            "must be positive"
        )
    source_min = _surface_vector(
        value.get("source_min"), label="reference_normalization.source_min"
    )
    source_max = _surface_vector(
        value.get("source_max"), label="reference_normalization.source_max"
    )
    if any(low > high for low, high in zip(source_min, source_max, strict=True)):
        raise ValueError(
            "surface comparison reference_normalization.source_min cannot exceed source_max"
        )
    return {
        "source_min": source_min,
        "source_max": source_max,
        "scale": scale,
        "translation": _surface_vector(
            value.get("translation"), label="reference_normalization.translation"
        ),
        "longest_dimension": longest_dimension,
    }


def _surface_inventory(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"surface comparison surfaces.{label} must be a JSON object")
    area = _surface_distance(value.get("area"), label=f"surfaces.{label}.area")
    if area <= 0.0:
        raise ValueError(f"surface comparison surfaces.{label}.area must be positive")
    result = {
        "vertices": _surface_positive_integer(
            value.get("vertices"), label=f"surfaces.{label}.vertices"
        ),
        "triangles": _surface_positive_integer(
            value.get("triangles"), label=f"surfaces.{label}.triangles"
        ),
        "area": area,
    }
    if "objects" in value:
        objects = value["objects"]
        if not isinstance(objects, list):
            raise ValueError(
                f"surface comparison surfaces.{label}.objects must be an array"
            )
        normalized_objects = []
        for index, record in enumerate(objects):
            record_label = f"surfaces.{label}.objects[{index}]"
            if not isinstance(record, dict):
                raise ValueError(
                    f"surface comparison {record_label} must be a JSON object"
                )
            normalized_objects.append(
                {
                    "name": _surface_string(
                        record.get("name"), label=f"{record_label}.name"
                    ),
                    "object_index": _surface_nonnegative_integer(
                        record.get("object_index"),
                        label=f"{record_label}.object_index",
                    ),
                    "vertices": _surface_nonnegative_integer(
                        record.get("vertices"), label=f"{record_label}.vertices"
                    ),
                    "triangles": _surface_nonnegative_integer(
                        record.get("triangles"), label=f"{record_label}.triangles"
                    ),
                    "area": _surface_distance(
                        record.get("area"), label=f"{record_label}.area"
                    ),
                }
            )
        result["objects"] = normalized_objects
    return result


def _surface_inventories(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("surface comparison surfaces must be a JSON object")
    return {
        label: _surface_inventory(value.get(label), label=label)
        for label in ("reference", "candidate")
    }


def _surface_comparison(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"surface comparison is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError("surface comparison must contain a JSON object")
    if type(document.get("schema_version")) is not int or document["schema_version"] != 1:
        raise ValueError("surface comparison must use schema_version 1")

    units = _surface_string(document.get("units"), label="units")
    if units != "normalized-scene-units":
        raise ValueError(
            "surface comparison units must be 'normalized-scene-units'"
        )
    sampling = _surface_sampling(document.get("sampling"))
    normalization = _surface_normalization(document.get("reference_normalization"))
    surfaces = _surface_inventories(document.get("surfaces"))
    candidate_to_reference = _surface_statistics(
        document.get("candidate_to_reference"),
        label="candidate_to_reference",
        include_samples=True,
        require_worst_samples=True,
    )
    reference_to_candidate = _surface_statistics(
        document.get("reference_to_candidate"),
        label="reference_to_candidate",
        include_samples=True,
        require_worst_samples=True,
    )
    for normalized, raw, label in (
        (
            candidate_to_reference,
            document.get("candidate_to_reference"),
            "candidate_to_reference",
        ),
        (
            reference_to_candidate,
            document.get("reference_to_candidate"),
            "reference_to_candidate",
        ),
    ):
        assert isinstance(raw, dict)
        for name in ("normal_metrics", "coverage", "visible_external_proxy"):
            if name in raw:
                section = raw[name]
                if not isinstance(section, dict):
                    raise ValueError(
                        f"surface comparison {label}.{name} must be a JSON object"
                    )
                normalized[name] = section
        if "per_source_object" in raw:
            records = raw["per_source_object"]
            if not isinstance(records, list):
                raise ValueError(
                    f"surface comparison {label}.per_source_object must be an array"
                )
            normalized["per_source_object"] = records
    requested_samples = sampling["requested_samples_per_direction"]
    worst_sample_limit = sampling["worst_sample_limit_per_direction"]
    for label, statistics, source_label in (
        ("candidate_to_reference", candidate_to_reference, "candidate"),
        ("reference_to_candidate", reference_to_candidate, "reference"),
    ):
        if statistics["samples"] != requested_samples:
            raise ValueError(
                f"surface comparison {label}.samples must equal "
                "sampling.requested_samples_per_direction"
            )
        if len(statistics["worst_samples"]) > worst_sample_limit:
            raise ValueError(
                f"surface comparison {label}.worst_samples exceeds the recorded limit"
            )
        expected_area = surfaces[source_label]["area"]
        tolerance = max(1.0e-12, expected_area * 1.0e-9)
        if abs(statistics["source_surface_area"] - expected_area) > tolerance:
            raise ValueError(
                f"surface comparison {label}.source_surface_area does not match "
                f"surfaces.{source_label}.area"
            )
    symmetric = _surface_statistics(
        document.get("symmetric"),
        label="symmetric",
        include_samples=False,
        require_worst_samples=False,
    )
    result = {
        "schema_version": 1,
        "units": units,
        "pose_policy": _surface_string(
            document.get("pose_policy"), label="pose_policy"
        ),
        "sampling": sampling,
        "reference_normalization": normalization,
        "surfaces": surfaces,
        "candidate_to_reference": candidate_to_reference,
        "reference_to_candidate": reference_to_candidate,
        "symmetric": symmetric,
    }
    for name in (
        "area_comparison",
        "normal_aware",
        "visible_external_proxy",
        "residual_artifacts",
    ):
        if name in document:
            section = document[name]
            if not isinstance(section, dict):
                raise ValueError(f"surface comparison {name} must be a JSON object")
            result[name] = section
    return result


def _get_bit(data: bytes, index: int) -> bool:
    return bool(data[index // 8] & (1 << (7 - index % 8)))


def _decode_base64(record: dict[str, Any], field: str) -> bytes:
    value = record.get(field)
    if not isinstance(value, str):
        raise ValueError(f"mask field {field!r} must be a base64 string")
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"mask field {field!r} is not valid base64") from exc


def _decode_mask(record: dict[str, Any]) -> tuple[int, int, bytes, bytes]:
    if not isinstance(record, dict):
        raise ValueError("mask record must be a JSON object")
    width = record.get("width")
    height = record.get("height")
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ValueError("mask dimensions must be positive integers")
    if record.get("encoding") != "base64-msb-packbits":
        raise ValueError(f"unsupported mask encoding: {record.get('encoding')!r}")
    if record.get("rgb_encoding") != _RGB_ENCODING:
        raise ValueError(f"unsupported RGB encoding: {record.get('rgb_encoding')!r}")

    pixel_count = width * height
    mask_data = _decode_base64(record, "data")
    expected_mask_bytes = (pixel_count + 7) // 8
    if len(mask_data) != expected_mask_bytes:
        raise ValueError(
            f"mask data has {len(mask_data)} bytes; expected {expected_mask_bytes}"
        )
    if any(_get_bit(mask_data, index) for index in range(pixel_count, expected_mask_bytes * 8)):
        raise ValueError("mask data has non-zero padding bits")

    foreground = sum(_get_bit(mask_data, index) for index in range(pixel_count))
    if type(record.get("foreground_pixels")) is not int or record["foreground_pixels"] != foreground:
        raise ValueError(
            f"foreground_pixels is {record.get('foreground_pixels')!r}; expected {foreground}"
        )

    rgb_data = _decode_base64(record, "rgb_data")
    expected_rgb_bytes = pixel_count * 3
    if len(rgb_data) != expected_rgb_bytes:
        raise ValueError(
            f"RGB data has {len(rgb_data)} bytes; expected {expected_rgb_bytes}"
        )
    return width, height, mask_data, rgb_data


def mask_metrics(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Compare aligned silhouette and RGB samples for one canonical view.

    RGB similarity is accumulated at matching foreground pixels and normalized
    by the silhouette union.  It is therefore spatial (a color appearing at the
    wrong pixel does not receive histogram credit), symmetric, and naturally
    penalizes missing or extra geometry.
    """

    width, height, reference_data, reference_rgb = _decode_mask(reference)
    other_width, other_height, candidate_data, candidate_rgb = _decode_mask(candidate)
    if (width, height) != (other_width, other_height):
        raise ValueError("mask resolutions do not match")
    intersection = union = reference_count = candidate_count = 0
    rgb_similarity_sum = 0.0
    reference_palette = [0] * 64
    candidate_palette = [0] * 64
    ref_bounds = [width, height, -1, -1]
    candidate_bounds = [width, height, -1, -1]
    for index in range(width * height):
        ref = _get_bit(reference_data, index)
        cand = _get_bit(candidate_data, index)
        if ref:
            reference_count += 1
            offset = index * 3
            palette_index = (
                (reference_rgb[offset] // 64) * 16
                + (reference_rgb[offset + 1] // 64) * 4
                + reference_rgb[offset + 2] // 64
            )
            reference_palette[palette_index] += 1
            x, y = index % width, index // width
            ref_bounds = [min(ref_bounds[0], x), min(ref_bounds[1], y), max(ref_bounds[2], x), max(ref_bounds[3], y)]
        if cand:
            candidate_count += 1
            offset = index * 3
            palette_index = (
                (candidate_rgb[offset] // 64) * 16
                + (candidate_rgb[offset + 1] // 64) * 4
                + candidate_rgb[offset + 2] // 64
            )
            candidate_palette[palette_index] += 1
            x, y = index % width, index // width
            candidate_bounds = [
                min(candidate_bounds[0], x),
                min(candidate_bounds[1], y),
                max(candidate_bounds[2], x),
                max(candidate_bounds[3], y),
            ]
        if ref and cand:
            intersection += 1
            offset = index * 3
            channel_error = sum(
                abs(reference_rgb[offset + channel] - candidate_rgb[offset + channel])
                for channel in range(3)
            )
            rgb_similarity_sum += 1.0 - channel_error / (3.0 * 255.0)
        union += int(ref or cand)
    iou = intersection / union if union else 1.0
    spatial_rgb_similarity = rgb_similarity_sum / union if union else 1.0
    if reference_count and candidate_count:
        palette_similarity = sum(
            min(ref_count / reference_count, candidate_count_value / candidate_count)
            for ref_count, candidate_count_value in zip(
                reference_palette, candidate_palette, strict=True
            )
        )
    else:
        palette_similarity = 1.0 if reference_count == candidate_count else 0.0
    area_similarity = (
        min(reference_count, candidate_count) / max(reference_count, candidate_count)
        if max(reference_count, candidate_count)
        else 1.0
    )
    return {
        "iou": iou,
        "area_similarity": area_similarity,
        "spatial_rgb_similarity": spatial_rgb_similarity,
        "palette_similarity": _bounded_score(
            "palette similarity", palette_similarity
        ),
        "reference_pixels": reference_count,
        "candidate_pixels": candidate_count,
        "reference_bbox": ref_bounds if reference_count else None,
        "candidate_bbox": candidate_bounds if candidate_count else None,
    }


def _canonical_views(path: Path, *, label: str) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema_version") != 2:
        raise ValueError(f"{label} masks must use schema_version 2")
    views = document.get("views")
    if not isinstance(views, dict):
        raise ValueError(f"{label} masks must contain a views object")

    required = set(CANONICAL_VIEW_NAMES)
    actual = set(views)
    if actual != required:
        missing = sorted(required - actual)
        unexpected = sorted(actual - required)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise ValueError(f"{label} canonical view set mismatch: {'; '.join(details)}")
    return views


def _finite_vector(value: Any, *, label: str, field: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{label} scene bounds.{field} must be a 3-element array")
    result = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(
                f"{label} scene bounds.{field}[{index}] must be a finite number"
            )
        try:
            number = float(item)
        except OverflowError as exc:
            raise ValueError(
                f"{label} scene bounds.{field}[{index}] must be a finite number"
            ) from exc
        if not math.isfinite(number):
            raise ValueError(
                f"{label} scene bounds.{field}[{index}] must be a finite number"
            )
        result.append(number)
    return result


def _scene_bounds(path: Path, *, label: str) -> dict[str, list[float]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{label} scene must contain a JSON object")
    bounds = document.get("bounds")
    if not isinstance(bounds, dict):
        raise ValueError(f"{label} scene must contain a bounds object")

    minimum = _finite_vector(bounds.get("min"), label=label, field="min")
    maximum = _finite_vector(bounds.get("max"), label=label, field="max")
    dimensions = _finite_vector(bounds.get("dimensions"), label=label, field="dimensions")
    center = _finite_vector(bounds.get("center"), label=label, field="center")
    for axis, (low, high, dimension, midpoint) in enumerate(
        zip(minimum, maximum, dimensions, center, strict=True)
    ):
        if high < low:
            raise ValueError(
                f"{label} scene bounds.min[{axis}] cannot exceed bounds.max[{axis}]"
            )
        span = high - low
        if not math.isfinite(span):
            raise ValueError(f"{label} scene bounds span on axis {axis} is not finite")
        if dimension < 0.0:
            raise ValueError(f"{label} scene bounds.dimensions[{axis}] cannot be negative")
        tolerance = max(1.0e-9, span * 1.0e-7)
        if abs(dimension - span) > tolerance:
            raise ValueError(
                f"{label} scene bounds.dimensions[{axis}] does not match max - min"
            )
        expected_center = low + span * 0.5
        if abs(midpoint - expected_center) > tolerance:
            raise ValueError(
                f"{label} scene bounds.center[{axis}] is not the bounds midpoint"
            )
    if not any(dimension > 0.0 for dimension in dimensions):
        raise ValueError(f"{label} scene bounds must span at least one axis")
    return {
        "min": minimum,
        "max": maximum,
        "dimensions": dimensions,
        "center": center,
    }


def _score_threshold(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("min_score must be a finite number between 0 and 1")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError("min_score must be a finite number between 0 and 1") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("min_score must be a finite number between 0 and 1")
    return result


def _bounded_score(name: str, value: float) -> float:
    if not math.isfinite(value) or value < -1.0e-12 or value > 1.0 + 1.0e-12:
        raise ValueError(f"computed {name} must be finite and between 0 and 1")
    return min(1.0, max(0.0, value))


def _vector_similarity(reference: list[float], candidate: list[float]) -> float:
    values = []
    for expected, actual in zip(reference, candidate, strict=True):
        if expected == 0.0 and actual == 0.0:
            values.append(1.0)
        elif expected <= 0.0 or actual <= 0.0:
            values.append(0.0)
        else:
            values.append(math.exp(-abs(math.log(actual) - math.log(expected))))
    return _bounded_score("dimension similarity", sum(values) / len(values))


def _hard_gate_result(
    *,
    value: float,
    threshold: float,
    operator: str,
    passed: bool,
    message: str,
    view: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "value": value,
        "operator": operator,
        "threshold": threshold,
        "passed": passed,
        "message": message,
    }
    if view is not None:
        result["view"] = view
    return result


def _evaluate_hard_gates(
    *,
    views: dict[str, dict[str, Any]],
    mean_iou: float,
    mean_rgb: float,
    mean_palette: float,
    center_distance: float,
    reference_geometry: dict[str, list[float]],
    candidate_geometry: dict[str, list[float]],
    thresholds: FidelityGateThresholds,
) -> dict[str, Any]:
    worst_iou_view, worst_iou_metrics = min(
        views.items(), key=lambda item: item[1]["iou"]
    )
    worst_area_view, worst_area_metrics = min(
        views.items(), key=lambda item: item[1]["area_similarity"]
    )
    ground_offset = abs(
        candidate_geometry["min"][2] - reference_geometry["min"][2]
    )
    if not math.isfinite(ground_offset):
        raise ValueError("reference and candidate ground offset is not finite")

    mean_iou_passed = mean_iou >= thresholds.min_mean_silhouette_iou
    worst_iou = float(worst_iou_metrics["iou"])
    worst_iou_passed = worst_iou >= thresholds.min_view_silhouette_iou
    worst_area = float(worst_area_metrics["area_similarity"])
    worst_area_passed = worst_area >= thresholds.min_view_area_similarity
    center_passed = center_distance <= thresholds.max_center_distance
    ground_passed = ground_offset <= thresholds.max_ground_offset
    rgb_passed = mean_rgb >= thresholds.min_mean_spatial_rgb_similarity
    palette_passed = mean_palette >= thresholds.min_mean_palette_similarity

    results = {
        "mean_silhouette_iou": _hard_gate_result(
            value=mean_iou,
            threshold=thresholds.min_mean_silhouette_iou,
            operator=">=",
            passed=mean_iou_passed,
            message=(
                f"mean silhouette IoU {mean_iou:.4f} must be at least "
                f"{thresholds.min_mean_silhouette_iou:.4f}"
            ),
        ),
        "minimum_view_silhouette_iou": _hard_gate_result(
            value=worst_iou,
            threshold=thresholds.min_view_silhouette_iou,
            operator=">=",
            passed=worst_iou_passed,
            view=worst_iou_view,
            message=(
                f"{worst_iou_view} silhouette IoU {worst_iou:.4f} must be at least "
                f"{thresholds.min_view_silhouette_iou:.4f}"
            ),
        ),
        "minimum_view_area_similarity": _hard_gate_result(
            value=worst_area,
            threshold=thresholds.min_view_area_similarity,
            operator=">=",
            passed=worst_area_passed,
            view=worst_area_view,
            message=(
                f"{worst_area_view} foreground-area similarity {worst_area:.4f} "
                f"must be at least {thresholds.min_view_area_similarity:.4f}"
            ),
        ),
        "center_distance": _hard_gate_result(
            value=center_distance,
            threshold=thresholds.max_center_distance,
            operator="<=",
            passed=center_passed,
            message=(
                f"center distance {center_distance:.4f} must be at most "
                f"{thresholds.max_center_distance:.4f}"
            ),
        ),
        "ground_offset": _hard_gate_result(
            value=ground_offset,
            threshold=thresholds.max_ground_offset,
            operator="<=",
            passed=ground_passed,
            message=(
                f"ground-plane offset {ground_offset:.4f} must be at most "
                f"{thresholds.max_ground_offset:.4f}"
            ),
        ),
        "mean_spatial_rgb_similarity": _hard_gate_result(
            value=mean_rgb,
            threshold=thresholds.min_mean_spatial_rgb_similarity,
            operator=">=",
            passed=rgb_passed,
            message=(
                f"mean spatial RGB similarity {mean_rgb:.4f} must be at least "
                f"{thresholds.min_mean_spatial_rgb_similarity:.4f}"
            ),
        ),
        "mean_palette_similarity": _hard_gate_result(
            value=mean_palette,
            threshold=thresholds.min_mean_palette_similarity,
            operator=">=",
            passed=palette_passed,
            message=(
                f"mean foreground palette similarity {mean_palette:.4f} must be at least "
                f"{thresholds.min_mean_palette_similarity:.4f}"
            ),
        ),
    }
    failures = [
        {"gate": name, **result}
        for name, result in results.items()
        if not result["passed"]
    ]
    return {
        "passed": not failures,
        "thresholds": asdict(thresholds),
        "results": results,
        "failures": failures,
    }


def _add_surface_hard_gates(
    hard_gates: dict[str, Any],
    *,
    surface: dict[str, Any],
    thresholds: SurfaceGateThresholds,
) -> None:
    symmetric = surface["symmetric"]
    mean_distance = float(symmetric["mean"])
    p95_distance = float(symmetric["p95"])
    mean_passed = mean_distance <= thresholds.max_mean_surface_distance
    p95_passed = p95_distance <= thresholds.max_p95_surface_distance
    results = {
        "mean_surface_distance": _hard_gate_result(
            value=mean_distance,
            threshold=thresholds.max_mean_surface_distance,
            operator="<=",
            passed=mean_passed,
            message=(
                f"mean symmetric surface distance {mean_distance:.6f} must be at most "
                f"{thresholds.max_mean_surface_distance:.6f}"
            ),
        ),
        "p95_surface_distance": _hard_gate_result(
            value=p95_distance,
            threshold=thresholds.max_p95_surface_distance,
            operator="<=",
            passed=p95_passed,
            message=(
                f"p95 symmetric surface distance {p95_distance:.6f} must be at most "
                f"{thresholds.max_p95_surface_distance:.6f}"
            ),
        ),
    }
    if thresholds.max_mean_normal_angle_degrees is not None:
        normal_aware = surface.get("normal_aware")
        normal_angles = (
            normal_aware.get("normal_angle_degrees")
            if isinstance(normal_aware, dict)
            else None
        )
        if not isinstance(normal_angles, dict):
            raise ValueError(
                "surface comparison lacks normal_aware.normal_angle_degrees"
            )
        mean_angle = _surface_distance(
            normal_angles.get("mean"), label="normal_aware.normal_angle_degrees.mean"
        )
        normal_passed = mean_angle <= thresholds.max_mean_normal_angle_degrees
        results["mean_normal_angle_degrees"] = _hard_gate_result(
            value=mean_angle,
            threshold=thresholds.max_mean_normal_angle_degrees,
            operator="<=",
            passed=normal_passed,
            message=(
                f"mean oriented normal angle {mean_angle:.4f} degrees must be at most "
                f"{thresholds.max_mean_normal_angle_degrees:.4f}"
            ),
        )
    if thresholds.min_visible_coverage is not None:
        directed_coverages = []
        for direction_name in (
            "candidate_to_reference",
            "reference_to_candidate",
        ):
            directed = surface.get(direction_name)
            visible = (
                directed.get("visible_external_proxy")
                if isinstance(directed, dict)
                else None
            )
            coverage = visible.get("coverage") if isinstance(visible, dict) else None
            entries = coverage.get("thresholds") if isinstance(coverage, dict) else None
            if not isinstance(entries, list) or not entries:
                raise ValueError(
                    f"surface comparison lacks {direction_name} visible coverage thresholds"
                )
            valid_entries = []
            for index, entry in enumerate(entries):
                if not isinstance(entry, dict):
                    raise ValueError(
                        f"surface comparison {direction_name} coverage threshold {index} "
                        "must be an object"
                    )
                distance = _surface_distance(
                    entry.get("distance"),
                    label=f"{direction_name}.visible.coverage.thresholds[{index}].distance",
                )
                fraction = entry.get("distance_and_normal_aligned_fraction")
                _validate_gate_threshold(
                    f"{direction_name}.visible coverage fraction",
                    fraction,
                    maximum=1.0,
                )
                valid_entries.append((distance, float(fraction)))
            _distance, fraction = min(
                valid_entries,
                key=lambda item: abs(item[0] - thresholds.max_p95_surface_distance),
            )
            directed_coverages.append(fraction)
        visible_coverage = min(directed_coverages)
        coverage_passed = visible_coverage >= thresholds.min_visible_coverage
        results["visible_surface_coverage"] = _hard_gate_result(
            value=visible_coverage,
            threshold=thresholds.min_visible_coverage,
            operator=">=",
            passed=coverage_passed,
            message=(
                f"worst directed visible distance-and-normal coverage "
                f"{visible_coverage:.4f} must be at least "
                f"{thresholds.min_visible_coverage:.4f}"
            ),
        )
    area_ratio = (
        surface["surfaces"]["candidate"]["area"]
        / surface["surfaces"]["reference"]["area"]
    )
    if thresholds.min_surface_area_ratio is not None:
        minimum_passed = area_ratio >= thresholds.min_surface_area_ratio
        results["minimum_surface_area_ratio"] = _hard_gate_result(
            value=area_ratio,
            threshold=thresholds.min_surface_area_ratio,
            operator=">=",
            passed=minimum_passed,
            message=(
                f"candidate/reference surface-area ratio {area_ratio:.4f} must be at least "
                f"{thresholds.min_surface_area_ratio:.4f}"
            ),
        )
    if thresholds.max_surface_area_ratio is not None:
        maximum_passed = area_ratio <= thresholds.max_surface_area_ratio
        results["maximum_surface_area_ratio"] = _hard_gate_result(
            value=area_ratio,
            threshold=thresholds.max_surface_area_ratio,
            operator="<=",
            passed=maximum_passed,
            message=(
                f"candidate/reference surface-area ratio {area_ratio:.4f} must be at most "
                f"{thresholds.max_surface_area_ratio:.4f}"
            ),
        )
    hard_gates["thresholds"].update(asdict(thresholds))
    hard_gates["results"].update(results)
    hard_gates["failures"].extend(
        {"gate": name, **result}
        for name, result in results.items()
        if not result["passed"]
    )
    hard_gates["passed"] = not hard_gates["failures"]


def _read_report(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _merge_gate_results(
    hard_gates: dict[str, Any],
    *,
    thresholds: Any,
    results: dict[str, dict[str, Any]],
) -> None:
    hard_gates["thresholds"].update(asdict(thresholds))
    hard_gates["results"].update(results)
    hard_gates["failures"].extend(
        {"gate": name, **result}
        for name, result in results.items()
        if not result["passed"]
    )
    hard_gates["passed"] = not hard_gates["failures"]


def _add_material_hard_gates(
    hard_gates: dict[str, Any],
    *,
    candidate_probe: dict[str, Any],
    thresholds: MaterialGateThresholds,
) -> dict[str, Any]:
    diagnostics = candidate_probe.get("material_diagnostics")
    scene = candidate_probe.get("scene")
    if not isinstance(diagnostics, dict) or not isinstance(scene, dict):
        raise ValueError(
            "candidate GLB probe must contain scene and material_diagnostics objects"
        )
    primitive_count = scene.get("primitive_count")
    at_risk = diagnostics.get("primitive_count_at_default_white_risk")
    if type(primitive_count) is not int or primitive_count < 0:
        raise ValueError("candidate GLB probe scene.primitive_count must be non-negative")
    if type(at_risk) is not int or not 0 <= at_risk <= primitive_count:
        raise ValueError(
            "candidate GLB probe default-white primitive count is invalid"
        )
    fraction = at_risk / primitive_count if primitive_count else 0.0
    passed = fraction <= thresholds.max_default_white_primitive_fraction
    results = {
        "default_white_primitive_fraction": _hard_gate_result(
            value=fraction,
            threshold=thresholds.max_default_white_primitive_fraction,
            operator="<=",
            passed=passed,
            message=(
                f"implicit-white-risk primitives {at_risk}/{primitive_count} "
                f"({fraction:.4f}) must be at most "
                f"{thresholds.max_default_white_primitive_fraction:.4f}"
            ),
        )
    }
    _merge_gate_results(hard_gates, thresholds=thresholds, results=results)
    return {
        "primitive_count": primitive_count,
        "primitive_count_at_default_white_risk": at_risk,
        "default_white_primitive_fraction": fraction,
        "declared_base_color_palette": diagnostics.get(
            "declared_base_color_palette", []
        ),
    }


def _probe_triangle_count(value: dict[str, Any], *, label: str) -> int:
    scene = value.get("scene")
    count = scene.get("triangle_count") if isinstance(scene, dict) else None
    if type(count) is not int or count < 0:
        raise ValueError(f"{label} GLB probe scene.triangle_count must be non-negative")
    return count


def _object_key(value: str) -> str:
    value = re.sub(r"\.\d{3}$", "", value.strip().lower())
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_")


def _matching_object_names(declared: Any, actual: list[str]) -> list[str]:
    if not isinstance(declared, list):
        return []
    expected = {
        _object_key(item)
        for item in declared
        if isinstance(item, str) and _object_key(item)
    }
    matches = []
    for name in actual:
        key = _object_key(name)
        if any(
            key == item or key.startswith(item + "_") or item.startswith(key + "_")
            for item in expected
        ):
            matches.append(name)
    return matches


def _add_detail_hard_gates(
    hard_gates: dict[str, Any],
    *,
    reference_probe: dict[str, Any],
    candidate_probe: dict[str, Any],
    candidate_scene: dict[str, Any],
    plan: dict[str, Any],
    thresholds: DetailGateThresholds,
) -> dict[str, Any]:
    reference_triangles = _probe_triangle_count(reference_probe, label="reference")
    candidate_triangles = _probe_triangle_count(candidate_probe, label="candidate")
    triangle_ratio = (
        min(1.0, candidate_triangles / reference_triangles)
        if reference_triangles
        else 1.0
    )
    scene_objects = candidate_scene.get("objects")
    if not isinstance(scene_objects, list):
        raise ValueError("candidate scene report must contain an objects array")
    actual_names = [
        item["name"]
        for item in scene_objects
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    ]
    parts = plan.get("parts")
    if not isinstance(parts, list) or not parts:
        raise ValueError("normalized plan must contain a non-empty parts array")
    covered_ids = []
    missing_ids = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        identifier = str(part.get("id") or "unknown")
        if _matching_object_names(part.get("object_names"), actual_names):
            covered_ids.append(identifier)
        else:
            missing_ids.append(identifier)
    semantic_coverage = len(covered_ids) / len(parts)
    triangle_passed = triangle_ratio >= thresholds.min_triangle_ratio
    coverage_passed = semantic_coverage >= thresholds.min_semantic_part_coverage
    results = {
        "triangle_richness_ratio": _hard_gate_result(
            value=triangle_ratio,
            threshold=thresholds.min_triangle_ratio,
            operator=">=",
            passed=triangle_passed,
            message=(
                f"candidate/reference triangle richness {candidate_triangles}/"
                f"{reference_triangles} ({triangle_ratio:.4f}) must be at least "
                f"{thresholds.min_triangle_ratio:.4f}"
            ),
        ),
        "semantic_part_coverage": _hard_gate_result(
            value=semantic_coverage,
            threshold=thresholds.min_semantic_part_coverage,
            operator=">=",
            passed=coverage_passed,
            message=(
                f"semantic object coverage {len(covered_ids)}/{len(parts)} "
                f"({semantic_coverage:.4f}) must be at least "
                f"{thresholds.min_semantic_part_coverage:.4f}"
            ),
        ),
    }
    _merge_gate_results(hard_gates, thresholds=thresholds, results=results)
    return {
        "reference_triangles": reference_triangles,
        "candidate_triangles": candidate_triangles,
        "triangle_ratio": triangle_ratio,
        "covered_part_ids": covered_ids,
        "missing_part_ids": missing_ids,
        "semantic_part_coverage": semantic_coverage,
    }


def _pair_key(first: str, second: str) -> tuple[str, str]:
    return tuple(sorted((first, second)))


def _reported_pairs(records: Any) -> set[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    if not isinstance(records, list):
        return result
    for record in records:
        objects = record.get("objects") if isinstance(record, dict) else None
        if (
            isinstance(objects, list)
            and len(objects) == 2
            and all(isinstance(item, str) for item in objects)
        ):
            result.add(_pair_key(objects[0], objects[1]))
    return result


def _attachment_bounds_measurement(
    first: dict[str, list[float]],
    second: dict[str, list[float]],
    contact_region: dict[str, list[float]],
) -> dict[str, float]:
    gaps = [
        max(
            0.0,
            first["min"][axis] - second["max"][axis],
            second["min"][axis] - first["max"][axis],
        )
        for axis in range(3)
    ]
    overlap = [
        max(
            0.0,
            min(first["max"][axis], second["max"][axis])
            - max(first["min"][axis], second["min"][axis]),
        )
        for axis in range(3)
    ]
    contact_min = [
        max(first["min"][axis], second["min"][axis]) for axis in range(3)
    ]
    contact_max = [
        min(first["max"][axis], second["max"][axis]) for axis in range(3)
    ]
    for axis in range(3):
        if contact_min[axis] > contact_max[axis]:
            midpoint = (contact_min[axis] + contact_max[axis]) * 0.5
            contact_min[axis] = midpoint
            contact_max[axis] = midpoint
    region_gaps = [
        max(
            0.0,
            contact_min[axis] - contact_region["max"][axis],
            contact_region["min"][axis] - contact_max[axis],
        )
        for axis in range(3)
    ]
    positive_overlap = sorted(overlap, reverse=True)
    return {
        "gap": math.sqrt(math.fsum(value * value for value in gaps)),
        "penetration": min(overlap) if all(value > 0.0 for value in overlap) else 0.0,
        "contact_area_proxy": positive_overlap[0] * positive_overlap[1],
        "contact_region_distance": math.sqrt(
            math.fsum(value * value for value in region_gaps)
        ),
    }


def _structural_measurements(
    candidate_scene: dict[str, Any],
    plan: dict[str, Any],
) -> dict[str, Any]:
    objects = candidate_scene.get("objects")
    structure = candidate_scene.get("structure")
    if not isinstance(objects, list) or not isinstance(structure, dict):
        raise ValueError("candidate scene report lacks structural diagnostics")
    topology_totals = {
        "edges": 0,
        "boundary_edges": 0,
        "non_manifold_edges": 0,
        "manifold_edges": 0,
        "inconsistent_winding_edges": 0,
    }
    total_triangles = 0
    degenerate_triangles = 0
    low_quality_triangles = 0
    inverted_objects = 0
    self_intersection_pairs = 0
    actual_names: list[str] = []
    actual_bounds: dict[str, dict[str, list[float]]] = {}
    for item in objects:
        if not isinstance(item, dict):
            continue
        if isinstance(item.get("name"), str):
            object_name = item["name"]
            actual_names.append(object_name)
            bounds = item.get("bounds")
            if isinstance(bounds, dict):
                minimum = bounds.get("min")
                maximum = bounds.get("max")
                if (
                    isinstance(minimum, list)
                    and isinstance(maximum, list)
                    and len(minimum) == len(maximum) == 3
                    and all(
                        isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and math.isfinite(value)
                        for value in (*minimum, *maximum)
                    )
                    and all(low <= high for low, high in zip(minimum, maximum, strict=True))
                ):
                    actual_bounds[object_name] = {
                        "min": [float(value) for value in minimum],
                        "max": [float(value) for value in maximum],
                    }
        object_structure = item.get("structure")
        if not isinstance(object_structure, dict):
            raise ValueError("candidate object lacks structure diagnostics")
        topology = object_structure.get("topology")
        quality = object_structure.get("triangle_quality")
        normals = object_structure.get("normal_consistency")
        intersection = object_structure.get("self_intersection_proxy")
        if not all(isinstance(value, dict) for value in (topology, quality, normals, intersection)):
            raise ValueError("candidate object has incomplete structural diagnostics")
        for name in topology_totals:
            count = topology.get(name)
            if type(count) is not int or count < 0:
                raise ValueError(f"candidate topology.{name} must be non-negative")
            topology_totals[name] += count
        triangles = item.get("triangles")
        if type(triangles) is not int or triangles < 0:
            raise ValueError("candidate object triangles must be non-negative")
        total_triangles += triangles
        degenerate = quality.get("degenerate_triangles")
        low_quality = quality.get("low_quality_below_0_05")
        if type(degenerate) is not int or type(low_quality) is not int:
            raise ValueError("candidate triangle-quality counts must be integers")
        degenerate_triangles += degenerate
        low_quality_triangles += low_quality
        if normals.get("outward_orientation_proxy") is False:
            inverted_objects += 1
        pairs = intersection.get("triangle_pairs")
        if type(pairs) is int and pairs > 0:
            self_intersection_pairs += pairs

    contact = structure.get("contact_intersection_proxy")
    if not isinstance(contact, dict):
        raise ValueError("candidate scene lacks contact/intersection diagnostics")
    isolated = contact.get("isolated_objects")
    if not isinstance(isolated, list):
        raise ValueError("candidate scene isolated_objects must be an array")
    near_pairs = _reported_pairs(contact.get("near_contact_pairs"))
    intersection_pairs = _reported_pairs(contact.get("triangle_intersection_pairs"))
    touching_pairs = near_pairs | intersection_pairs

    parts = plan.get("parts")
    if not isinstance(parts, list):
        raise ValueError("normalized plan parts must be an array")
    part_by_id = {
        str(part.get("id")): part for part in parts if isinstance(part, dict)
    }
    resolved_objects = {
        identifier: _matching_object_names(part.get("object_names"), actual_names)
        for identifier, part in part_by_id.items()
    }
    declared_isolated_object_names = {
        name
        for identifier, part in part_by_id.items()
        if isinstance(part.get("attachment"), dict)
        and part["attachment"].get("type") in {"root", "intentional-gap"}
        for name in resolved_objects.get(identifier, [])
    }
    loose_isolated = [
        name for name in isolated if name not in declared_isolated_object_names
    ]
    attachments_checked = 0
    unjoined = []
    gap_violations = []
    penetration_violations = []
    contact_area_violations = []
    contact_region_violations = []
    allowed_intersections: set[tuple[str, str]] = set()
    for identifier, part in part_by_id.items():
        attachment = part.get("attachment")
        if not isinstance(attachment, dict) or attachment.get("type") == "root":
            continue
        attachments_checked += 1
        parent_id = str(attachment.get("parent_id"))
        child_names = resolved_objects.get(identifier, [])
        parent_names = resolved_objects.get(parent_id, [])
        possible_pairs = {
            _pair_key(child, parent)
            for child in child_names
            for parent in parent_names
            if child != parent
        }
        kind = attachment.get("type")
        contact_region = attachment.get("contact_region")
        max_gap = float(attachment.get("max_gap", 0.0))
        max_penetration = float(attachment.get("max_penetration", 0.0))
        min_contact_area = float(attachment.get("min_contact_area", 0.0))
        pair_measurements = []
        if isinstance(contact_region, dict):
            for first, second in possible_pairs:
                if first not in actual_bounds or second not in actual_bounds:
                    continue
                measurement = _attachment_bounds_measurement(
                    actual_bounds[first], actual_bounds[second], contact_region
                )
                pair = _pair_key(first, second)
                contact_required = kind != "intentional-gap"
                contact_ok = not contact_required or pair in touching_pairs
                intersection_required = kind in {"fused", "embedded"}
                if intersection_required:
                    contact_ok = pair in intersection_pairs
                pair_measurements.append(
                    {
                        **measurement,
                        "pair": pair,
                        "gap_ok": measurement["gap"] <= max_gap and contact_ok,
                        "penetration_ok": (
                            measurement["penetration"] <= max_penetration
                        ),
                        "contact_area_ok": (
                            measurement["contact_area_proxy"] >= min_contact_area
                        ),
                        "contact_region_ok": (
                            measurement["contact_region_distance"] <= max_gap
                        ),
                    }
                )
        allowed_intersections.update(
            item["pair"]
            for item in pair_measurements
            if item["pair"] in intersection_pairs and item["penetration_ok"]
        )
        gap_ok = any(item["gap_ok"] for item in pair_measurements)
        penetration_ok = any(item["penetration_ok"] for item in pair_measurements)
        contact_area_ok = any(item["contact_area_ok"] for item in pair_measurements)
        contact_region_ok = any(item["contact_region_ok"] for item in pair_measurements)
        joined = any(
            item["gap_ok"]
            and item["penetration_ok"]
            and item["contact_area_ok"]
            and item["contact_region_ok"]
            for item in pair_measurements
        )
        if not gap_ok:
            gap_violations.append(identifier)
        if not penetration_ok:
            penetration_violations.append(identifier)
        if not contact_area_ok:
            contact_area_violations.append(identifier)
        if not contact_region_ok:
            contact_region_violations.append(identifier)
        if not joined:
            unjoined.append(identifier)

    unexpected_intersections = intersection_pairs - allowed_intersections
    edges = topology_totals["edges"]
    manifold_edges = topology_totals["manifold_edges"]
    object_count = len(actual_names)
    broad_phase_pairs = contact.get("broad_phase_pairs")
    if type(broad_phase_pairs) is not int or broad_phase_pairs < 0:
        raise ValueError("candidate scene broad_phase_pairs must be non-negative")
    cross_intersection_fraction = len(unexpected_intersections) / max(
        1, broad_phase_pairs, len(intersection_pairs)
    )
    self_intersection_fraction = self_intersection_pairs / max(1, total_triangles)
    return {
        "loose_component_fraction": len(loose_isolated) / max(1, object_count),
        "boundary_edge_fraction": topology_totals["boundary_edges"] / max(1, edges),
        "non_manifold_edge_fraction": topology_totals["non_manifold_edges"] / max(1, edges),
        "inverted_normal_fraction": max(
            topology_totals["inconsistent_winding_edges"] / max(1, manifold_edges),
            inverted_objects / max(1, object_count),
        ),
        "degenerate_triangle_fraction": degenerate_triangles / max(1, total_triangles),
        "low_quality_triangle_fraction": low_quality_triangles / max(1, total_triangles),
        "unjoined_attachment_fraction": len(unjoined) / max(1, attachments_checked),
        "attachment_gap_violation_fraction": len(gap_violations)
        / max(1, attachments_checked),
        "attachment_penetration_violation_fraction": len(penetration_violations)
        / max(1, attachments_checked),
        "attachment_contact_area_violation_fraction": len(contact_area_violations)
        / max(1, attachments_checked),
        "attachment_contact_region_violation_fraction": len(contact_region_violations)
        / max(1, attachments_checked),
        "intersection_fraction": max(
            cross_intersection_fraction, self_intersection_fraction
        ),
        "isolated_objects": sorted(str(item) for item in loose_isolated),
        "unjoined_part_ids": unjoined,
        "attachment_gap_violation_part_ids": gap_violations,
        "attachment_penetration_violation_part_ids": penetration_violations,
        "attachment_contact_area_violation_part_ids": contact_area_violations,
        "attachment_contact_region_violation_part_ids": contact_region_violations,
        "unexpected_intersection_pairs": [
            list(pair) for pair in sorted(unexpected_intersections)
        ],
        "attachments_checked": attachments_checked,
        "total_edges": edges,
        "total_triangles": total_triangles,
    }


def _add_structural_hard_gates(
    hard_gates: dict[str, Any],
    *,
    candidate_scene: dict[str, Any],
    plan: dict[str, Any],
    thresholds: StructuralGateThresholds,
) -> dict[str, Any]:
    diagnostics = _structural_measurements(candidate_scene, plan)
    fields = (
        ("loose_component_fraction", "max_loose_component_fraction"),
        ("boundary_edge_fraction", "max_boundary_edge_fraction"),
        ("non_manifold_edge_fraction", "max_non_manifold_edge_fraction"),
        ("inverted_normal_fraction", "max_inverted_normal_fraction"),
        ("degenerate_triangle_fraction", "max_degenerate_triangle_fraction"),
        ("low_quality_triangle_fraction", "max_low_quality_triangle_fraction"),
        ("unjoined_attachment_fraction", "max_unjoined_attachment_fraction"),
        ("attachment_gap_violation_fraction", "max_unjoined_attachment_fraction"),
        (
            "attachment_penetration_violation_fraction",
            "max_unjoined_attachment_fraction",
        ),
        (
            "attachment_contact_area_violation_fraction",
            "max_unjoined_attachment_fraction",
        ),
        (
            "attachment_contact_region_violation_fraction",
            "max_unjoined_attachment_fraction",
        ),
        ("intersection_fraction", "max_intersection_fraction"),
    )
    results = {}
    for metric_name, threshold_name in fields:
        value = float(diagnostics[metric_name])
        threshold = float(getattr(thresholds, threshold_name))
        results[metric_name] = _hard_gate_result(
            value=value,
            threshold=threshold,
            operator="<=",
            passed=value <= threshold,
            message=f"{metric_name} {value:.6f} must be at most {threshold:.6f}",
        )
    _merge_gate_results(hard_gates, thresholds=thresholds, results=results)
    return diagnostics


def compare_workspace(
    *,
    reference_masks: Path,
    candidate_masks: Path,
    reference_scene: Path,
    candidate_scene: Path,
    output: Path,
    min_score: float,
    gate_thresholds: FidelityGateThresholds | None = None,
    surface_comparison: Path | None = None,
    surface_gate_thresholds: SurfaceGateThresholds | None = None,
    reference_probe: Path | None = None,
    candidate_probe: Path | None = None,
    plan: Path | None = None,
    material_gate_thresholds: MaterialGateThresholds | None = None,
    detail_gate_thresholds: DetailGateThresholds | None = None,
    structural_gate_thresholds: StructuralGateThresholds | None = None,
) -> dict[str, Any]:
    min_score = _score_threshold(min_score)
    if gate_thresholds is None:
        gate_thresholds = FidelityGateThresholds()
    elif not isinstance(gate_thresholds, FidelityGateThresholds):
        raise TypeError("gate_thresholds must be a FidelityGateThresholds instance")
    if surface_comparison is None:
        if surface_gate_thresholds is not None:
            raise ValueError(
                "surface_gate_thresholds requires a surface_comparison path"
            )
        surface = None
    else:
        if surface_gate_thresholds is None:
            surface_gate_thresholds = SurfaceGateThresholds()
        elif not isinstance(surface_gate_thresholds, SurfaceGateThresholds):
            raise TypeError(
                "surface_gate_thresholds must be a SurfaceGateThresholds instance"
            )
        surface = _surface_comparison(surface_comparison)
    if material_gate_thresholds is not None and not isinstance(
        material_gate_thresholds, MaterialGateThresholds
    ):
        raise TypeError(
            "material_gate_thresholds must be a MaterialGateThresholds instance"
        )
    if detail_gate_thresholds is not None and not isinstance(
        detail_gate_thresholds, DetailGateThresholds
    ):
        raise TypeError("detail_gate_thresholds must be a DetailGateThresholds instance")
    if structural_gate_thresholds is not None and not isinstance(
        structural_gate_thresholds, StructuralGateThresholds
    ):
        raise TypeError(
            "structural_gate_thresholds must be a StructuralGateThresholds instance"
        )
    if material_gate_thresholds is not None and candidate_probe is None:
        raise ValueError("material_gate_thresholds requires a candidate_probe path")
    if detail_gate_thresholds is not None and (
        reference_probe is None or candidate_probe is None or plan is None
    ):
        raise ValueError(
            "detail_gate_thresholds requires reference_probe, candidate_probe, and plan paths"
        )
    if structural_gate_thresholds is not None and plan is None:
        raise ValueError("structural_gate_thresholds requires a plan path")
    reference_mask_data = _canonical_views(reference_masks, label="reference")
    candidate_mask_data = _canonical_views(candidate_masks, label="candidate")
    views = {
        name: mask_metrics(reference_mask_data[name], candidate_mask_data[name])
        for name in CANONICAL_VIEW_NAMES
    }
    mean_iou = _bounded_score(
        "mean silhouette IoU", sum(value["iou"] for value in views.values()) / len(views)
    )
    mean_area = _bounded_score(
        "mean area similarity",
        sum(value["area_similarity"] for value in views.values()) / len(views),
    )
    mean_rgb = _bounded_score(
        "mean spatial RGB similarity",
        sum(value["spatial_rgb_similarity"] for value in views.values()) / len(views),
    )
    mean_palette = _bounded_score(
        "mean palette similarity",
        sum(value["palette_similarity"] for value in views.values()) / len(views),
    )

    reference_geometry = _scene_bounds(reference_scene, label="reference")
    candidate_geometry = _scene_bounds(candidate_scene, label="candidate")
    candidate_scene_document = _read_report(candidate_scene, label="candidate scene report")
    dimension_similarity = _vector_similarity(reference_geometry["dimensions"], candidate_geometry["dimensions"])
    center_distance = math.dist(reference_geometry["center"], candidate_geometry["center"])
    if not math.isfinite(center_distance):
        raise ValueError("reference and candidate center distance is not finite")
    center_similarity = _bounded_score("center similarity", math.exp(-center_distance / 0.35))
    score = _bounded_score(
        "fidelity score",
        _SCORE_WEIGHTS["silhouette_iou"] * mean_iou
        + _SCORE_WEIGHTS["area_similarity"] * mean_area
        + _SCORE_WEIGHTS["spatial_rgb_similarity"] * mean_rgb
        + _SCORE_WEIGHTS["dimension_similarity"] * dimension_similarity
        + _SCORE_WEIGHTS["center_similarity"] * center_similarity,
    )
    hard_gates = _evaluate_hard_gates(
        views=views,
        mean_iou=mean_iou,
        mean_rgb=mean_rgb,
        mean_palette=mean_palette,
        center_distance=center_distance,
        reference_geometry=reference_geometry,
        candidate_geometry=candidate_geometry,
        thresholds=gate_thresholds,
    )
    if surface is not None:
        assert surface_gate_thresholds is not None
        _add_surface_hard_gates(
            hard_gates,
            surface=surface,
            thresholds=surface_gate_thresholds,
        )
    material_diagnostics = None
    detail_diagnostics = None
    structural_diagnostics = None
    candidate_probe_document = None
    normalized_plan = None
    if candidate_probe is not None and (
        material_gate_thresholds is not None or detail_gate_thresholds is not None
    ):
        candidate_probe_document = _read_report(
            candidate_probe, label="candidate GLB probe"
        )
    if plan is not None and (
        detail_gate_thresholds is not None or structural_gate_thresholds is not None
    ):
        normalized_plan = validate_plan_document(
            _read_report(plan, label="reconstruction plan")
        )
    if material_gate_thresholds is not None:
        assert candidate_probe_document is not None
        material_diagnostics = _add_material_hard_gates(
            hard_gates,
            candidate_probe=candidate_probe_document,
            thresholds=material_gate_thresholds,
        )
    if detail_gate_thresholds is not None:
        assert reference_probe is not None
        assert candidate_probe_document is not None
        assert normalized_plan is not None
        detail_diagnostics = _add_detail_hard_gates(
            hard_gates,
            reference_probe=_read_report(reference_probe, label="reference GLB probe"),
            candidate_probe=candidate_probe_document,
            candidate_scene=candidate_scene_document,
            plan=normalized_plan,
            thresholds=detail_gate_thresholds,
        )
    if structural_gate_thresholds is not None:
        assert normalized_plan is not None
        structural_diagnostics = _add_structural_hard_gates(
            hard_gates,
            candidate_scene=candidate_scene_document,
            plan=normalized_plan,
            thresholds=structural_gate_thresholds,
        )
    score_passed = score >= min_score
    report = {
        "schema_version": 2,
        "score": score,
        "score_weights": dict(_SCORE_WEIGHTS),
        "min_score": min_score,
        "score_passed": score_passed,
        "passed": score_passed and hard_gates["passed"],
        "hard_gates": hard_gates,
        "summary": {
            "mean_silhouette_iou": mean_iou,
            "mean_area_similarity": mean_area,
            "mean_spatial_rgb_similarity": mean_rgb,
            "mean_palette_similarity": mean_palette,
            "dimension_similarity": dimension_similarity,
            "center_similarity": center_similarity,
            "center_distance": center_distance,
        },
        "views": views,
        "reference_bounds": reference_geometry,
        "candidate_bounds": candidate_geometry,
    }
    if surface is not None:
        report["summary"]["mean_surface_distance"] = surface["symmetric"]["mean"]
        report["summary"]["p95_surface_distance"] = surface["symmetric"]["p95"]
        for gate_name, summary_name in (
            ("mean_normal_angle_degrees", "mean_normal_angle_degrees"),
            ("visible_surface_coverage", "visible_surface_coverage"),
            ("minimum_surface_area_ratio", "surface_area_ratio"),
            ("maximum_surface_area_ratio", "surface_area_ratio"),
        ):
            gate = hard_gates["results"].get(gate_name)
            if isinstance(gate, dict):
                report["summary"][summary_name] = gate["value"]
        report["surface_comparison"] = surface
    if material_diagnostics is not None:
        report["material_diagnostics"] = material_diagnostics
    if detail_diagnostics is not None:
        report["detail_diagnostics"] = detail_diagnostics
    if structural_diagnostics is not None:
        report["structural_diagnostics"] = structural_diagnostics
    write_json(output, report)
    return report
