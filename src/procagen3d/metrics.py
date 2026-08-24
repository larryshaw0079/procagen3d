"""Deterministic comparison of canonical reference and generated renders."""

from __future__ import annotations

import base64
import binascii
import json
import math
from pathlib import Path
from typing import Any

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
    ref_bounds = [width, height, -1, -1]
    candidate_bounds = [width, height, -1, -1]
    for index in range(width * height):
        ref = _get_bit(reference_data, index)
        cand = _get_bit(candidate_data, index)
        if ref:
            reference_count += 1
            x, y = index % width, index // width
            ref_bounds = [min(ref_bounds[0], x), min(ref_bounds[1], y), max(ref_bounds[2], x), max(ref_bounds[3], y)]
        if cand:
            candidate_count += 1
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
    area_similarity = (
        min(reference_count, candidate_count) / max(reference_count, candidate_count)
        if max(reference_count, candidate_count)
        else 1.0
    )
    return {
        "iou": iou,
        "area_similarity": area_similarity,
        "spatial_rgb_similarity": spatial_rgb_similarity,
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


def compare_workspace(
    *,
    reference_masks: Path,
    candidate_masks: Path,
    reference_scene: Path,
    candidate_scene: Path,
    output: Path,
    min_score: float,
) -> dict[str, Any]:
    min_score = _score_threshold(min_score)
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

    reference_geometry = _scene_bounds(reference_scene, label="reference")
    candidate_geometry = _scene_bounds(candidate_scene, label="candidate")
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
    report = {
        "schema_version": 2,
        "score": score,
        "score_weights": dict(_SCORE_WEIGHTS),
        "min_score": min_score,
        "passed": score >= min_score,
        "summary": {
            "mean_silhouette_iou": mean_iou,
            "mean_area_similarity": mean_area,
            "mean_spatial_rgb_similarity": mean_rgb,
            "dimension_similarity": dimension_similarity,
            "center_similarity": center_similarity,
            "center_distance": center_distance,
        },
        "views": views,
        "reference_bounds": reference_geometry,
        "candidate_bounds": candidate_geometry,
    }
    write_json(output, report)
    return report
