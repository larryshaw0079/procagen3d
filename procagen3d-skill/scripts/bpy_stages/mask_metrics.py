"""Deterministic, local mask metrics for registered image fitting.

The Blender fit stage supplies a capped-resolution binary grid.  This module
stays stdlib-only so its geometry and threshold policy can be tested without
Blender or NumPy.
"""

from __future__ import annotations

import math


MASK_METRIC_SUITE_VERSION = 1
MASK_GATE_IDS = frozenset({
    "mask_iou",
    "mask_precision",
    "mask_recall",
    "mask_boundary_f1",
    "mask_boundary_chamfer",
    "mask_boundary_p95",
    "mask_regional_iou_mean",
    "mask_regional_iou_p10",
    "mask_regional_occupancy",
    "mask_bbox",
    "mask_centroid",
    "mask_area_ratio",
})

# Defaults are the normal contract for a clean, registered subject mask.
DEFAULTS = {
    "resolution": 256,
    "grid_size": 4,
    "min_region_coverage": 0.005,
    "boundary_tolerance_uv": 0.012,
    "min_iou": 0.75,
    "min_precision": 0.80,
    "min_recall": 0.80,
    "min_boundary_f1": 0.75,
    "max_boundary_chamfer": 0.015,
    "max_boundary_p95": 0.040,
    "min_regional_iou_mean": 0.58,
    "min_regional_iou_p10": 0.25,
    "max_regional_occupancy_error": 0.18,
    "max_bbox_error": 0.04,
    "max_centroid_error": 0.03,
    "max_area_ratio_error": 0.20,
}

# Gate-mode overrides may tighten the contract, but cannot make it weaker than
# this safety envelope.  Low-confidence masks use mode=diagnostic instead of
# laundering uncertainty through permissive numbers.
MINIMUM_SCORES = {
    "min_iou": 0.70,
    "min_precision": 0.75,
    "min_recall": 0.75,
    "min_boundary_f1": 0.65,
    "min_regional_iou_mean": 0.48,
    "min_regional_iou_p10": 0.10,
}
MAXIMUM_ERRORS = {
    "boundary_tolerance_uv": 0.020,
    "max_boundary_chamfer": 0.025,
    "max_boundary_p95": 0.060,
    "max_regional_occupancy_error": 0.30,
    "max_bbox_error": 0.06,
    "max_centroid_error": 0.05,
    "max_area_ratio_error": 0.30,
}
MINIMUM_SAMPLING = {
    "resolution": 192,
    "grid_size": 4,
}
MAXIMUM_SAMPLING = {
    "min_region_coverage": 0.010,
}

GLOBAL_THRESHOLD_ALIASES = {
    "min_iou": "mask_min_iou",
    "min_precision": "mask_min_precision",
    "min_recall": "mask_min_recall",
    "min_boundary_f1": "mask_min_boundary_f1",
    "max_boundary_chamfer": "mask_max_boundary_chamfer",
    "max_boundary_p95": "mask_max_boundary_p95",
    "min_regional_iou_mean": "mask_min_regional_iou_mean",
    "min_regional_iou_p10": "mask_min_regional_iou_p10",
    "max_regional_occupancy_error": "mask_max_regional_occupancy_error",
    "max_bbox_error": "bbox_max_error",
    "max_centroid_error": "centroid_max_error",
    "max_area_ratio_error": "area_ratio_max_error",
}


def _finite_number(value, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _integer(value, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{label} must be within [{minimum}, {maximum}]")
    return value


def resolve_metric_config(mask_config: dict, thresholds: dict | None = None) -> dict:
    """Resolve legacy/new settings and enforce the non-permissive gate floor."""

    if not isinstance(mask_config, dict):
        raise ValueError("mask must be an object")
    if thresholds is None:
        thresholds = {}
    if not isinstance(thresholds, dict):
        raise ValueError("thresholds must be an object")
    nested = mask_config.get("metrics", {})
    if not isinstance(nested, dict):
        raise ValueError("mask.metrics must be an object")
    mode = str(mask_config.get("mode", "gate")).lower()
    if mode not in {"gate", "diagnostic"}:
        raise ValueError("mask.mode must be gate or diagnostic")

    requested = {}
    for name, default in DEFAULTS.items():
        global_alias = GLOBAL_THRESHOLD_ALIASES.get(name)
        raw = nested.get(
            name,
            mask_config.get(
                name,
                thresholds.get(global_alias, default) if global_alias else default,
            ),
        )
        if name == "resolution":
            requested[name] = _integer(raw, "mask.metrics.resolution", 64, 512)
        elif name == "grid_size":
            requested[name] = _integer(raw, "mask.metrics.grid_size", 2, 16)
        else:
            requested[name] = _finite_number(raw, f"mask.metrics.{name}")

    unit_interval = {
        "min_region_coverage",
        "boundary_tolerance_uv",
        "min_iou",
        "min_precision",
        "min_recall",
        "min_boundary_f1",
        "max_boundary_chamfer",
        "max_boundary_p95",
        "min_regional_iou_mean",
        "min_regional_iou_p10",
        "max_regional_occupancy_error",
        "max_bbox_error",
        "max_centroid_error",
    }
    for name in unit_interval:
        if not 0.0 <= requested[name] <= 1.0:
            raise ValueError(f"mask.metrics.{name} must be within [0, 1]")
    if not 0.0 < requested["min_region_coverage"]:
        raise ValueError("mask.metrics.min_region_coverage must be positive")
    if not 0.0 < requested["boundary_tolerance_uv"]:
        raise ValueError("mask.metrics.boundary_tolerance_uv must be positive")
    if not 0.0 <= requested["max_area_ratio_error"] <= 2.0:
        raise ValueError(
            "mask.metrics.max_area_ratio_error must be within [0, 2]")

    effective = dict(requested)
    adjustments = []
    if mode == "gate":
        for name, floor in MINIMUM_SAMPLING.items():
            if effective[name] < floor:
                adjustments.append({
                    "field": name,
                    "requested": effective[name],
                    "effective": floor,
                    "reason": "gate sampling floor",
                })
                effective[name] = floor
        for name, ceiling in MAXIMUM_SAMPLING.items():
            if effective[name] > ceiling:
                adjustments.append({
                    "field": name,
                    "requested": effective[name],
                    "effective": ceiling,
                    "reason": "gate sampling ceiling",
                })
                effective[name] = ceiling
        for name, floor in MINIMUM_SCORES.items():
            if effective[name] < floor:
                adjustments.append({
                    "field": name,
                    "requested": effective[name],
                    "effective": floor,
                    "reason": "gate safety floor",
                })
                effective[name] = floor
        for name, ceiling in MAXIMUM_ERRORS.items():
            if effective[name] > ceiling:
                adjustments.append({
                    "field": name,
                    "requested": effective[name],
                    "effective": ceiling,
                    "reason": "gate safety ceiling",
                })
                effective[name] = ceiling

    return {
        "version": MASK_METRIC_SUITE_VERSION,
        "mode": mode,
        "requested": requested,
        "effective": effective,
        "adjustments": adjustments,
    }


def _boundary(mask: bytes, width: int, height: int) -> list[int]:
    result = []
    for y in range(height):
        row = y * width
        for x in range(width):
            index = row + x
            if not mask[index]:
                continue
            if (x == 0 or x == width - 1 or y == 0 or y == height - 1
                    or not mask[index - 1] or not mask[index + 1]
                    or not mask[index - width] or not mask[index + width]):
                result.append(index)
    return result


def _distance_field(
    boundary: list[int],
    width: int,
    height: int,
) -> list[float]:
    """Two-pass chamfer field in normalized image coordinates."""

    infinity = 4.0
    distance = [infinity] * (width * height)
    for index in boundary:
        distance[index] = 0.0
    dx = 1.0 / width
    dy = 1.0 / height
    diagonal = math.hypot(dx, dy)

    for y in range(height):
        row = y * width
        for x in range(width):
            index = row + x
            best = distance[index]
            if x:
                best = min(best, distance[index - 1] + dx)
            if y:
                best = min(best, distance[index - width] + dy)
                if x:
                    best = min(best, distance[index - width - 1] + diagonal)
                if x + 1 < width:
                    best = min(best, distance[index - width + 1] + diagonal)
            distance[index] = best

    for y in range(height - 1, -1, -1):
        row = y * width
        for x in range(width - 1, -1, -1):
            index = row + x
            best = distance[index]
            if x + 1 < width:
                best = min(best, distance[index + 1] + dx)
            if y + 1 < height:
                best = min(best, distance[index + width] + dy)
                if x:
                    best = min(best, distance[index + width - 1] + diagonal)
                if x + 1 < width:
                    best = min(best, distance[index + width + 1] + diagonal)
            distance[index] = best
    return distance


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _regions(
    reference: bytes,
    render: bytes,
    width: int,
    height: int,
    grid_size: int,
    minimum_coverage: float,
) -> tuple[list[dict], list[dict]]:
    all_regions = []
    active_regions = []
    for row_index in range(grid_size):
        top = row_index * height // grid_size
        bottom = (row_index + 1) * height // grid_size
        for column_index in range(grid_size):
            left = column_index * width // grid_size
            right = (column_index + 1) * width // grid_size
            area = max(1, (right - left) * (bottom - top))
            reference_count = 0
            render_count = 0
            intersection = 0
            union = 0
            for y in range(top, bottom):
                offset = y * width
                for x in range(left, right):
                    index = offset + x
                    ref_value = bool(reference[index])
                    render_value = bool(render[index])
                    reference_count += ref_value
                    render_count += render_value
                    intersection += ref_value and render_value
                    union += ref_value or render_value
            reference_fraction = reference_count / area
            render_fraction = render_count / area
            record = {
                "row": row_index,
                "column": column_index,
                "reference_fraction": reference_fraction,
                "render_fraction": render_fraction,
                "occupancy_error": abs(reference_fraction - render_fraction),
                "iou": intersection / union if union else 1.0,
                "active": max(reference_fraction, render_fraction) >= minimum_coverage,
            }
            all_regions.append(record)
            if record["active"]:
                active_regions.append(record)
    return all_regions, active_regions


def _component_count(mask: bytes, width: int, height: int) -> int:
    visited = bytearray(width * height)
    components = 0
    for start, value in enumerate(mask):
        if not value or visited[start]:
            continue
        components += 1
        visited[start] = 1
        stack = [start]
        while stack:
            index = stack.pop()
            y, x = divmod(index, width)
            for ny in range(max(0, y - 1), min(height, y + 2)):
                row = ny * width
                for nx in range(max(0, x - 1), min(width, x + 2)):
                    neighbor = row + nx
                    if mask[neighbor] and not visited[neighbor]:
                        visited[neighbor] = 1
                        stack.append(neighbor)
    return components


def _hole_count(mask: bytes, width: int, height: int) -> int:
    visited = bytearray(width * height)
    holes = 0
    for start, value in enumerate(mask):
        if value or visited[start]:
            continue
        visited[start] = 1
        stack = [start]
        touches_border = False
        while stack:
            index = stack.pop()
            y, x = divmod(index, width)
            touches_border = touches_border or (
                x == 0 or y == 0 or x == width - 1 or y == height - 1)
            neighbors = []
            if x:
                neighbors.append(index - 1)
            if x + 1 < width:
                neighbors.append(index + 1)
            if y:
                neighbors.append(index - width)
            if y + 1 < height:
                neighbors.append(index + width)
            for neighbor in neighbors:
                if not mask[neighbor] and not visited[neighbor]:
                    visited[neighbor] = 1
                    stack.append(neighbor)
        if not touches_border:
            holes += 1
    return holes


def measure_grid_suite(
    reference: bytes,
    render: bytes,
    width: int,
    height: int,
    *,
    boundary_tolerance_uv: float,
    grid_size: int,
    min_region_coverage: float,
) -> dict:
    """Measure contour, regional, and topology signals on equal binary grids."""

    expected = width * height
    if width < 2 or height < 2 or len(reference) != expected or len(render) != expected:
        raise ValueError("mask metric grids have invalid dimensions")
    reference_boundary = _boundary(reference, width, height)
    render_boundary = _boundary(render, width, height)
    if not reference_boundary or not render_boundary:
        raise ValueError("mask metric grids require non-empty boundaries")
    reference_distance = _distance_field(reference_boundary, width, height)
    render_distance = _distance_field(render_boundary, width, height)
    render_to_reference = [reference_distance[index] for index in render_boundary]
    reference_to_render = [render_distance[index] for index in reference_boundary]

    boundary_precision = sum(
        distance <= boundary_tolerance_uv for distance in render_to_reference
    ) / len(render_to_reference)
    boundary_recall = sum(
        distance <= boundary_tolerance_uv for distance in reference_to_render
    ) / len(reference_to_render)
    denominator = boundary_precision + boundary_recall
    boundary_f1 = (
        2.0 * boundary_precision * boundary_recall / denominator
        if denominator else 0.0
    )
    chamfer = 0.5 * (
        sum(render_to_reference) / len(render_to_reference)
        + sum(reference_to_render) / len(reference_to_render)
    )
    boundary_p95 = max(
        _percentile(render_to_reference, 0.95),
        _percentile(reference_to_render, 0.95),
    )

    regions, active_regions = _regions(
        reference,
        render,
        width,
        height,
        grid_size,
        min_region_coverage,
    )
    if not active_regions:
        raise ValueError("mask metric grid has no active regions")
    regional_ious = [record["iou"] for record in active_regions]
    regional_occupancy = [record["occupancy_error"] for record in active_regions]
    worst_regions = sorted(
        active_regions,
        key=lambda record: (record["iou"], -record["occupancy_error"]),
    )[:3]

    reference_components = _component_count(reference, width, height)
    render_components = _component_count(render, width, height)
    reference_holes = _hole_count(reference, width, height)
    render_holes = _hole_count(render, width, height)
    return {
        "resolution_px": [width, height],
        "boundary": {
            "reference_pixels": len(reference_boundary),
            "render_pixels": len(render_boundary),
            "precision": boundary_precision,
            "recall": boundary_recall,
            "f1": boundary_f1,
            "chamfer_uv": chamfer,
            "p95_uv": boundary_p95,
            "render_to_reference_mean_uv": (
                sum(render_to_reference) / len(render_to_reference)),
            "reference_to_render_mean_uv": (
                sum(reference_to_render) / len(reference_to_render)),
        },
        "regional": {
            "grid_size": grid_size,
            "active_regions": len(active_regions),
            "iou_mean": sum(regional_ious) / len(regional_ious),
            "iou_p10": _percentile(regional_ious, 0.10),
            "occupancy_error_mean": (
                sum(regional_occupancy) / len(regional_occupancy)),
            "occupancy_error_max": max(regional_occupancy),
            "regions": regions,
            "worst_regions": worst_regions,
        },
        "topology": {
            "reference_components": reference_components,
            "render_components": render_components,
            "component_count_delta": abs(render_components - reference_components),
            "reference_holes": reference_holes,
            "render_holes": render_holes,
            "hole_count_delta": abs(render_holes - reference_holes),
        },
    }
