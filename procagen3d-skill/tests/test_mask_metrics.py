"""Regression coverage for the registered mask metric suite."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = SKILL_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from bpy_stages.mask_metrics import (  # noqa: E402
    DEFAULTS,
    MAXIMUM_ERRORS,
    MAXIMUM_SAMPLING,
    MINIMUM_SCORES,
    MINIMUM_SAMPLING,
    measure_grid_suite,
    resolve_metric_config,
)


def mask(width, height, rectangles):
    values = bytearray(width * height)
    for left, top, right, bottom in rectangles:
        for y in range(top, bottom):
            start = y * width + left
            values[start:start + right - left] = b"\x01" * (right - left)
    return bytes(values)


def measure(reference, render, width=64, height=64):
    return measure_grid_suite(
        reference,
        render,
        width,
        height,
        boundary_tolerance_uv=DEFAULTS["boundary_tolerance_uv"],
        grid_size=DEFAULTS["grid_size"],
        min_region_coverage=DEFAULTS["min_region_coverage"],
    )


class MaskMetricConfigTests(unittest.TestCase):
    def test_gate_mode_clamps_permissive_legacy_thresholds(self):
        result = resolve_metric_config({
            "min_iou": 0.24,
            "max_bbox_error": 0.90,
            "metrics": {
                "min_precision": 0.10,
                "boundary_tolerance_uv": 0.50,
                "resolution": 64,
                "grid_size": 2,
                "min_region_coverage": 0.50,
            },
        })

        self.assertEqual(result["mode"], "gate")
        self.assertEqual(result["effective"]["min_iou"], MINIMUM_SCORES["min_iou"])
        self.assertEqual(
            result["effective"]["min_precision"],
            MINIMUM_SCORES["min_precision"],
        )
        self.assertEqual(
            result["effective"]["max_bbox_error"],
            MAXIMUM_ERRORS["max_bbox_error"],
        )
        self.assertEqual(
            result["effective"]["boundary_tolerance_uv"],
            MAXIMUM_ERRORS["boundary_tolerance_uv"],
        )
        self.assertEqual(
            result["effective"]["resolution"],
            MINIMUM_SAMPLING["resolution"],
        )
        self.assertEqual(
            result["effective"]["grid_size"],
            MINIMUM_SAMPLING["grid_size"],
        )
        self.assertEqual(
            result["effective"]["min_region_coverage"],
            MAXIMUM_SAMPLING["min_region_coverage"],
        )
        self.assertEqual(
            {item["field"] for item in result["adjustments"]},
            {
                "min_iou",
                "min_precision",
                "max_bbox_error",
                "boundary_tolerance_uv",
                "resolution",
                "grid_size",
                "min_region_coverage",
            },
        )

    def test_diagnostic_mode_preserves_declared_thresholds(self):
        result = resolve_metric_config({
            "mode": "diagnostic",
            "metrics": {"min_iou": 0.20, "max_bbox_error": 0.80},
        })

        self.assertEqual(result["effective"]["min_iou"], 0.20)
        self.assertEqual(result["effective"]["max_bbox_error"], 0.80)
        self.assertEqual(result["adjustments"], [])

    def test_rejects_invalid_numeric_configuration(self):
        with self.assertRaisesRegex(ValueError, "must be finite"):
            resolve_metric_config({"metrics": {"min_iou": float("nan")}})
        with self.assertRaisesRegex(ValueError, "grid_size"):
            resolve_metric_config({"metrics": {"grid_size": 1}})
        with self.assertRaisesRegex(ValueError, "min_region_coverage"):
            resolve_metric_config({"metrics": {"min_region_coverage": 0.0}})


class MaskMetricMeasurementTests(unittest.TestCase):
    def test_identical_masks_are_perfect(self):
        reference = mask(64, 64, [(12, 10, 52, 54)])
        result = measure(reference, reference)

        self.assertEqual(result["boundary"]["f1"], 1.0)
        self.assertEqual(result["boundary"]["chamfer_uv"], 0.0)
        self.assertEqual(result["boundary"]["p95_uv"], 0.0)
        self.assertEqual(result["regional"]["iou_mean"], 1.0)
        self.assertEqual(result["regional"]["iou_p10"], 1.0)
        self.assertEqual(result["regional"]["occupancy_error_max"], 0.0)
        self.assertEqual(result["topology"]["component_count_delta"], 0)
        self.assertEqual(result["topology"]["hole_count_delta"], 0)

    def test_local_rearrangement_defeats_high_whole_mask_iou(self):
        body = (16, 16, 48, 48)
        reference = mask(64, 64, [
            body,
            (8, 20, 16, 28),
            (48, 36, 56, 44),
        ])
        render = mask(64, 64, [
            body,
            (8, 36, 16, 44),
            (48, 20, 56, 28),
        ])
        intersection = sum(a and b for a, b in zip(reference, render))
        union = sum(a or b for a, b in zip(reference, render))
        result = measure(reference, render)

        # Area, outer bbox, and centroid are unchanged; even IoU is 0.8. Local
        # and tail metrics still expose the relocated identity-bearing tabs.
        self.assertAlmostEqual(intersection / union, 0.8)
        self.assertEqual(sum(reference), sum(render))
        self.assertEqual(result["regional"]["iou_p10"], 0.0)
        self.assertGreater(result["boundary"]["p95_uv"], 0.04)
        self.assertGreater(result["regional"]["occupancy_error_max"], 0.18)

    def test_topology_is_reported_without_becoming_a_fragile_gate(self):
        reference = mask(64, 64, [(12, 12, 52, 52)])
        render_values = bytearray(reference)
        for y in range(26, 38):
            render_values[y * 64 + 26:y * 64 + 38] = b"\x00" * 12
        result = measure(reference, bytes(render_values))

        self.assertEqual(result["topology"]["reference_holes"], 0)
        self.assertEqual(result["topology"]["render_holes"], 1)
        self.assertEqual(result["topology"]["hole_count_delta"], 1)


if __name__ == "__main__":
    unittest.main()
