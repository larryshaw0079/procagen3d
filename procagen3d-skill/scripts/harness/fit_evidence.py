"""Validate hash-bound registered-fit evidence for image-conditioned assets."""

import json
from pathlib import Path

from bpy_stages.mask_metrics import (
    DEFAULTS,
    MASK_GATE_IDS,
    MASK_METRIC_SUITE_VERSION,
)

from .graph import sha256_file


FIT_REPORT_VERSION = 2
METRIC_CONFIG_FIELDS = set(DEFAULTS)
GLOBAL_METRIC_FIELDS = {
    "reference_pixels",
    "render_pixels",
    "intersection_pixels",
    "union_pixels",
    "iou",
    "precision",
    "recall",
    "dice",
    "bbox_error",
    "centroid_error",
    "area_ratio_error",
}
SAMPLED_METRIC_FIELDS = {
    "boundary": {"f1", "chamfer_uv", "p95_uv"},
    "regional": {
        "iou_mean",
        "iou_p10",
        "occupancy_error_max",
        "worst_regions",
    },
    "topology": {"component_count_delta", "hole_count_delta"},
}


def _require_fields(errors, value, fields, label):
    if not isinstance(value, dict):
        errors.append(f"fit report mask metric suite has no {label} object")
        return
    missing = sorted(fields - value.keys())
    if missing:
        errors.append(
            f"fit report mask metric {label} is missing: " + ", ".join(missing)
        )


def _suite_errors(spec, report):
    errors = []
    if report.get("procagen3d_fit_version") != FIT_REPORT_VERSION:
        return [
            "legacy or unsupported fit report; rerun `procagen3d fit` for "
            "metric-suite evidence"
        ]
    mask_report = report.get("mask")
    suite = mask_report.get("metric_suite") if isinstance(mask_report, dict) else None
    if not isinstance(suite, dict):
        errors.append("fit report has no registered mask metric suite")
        return errors
    if suite.get("version") != MASK_METRIC_SUITE_VERSION:
        errors.append("fit report has an unsupported mask metric suite version")
    mode = suite.get("mode")
    if mode not in {"gate", "diagnostic"}:
        errors.append("fit report has an invalid mask metric mode")
    mask_config = spec.get("mask")
    if isinstance(mask_config, dict):
        declared_mode = str(mask_config.get("mode", "gate")).lower()
        if mode != declared_mode:
            errors.append("fit report mask metric mode conflicts with fit spec")
    if report.get("fit_spec_version") != spec.get("version"):
        errors.append("fit report fit-spec version conflicts with fit spec")
    _require_fields(
        errors, suite.get("requested"), METRIC_CONFIG_FIELDS, "requested")
    _require_fields(
        errors, suite.get("effective"), METRIC_CONFIG_FIELDS, "effective")
    _require_fields(
        errors, suite.get("global"), GLOBAL_METRIC_FIELDS, "global")
    if not isinstance(suite.get("adjustments"), list):
        errors.append("fit report mask metric suite has no adjustments list")
    sampled = suite.get("sampled", {})
    if not isinstance(sampled, dict):
        errors.append("fit report mask metric suite has no sampled object")
    else:
        for field, required in SAMPLED_METRIC_FIELDS.items():
            _require_fields(errors, sampled.get(field), required, f"sampled.{field}")

    gates = report.get("gates")
    if not isinstance(gates, list) or any(not isinstance(gate, dict) for gate in gates):
        errors.append("fit report gates must be a list of objects")
        return errors
    gate_map = {
        gate.get("id"): gate for gate in gates
        if gate.get("kind") == "mask" and isinstance(gate.get("id"), str)
    }
    missing = sorted(MASK_GATE_IDS - gate_map.keys())
    if missing:
        errors.append("fit report is missing mask gates: " + ", ".join(missing))
    if any(type(gate.get("blocking")) is not bool for gate in gates):
        errors.append("fit report gates must declare boolean blocking values")
    if any(type(gate.get("pass")) is not bool for gate in gates):
        errors.append("fit report gates must declare boolean pass values")
    blocking = [gate for gate in gates if gate.get("blocking") is True]
    diagnostics = [gate for gate in gates if gate.get("blocking") is False]
    if not blocking:
        errors.append("fit report has no blocking gates")
    if any(gate.get("pass") is not True for gate in blocking):
        errors.append("fit report marks a blocking gate as failed")
    if mode in {"gate", "diagnostic"}:
        expected_mask_blocking = mode == "gate"
        for gate_id in MASK_GATE_IDS & gate_map.keys():
            if gate_map[gate_id].get("blocking") is not expected_mask_blocking:
                errors.append(
                    f"fit report mask gate {gate_id} conflicts with {mode!r} mode"
                )

    summary = report.get("summary")
    if not isinstance(summary, dict):
        errors.append("fit report has no summary object")
    else:
        passed = sum(gate.get("pass") is True for gate in blocking)
        expected = {
            "passed": passed,
            "total": len(blocking),
            "failures": len(blocking) - passed,
        }
        for key, value in expected.items():
            if summary.get(key) != value:
                errors.append(f"fit report summary.{key} is inconsistent with gates")
        diagnostic_summary = summary.get("diagnostics")
        if not isinstance(diagnostic_summary, dict):
            errors.append("fit report summary has no diagnostics object")
        else:
            diagnostic_passed = sum(
                gate.get("pass") is True for gate in diagnostics)
            diagnostic_expected = {
                "passed": diagnostic_passed,
                "total": len(diagnostics),
                "failures": len(diagnostics) - diagnostic_passed,
            }
            for key, value in diagnostic_expected.items():
                if diagnostic_summary.get(key) != value:
                    errors.append(
                        f"fit report summary.diagnostics.{key} is inconsistent "
                        "with gates"
                    )
    return errors


def image_fit_errors(dir_path):
    """Validate required, passing, hash-bound fit evidence for image inputs."""
    root = Path(dir_path)
    references = sorted(root.glob("reference_[0-9][0-9].*"))
    if not references:
        return []
    spec_path = root / "fit_spec.json"
    report_path = root / "fit_report.json"
    errors = []
    if not spec_path.is_file():
        errors.append("fit_spec.json missing")
    if not report_path.is_file():
        errors.append("fit_report.json missing (run `procagen3d fit`)")
    if errors:
        return errors
    try:
        spec = json.loads(spec_path.read_text())
        report = json.loads(report_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return [f"invalid fit evidence JSON: {exc}"]
    if not isinstance(spec, dict) or not isinstance(report, dict):
        return ["fit evidence JSON roots must be objects"]
    errors.extend(_suite_errors(spec, report))
    reference_name = spec.get("reference_image")
    reference_path = root / reference_name if isinstance(reference_name, str) else None
    if reference_path is None or not reference_path.is_file():
        errors.append(f"fit reference is missing: {reference_name!r}")
    if report.get("passed") is not True:
        summary = report.get("summary", {})
        if not isinstance(summary, dict):
            summary = {}
        errors.append(
            f"registered fit did not pass ({summary.get('passed', 0)}/"
            f"{summary.get('total', '?')} gates)")
    inputs = report.get("inputs")
    if not isinstance(inputs, dict):
        errors.append("fit report has no input hashes")
        return errors
    expected = {
        "fit_spec_sha256": spec_path,
        "scene_graph_sha256": root / "scene_graph.json",
        "scene_blend_sha256": root / "scene.blend",
    }
    if reference_path is not None and reference_path.is_file():
        expected["reference_sha256"] = reference_path
    mask = spec.get("mask")
    if not isinstance(mask, dict):
        errors.append("fit_spec.json has no registered mask object")
    if isinstance(mask, dict) and str(mask.get("source", "auto")).lower() == "file":
        mask_name = mask.get("path")
        mask_path = root / mask_name if isinstance(mask_name, str) else None
        if mask_path is None or not mask_path.is_file():
            errors.append(f"fit mask is missing: {mask_name!r}")
        else:
            expected["mask_sha256"] = mask_path
    for key, path in expected.items():
        if inputs.get(key) != sha256_file(path):
            errors.append(f"stale fit evidence: {key} does not match {path.name}")
    return errors
