"""Validate hash-bound registered-fit evidence for image-conditioned assets."""

import json
from pathlib import Path

from .graph import sha256_file


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
    reference_name = spec.get("reference_image")
    reference_path = root / reference_name if isinstance(reference_name, str) else None
    if reference_path is None or not reference_path.is_file():
        errors.append(f"fit reference is missing: {reference_name!r}")
    if not report.get("passed"):
        summary = report.get("summary", {})
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
