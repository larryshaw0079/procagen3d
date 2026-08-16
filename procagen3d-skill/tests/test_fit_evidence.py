"""Tests for versioned, hash-bound registered fit evidence."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = SKILL_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from harness.fit_evidence import (  # noqa: E402
    GLOBAL_METRIC_FIELDS,
    MASK_GATE_IDS,
    METRIC_CONFIG_FIELDS,
    image_fit_errors,
)
from harness.graph import sha256_file  # noqa: E402


def write(path, content):
    path.write_bytes(content if isinstance(content, bytes) else content.encode())
    return path


def suite_report(root, *, mode="gate"):
    spec_path = root / "fit_spec.json"
    spec = json.loads(spec_path.read_text())
    spec["mask"]["mode"] = mode
    spec_path.write_text(json.dumps(spec))
    config = {field: 0.0 for field in METRIC_CONFIG_FIELDS}
    config.update({"resolution": 256, "grid_size": 4})
    global_metrics = {field: 0.0 for field in GLOBAL_METRIC_FIELDS}
    gates = [
        {
            "id": gate_id,
            "kind": "mask",
            "target": ">= 0.0",
            "measured": 1.0,
            "pass": True,
            "blocking": mode == "gate",
            "note": "",
        }
        for gate_id in sorted(MASK_GATE_IDS)
    ]
    if mode == "diagnostic":
        gates.append({
            "id": "semantic_anchor",
            "kind": "landmark",
            "target": "<= 0.04",
            "measured": 0.01,
            "pass": True,
            "blocking": True,
            "note": "",
        })
    blocking = [gate for gate in gates if gate["blocking"]]
    return {
        "procagen3d_fit_version": 2,
        "fit_spec_version": 2,
        "fit_spec": "fit_spec.json",
        "passed": True,
        "gates": gates,
        "summary": {
            "passed": len(blocking),
            "total": len(blocking),
            "failures": 0,
            "diagnostics": {
                "passed": len(gates) - len(blocking),
                "total": len(gates) - len(blocking),
                "failures": 0,
            },
        },
        "mask": {
            "metric_suite": {
                "version": 1,
                "mode": mode,
                "requested": dict(config),
                "effective": dict(config),
                "adjustments": [],
                "global": global_metrics,
                "sampled": {
                    "boundary": {
                        "f1": 1.0,
                        "chamfer_uv": 0.0,
                        "p95_uv": 0.0,
                    },
                    "regional": {
                        "iou_mean": 1.0,
                        "iou_p10": 1.0,
                        "occupancy_error_max": 0.0,
                        "worst_regions": [],
                    },
                    "topology": {
                        "component_count_delta": 0,
                        "hole_count_delta": 0,
                    },
                },
            },
        },
        "inputs": {
            "fit_spec_sha256": sha256_file(root / "fit_spec.json"),
            "reference_sha256": sha256_file(root / "reference_01.png"),
            "scene_graph_sha256": sha256_file(root / "scene_graph.json"),
            "scene_blend_sha256": sha256_file(root / "scene.blend"),
        },
    }


class FitEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        write(self.root / "reference_01.png", b"reference")
        write(self.root / "scene_graph.json", "{}")
        write(self.root / "scene.blend", b"blend")
        (self.root / "fit_spec.json").write_text(json.dumps({
            "version": 2,
            "reference_image": "reference_01.png",
            "mask": {"source": "auto"},
        }))

    def tearDown(self):
        self.temp_dir.cleanup()

    def save_report(self, report):
        (self.root / "fit_report.json").write_text(json.dumps(report))

    def test_accepts_complete_gate_suite(self):
        self.save_report(suite_report(self.root))
        self.assertEqual(image_fit_errors(self.root), [])

    def test_accepts_diagnostic_mask_with_semantic_blocking_gate(self):
        self.save_report(suite_report(self.root, mode="diagnostic"))
        self.assertEqual(image_fit_errors(self.root), [])

    def test_rejects_legacy_report_even_when_it_claims_pass(self):
        report = suite_report(self.root)
        report["procagen3d_fit_version"] = 1
        self.save_report(report)

        self.assertTrue(any(
            "legacy or unsupported" in error
            for error in image_fit_errors(self.root)
        ))

    def test_rejects_missing_mask_gate(self):
        report = suite_report(self.root)
        report["gates"] = report["gates"][:-1]
        report["summary"]["passed"] -= 1
        report["summary"]["total"] -= 1
        self.save_report(report)

        self.assertTrue(any(
            "missing mask gates" in error for error in image_fit_errors(self.root)
        ))

    def test_rejects_incomplete_metric_suite_section(self):
        report = suite_report(self.root)
        del report["mask"]["metric_suite"]["sampled"]["boundary"]["p95_uv"]
        self.save_report(report)

        self.assertTrue(any(
            "sampled.boundary is missing" in error
            for error in image_fit_errors(self.root)
        ))

    def test_rejects_diagnostic_mask_as_only_evidence(self):
        report = suite_report(self.root, mode="diagnostic")
        report["gates"] = report["gates"][:-1]
        report["summary"].update({"passed": 0, "total": 0, "failures": 0})
        self.save_report(report)

        self.assertIn("fit report has no blocking gates", image_fit_errors(self.root))


if __name__ == "__main__":
    unittest.main()
