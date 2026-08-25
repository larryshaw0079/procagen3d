from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from procagen3d.metrics import (
    CANONICAL_VIEW_NAMES,
    DetailGateThresholds,
    FidelityGateThresholds,
    MaterialGateThresholds,
    StructuralGateThresholds,
    compare_workspace,
)


def _write_json(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _mask_record(colors: list[tuple[int, int, int]]) -> dict:
    width = len(colors)
    bits = bytearray((width + 7) // 8)
    for index in range(width):
        bits[index // 8] |= 1 << (7 - index % 8)
    return {
        "width": width,
        "height": 1,
        "foreground_pixels": width,
        "encoding": "base64-msb-packbits",
        "data": base64.b64encode(bits).decode("ascii"),
        "rgb_encoding": "base64-rgb8",
        "rgb_data": base64.b64encode(
            bytes(channel for color in colors for channel in color)
        ).decode("ascii"),
    }


def _mask_document(colors: list[tuple[int, int, int]]) -> dict:
    record = _mask_record(colors)
    return {
        "schema_version": 2,
        "views": {name: dict(record) for name in CANONICAL_VIEW_NAMES},
    }


def _bounds() -> dict:
    return {
        "bounds": {
            "min": [0.0, 0.0, 0.0],
            "max": [1.0, 2.0, 3.0],
            "dimensions": [1.0, 2.0, 3.0],
            "center": [0.5, 1.0, 1.5],
        }
    }


def _attachment(
    parent_id: str,
    kind: str,
    *,
    minimum: list[float],
    maximum: list[float],
) -> dict:
    return {
        "parent_id": parent_id,
        "type": kind,
        "contact_region": {"min": minimum, "max": maximum},
        "max_gap": 0.02,
        "max_penetration": 0.02,
        "min_contact_area": 0.0,
    }


def _part(
    identifier: str,
    object_name: str,
    *,
    attachment: dict,
) -> dict:
    return {
        "id": identifier,
        "name": object_name,
        "shape_family": "fixture",
        "approximate_bounds": {
            "min": [0.0, 0.0, 0.0],
            "max": [1.0, 2.0, 3.0],
        },
        "visual_role": "primary" if attachment["type"] == "root" else "secondary",
        "object_names": [object_name],
        "attachment": attachment,
    }


def _plan(*, include_wing: bool = True) -> dict:
    parts = [
        _part(
            "body",
            "Body",
            attachment=_attachment(
                "__root__",
                "root",
                minimum=[0.0, 0.0, 0.0],
                maximum=[1.0, 2.0, 3.0],
            ),
        )
    ]
    if include_wing:
        parts.append(
            _part(
                "wing",
                "Wing",
                attachment=_attachment(
                    "body",
                    "surface-contact",
                    minimum=[0.8, 0.5, 1.0],
                    maximum=[1.0, 1.5, 2.0],
                ),
            )
        )
    return {
        "subject": "quality gate fixture",
        "subject_kind": "object",
        "reconstruction_mode": "procedural",
        "granularity": "surface",
        "quality_profile": {
            "surface_fidelity": "strict",
            "detail_richness": "maximum",
            "material_fidelity": "strict",
            "structural_coherence": "strict",
        },
        "coordinate_frame": {"up": "+Z"},
        "dimensions": [1.0, 2.0, 3.0],
        "parts": parts,
        "materials": [],
        "construction_strategy": "deterministic test geometry",
        "identity_features": [],
        "limitations": [],
    }


def _object_structure(
    *,
    edges: int = 30,
    boundary_edges: int = 0,
    non_manifold_edges: int = 0,
    inconsistent_winding_edges: int = 0,
    triangles: int = 100,
    degenerate_triangles: int = 0,
    low_quality_triangles: int = 0,
    outward: bool = True,
    self_intersections: int = 0,
) -> dict:
    return {
        "triangles": triangles,
        "structure": {
            "topology": {
                "edges": edges,
                "boundary_edges": boundary_edges,
                "non_manifold_edges": non_manifold_edges,
                "manifold_edges": max(0, edges - non_manifold_edges),
                "inconsistent_winding_edges": inconsistent_winding_edges,
            },
            "triangle_quality": {
                "degenerate_triangles": degenerate_triangles,
                "low_quality_below_0_05": low_quality_triangles,
            },
            "normal_consistency": {"outward_orientation_proxy": outward},
            "self_intersection_proxy": {"triangle_pairs": self_intersections},
        },
    }


def _scene_document(
    objects: list[dict] | None = None,
    *,
    near_contacts: list[list[str]] | None = None,
    intersections: list[list[str]] | None = None,
    isolated: list[str] | None = None,
) -> dict:
    scene = _bounds()
    if objects is None:
        return scene
    intersection_pairs = intersections or []
    scene["objects"] = objects
    scene["structure"] = {
        "contact_intersection_proxy": {
            "broad_phase_pairs": len(near_contacts or []) + len(intersection_pairs),
            "near_contact_pairs": [
                {"objects": pair} for pair in (near_contacts or [])
            ],
            "triangle_intersection_pairs": [
                {"objects": pair} for pair in intersection_pairs
            ],
            "isolated_objects": list(isolated or []),
        }
    }
    return scene


def _named_object(name: str, **structure_overrides: int | bool) -> dict:
    bounds = {
        "Body": {"min": [0.0, 0.0, 0.0], "max": [1.0, 2.0, 3.0]},
        "Wing": {"min": [0.99, 0.5, 1.0], "max": [1.5, 1.5, 2.0]},
        "LooseTrim": {"min": [0.4, 0.4, 0.4], "max": [0.6, 0.6, 0.6]},
    }.get(name, {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]})
    return {"name": name, "bounds": bounds, **_object_structure(**structure_overrides)}


def _probe(*, triangles: int, primitives: int = 0, white_risk: int = 0) -> dict:
    return {
        "scene": {
            "triangle_count": triangles,
            "primitive_count": primitives,
        },
        "material_diagnostics": {
            "primitive_count_at_default_white_risk": white_risk,
            "declared_base_color_palette": [],
        },
    }


def _compare(
    tmp_path: Path,
    *,
    reference_colors: list[tuple[int, int, int]] | None = None,
    candidate_colors: list[tuple[int, int, int]] | None = None,
    candidate_scene: dict | None = None,
    reference_probe: dict | None = None,
    candidate_probe: dict | None = None,
    plan: dict | None = None,
    gate_thresholds: FidelityGateThresholds | None = None,
    material_thresholds: MaterialGateThresholds | None = None,
    detail_thresholds: DetailGateThresholds | None = None,
    structural_thresholds: StructuralGateThresholds | None = None,
    min_score: float = 0.0,
) -> dict:
    neutral = [(128, 128, 128)] * 4
    reference_masks = _write_json(
        tmp_path / "reference-masks.json",
        _mask_document(reference_colors or neutral),
    )
    candidate_masks = _write_json(
        tmp_path / "candidate-masks.json",
        _mask_document(candidate_colors or neutral),
    )
    kwargs = {
        "reference_masks": reference_masks,
        "candidate_masks": candidate_masks,
        "reference_scene": _write_json(tmp_path / "reference-scene.json", _bounds()),
        "candidate_scene": _write_json(
            tmp_path / "candidate-scene.json", candidate_scene or _bounds()
        ),
        "output": tmp_path / "comparison.json",
        "min_score": min_score,
        "gate_thresholds": gate_thresholds,
        "material_gate_thresholds": material_thresholds,
        "detail_gate_thresholds": detail_thresholds,
        "structural_gate_thresholds": structural_thresholds,
    }
    if reference_probe is not None:
        kwargs["reference_probe"] = _write_json(
            tmp_path / "reference-probe.json", reference_probe
        )
    if candidate_probe is not None:
        kwargs["candidate_probe"] = _write_json(
            tmp_path / "candidate-probe.json", candidate_probe
        )
    if plan is not None:
        kwargs["plan"] = _write_json(tmp_path / "plan.json", plan)
    return compare_workspace(**kwargs)


def _failed_gates(report: dict) -> set[str]:
    return {failure["gate"] for failure in report["hard_gates"]["failures"]}


def test_pure_white_candidate_fails_palette_gate_even_with_perfect_geometry(
    tmp_path: Path,
) -> None:
    report = _compare(
        tmp_path,
        reference_colors=[(220, 30, 20)] * 4,
        candidate_colors=[(255, 255, 255)] * 4,
        gate_thresholds=FidelityGateThresholds(
            min_mean_spatial_rgb_similarity=0.0,
            min_mean_palette_similarity=0.50,
        ),
    )

    assert report["score_passed"] is True
    assert report["summary"]["mean_silhouette_iou"] == 1.0
    assert report["summary"]["mean_palette_similarity"] == 0.0
    assert _failed_gates(report) == {"mean_palette_similarity"}
    assert report["passed"] is False


def test_default_white_primitives_fail_material_gate(tmp_path: Path) -> None:
    report = _compare(
        tmp_path,
        candidate_probe=_probe(triangles=100, primitives=10, white_risk=6),
        material_thresholds=MaterialGateThresholds(
            max_default_white_primitive_fraction=0.10
        ),
    )

    gate = report["hard_gates"]["results"][
        "default_white_primitive_fraction"
    ]
    assert gate["value"] == pytest.approx(0.60)
    assert gate["threshold"] == 0.10
    assert gate["passed"] is False
    assert _failed_gates(report) == {"default_white_primitive_fraction"}
    assert report["material_diagnostics"][
        "primitive_count_at_default_white_risk"
    ] == 6


def test_low_triangle_and_semantic_richness_fail_detail_gates(
    tmp_path: Path,
) -> None:
    candidate_scene = _scene_document([{"name": "Body"}])
    report = _compare(
        tmp_path,
        candidate_scene=candidate_scene,
        reference_probe=_probe(triangles=1_000),
        candidate_probe=_probe(triangles=100),
        plan=_plan(),
        detail_thresholds=DetailGateThresholds(
            min_triangle_ratio=0.50,
            min_semantic_part_coverage=0.75,
        ),
    )

    assert report["detail_diagnostics"]["triangle_ratio"] == pytest.approx(0.10)
    assert report["detail_diagnostics"]["semantic_part_coverage"] == 0.50
    assert report["detail_diagnostics"]["covered_part_ids"] == ["body"]
    assert report["detail_diagnostics"]["missing_part_ids"] == ["wing"]
    assert _failed_gates(report) == {
        "triangle_richness_ratio",
        "semantic_part_coverage",
    }


def test_topology_attachment_and_intersection_defects_fail_structural_gates(
    tmp_path: Path,
) -> None:
    candidate_scene = _scene_document(
        [
            _named_object(
                "Body", boundary_edges=9, non_manifold_edges=6
            ),
            _named_object("Wing"),
            _named_object("LooseTrim"),
        ],
        intersections=[["Body", "LooseTrim"]],
    )
    report = _compare(
        tmp_path,
        candidate_scene=candidate_scene,
        plan=_plan(),
        structural_thresholds=StructuralGateThresholds(
            max_boundary_edge_fraction=0.05,
            max_non_manifold_edge_fraction=0.05,
        ),
    )

    failures = _failed_gates(report)
    assert {
        "boundary_edge_fraction",
        "non_manifold_edge_fraction",
        "unjoined_attachment_fraction",
        "attachment_gap_violation_fraction",
        "intersection_fraction",
    } <= failures
    assert report["structural_diagnostics"]["unjoined_part_ids"] == ["wing"]
    assert report["structural_diagnostics"]["unexpected_intersection_pairs"] == [
        ["Body", "LooseTrim"]
    ]
    assert report["structural_diagnostics"][
        "unjoined_attachment_fraction"
    ] == 1.0
    assert report["structural_diagnostics"]["intersection_fraction"] == 1.0


def test_clean_candidate_passes_all_material_detail_and_structural_gates(
    tmp_path: Path,
) -> None:
    colors = [(205, 35, 25), (30, 90, 210), (25, 170, 70), (180, 130, 40)]
    candidate_scene = _scene_document(
        [
            _named_object("Body", triangles=600),
            _named_object("Wing", triangles=400),
        ],
        near_contacts=[["Body", "Wing"]],
    )
    report = _compare(
        tmp_path,
        reference_colors=colors,
        candidate_colors=colors,
        candidate_scene=candidate_scene,
        reference_probe=_probe(triangles=1_000),
        candidate_probe=_probe(triangles=1_000, primitives=4, white_risk=0),
        plan=_plan(),
        material_thresholds=MaterialGateThresholds(
            max_default_white_primitive_fraction=0.05
        ),
        detail_thresholds=DetailGateThresholds(
            min_triangle_ratio=0.40,
            min_semantic_part_coverage=0.95,
        ),
        structural_thresholds=StructuralGateThresholds(),
        min_score=1.0,
    )

    assert report["score"] == pytest.approx(1.0)
    assert report["score_passed"] is True
    assert report["hard_gates"]["passed"] is True
    assert report["hard_gates"]["failures"] == []
    assert all(
        gate["passed"] for gate in report["hard_gates"]["results"].values()
    )
    assert report["material_diagnostics"]["default_white_primitive_fraction"] == 0.0
    assert report["detail_diagnostics"]["semantic_part_coverage"] == 1.0
    assert report["structural_diagnostics"]["unjoined_part_ids"] == []
    assert report["structural_diagnostics"]["unexpected_intersection_pairs"] == []
    assert report["passed"] is True


def test_single_root_object_is_not_treated_as_a_loose_component(
    tmp_path: Path,
) -> None:
    candidate_scene = _scene_document(
        [_named_object("Body")],
        isolated=["Body"],
    )
    report = _compare(
        tmp_path,
        candidate_scene=candidate_scene,
        plan=_plan(include_wing=False),
        structural_thresholds=StructuralGateThresholds(
            max_loose_component_fraction=0.0
        ),
    )

    assert report["structural_diagnostics"]["loose_component_fraction"] == 0.0
    assert report["structural_diagnostics"]["isolated_objects"] == []
    assert report["hard_gates"]["results"]["loose_component_fraction"][
        "passed"
    ] is True


def test_declared_intentional_gap_is_not_treated_as_a_loose_component(
    tmp_path: Path,
) -> None:
    plan = _plan()
    plan["parts"][1]["attachment"].update(type="intentional-gap")
    candidate_scene = _scene_document(
        [_named_object("Body"), _named_object("Wing")],
        isolated=["Body", "Wing"],
    )
    report = _compare(
        tmp_path,
        candidate_scene=candidate_scene,
        plan=plan,
        structural_thresholds=StructuralGateThresholds(
            max_loose_component_fraction=0.0
        ),
    )

    assert report["structural_diagnostics"]["loose_component_fraction"] == 0.0
    assert report["structural_diagnostics"]["isolated_objects"] == []
