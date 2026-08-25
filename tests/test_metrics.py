from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from procagen3d.metrics import (
    CANONICAL_VIEW_NAMES,
    FidelityGateThresholds,
    SurfaceGateThresholds,
    compare_workspace,
    mask_metrics,
)


def packed(
    rows: list[str],
    *,
    colors: list[tuple[int, int, int]] | None = None,
) -> dict:
    height = len(rows)
    width = len(rows[0])
    values = "".join(rows)
    bits = bytearray((width * height + 7) // 8)
    count = 0
    for index, value in enumerate(values):
        if value == "1":
            bits[index // 8] |= 1 << (7 - index % 8)
            count += 1

    colors = colors or [(128, 128, 128)] * (width * height)
    if len(colors) != width * height:
        raise ValueError("one RGB tuple is required per pixel")
    rgb = bytes(channel for color in colors for channel in color)
    return {
        "width": width,
        "height": height,
        "foreground_pixels": count,
        "encoding": "base64-msb-packbits",
        "data": base64.b64encode(bits).decode("ascii"),
        "rgb_encoding": "base64-rgb8",
        "rgb_data": base64.b64encode(rgb).decode("ascii"),
    }


def mask_document(record: dict) -> dict:
    return {
        "schema_version": 2,
        "views": {name: dict(record) for name in CANONICAL_VIEW_NAMES},
    }


def geometry_document() -> dict:
    return {
        "bounds": {
            "min": [0.0, 0.0, 0.0],
            "max": [1.0, 2.0, 3.0],
            "dimensions": [1.0, 2.0, 3.0],
            "center": [0.5, 1.0, 1.5],
        }
    }


def surface_document(
    *,
    symmetric_mean: float = 0.020,
    symmetric_p95: float = 0.050,
) -> dict:
    return {
        "schema_version": 1,
        "units": "normalized-scene-units",
        "pose_policy": "frame-0, armatures-in-rest-position",
        "sampling": {
            "strategy": "deterministic fixture",
            "requested_samples_per_direction": 4,
            "percentile_method": "linear fixture",
            "worst_sample_limit_per_direction": 2,
        },
        "reference_normalization": {
            "source_min": [-0.5, -0.5, 0.0],
            "source_max": [0.5, 0.5, 1.0],
            "scale": 2.0,
            "translation": [0.0, 0.0, 0.0],
            "longest_dimension": 2.0,
        },
        "surfaces": {
            "reference": {"vertices": 8, "triangles": 12, "area": 6.0},
            "candidate": {"vertices": 12, "triangles": 20, "area": 7.0},
        },
        "candidate_to_reference": {
            "samples": 4,
            "source_surface_area": 7.0,
            "mean": 0.018,
            "rms": 0.025,
            "p95": 0.045,
            "max": 0.060,
            "worst_samples": [
                {
                    "sample_index": 1,
                    "distance": 0.060,
                    "source": [0.0, 0.0, 1.0],
                    "nearest": [0.0, 0.0, 0.94],
                }
            ],
        },
        "reference_to_candidate": {
            "samples": 4,
            "source_surface_area": 6.0,
            "mean": 0.022,
            "rms": 0.030,
            "p95": 0.055,
            "max": 0.070,
            "worst_samples": [
                {
                    "sample_index": 2,
                    "distance": 0.070,
                    "source": [0.0, 0.0, 0.0],
                    "nearest": [0.0, 0.0, 0.07],
                }
            ],
        },
        "symmetric": {
            "mean": symmetric_mean,
            "rms": max(symmetric_mean, 0.028),
            "p95": symmetric_p95,
            "max": max(symmetric_p95, 0.070),
        },
    }


def enhanced_surface_document(
    *,
    mean_normal_angle: float = 12.0,
    visible_coverage: float = 0.95,
    candidate_area: float = 7.0,
) -> dict:
    value = surface_document()
    value["surfaces"]["candidate"]["area"] = candidate_area
    value["candidate_to_reference"]["source_surface_area"] = candidate_area
    value["normal_aware"] = {
        "normal_angle_degrees": {
            "mean": mean_normal_angle,
            "rms": max(mean_normal_angle, 14.0),
            "p95": max(mean_normal_angle, 18.0),
            "max": max(mean_normal_angle, 25.0),
        }
    }
    for direction in ("candidate_to_reference", "reference_to_candidate"):
        value[direction]["visible_external_proxy"] = {
            "coverage": {
                "thresholds": [
                    {
                        "distance": threshold,
                        "distance_and_normal_aligned_fraction": visible_coverage,
                    }
                    for threshold in (0.005, 0.010, 0.020, 0.040, 0.080)
                ]
            }
        }
    return value


def test_surface_comparison_preserves_diagnostic_identities(
    tmp_path: Path,
) -> None:
    document = enhanced_surface_document()
    document["surfaces"]["candidate"]["objects"] = [
        {
            "name": "Body",
            "object_index": 0,
            "vertices": 12,
            "triangles": 20,
            "area": 7.0,
        }
    ]
    sample = document["candidate_to_reference"]["worst_samples"][0]
    sample.update(
        source_identity={
            "object": "Body",
            "object_index": 0,
            "polygon_index": 3,
            "triangle_index_in_object": 4,
            "surface_triangle_index": 4,
        },
        target_identity={
            "object": "ReferenceBody",
            "object_index": 0,
            "polygon_index": 2,
            "triangle_index_in_object": 3,
            "surface_triangle_index": 3,
        },
        source_normal=[0.0, 0.0, 1.0],
        target_normal=[0.0, 0.1, 0.995],
        normal_cosine=0.995,
        normal_angle_degrees=5.73,
        unoriented_normal_angle_degrees=5.73,
        signed_target_plane_offset=0.01,
        point_to_plane_distance=0.01,
        normal_aware_distance=0.061,
        visible_from=["front", "top"],
    )
    path = write_json(tmp_path / "surface-identities.json", document)

    report = compare_scenes(
        tmp_path,
        reference=geometry_document(),
        candidate=geometry_document(),
        surface_comparison=path,
    )

    normalized = report["surface_comparison"]
    assert normalized["surfaces"]["candidate"]["objects"][0]["name"] == "Body"
    worst = normalized["candidate_to_reference"]["worst_samples"][0]
    assert worst["source_identity"]["object"] == "Body"
    assert worst["target_identity"]["polygon_index"] == 2
    assert worst["visible_from"] == ["front", "top"]


def write_json(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def compare_scenes(
    tmp_path: Path,
    *,
    reference: object,
    candidate: object,
    min_score: object = 0.5,
    gate_thresholds: FidelityGateThresholds | None = None,
    surface_comparison: Path | None = None,
    surface_gate_thresholds: SurfaceGateThresholds | None = None,
) -> dict:
    record = packed(["11", "11"])
    masks = write_json(tmp_path / "valid-masks.json", mask_document(record))
    reference_path = write_json(tmp_path / "reference-scene.json", reference)
    candidate_path = write_json(tmp_path / "candidate-scene.json", candidate)
    return compare_workspace(
        reference_masks=masks,
        candidate_masks=masks,
        reference_scene=reference_path,
        candidate_scene=candidate_path,
        output=tmp_path / "comparison.json",
        min_score=min_score,
        gate_thresholds=gate_thresholds,
        surface_comparison=surface_comparison,
        surface_gate_thresholds=surface_gate_thresholds,
    )


def test_mask_metrics_identical_and_disjoint() -> None:
    left = packed(["1100", "1100"])
    same = mask_metrics(left, left)
    assert same["iou"] == 1.0
    assert same["area_similarity"] == 1.0
    assert same["spatial_rgb_similarity"] == 1.0
    assert same["reference_bbox"] == [0, 0, 1, 1]

    right = packed(["0011", "0011"])
    other = mask_metrics(left, right)
    assert other["iou"] == 0.0
    assert other["area_similarity"] == 1.0
    assert other["spatial_rgb_similarity"] == 0.0


def test_rgb_similarity_is_spatial_not_histogram_only() -> None:
    red = (255, 0, 0)
    blue = (0, 0, 255)
    reference = packed(["11"], colors=[red, blue])
    swapped = packed(["11"], colors=[blue, red])

    result = mask_metrics(reference, swapped)

    assert result["iou"] == 1.0
    assert result["spatial_rgb_similarity"] == pytest.approx(1.0 / 3.0)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("rgb_data", base64.b64encode(b"short").decode("ascii"), "RGB data"),
        ("data", "not base64!", "not valid base64"),
        ("foreground_pixels", 999, "foreground_pixels"),
    ],
)
def test_mask_metrics_rejects_malformed_records(field: str, value: object, message: str) -> None:
    valid = packed(["11", "11"])
    malformed = dict(valid)
    malformed[field] = value

    with pytest.raises(ValueError, match=message):
        mask_metrics(valid, malformed)


def test_compare_workspace_requires_exact_canonical_view_sets(tmp_path: Path) -> None:
    record = packed(["11", "11"])
    reference = mask_document(record)
    candidate = mask_document(record)
    reference["views"].pop("iso")
    candidate["views"]["three-quarter"] = dict(record)
    reference_path = write_json(tmp_path / "reference.json", reference)
    candidate_path = write_json(tmp_path / "candidate.json", candidate)

    with pytest.raises(ValueError, match=r"reference canonical view set mismatch: missing iso"):
        compare_workspace(
            reference_masks=reference_path,
            candidate_masks=candidate_path,
            reference_scene=tmp_path / "unused-reference-scene.json",
            candidate_scene=tmp_path / "unused-candidate-scene.json",
            output=tmp_path / "unused-comparison.json",
            min_score=0.0,
        )

    reference["views"]["iso"] = dict(record)
    write_json(reference_path, reference)
    with pytest.raises(ValueError, match=r"candidate canonical view set mismatch: unexpected three-quarter"):
        compare_workspace(
            reference_masks=reference_path,
            candidate_masks=candidate_path,
            reference_scene=tmp_path / "unused-reference-scene.json",
            candidate_scene=tmp_path / "unused-candidate-scene.json",
            output=tmp_path / "unused-comparison.json",
            min_score=0.0,
        )


def test_compare_workspace_writes_weighted_rgb_report(tmp_path: Path) -> None:
    reference_record = packed(["11", "11"], colors=[(0, 0, 0)] * 4)
    candidate_record = packed(["11", "11"], colors=[(255, 255, 255)] * 4)
    ref_masks = write_json(tmp_path / "ref_masks.json", mask_document(reference_record))
    cand_masks = write_json(tmp_path / "cand_masks.json", mask_document(candidate_record))
    geometry = geometry_document()
    ref_scene = write_json(tmp_path / "ref_scene.json", geometry)
    cand_scene = write_json(tmp_path / "cand_scene.json", geometry)
    output = tmp_path / "comparison.json"

    report = compare_workspace(
        reference_masks=ref_masks,
        candidate_masks=cand_masks,
        reference_scene=ref_scene,
        candidate_scene=cand_scene,
        output=output,
        min_score=0.90,
    )

    assert report["schema_version"] == 2
    assert report["score_weights"]["spatial_rgb_similarity"] == 0.15
    assert report["summary"]["mean_spatial_rgb_similarity"] == 0.0
    assert report["score"] == pytest.approx(0.85)
    assert report["passed"] is False
    assert list(report["views"]) == list(CANONICAL_VIEW_NAMES)
    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_compare_workspace_identical_inputs_score_one(tmp_path: Path) -> None:
    record = packed(
        ["10", "01"],
        colors=[(255, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 255)],
    )
    masks = write_json(tmp_path / "masks.json", mask_document(record))
    scene = write_json(tmp_path / "scene.json", geometry_document())

    report = compare_workspace(
        reference_masks=masks,
        candidate_masks=masks,
        reference_scene=scene,
        candidate_scene=scene,
        output=tmp_path / "comparison.json",
        min_score=1.0,
    )

    assert report["score"] == pytest.approx(1.0)
    assert report["score_passed"] is True
    assert report["hard_gates"]["passed"] is True
    assert report["hard_gates"]["failures"] == []
    assert report["passed"] is True


def test_omitting_surface_data_preserves_the_exact_legacy_report(tmp_path: Path) -> None:
    record = packed(["10", "01"])
    masks = write_json(tmp_path / "masks.json", mask_document(record))
    scene = write_json(tmp_path / "scene.json", geometry_document())

    legacy = compare_workspace(
        reference_masks=masks,
        candidate_masks=masks,
        reference_scene=scene,
        candidate_scene=scene,
        output=tmp_path / "legacy.json",
        min_score=0.5,
    )
    explicit_none = compare_workspace(
        reference_masks=masks,
        candidate_masks=masks,
        reference_scene=scene,
        candidate_scene=scene,
        output=tmp_path / "explicit-none.json",
        min_score=0.5,
        surface_comparison=None,
        surface_gate_thresholds=None,
    )

    assert explicit_none == legacy
    assert "surface_comparison" not in legacy
    assert "mean_surface_distance" not in legacy["summary"]
    assert "max_mean_surface_distance" not in legacy["hard_gates"]["thresholds"]


def test_surface_distances_add_diagnostics_and_noncompensating_gates(
    tmp_path: Path,
) -> None:
    surface_path = write_json(tmp_path / "surface.json", surface_document())

    accepted = compare_scenes(
        tmp_path,
        reference=geometry_document(),
        candidate=geometry_document(),
        min_score=1.0,
        surface_comparison=surface_path,
    )

    assert accepted["score"] == 1.0
    assert accepted["score_weights"] == {
        "silhouette_iou": 0.50,
        "area_similarity": 0.10,
        "spatial_rgb_similarity": 0.15,
        "dimension_similarity": 0.15,
        "center_similarity": 0.10,
    }
    assert accepted["summary"]["mean_surface_distance"] == 0.020
    assert accepted["summary"]["p95_surface_distance"] == 0.050
    assert accepted["surface_comparison"]["sampling"][
        "requested_samples_per_direction"
    ] == 4
    assert accepted["hard_gates"]["results"]["mean_surface_distance"]["passed"]
    assert accepted["hard_gates"]["results"]["p95_surface_distance"]["passed"]
    assert accepted["passed"] is True

    rejected = compare_scenes(
        tmp_path,
        reference=geometry_document(),
        candidate=geometry_document(),
        min_score=1.0,
        surface_comparison=surface_path,
        surface_gate_thresholds=SurfaceGateThresholds(
            max_mean_surface_distance=0.015,
            max_p95_surface_distance=0.040,
        ),
    )

    assert rejected["score"] == accepted["score"]
    assert rejected["score_passed"] is True
    assert rejected["passed"] is False
    assert rejected["hard_gates"]["thresholds"]["max_mean_surface_distance"] == 0.015
    assert rejected["hard_gates"]["thresholds"]["max_p95_surface_distance"] == 0.040
    assert [failure["gate"] for failure in rejected["hard_gates"]["failures"]] == [
        "mean_surface_distance",
        "p95_surface_distance",
    ]


def test_enhanced_surface_gates_cover_normals_visibility_and_area(
    tmp_path: Path,
) -> None:
    thresholds = SurfaceGateThresholds(
        max_mean_surface_distance=0.035,
        max_p95_surface_distance=0.080,
        max_mean_normal_angle_degrees=20.0,
        min_visible_coverage=0.90,
        min_surface_area_ratio=0.80,
        max_surface_area_ratio=1.25,
    )
    accepted_path = write_json(
        tmp_path / "enhanced-surface.json", enhanced_surface_document()
    )
    accepted = compare_scenes(
        tmp_path,
        reference=geometry_document(),
        candidate=geometry_document(),
        surface_comparison=accepted_path,
        surface_gate_thresholds=thresholds,
    )
    assert accepted["passed"]
    assert accepted["summary"]["mean_normal_angle_degrees"] == 12.0
    assert accepted["summary"]["visible_surface_coverage"] == 0.95
    assert accepted["summary"]["surface_area_ratio"] == pytest.approx(7.0 / 6.0)

    rejected_path = write_json(
        tmp_path / "bad-enhanced-surface.json",
        enhanced_surface_document(
            mean_normal_angle=35.0,
            visible_coverage=0.50,
            candidate_area=9.0,
        ),
    )
    rejected = compare_scenes(
        tmp_path,
        reference=geometry_document(),
        candidate=geometry_document(),
        surface_comparison=rejected_path,
        surface_gate_thresholds=thresholds,
    )
    failed = {
        item["gate"] for item in rejected["hard_gates"]["failures"]
    }
    assert {
        "mean_normal_angle_degrees",
        "visible_surface_coverage",
        "maximum_surface_area_ratio",
    } <= failed


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value.update(schema_version=2),
            "must use schema_version 1",
        ),
        (
            lambda value: value["sampling"].update(
                requested_samples_per_direction=True
            ),
            "requested_samples_per_direction must be a positive integer",
        ),
        (
            lambda value: value["candidate_to_reference"].update(samples=3),
            "samples must equal",
        ),
        (
            lambda value: value["reference_to_candidate"].update(
                source_surface_area=5.0
            ),
            "source_surface_area does not match",
        ),
        (
            lambda value: value["candidate_to_reference"].update(mean=-0.1),
            "mean must be a finite non-negative number",
        ),
        (
            lambda value: value["symmetric"].update(p95=0.2, max=0.1),
            "p95 cannot exceed symmetric.max",
        ),
        (
            lambda value: value["candidate_to_reference"].update(
                worst_samples="invalid"
            ),
            "worst_samples must be an array",
        ),
        (
            lambda value: value["reference_normalization"].update(
                source_min=[0.0, 0.0]
            ),
            "source_min must be a 3-element array",
        ),
    ],
)
def test_surface_comparison_schema_is_validated_exhaustively(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    document = surface_document()
    mutation(document)
    surface_path = write_json(tmp_path / "invalid-surface.json", document)

    with pytest.raises(ValueError, match=message):
        compare_scenes(
            tmp_path,
            reference=geometry_document(),
            candidate=geometry_document(),
            surface_comparison=surface_path,
        )


@pytest.mark.parametrize(
    "thresholds",
    [
        {"max_mean_surface_distance": -0.1},
        {"max_p95_surface_distance": float("nan")},
        {"max_mean_surface_distance": True},
    ],
)
def test_surface_gate_thresholds_reject_invalid_limits(thresholds: dict) -> None:
    with pytest.raises(ValueError, match="must be finite and non-negative"):
        SurfaceGateThresholds(**thresholds)


def test_surface_thresholds_require_surface_data(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires a surface_comparison path"):
        compare_scenes(
            tmp_path,
            reference=geometry_document(),
            candidate=geometry_document(),
            surface_gate_thresholds=SurfaceGateThresholds(),
        )


def test_low_single_view_fails_hard_gate_despite_passing_score(tmp_path: Path) -> None:
    reference_record = packed(["1100", "1100"])
    candidate_document = mask_document(reference_record)
    candidate_document["views"]["left"] = packed(["0011", "0011"])
    reference_masks = write_json(
        tmp_path / "reference-masks.json", mask_document(reference_record)
    )
    candidate_masks = write_json(tmp_path / "candidate-masks.json", candidate_document)
    scene = write_json(tmp_path / "scene.json", geometry_document())
    output = tmp_path / "comparison.json"

    report = compare_workspace(
        reference_masks=reference_masks,
        candidate_masks=candidate_masks,
        reference_scene=scene,
        candidate_scene=scene,
        output=output,
        min_score=0.50,
    )

    assert report["score_passed"] is True
    assert report["passed"] is False
    gate = report["hard_gates"]["results"]["minimum_view_silhouette_iou"]
    assert gate == {
        "value": 0.0,
        "operator": ">=",
        "threshold": 0.30,
        "passed": False,
        "message": "left silhouette IoU 0.0000 must be at least 0.3000",
        "view": "left",
    }
    assert [failure["gate"] for failure in report["hard_gates"]["failures"]] == [
        "minimum_view_silhouette_iou"
    ]
    assert json.loads(output.read_text(encoding="utf-8")) == report


@pytest.mark.parametrize(
    ("translation", "failed_gate", "value"),
    [
        ((0.0, 0.0, -0.20), "ground_offset", 0.20),
        ((0.40, 0.0, 0.0), "center_distance", 0.40),
    ],
)
def test_spatial_hard_gates_reject_displaced_geometry(
    tmp_path: Path,
    translation: tuple[float, float, float],
    failed_gate: str,
    value: float,
) -> None:
    candidate = geometry_document()
    for field in ("min", "max", "center"):
        candidate["bounds"][field] = [
            coordinate + offset
            for coordinate, offset in zip(
                candidate["bounds"][field], translation, strict=True
            )
        ]

    report = compare_scenes(
        tmp_path,
        reference=geometry_document(),
        candidate=candidate,
        min_score=0.50,
    )

    assert report["score_passed"] is True
    assert report["passed"] is False
    assert report["hard_gates"]["results"][failed_gate]["value"] == pytest.approx(value)
    assert [failure["gate"] for failure in report["hard_gates"]["failures"]] == [
        failed_gate
    ]


def test_custom_hard_gate_thresholds_can_relax_a_profile(tmp_path: Path) -> None:
    candidate = geometry_document()
    candidate["bounds"]["min"][0] += 0.40
    candidate["bounds"]["max"][0] += 0.40
    candidate["bounds"]["center"][0] += 0.40
    thresholds = FidelityGateThresholds(max_center_distance=0.50)

    report = compare_scenes(
        tmp_path,
        reference=geometry_document(),
        candidate=candidate,
        min_score=0.50,
        gate_thresholds=thresholds,
    )

    assert report["hard_gates"]["thresholds"]["max_center_distance"] == 0.50
    assert report["hard_gates"]["passed"] is True
    assert report["passed"] is True


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("min", [0.0, 0.0], r"bounds\.min must be a 3-element array"),
        ("max", [1.0, 2.0, float("nan")], r"bounds\.max\[2\] must be a finite number"),
        (
            "dimensions",
            [1.0, float("inf"), 3.0],
            r"bounds\.dimensions\[1\] must be a finite number",
        ),
        ("center", [0.5, True, 1.5], r"bounds\.center\[1\] must be a finite number"),
    ],
)
def test_compare_workspace_rejects_malformed_scene_vectors(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    candidate = geometry_document()
    candidate["bounds"][field] = value

    with pytest.raises(ValueError, match="candidate scene " + message):
        compare_scenes(
            tmp_path,
            reference=geometry_document(),
            candidate=candidate,
        )


@pytest.mark.parametrize(
    ("bounds", "message"),
    [
        (
            {
                "min": [2.0, 0.0, 0.0],
                "max": [1.0, 2.0, 3.0],
                "dimensions": [1.0, 2.0, 3.0],
                "center": [1.5, 1.0, 1.5],
            },
            r"bounds\.min\[0\] cannot exceed bounds\.max\[0\]",
        ),
        (
            {
                "min": [0.0, 0.0, 0.0],
                "max": [1.0, 2.0, 3.0],
                "dimensions": [-1.0, 2.0, 3.0],
                "center": [0.5, 1.0, 1.5],
            },
            r"bounds\.dimensions\[0\] cannot be negative",
        ),
        (
            {
                "min": [0.0, 0.0, 0.0],
                "max": [1.0, 2.0, 3.0],
                "dimensions": [1.5, 2.0, 3.0],
                "center": [0.5, 1.0, 1.5],
            },
            r"bounds\.dimensions\[0\] does not match max - min",
        ),
        (
            {
                "min": [0.0, 0.0, 0.0],
                "max": [1.0, 2.0, 3.0],
                "dimensions": [1.0, 2.0, 3.0],
                "center": [0.75, 1.0, 1.5],
            },
            r"bounds\.center\[0\] is not the bounds midpoint",
        ),
        (
            {
                "min": [0.0, 0.0, 0.0],
                "max": [0.0, 0.0, 0.0],
                "dimensions": [0.0, 0.0, 0.0],
                "center": [0.0, 0.0, 0.0],
            },
            r"bounds must span at least one axis",
        ),
    ],
)
def test_compare_workspace_rejects_incoherent_scene_bounds(
    tmp_path: Path,
    bounds: dict,
    message: str,
) -> None:
    with pytest.raises(ValueError, match="candidate scene " + message):
        compare_scenes(
            tmp_path,
            reference=geometry_document(),
            candidate={"bounds": bounds},
        )


@pytest.mark.parametrize("scene", [None, [], {}, {"bounds": []}])
def test_compare_workspace_requires_scene_and_bounds_objects(
    tmp_path: Path,
    scene: object,
) -> None:
    with pytest.raises(ValueError, match=r"candidate scene (must contain|bounds)"):
        compare_scenes(
            tmp_path,
            reference=geometry_document(),
            candidate=scene,
        )


@pytest.mark.parametrize(
    "min_score",
    [float("nan"), float("inf"), float("-inf"), -0.01, 1.01, True, "0.5"],
)
def test_compare_workspace_rejects_invalid_score_threshold(
    tmp_path: Path,
    min_score: object,
) -> None:
    with pytest.raises(ValueError, match="min_score must be a finite number between 0 and 1"):
        compare_scenes(
            tmp_path,
            reference=geometry_document(),
            candidate=geometry_document(),
            min_score=min_score,
        )


def test_compare_workspace_rejects_non_finite_derived_center_distance(tmp_path: Path) -> None:
    span = 4.0e292
    reference_bounds = {
        "min": [1.0e308 - span * 0.5, 0.0, 0.0],
        "max": [1.0e308 + span * 0.5, 1.0, 1.0],
    }
    candidate_bounds = {
        "min": [-1.0e308 - span * 0.5, 0.0, 0.0],
        "max": [-1.0e308 + span * 0.5, 1.0, 1.0],
    }
    for bounds in (reference_bounds, candidate_bounds):
        bounds["dimensions"] = [
            bounds["max"][index] - bounds["min"][index] for index in range(3)
        ]
        bounds["center"] = [
            bounds["min"][index] + bounds["dimensions"][index] * 0.5
            for index in range(3)
        ]

    with pytest.raises(ValueError, match="center distance is not finite"):
        compare_scenes(
            tmp_path,
            reference={"bounds": reference_bounds},
            candidate={"bounds": candidate_bounds},
        )
