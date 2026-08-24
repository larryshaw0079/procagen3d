from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from procagen3d.metrics import (
    CANONICAL_VIEW_NAMES,
    FidelityGateThresholds,
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
