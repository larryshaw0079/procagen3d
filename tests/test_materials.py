from __future__ import annotations

import copy
import json

import pytest

from procagen3d.materials import (
    GeometrySnapshot,
    MaterialAssignment,
    MaterialGeometryChangeError,
    MaterialPlan,
    MaterialPlanError,
    PBRMaterial,
    build_material_pass_context,
    compare_material_pass_geometry,
    enforce_material_only_change,
    geometry_snapshot_from_report,
    material_plan_from_document,
    summarize_material_evidence,
)


def _material(
    material_id: str = "paint",
    *,
    color: tuple[float, float, float, float] = (0.12, 0.25, 0.8, 1.0),
) -> PBRMaterial:
    return PBRMaterial(
        material_id=material_id,
        name="Deep blue paint" if material_id == "paint" else None,
        base_color_rgba=color,
        metallic=0.72,
        roughness=0.24,
    )


def _geometry_report() -> dict:
    return {
        "coordinate_system": {"up": "+Z"},
        "bounds": {
            "min": [-1.0, -0.5, 0.0],
            "max": [1.0, 0.5, 1.5],
            "dimensions": [2.0, 1.0, 1.5],
        },
        "geometry_object_count": 2,
        "mesh_count": 2,
        "objects": [
            {
                "name": "Body",
                "type": "MESH",
                "vertices": 100,
                "triangles": 180,
                "materials": ["Gray"],
                "bounds": {
                    "min": [-1.0, -0.5, 0.0],
                    "max": [1.0, 0.5, 1.0],
                    "dimensions": [2.0, 1.0, 1.0],
                },
                "structure": {"connected_components": {"count": 1}},
            },
            {
                "name": "Lamp",
                "type": "MESH",
                "vertices": 24,
                "triangles": 40,
                "materials": ["White"],
                "bounds": {
                    "min": [-0.2, -0.5, 1.0],
                    "max": [0.2, -0.3, 1.5],
                    "dimensions": [0.4, 0.2, 0.5],
                },
                "structure": {"connected_components": {"count": 1}},
            },
        ],
        "cross_sections_x": [{"x": 0.0, "samples": 12}],
        "cross_sections_y": [],
        "cross_sections_z": [],
        "structure": {"isolated_objects": []},
        "welded_components": {"Body": 1, "Lamp": 1},
        "artifact": "model.glb",
        "canonical_evidence": {"renders": "renders"},
    }


def test_pbr_material_normalizes_and_serializes_optional_channels() -> None:
    material = PBRMaterial(
        material_id="lamp_glass",
        name="Lamp glass",
        base_color_rgba=(1, 0.8, 0.3, 0.45),
        metallic=0,
        roughness=0.18,
        emissive_rgb=(1.0, 0.3, 0.05),
        alpha_mode="blend",
    )

    assert material.base_color_rgba == (1.0, 0.8, 0.3, 0.45)
    assert material.alpha_mode == "BLEND"
    assert material.effective_name == "Lamp glass"
    assert PBRMaterial.from_dict(material.as_dict()) == material


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"base_color_rgba": (1.1, 0, 0, 1)}, "at most 1"),
        ({"metallic": True}, "finite number"),
        ({"roughness": float("nan")}, "finite number"),
        ({"emissive_rgb": (0.2, 0.3)}, "exactly 3"),
        ({"alpha_mode": "HASHED"}, "alpha_mode"),
        ({"alpha_cutoff": 0.5}, "requires alpha_mode MASK"),
    ],
)
def test_pbr_material_rejects_invalid_values(changes: dict, message: str) -> None:
    values = {
        "material_id": "paint",
        "base_color_rgba": (0.2, 0.3, 0.4, 1.0),
        "metallic": 0.1,
        "roughness": 0.6,
    }
    values.update(changes)
    with pytest.raises(MaterialPlanError, match=message):
        PBRMaterial(**values)


def test_material_plan_round_trip_with_part_and_subpart_overrides() -> None:
    plan = MaterialPlan(
        materials=(_material(), _material("chrome", color=(0.7, 0.7, 0.72, 1))),
        assignments=(
            MaterialAssignment(part_id="body", material_id="paint"),
            MaterialAssignment(
                part_id="body",
                subpart_id="window_trim",
                material_id="chrome",
                object_names=("WindowTrim.L", "WindowTrim.R"),
                selector="thin trim bordering the side windows",
            ),
        ),
    )

    payload = plan.to_json()
    assert payload.endswith("\n")
    assert json.loads(payload)["schema_version"] == 1
    assert MaterialPlan.from_json(payload) == plan
    plan.validate_part_ids(["body", "wheel"])


def test_empty_legacy_material_plans_remain_disabled() -> None:
    expected = MaterialPlan()
    assert material_plan_from_document(None) == expected
    assert material_plan_from_document([]) == expected
    assert material_plan_from_document({}) == expected
    assert material_plan_from_document({"materials": []}) == expected
    assert not expected.enabled

    with pytest.raises(MaterialPlanError, match="non-empty legacy"):
        material_plan_from_document([{"name": "old paint"}])


def test_material_plan_rejects_ambiguous_or_unknown_assignments() -> None:
    with pytest.raises(MaterialPlanError, match="unknown material"):
        MaterialPlan(
            materials=(_material(),),
            assignments=(MaterialAssignment("body", "missing"),),
        )

    with pytest.raises(MaterialPlanError, match="duplicate material assignment"):
        MaterialPlan(
            materials=(_material(),),
            assignments=(
                MaterialAssignment("body", "paint"),
                MaterialAssignment("body", "paint"),
            ),
        )

    plan = MaterialPlan(
        materials=(_material(),),
        assignments=(MaterialAssignment("body", "paint"),),
    )
    with pytest.raises(MaterialPlanError, match="unknown parts: body"):
        plan.validate_part_ids(["wheel"])


def test_subpart_object_rule_may_override_whole_part_but_not_another_subpart() -> None:
    materials = (_material(), _material("chrome"), _material("rubber"))
    plan = MaterialPlan(
        materials=materials,
        assignments=(
            MaterialAssignment("body", "paint", object_names=("Body", "Trim")),
            MaterialAssignment(
                "body", "chrome", subpart_id="trim", object_names=("Trim",)
            ),
        ),
    )
    assert plan.enabled

    with pytest.raises(MaterialPlanError, match="more than one subpart"):
        MaterialPlan(
            materials=materials,
            assignments=plan.assignments
            + (
                MaterialAssignment(
                    "body", "rubber", subpart_id="seal", object_names=("Trim",)
                ),
            ),
        )


def test_subpart_assignment_requires_a_bounded_target() -> None:
    with pytest.raises(MaterialPlanError, match="object_names or a selector"):
        MaterialAssignment("body", "paint", subpart_id="stripe")
    with pytest.raises(MaterialPlanError, match="whole-part assignments"):
        MaterialAssignment("body", "paint", selector="all visible faces")


def test_geometry_snapshot_is_stable_across_object_order_and_material_changes() -> None:
    before = _geometry_report()
    after = copy.deepcopy(before)
    after["objects"].reverse()
    after["objects"][0]["materials"] = ["Emissive lamp"]
    after["objects"][1]["materials"] = ["Blue paint", "Chrome"]
    after["artifact"] = "painted.glb"

    first = geometry_snapshot_from_report(before)
    second = geometry_snapshot_from_report(after)
    assert isinstance(first, GeometrySnapshot)
    assert first == second
    result = enforce_material_only_change(first, second)
    assert result.passed
    assert result.violations == ()


def test_material_guard_reports_triangle_and_object_changes_deterministically() -> None:
    before = _geometry_report()
    after = copy.deepcopy(before)
    after["objects"][0]["triangles"] += 2
    after["objects"][1]["name"] = "Headlamp"

    result = compare_material_pass_geometry(before, after)

    assert not result.passed
    fields = [violation.field for violation in result.violations]
    assert fields == [
        "triangle_count",
        "object_names",
        "objects.Body.triangles",
        "geometry_digest",
    ]
    with pytest.raises(MaterialGeometryChangeError) as raised:
        enforce_material_only_change(before, after)
    assert raised.value.result == result


def test_material_guard_detects_count_preserving_internal_geometry_change() -> None:
    before = _geometry_report()
    after = copy.deepcopy(before)
    after["objects"][0]["structure"]["connected_components"]["count"] = 2

    result = compare_material_pass_geometry(before, after)

    assert not result.passed
    assert [item.field for item in result.violations] == ["geometry_digest"]


def test_material_triangle_fingerprint_allows_material_seam_vertex_splits() -> None:
    before = _geometry_report()
    after = copy.deepcopy(before)
    after["objects"][0]["vertices"] += 6
    fingerprint = {
        "schema_version": 1,
        "algorithm": "oriented-world-triangle-multiset-sha256-v1",
        "objects": [],
        "digest": "a" * 64,
    }
    before["material_geometry_fingerprint"] = copy.deepcopy(fingerprint)
    after["material_geometry_fingerprint"] = copy.deepcopy(fingerprint)

    result = compare_material_pass_geometry(before, after)

    assert result.passed
    assert result.before_digest == "a" * 64
    assert result.after_digest == "a" * 64

    after["material_geometry_fingerprint"]["digest"] = "b" * 64
    changed = compare_material_pass_geometry(before, after)
    assert not changed.passed
    assert [item.field for item in changed.violations] == [
        "material_geometry_fingerprint"
    ]


def test_geometry_snapshot_validates_report_consistency() -> None:
    report = _geometry_report()
    report["geometry_object_count"] = 3
    with pytest.raises(MaterialPlanError, match="does not match"):
        geometry_snapshot_from_report(report)


def test_material_evidence_summary_is_bounded_and_usage_sorted() -> None:
    report = {
        "materials": [
            {
                "index": 0,
                "name": "Unused",
                "base_color_factor": [1, 1, 1, 1],
                "metallic_factor": 1.0,
                "roughness_factor": 1.0,
                "primitive_usage_count": 0,
                "default_white_risk": False,
            },
            {
                "index": 1,
                "name": "Paint",
                "base_color_factor": [0.1, 0.2, 0.8, 1],
                "metallic_factor": 0.7,
                "roughness_factor": 0.2,
                "primitive_usage_count": 4,
                "default_white_risk": False,
            },
            {
                "index": 2,
                "name": "Implicit",
                "base_color_factor": None,
                "metallic_factor": 1.0,
                "roughness_factor": 1.0,
                "primitive_usage_count": 2,
                "default_white_risk": True,
            },
        ],
        "material_diagnostics": {
            "used_material_count": 2,
            "default_white_risk": True,
        },
    }

    summary = summarize_material_evidence(report, max_materials=2)

    assert summary["material_count"] == 3
    assert summary["used_material_count"] == 2
    assert summary["default_white_risk"] is True
    assert [item["name"] for item in summary["materials"]] == ["Paint", "Implicit"]
    assert summary["materials_truncated"] == 1


def test_scene_material_slot_names_are_usable_as_weak_evidence() -> None:
    summary = summarize_material_evidence(_geometry_report())
    assert summary["material_count"] == 2
    assert summary["material_slot_names"] == ["Gray", "White"]


def test_material_context_contains_plan_evidence_and_protected_geometry() -> None:
    plan = MaterialPlan(
        materials=(_material(),),
        assignments=(MaterialAssignment("body", "paint"),),
    )
    context = build_material_pass_context(
        plan,
        part_ids=["wheel", "body", "body"],
        reference_report={"materials": []},
        pre_material_geometry_report=_geometry_report(),
    )

    assert context["stage"] == "dedicated-pbr-material-pass"
    assert context["enabled"] is True
    assert context["part_ids"] == ["body", "wheel"]
    assert context["material_plan"] == plan.as_dict()
    assert len(context["pre_material_geometry"]["geometry_digest"]) == 64
    assert "Do not create" in context["invariants"][1]


def test_disabled_material_context_is_backward_compatible() -> None:
    context = build_material_pass_context([])
    assert context["enabled"] is False
    assert context["material_plan"] == {
        "schema_version": 1,
        "materials": [],
        "assignments": [],
    }
    assert context["pre_material_geometry"] is None
