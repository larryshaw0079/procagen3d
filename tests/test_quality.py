from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from procagen3d.quality import (
    DETAIL_QUALITY_SETTINGS,
    GRANULARITY_QUALITY_PRESETS,
    MATERIAL_QUALITY_SETTINGS,
    SURFACE_QUALITY_SETTINGS,
    QualityProfile,
    quality_profile_from_mapping,
    resolve_quality_profile,
)


@pytest.mark.parametrize(
    ("granularity", "expected"),
    [
        ("coarse", QualityProfile("off", "basic", "basic", "basic")),
        ("medium", QualityProfile("off", "standard", "faithful", "coherent")),
        ("fine", QualityProfile("balanced", "rich", "faithful", "coherent")),
        ("surface", QualityProfile("strict", "maximum", "strict", "strict")),
    ],
)
def test_granularity_presets_resolve_all_independent_axes(
    granularity: str,
    expected: QualityProfile,
) -> None:
    assert resolve_quality_profile(granularity) == expected
    assert GRANULARITY_QUALITY_PRESETS[granularity] == expected


def test_independent_overrides_do_not_change_unrelated_axes() -> None:
    resolved = resolve_quality_profile(
        "fine",
        surface_fidelity="strict",
        material_fidelity="basic",
    )

    assert resolved == QualityProfile("strict", "rich", "basic", "coherent")
    assert resolved.as_dict() == {
        "surface_fidelity": "strict",
        "detail_richness": "rich",
        "material_fidelity": "basic",
        "structural_coherence": "coherent",
    }


def test_partial_serialized_profile_inherits_from_granularity() -> None:
    resolved = quality_profile_from_mapping(
        {"detail_richness": "maximum"},
        granularity="medium",
    )

    assert resolved == QualityProfile("off", "maximum", "faithful", "coherent")


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({"surface_fidelity": ""}, "surface_fidelity"),
        ({"detail_richness": 2}, "detail_richness"),
        ({"unknown_axis": "strict"}, "unknown fields"),
        ({1: "strict", "unknown_axis": "strict"}, "unknown fields"),
        ([], "must be an object"),
    ],
)
def test_serialized_profile_rejects_invalid_explicit_values(
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        quality_profile_from_mapping(value, granularity="medium")  # type: ignore[arg-type]


def test_profile_and_registries_are_immutable() -> None:
    profile = resolve_quality_profile()
    with pytest.raises(FrozenInstanceError):
        profile.surface_fidelity = "strict"  # type: ignore[misc]
    with pytest.raises(TypeError):
        GRANULARITY_QUALITY_PRESETS["medium"] = profile  # type: ignore[index]


def test_acceptance_tables_are_axis_specific() -> None:
    assert not SURFACE_QUALITY_SETTINGS["off"].enabled
    assert SURFACE_QUALITY_SETTINGS["strict"].sample_budget > (
        SURFACE_QUALITY_SETTINGS["balanced"].sample_budget
    )
    assert DETAIL_QUALITY_SETTINGS["maximum"].min_semantic_part_coverage > (
        DETAIL_QUALITY_SETTINGS["basic"].min_semantic_part_coverage
    )
    assert MATERIAL_QUALITY_SETTINGS["strict"].max_default_white_primitive_fraction < (
        MATERIAL_QUALITY_SETTINGS["basic"].max_default_white_primitive_fraction
    )
