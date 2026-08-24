from __future__ import annotations

import pytest

from procagen3d.granularity import (
    DEFAULT_GRANULARITY,
    GRANULARITY_LEVELS,
    GRANULARITY_PROFILES,
    get_granularity_profile,
    validate_granularity,
)


def test_granularity_profiles_preserve_medium_and_gate_fine_levels() -> None:
    assert GRANULARITY_LEVELS == ("coarse", "medium", "fine", "surface")
    assert DEFAULT_GRANULARITY == "medium"
    assert not get_granularity_profile("coarse").surface_evaluation_enabled
    assert not get_granularity_profile("medium").surface_evaluation_enabled

    fine = get_granularity_profile("fine")
    surface = get_granularity_profile("surface")
    assert fine.surface_evaluation_enabled
    assert surface.surface_sample_budget > fine.surface_sample_budget
    assert surface.max_mean_surface_distance < fine.max_mean_surface_distance
    assert surface.max_p95_surface_distance < fine.max_p95_surface_distance


def test_granularity_registry_is_immutable_and_validation_is_actionable() -> None:
    with pytest.raises(TypeError):
        GRANULARITY_PROFILES["custom"] = GRANULARITY_PROFILES["medium"]
    with pytest.raises(ValueError, match="coarse, medium, fine, surface"):
        validate_granularity("microscopic")

