"""Granularity profiles for procedural reconstruction and surface scoring."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


GRANULARITY_LEVELS = ("coarse", "medium", "fine", "surface")
DEFAULT_GRANULARITY = "medium"


@dataclass(frozen=True, slots=True)
class GranularityProfile:
    """Immutable runtime settings for one reconstruction granularity level."""

    level: str
    surface_sample_budget: int
    max_mean_surface_distance: float | None
    max_p95_surface_distance: float | None

    def __post_init__(self) -> None:
        if self.level not in GRANULARITY_LEVELS:
            raise ValueError(f"unknown granularity level {self.level!r}")
        if (
            isinstance(self.surface_sample_budget, bool)
            or not isinstance(self.surface_sample_budget, int)
            or self.surface_sample_budget < 0
        ):
            raise ValueError("surface_sample_budget must be a non-negative integer")

        thresholds = (
            self.max_mean_surface_distance,
            self.max_p95_surface_distance,
        )
        if self.surface_sample_budget == 0:
            if thresholds != (None, None):
                raise ValueError("disabled surface evaluation must not define thresholds")
            return
        for name, value in zip(
            ("max_mean_surface_distance", "max_p95_surface_distance"),
            thresholds,
            strict=True,
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"{name} must be a positive finite number")
        if self.max_mean_surface_distance > self.max_p95_surface_distance:
            raise ValueError("mean surface-distance threshold cannot exceed p95")

    @property
    def surface_evaluation_enabled(self) -> bool:
        """Whether this profile requires deterministic 3D surface comparison."""

        return self.surface_sample_budget > 0


GRANULARITY_PROFILES: Mapping[str, GranularityProfile] = MappingProxyType(
    {
        "coarse": GranularityProfile("coarse", 0, None, None),
        "medium": GranularityProfile("medium", 0, None, None),
        "fine": GranularityProfile("fine", 20_000, 0.035, 0.080),
        "surface": GranularityProfile("surface", 80_000, 0.015, 0.040),
    }
)


def validate_granularity(value: str) -> str:
    """Return a supported granularity level or raise a clear validation error."""

    if value not in GRANULARITY_LEVELS:
        allowed = ", ".join(GRANULARITY_LEVELS)
        raise ValueError(f"granularity must be one of: {allowed}")
    return value


def get_granularity_profile(value: str) -> GranularityProfile:
    """Return the immutable profile for a validated granularity level."""

    return GRANULARITY_PROFILES[validate_granularity(value)]
