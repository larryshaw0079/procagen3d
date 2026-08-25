"""Independent quality axes and acceptance settings for reconstruction builds.

``granularity`` remains a convenient authoring preset, but it no longer has to
stand in for every kind of quality.  A run can independently request surface
fidelity, detail richness, material fidelity, and structural coherence.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any, Mapping

from .granularity import DEFAULT_GRANULARITY, GRANULARITY_LEVELS, validate_granularity


SURFACE_FIDELITY_LEVELS = ("off", "balanced", "strict")
DETAIL_RICHNESS_LEVELS = ("basic", "standard", "rich", "maximum")
MATERIAL_FIDELITY_LEVELS = ("basic", "faithful", "strict")
STRUCTURAL_COHERENCE_LEVELS = ("basic", "coherent", "strict")
QUALITY_PROFILE_FIELDS = (
    "surface_fidelity",
    "detail_richness",
    "material_fidelity",
    "structural_coherence",
)


def _choice(value: str, *, name: str, allowed: tuple[str, ...]) -> str:
    if value not in allowed:
        raise ValueError(f"{name} must be one of: {', '.join(allowed)}")
    return value


@dataclass(frozen=True, slots=True)
class QualityProfile:
    """Four independently selectable quality requirements."""

    surface_fidelity: str
    detail_richness: str
    material_fidelity: str
    structural_coherence: str

    def __post_init__(self) -> None:
        _choice(
            self.surface_fidelity,
            name="surface_fidelity",
            allowed=SURFACE_FIDELITY_LEVELS,
        )
        _choice(
            self.detail_richness,
            name="detail_richness",
            allowed=DETAIL_RICHNESS_LEVELS,
        )
        _choice(
            self.material_fidelity,
            name="material_fidelity",
            allowed=MATERIAL_FIDELITY_LEVELS,
        )
        _choice(
            self.structural_coherence,
            name="structural_coherence",
            allowed=STRUCTURAL_COHERENCE_LEVELS,
        )

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SurfaceQualitySettings:
    sample_budget: int
    max_mean_distance: float | None
    max_p95_distance: float | None
    max_mean_normal_angle_degrees: float | None
    min_visible_coverage: float | None
    min_surface_area_ratio: float | None
    max_surface_area_ratio: float | None

    def __post_init__(self) -> None:
        if (
            isinstance(self.sample_budget, bool)
            or not isinstance(self.sample_budget, int)
            or self.sample_budget < 0
        ):
            raise ValueError("sample_budget must be a non-negative integer")
        limits = (
            self.max_mean_distance,
            self.max_p95_distance,
            self.max_mean_normal_angle_degrees,
            self.min_visible_coverage,
            self.min_surface_area_ratio,
            self.max_surface_area_ratio,
        )
        if self.sample_budget == 0:
            if any(value is not None for value in limits):
                raise ValueError("disabled surface quality cannot define thresholds")
            return
        if any(
            value is None
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0.0
            for value in limits
        ):
            raise ValueError("enabled surface quality requires positive finite thresholds")
        assert self.max_mean_distance is not None
        assert self.max_p95_distance is not None
        assert self.max_mean_normal_angle_degrees is not None
        assert self.min_visible_coverage is not None
        assert self.min_surface_area_ratio is not None
        assert self.max_surface_area_ratio is not None
        if self.max_mean_distance > self.max_p95_distance:
            raise ValueError("mean surface-distance threshold cannot exceed p95")
        if self.max_mean_normal_angle_degrees > 180.0:
            raise ValueError("normal-angle threshold cannot exceed 180 degrees")
        if self.min_visible_coverage > 1.0:
            raise ValueError("visible coverage must not exceed one")
        if self.min_surface_area_ratio > self.max_surface_area_ratio:
            raise ValueError("minimum surface-area ratio cannot exceed maximum")

    @property
    def enabled(self) -> bool:
        return self.sample_budget > 0


@dataclass(frozen=True, slots=True)
class DetailQualitySettings:
    min_triangle_ratio: float
    min_semantic_part_coverage: float


@dataclass(frozen=True, slots=True)
class MaterialQualitySettings:
    min_palette_similarity: float
    min_spatial_rgb_similarity: float
    max_default_white_primitive_fraction: float


@dataclass(frozen=True, slots=True)
class StructuralQualitySettings:
    max_loose_component_fraction: float
    max_boundary_edge_fraction: float
    max_non_manifold_edge_fraction: float
    max_inverted_normal_fraction: float
    max_degenerate_triangle_fraction: float
    max_low_quality_triangle_fraction: float
    max_unjoined_attachment_fraction: float
    max_intersection_fraction: float


GRANULARITY_QUALITY_PRESETS: Mapping[str, QualityProfile] = MappingProxyType(
    {
        "coarse": QualityProfile("off", "basic", "basic", "basic"),
        "medium": QualityProfile("off", "standard", "faithful", "coherent"),
        "fine": QualityProfile("balanced", "rich", "faithful", "coherent"),
        "surface": QualityProfile("strict", "maximum", "strict", "strict"),
    }
)

SURFACE_QUALITY_SETTINGS: Mapping[str, SurfaceQualitySettings] = MappingProxyType(
    {
        "off": SurfaceQualitySettings(0, None, None, None, None, None, None),
        "balanced": SurfaceQualitySettings(
            20_000, 0.035, 0.080, 35.0, 0.75, 0.65, 1.50
        ),
        "strict": SurfaceQualitySettings(
            80_000, 0.015, 0.040, 20.0, 0.90, 0.80, 1.25
        ),
    }
)

DETAIL_QUALITY_SETTINGS: Mapping[str, DetailQualitySettings] = MappingProxyType(
    {
        "basic": DetailQualitySettings(0.05, 0.50),
        "standard": DetailQualitySettings(0.10, 0.65),
        "rich": DetailQualitySettings(0.20, 0.80),
        "maximum": DetailQualitySettings(0.40, 0.95),
    }
)

MATERIAL_QUALITY_SETTINGS: Mapping[str, MaterialQualitySettings] = MappingProxyType(
    {
        "basic": MaterialQualitySettings(0.15, 0.15, 0.75),
        "faithful": MaterialQualitySettings(0.35, 0.30, 0.25),
        "strict": MaterialQualitySettings(0.50, 0.45, 0.05),
    }
)

STRUCTURAL_QUALITY_SETTINGS: Mapping[str, StructuralQualitySettings] = MappingProxyType(
    {
        "basic": StructuralQualitySettings(
            0.25, 0.10, 0.10, 0.05, 0.01, 0.15, 0.25, 0.25
        ),
        "coherent": StructuralQualitySettings(
            0.08, 0.03, 0.03, 0.01, 0.002, 0.08, 0.05, 0.08
        ),
        "strict": StructuralQualitySettings(
            0.02, 0.005, 0.005, 0.002, 0.0005, 0.03, 0.0, 0.03
        ),
    }
)


def resolve_quality_profile(
    granularity: str = DEFAULT_GRANULARITY,
    *,
    surface_fidelity: str | None = None,
    detail_richness: str | None = None,
    material_fidelity: str | None = None,
    structural_coherence: str | None = None,
) -> QualityProfile:
    """Resolve optional per-axis overrides on top of a granularity preset."""

    preset = GRANULARITY_QUALITY_PRESETS[validate_granularity(granularity)]
    return QualityProfile(
        surface_fidelity=(
            preset.surface_fidelity if surface_fidelity is None else surface_fidelity
        ),
        detail_richness=(
            preset.detail_richness if detail_richness is None else detail_richness
        ),
        material_fidelity=(
            preset.material_fidelity if material_fidelity is None else material_fidelity
        ),
        structural_coherence=(
            preset.structural_coherence
            if structural_coherence is None
            else structural_coherence
        ),
    )


def quality_profile_from_mapping(
    value: Mapping[str, Any] | None,
    *,
    granularity: str = DEFAULT_GRANULARITY,
) -> QualityProfile:
    """Validate a serialized profile, filling absent axes from its preset."""

    if value is None:
        return resolve_quality_profile(granularity)
    if not isinstance(value, Mapping):
        raise ValueError("quality_profile must be an object")
    unexpected = [name for name in value if name not in QUALITY_PROFILE_FIELDS]
    if unexpected:
        raise ValueError(
            "quality_profile contains unknown fields: "
            + ", ".join(sorted(repr(name) for name in unexpected))
        )
    for name, raw in value.items():
        if not isinstance(raw, str):
            raise ValueError(f"quality_profile.{name} must be a string")
    return resolve_quality_profile(
        granularity,
        surface_fidelity=value.get("surface_fidelity"),
        detail_richness=value.get("detail_richness"),
        material_fidelity=value.get("material_fidelity"),
        structural_coherence=value.get("structural_coherence"),
    )


def _validate_settings() -> None:
    """Fail at import time if a profile table contains an invalid threshold."""

    if set(GRANULARITY_QUALITY_PRESETS) != set(GRANULARITY_LEVELS):
        raise RuntimeError("every granularity must define a quality preset")
    for table in (
        DETAIL_QUALITY_SETTINGS,
        MATERIAL_QUALITY_SETTINGS,
        STRUCTURAL_QUALITY_SETTINGS,
    ):
        for settings in table.values():
            for value in asdict(settings).values():
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise RuntimeError("quality thresholds must be finite")
                if not 0.0 <= float(value) <= 1.0:
                    raise RuntimeError("quality fractions must be between zero and one")


_validate_settings()
