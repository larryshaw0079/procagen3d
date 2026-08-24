"""Reconstruction-mode contracts shared by the CLI, prompts, and builder."""

from __future__ import annotations


RECONSTRUCTION_MODES = ("procedural", "glb-ref")
DEFAULT_RECONSTRUCTION_MODE = "procedural"


def validate_reconstruction_mode(value: str) -> str:
    if value not in RECONSTRUCTION_MODES:
        allowed = ", ".join(RECONSTRUCTION_MODES)
        raise ValueError(f"reconstruction_mode must be one of: {allowed}")
    return value
