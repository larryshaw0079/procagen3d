"""Command-line LLM backend registry."""

from __future__ import annotations

from typing import Any

from .base import AgentRunResult, CLIBackend, CLIInvocation, ExitReason, ParsedOutput
from .codex import CodexBackend, CodexCLIBackend
from .cursor import CursorBackend, CursorCLIBackend
from .grok import GrokBackend, GrokCLIBackend


BACKEND_NAMES = ("codex", "grok", "cursor")


def create_backend(name: str, **overrides: Any) -> CLIBackend:
    normalized = name.strip().lower().replace("_", "-")
    if normalized == "codex":
        return CodexBackend(**overrides)
    if normalized in {"grok", "grok-build"}:
        return GrokBackend(**overrides)
    if normalized in {"cursor", "cursor-agent"}:
        return CursorBackend(**overrides)
    choices = ", ".join(BACKEND_NAMES)
    raise ValueError(f"unknown backend {name!r}; choose one of: {choices}")


__all__ = [
    "AgentRunResult",
    "BACKEND_NAMES",
    "CLIBackend",
    "CLIInvocation",
    "CodexBackend",
    "CodexCLIBackend",
    "CursorBackend",
    "CursorCLIBackend",
    "ExitReason",
    "GrokBackend",
    "GrokCLIBackend",
    "ParsedOutput",
    "create_backend",
]
