"""Provider-neutral progress events for long-running pipeline stages."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Literal, Mapping


ProgressKind = Literal["start", "success", "failure", "info", "warning"]
CompletionKind = Literal["success", "info", "warning"]


@dataclass(frozen=True)
class ProgressEvent:
    kind: ProgressKind
    stage: str
    message: str
    elapsed_s: float | None = None
    details: Mapping[str, Any] = field(default_factory=dict)


ProgressReporter = Callable[[ProgressEvent], None]


@dataclass
class StepOutcome:
    """Mutable completion details filled by a reported stage."""

    message: str | None = None
    kind: CompletionKind = "success"
    details: dict[str, Any] = field(default_factory=dict)

    def complete(
        self,
        message: str,
        *,
        kind: CompletionKind = "success",
        **details: Any,
    ) -> None:
        self.message = message
        self.kind = kind
        self.details.update(details)


def emit_progress(
    reporter: ProgressReporter | None,
    kind: ProgressKind,
    stage: str,
    message: str,
    *,
    elapsed_s: float | None = None,
    **details: Any,
) -> None:
    """Emit one event without allowing a presentation failure to stop a build."""

    if reporter is None:
        return
    try:
        reporter(
            ProgressEvent(
                kind=kind,
                stage=stage,
                message=message,
                elapsed_s=elapsed_s,
                details=details,
            )
        )
    except (Exception, SystemExit):
        # Progress is observational. Terminal rendering must never alter the
        # source/artifact transaction or turn a successful build into a failure.
        return


@contextmanager
def progress_step(
    reporter: ProgressReporter | None,
    stage: str,
    message: str,
) -> Iterator[StepOutcome]:
    """Report a spinner-friendly start followed by success or failure."""

    started = time.monotonic()
    outcome = StepOutcome()
    emit_progress(reporter, "start", stage, message)
    try:
        yield outcome
    except BaseException as exc:
        emit_progress(
            reporter,
            "failure",
            stage,
            f"{message} failed",
            elapsed_s=time.monotonic() - started,
            error=str(exc),
        )
        raise
    else:
        emit_progress(
            reporter,
            outcome.kind,
            stage,
            outcome.message or message,
            elapsed_s=time.monotonic() - started,
            **outcome.details,
        )
