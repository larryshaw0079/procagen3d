from __future__ import annotations

from io import StringIO

import pytest
from rich.console import Console

from procagen3d.progress import ProgressEvent, emit_progress, progress_step
from procagen3d.rich_ui import ProgressConsole, RichProgressReporter


def test_progress_step_reports_start_and_custom_success() -> None:
    events: list[ProgressEvent] = []

    with progress_step(events.append, "compile", "Compiling Blender source") as outcome:
        outcome.complete("Compiled Blender source", model="artifacts/model.glb")

    assert [event.kind for event in events] == ["start", "success"]
    assert events[0] == ProgressEvent(
        kind="start",
        stage="compile",
        message="Compiling Blender source",
    )
    success = events[1]
    assert success.stage == "compile"
    assert success.message == "Compiled Blender source"
    assert success.elapsed_s is not None and success.elapsed_s >= 0.0
    assert success.details == {"model": "artifacts/model.glb"}


def test_progress_step_can_finish_with_a_warning() -> None:
    events: list[ProgressEvent] = []

    with progress_step(events.append, "cache", "Validating cache") as outcome:
        outcome.complete("Cache incomplete; rebuilding", kind="warning")

    assert [event.kind for event in events] == ["start", "warning"]
    assert events[-1].elapsed_s is not None


def test_progress_step_reports_failure_and_preserves_original_exception() -> None:
    events: list[ProgressEvent] = []

    with pytest.raises(RuntimeError, match="Blender exploded") as raised:
        with progress_step(events.append, "compile", "Compiling Blender source"):
            original = RuntimeError("Blender exploded")
            raise original

    assert raised.value is original
    assert [event.kind for event in events] == ["start", "failure"]
    failure = events[1]
    assert failure.stage == "compile"
    assert failure.message == "Compiling Blender source failed"
    assert failure.elapsed_s is not None and failure.elapsed_s >= 0.0
    assert failure.details == {"error": "Blender exploded"}


def test_reporter_exceptions_are_observational_only() -> None:
    calls: list[ProgressEvent] = []

    def broken_reporter(event: ProgressEvent) -> None:
        calls.append(event)
        raise RuntimeError("terminal renderer failed")

    emit_progress(broken_reporter, "info", "reference", "Inspecting reference")
    with progress_step(broken_reporter, "compile", "Compiling source"):
        pass

    assert [event.kind for event in calls] == ["info", "start", "success"]

    with pytest.raises(ValueError, match="pipeline failure"):
        with progress_step(broken_reporter, "probe", "Probing compiled GLB"):
            raise ValueError("pipeline failure")

    assert [event.kind for event in calls[-2:]] == ["start", "failure"]


def test_broken_progress_pipe_cannot_exit_the_pipeline() -> None:
    def closed_pipe(_event: ProgressEvent) -> None:
        raise SystemExit(1)

    emit_progress(closed_pipe, "info", "reference", "Inspecting reference")

    with progress_step(closed_pipe, "compile", "Compiling source"):
        pass


def test_broken_progress_stderr_does_not_rewire_stdout() -> None:
    class ClosedStream(StringIO):
        def write(self, _value: str) -> int:
            raise BrokenPipeError

    console = ProgressConsole(
        file=ClosedStream(),
        force_terminal=False,
        color_system=None,
    )
    reporter = RichProgressReporter(console)

    emit_progress(reporter, "info", "reference", "Inspecting reference")

    assert console.quiet is True


def test_rich_reporter_non_tty_prints_start_and_done_lines() -> None:
    stream = StringIO()
    console = Console(
        file=stream,
        force_terminal=False,
        color_system=None,
        width=120,
    )
    reporter = RichProgressReporter(console)

    reporter(ProgressEvent("start", "agent", "Running Codex agent"))
    reporter(
        ProgressEvent(
            "success",
            "agent",
            "Codex agent completed",
            elapsed_s=1.25,
        )
    )
    reporter.close()

    output = stream.getvalue()
    assert "Running Codex agent" in output
    assert "Codex agent completed" in output
    assert "1.2s" in output


def test_agent_activity_line_keeps_the_blocking_stage_active() -> None:
    stream = StringIO()
    reporter = RichProgressReporter(
        Console(file=stream, force_terminal=False, color_system=None, width=160)
    )

    reporter(ProgressEvent("start", "agent-00", "Authoring Blender source"))
    reporter(
        ProgressEvent(
            "info",
            "agent-00",
            "Codex process still running — 30s elapsed; plan.json 14.6 KiB",
        )
    )

    assert reporter._active_stage == "agent-00"
    assert "process still running" in stream.getvalue()

    reporter(ProgressEvent("success", "agent-00", "Codex produced source", elapsed_s=31.0))
    assert reporter._active_stage is None


def test_rich_cleanup_cannot_mask_pipeline_exception() -> None:
    class BrokenStatus:
        def stop(self) -> None:
            raise RuntimeError("terminal cleanup failed")

    reporter = RichProgressReporter(
        Console(file=StringIO(), force_terminal=False, color_system=None)
    )
    original = ValueError("pipeline failed")

    with pytest.raises(ValueError, match="pipeline failed") as raised:
        with reporter:
            reporter._status = BrokenStatus()  # type: ignore[assignment]
            reporter._active_stage = "build"
            raise original

    assert raised.value is original
    assert reporter._status is None
    assert reporter._active_stage is None
