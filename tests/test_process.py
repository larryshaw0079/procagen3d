from __future__ import annotations

import os
import sys
import time

import pytest

from procagen3d.process import run_process


def test_run_process_decodes_non_utf8_output_without_crashing(tmp_path) -> None:
    result = run_process(
        [
            sys.executable,
            "-c",
            "import sys; sys.stdout.buffer.write(b'good\\xffbad')",
        ],
        cwd=tmp_path,
        timeout_s=5,
    )

    assert result.ok
    assert result.stdout == "good\ufffdbad"


def test_run_process_marks_and_terminates_timeout(tmp_path) -> None:
    result = run_process(
        [sys.executable, "-c", "import time; print('started', flush=True); time.sleep(10)"],
        cwd=tmp_path,
        timeout_s=0.05,
    )

    assert result.timed_out is True
    assert result.returncode == 124
    assert "started" in result.stdout


def test_run_process_streams_lines_before_exit_and_preserves_exact_output(tmp_path) -> None:
    gate = tmp_path / "continue"
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    def observe_stdout(line: str) -> None:
        stdout_lines.append(line)
        if line == "ready":
            gate.write_text("continue", encoding="utf-8")

    script = "\n".join(
        (
            "import pathlib, sys, time",
            "gate = pathlib.Path(sys.argv[1])",
            "print('ready', flush=True)",
            "print('warning', file=sys.stderr, flush=True)",
            "deadline = time.monotonic() + 2",
            "while not gate.exists() and time.monotonic() < deadline: time.sleep(0.01)",
            "if not gate.exists(): raise SystemExit(7)",
            "sys.stdout.write('done')",
        )
    )
    result = run_process(
        [sys.executable, "-u", "-c", script, str(gate)],
        cwd=tmp_path,
        timeout_s=3,
        on_stdout_line=observe_stdout,
        on_stderr_line=stderr_lines.append,
    )

    assert result.ok
    assert result.stdout == "ready\ndone"
    assert result.stderr == "warning\n"
    assert stdout_lines == ["ready", "done"]
    assert stderr_lines == ["warning"]


@pytest.mark.parametrize("failure", [RuntimeError("renderer failed"), SystemExit(3)])
def test_process_observer_failures_are_observational(tmp_path, failure: BaseException) -> None:
    def broken_observer(_value) -> None:
        raise failure

    result = run_process(
        [
            sys.executable,
            "-u",
            "-c",
            "import sys; print('out'); print('err', file=sys.stderr)",
        ],
        cwd=tmp_path,
        timeout_s=3,
        on_stdout_line=broken_observer,
        on_stderr_line=broken_observer,
    )

    assert result.ok
    assert result.stdout == "out\n"
    assert result.stderr == "err\n"


def test_process_heartbeat_is_live_and_can_observe_a_running_child(tmp_path) -> None:
    gate = tmp_path / "heartbeat-seen"
    heartbeats: list[float] = []

    def heartbeat(elapsed_s: float) -> None:
        heartbeats.append(elapsed_s)
        gate.write_text("continue", encoding="utf-8")

    script = "\n".join(
        (
            "import pathlib, sys, time",
            "gate = pathlib.Path(sys.argv[1])",
            "print('waiting', flush=True)",
            "deadline = time.monotonic() + 2",
            "while not gate.exists() and time.monotonic() < deadline: time.sleep(0.01)",
            "if not gate.exists(): raise SystemExit(8)",
            "print('complete')",
        )
    )
    result = run_process(
        [sys.executable, "-u", "-c", script, str(gate)],
        cwd=tmp_path,
        timeout_s=3,
        on_heartbeat=heartbeat,
        heartbeat_interval_s=0.02,
    )

    assert result.ok
    assert result.stdout == "waiting\ncomplete\n"
    assert heartbeats and heartbeats[0] >= 0.015


def test_process_drains_large_stdout_stderr_while_writing_large_stdin(tmp_path) -> None:
    input_text = "p" * 300_000
    script = "\n".join(
        (
            "import sys, threading",
            "def write(stream, value):",
            "    stream.write(value * 300_000)",
            "    stream.flush()",
            "threads = [threading.Thread(target=write, args=(sys.stdout, 'o')),",
            "           threading.Thread(target=write, args=(sys.stderr, 'e'))]",
            "[thread.start() for thread in threads]",
            "data = sys.stdin.read()",
            "[thread.join() for thread in threads]",
            "print(len(data))",
        )
    )

    result = run_process(
        [sys.executable, "-u", "-c", script],
        cwd=tmp_path,
        timeout_s=5,
        input_text=input_text,
    )

    assert result.ok
    assert result.stdout == "o" * 300_000 + "300000\n"
    assert result.stderr == "e" * 300_000


@pytest.mark.skipif(os.name == "nt", reason="POSIX detached-process regression")
def test_timeout_does_not_wait_forever_for_an_escaped_pipe_holder(tmp_path) -> None:
    script = "\n".join(
        (
            "import subprocess, sys, time",
            "escaped = subprocess.Popen(",
            "    [sys.executable, '-c', 'import time; time.sleep(30)'],",
            "    start_new_session=True,",
            ")",
            "print(escaped.pid, flush=True)",
            "time.sleep(30)",
        )
    )
    started = time.monotonic()
    result = run_process(
        [sys.executable, "-u", "-c", script],
        cwd=tmp_path,
        timeout_s=0.2,
    )
    duration = time.monotonic() - started

    assert result.timed_out
    assert duration < 1.5
    escaped_pid = int(result.stdout.splitlines()[0])
    try:
        os.kill(escaped_pid, 9)
    except ProcessLookupError:
        pass
