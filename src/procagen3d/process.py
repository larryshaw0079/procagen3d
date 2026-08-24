"""Small, bounded subprocess helpers used by agent and Blender backends."""

from __future__ import annotations

import os
import queue
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence


_POST_KILL_DRAIN_S = 0.25


@dataclass(frozen=True)
class ProcessResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def run_process(
    command: Sequence[str],
    *,
    cwd: Path,
    timeout_s: float,
    input_text: str | None = None,
    env: Mapping[str, str] | None = None,
    inherit_env: bool = True,
    on_stdout_line: Callable[[str], None] | None = None,
    on_stderr_line: Callable[[str], None] | None = None,
    on_heartbeat: Callable[[float], None] | None = None,
    heartbeat_interval_s: float = 30.0,
) -> ProcessResult:
    """Run a command without a shell and retain its complete transcript.

    Stdout and stderr are drained concurrently. Optional observers receive
    complete decoded lines while the child is still running, but execute on a
    separate dispatcher so a broken renderer cannot fill a subprocess pipe or
    interfere with timeout supervision.
    """

    if timeout_s <= 0:
        raise ValueError("timeout_s must be greater than zero")
    if on_heartbeat is not None and heartbeat_interval_s <= 0:
        raise ValueError("heartbeat_interval_s must be greater than zero")

    merged_env = os.environ.copy() if inherit_env else {}
    if env:
        merged_env.update(env)
    started = time.monotonic()
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
    process = subprocess.Popen(
        list(command),
        cwd=str(cwd),
        env=merged_env,
        stdin=subprocess.PIPE if input_text is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        start_new_session=os.name != "nt",
        creationflags=creationflags,
    )
    assert process.stdout is not None
    assert process.stderr is not None

    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    observations: queue.SimpleQueue[tuple[str, str | float | None]] = queue.SimpleQueue()
    callbacks_enabled = threading.Event()
    callbacks_enabled.set()
    capture_enabled = threading.Event()
    capture_enabled.set()
    finished_streams = 0
    finished_streams_lock = threading.Lock()
    all_streams_finished = threading.Event()

    def drain_stream(channel: str, stream: object, parts: list[str]) -> None:
        nonlocal finished_streams
        try:
            readline = getattr(stream, "readline")
            while capture_enabled.is_set():
                line = readline()
                if not line or not capture_enabled.is_set():
                    break
                # Keep the transcript segment exactly as decoded, including
                # line endings. Only the observational copy drops CR/LF.
                parts.append(line)
                observations.put((channel, line.rstrip("\r\n")))
        finally:
            try:
                getattr(stream, "close")()
            except (OSError, ValueError):
                pass
            with finished_streams_lock:
                finished_streams += 1
                if finished_streams == 2:
                    all_streams_finished.set()

    def dispatch_observations() -> None:
        while True:
            channel, value = observations.get()
            if channel == "stop":
                return
            if not callbacks_enabled.is_set():
                continue
            callback: Callable[[str], None] | Callable[[float], None] | None
            if channel == "stdout":
                callback = on_stdout_line
            elif channel == "stderr":
                callback = on_stderr_line
            else:
                callback = on_heartbeat
            if callback is None:
                continue
            try:
                callback(value)  # type: ignore[arg-type]
            except BaseException:
                # Observers are presentation-only. Even SystemExit from a
                # custom reporter must not alter the subprocess result.
                continue

    stdout_thread = threading.Thread(
        target=drain_stream,
        args=("stdout", process.stdout, stdout_parts),
        name="procagen3d-stdout",
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=drain_stream,
        args=("stderr", process.stderr, stderr_parts),
        name="procagen3d-stderr",
        daemon=True,
    )
    dispatcher_thread = threading.Thread(
        target=dispatch_observations,
        name="procagen3d-observer",
        daemon=True,
    )

    def write_stdin() -> None:
        assert process.stdin is not None
        try:
            process.stdin.write(input_text or "")
            process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            pass
        finally:
            try:
                process.stdin.close()
            except (OSError, ValueError):
                pass

    stdout_thread.start()
    stderr_thread.start()
    dispatcher_thread.start()
    stdin_thread: threading.Thread | None = None
    if input_text is not None:
        stdin_thread = threading.Thread(
            target=write_stdin,
            name="procagen3d-stdin",
            daemon=True,
        )
        stdin_thread.start()

    deadline = started + timeout_s
    next_heartbeat = started + heartbeat_interval_s
    timed_out = False
    post_kill_deadline: float | None = None
    forced_capture_cutoff = False
    try:
        while process.poll() is None or not all_streams_finished.is_set():
            now = time.monotonic()
            if not timed_out and now >= deadline:
                timed_out = True
                _terminate_process_tree(process)
                post_kill_deadline = now + _POST_KILL_DRAIN_S

            if (
                timed_out
                and post_kill_deadline is not None
                and now >= post_kill_deadline
                and not all_streams_finished.is_set()
            ):
                # A malicious/detached descendant can escape the killed process
                # group while retaining inherited pipe handles. Stop retaining
                # late output rather than letting a bounded timeout hang forever.
                forced_capture_cutoff = True
                capture_enabled.clear()
                break

            if on_heartbeat is not None and not timed_out and now >= next_heartbeat:
                observations.put(("heartbeat", now - started))
                missed = int((now - next_heartbeat) // heartbeat_interval_s)
                next_heartbeat += (missed + 1) * heartbeat_interval_s

            if process.poll() is None:
                wake_at = post_kill_deadline if timed_out else deadline
                if on_heartbeat is not None and not timed_out:
                    wake_at = min(wake_at, next_heartbeat)
                assert wake_at is not None
                wait_s = max(0.001, min(0.1, wake_at - time.monotonic()))
                try:
                    process.wait(timeout=wait_s)
                except subprocess.TimeoutExpired:
                    pass
            else:
                all_streams_finished.wait(timeout=0.05)
    except BaseException:
        _terminate_process_tree(process)
        process.wait()
        raise
    finally:
        if process.poll() is None:
            _terminate_process_tree(process)
        process.wait()
        join_timeout = 0.2 if forced_capture_cutoff else 2.0
        stdout_thread.join(timeout=join_timeout)
        stderr_thread.join(timeout=join_timeout)
        if stdin_thread is not None:
            stdin_thread.join(timeout=join_timeout)
        # Reader events precede this sentinel in the FIFO queue. Fast normal
        # observers therefore finish before return, while a blocked custom
        # observer can delay the function by at most two seconds.
        if forced_capture_cutoff:
            callbacks_enabled.clear()
        observations.put(("stop", None))
        dispatcher_thread.join(timeout=join_timeout)
        callbacks_enabled.clear()

    return ProcessResult(
        command=tuple(str(part) for part in command),
        returncode=124 if timed_out else process.returncode,
        stdout="".join(stdout_parts),
        stderr="".join(stderr_parts),
        duration_s=time.monotonic() - started,
        timed_out=timed_out,
    )


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    """Best-effort termination for a command and all descendants."""

    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.kill()
    except ProcessLookupError:
        # The process-group operation may already have terminated the direct
        # child before this fallback.
        pass
