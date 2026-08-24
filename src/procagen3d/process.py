"""Small, bounded subprocess helpers used by agent and Blender backends."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


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
    timeout_s: int,
    input_text: str | None = None,
    env: Mapping[str, str] | None = None,
    inherit_env: bool = True,
) -> ProcessResult:
    """Run a command without a shell and retain its complete transcript."""

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
    try:
        stdout, stderr = process.communicate(input=input_text, timeout=timeout_s)
        return ProcessResult(
            command=tuple(str(part) for part in command),
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr,
            duration_s=time.monotonic() - started,
        )
    except subprocess.TimeoutExpired as exc:
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
            # Killing the POSIX process group (or taskkill on Windows) may
            # already have reaped the direct child before this fallback.
            pass
        stdout, stderr = process.communicate()
        if not stdout and exc.stdout:
            stdout = (
                exc.stdout.decode("utf-8", errors="replace")
                if isinstance(exc.stdout, bytes)
                else exc.stdout
            )
        if not stderr and exc.stderr:
            stderr = (
                exc.stderr.decode("utf-8", errors="replace")
                if isinstance(exc.stderr, bytes)
                else exc.stderr
            )
        return ProcessResult(
            command=tuple(str(part) for part in command),
            returncode=124,
            stdout=stdout,
            stderr=stderr,
            duration_s=time.monotonic() - started,
            timed_out=True,
        )
