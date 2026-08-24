from __future__ import annotations

import sys

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
