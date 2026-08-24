"""Discovery and isolated invocation of headless Blender stages."""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Sequence

from .process import ProcessResult, run_process


class BlenderError(RuntimeError):
    """Raised when a Blender stage cannot be launched or does not complete."""


@dataclass(frozen=True)
class BlenderRuntime:
    executable: Path

    @classmethod
    def discover(cls, explicit: Path | None = None) -> "BlenderRuntime":
        candidates: list[Path] = []
        if explicit is not None:
            candidates.append(explicit.expanduser())
        configured = os.environ.get("PROCAGEN3D_BLENDER")
        if configured:
            candidates.append(Path(configured).expanduser())
        on_path = shutil.which("blender")
        if on_path:
            candidates.append(Path(on_path))
        if sys.platform == "darwin":
            candidates.extend(
                [
                    Path("/Applications/Blender.app/Contents/MacOS/Blender"),
                    Path("/Applications/Blender.app/Contents/MacOS/blender"),
                ]
            )
        cache_root = Path.home() / ".cache" / "procagen3d"
        if cache_root.is_dir():
            candidates.extend(sorted(cache_root.glob("*/blender"), reverse=True))

        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved.is_file() and os.access(resolved, os.X_OK):
                return cls(resolved)
        searched = ", ".join(str(item) for item in candidates) or "PATH"
        raise BlenderError(
            "Blender was not found. Pass --blender, set PROCAGEN3D_BLENDER, "
            f"or install `blender` on PATH. Searched: {searched}"
        )

    def version(self, *, timeout_s: int = 30) -> ProcessResult:
        return run_process(
            [str(self.executable), "--version"],
            cwd=Path.cwd(),
            timeout_s=timeout_s,
        )

    def run_stage(
        self,
        stage: str,
        arguments: Sequence[str | Path],
        *,
        cwd: Path,
        timeout_s: int,
        clean_environment: bool = True,
    ) -> ProcessResult:
        resource = files("procagen3d").joinpath("blender_scripts", f"{stage}.py")
        script = Path(str(resource)).resolve()
        if not script.is_file():
            raise BlenderError(f"packaged Blender stage is missing: {script}")
        command = [
            str(self.executable),
            "--background",
            "--factory-startup",
            "--python-exit-code",
            "1",
            "--python",
            str(script),
            "--",
            *(str(item) for item in arguments),
        ]
        environment = None
        if clean_environment:
            allowed = {
                "LANG",
                "LC_ALL",
                "PATH",
                "TMPDIR",
                "TMP",
                "TEMP",
                "DISPLAY",
                "WAYLAND_DISPLAY",
                "XDG_RUNTIME_DIR",
                "DYLD_LIBRARY_PATH",
                "SYSTEMROOT",
                "WINDIR",
                "COMSPEC",
                "PATHEXT",
                "NUMBER_OF_PROCESSORS",
                "PROCESSOR_ARCHITECTURE",
            }
            environment = {key: value for key, value in os.environ.items() if key in allowed}
        return run_process(
            command,
            cwd=cwd,
            timeout_s=timeout_s,
            env=environment,
            inherit_env=not clean_environment,
        )


def require_success(result: ProcessResult, *, stage: str) -> None:
    if result.ok:
        return
    reason = "timed out" if result.timed_out else f"exited with {result.returncode}"
    diagnostics = "\n".join(
        section
        for section in (
            f"stdout:\n{result.stdout[-3000:]}" if result.stdout else "",
            f"stderr:\n{result.stderr[-3000:]}" if result.stderr else "",
        )
        if section
    )
    hint = ""
    if result.returncode < 0:
        hint = (
            "\nBlender terminated by a signal. On macOS, headless Blender may require "
            "GPU/Metal access unavailable inside an outer application sandbox."
        )
    raise BlenderError(f"Blender {stage} {reason}.{hint}\n{diagnostics}")
