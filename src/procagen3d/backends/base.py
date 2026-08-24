"""Provider-neutral command-line agent backend contracts.

The application treats coding CLIs as subprocesses, not Python SDKs.  This
module keeps their common lifecycle deterministic: prompts and complete
transcripts are persisted, workspace changes are measured, and provider
specific envelopes are normalized into one result type.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal, Mapping, Sequence

from procagen3d.process import ProcessResult, run_process


ExitReason = Literal["completed", "timeout", "quota", "error"]

_QUOTA_MARKERS = (
    "429",
    "capacity",
    "credit balance",
    "insufficient quota",
    "quota",
    "rate limit",
    "rate_limit",
    "too many requests",
    "usage limit",
)


@dataclass(frozen=True)
class CLIInvocation:
    """A fully resolved subprocess invocation.

    ``stdin`` is used by Codex so large prompts never enter argv.  Grok uses a
    prompt file, while Cursor currently documents only a positional prompt.
    """

    command: tuple[str, ...]
    stdin: str | None = None
    display_command: tuple[str, ...] | None = None


@dataclass(frozen=True)
class ParsedOutput:
    """Provider envelope projected onto the common backend vocabulary."""

    saw_terminal_event: bool = False
    terminal_success: bool | None = None
    final_message: str = ""
    session_id: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    model_usage: dict[str, Any] = field(default_factory=dict)
    cost_usd: float | None = None
    error: str | None = None


@dataclass(frozen=True)
class AgentRunResult:
    backend: str
    model: str
    command: tuple[str, ...]
    success: bool
    exit_reason: ExitReason
    returncode: int
    timed_out: bool
    duration_s: float
    stdout: str
    stderr: str
    final_message: str
    session_id: str | None
    usage: dict[str, Any]
    model_usage: dict[str, Any]
    cost_usd: float | None
    files_modified: tuple[Path, ...]
    prompt_path: Path
    transcript_path: Path
    stderr_path: Path
    result_path: Path
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.success


class CLIBackend(ABC):
    """Template for a one-shot coding CLI invocation."""

    name: ClassVar[str]
    model: str
    default_timeout_s: int

    @abstractmethod
    def build_command(
        self,
        prompt: str,
        workspace: Path,
        *,
        prompt_file: Path | None = None,
        image_paths: Sequence[Path] = (),
    ) -> tuple[str, ...]:
        """Return argv without spawning the provider."""

    def build_invocation(
        self,
        prompt: str,
        workspace: Path,
        *,
        prompt_file: Path,
        image_paths: Sequence[Path],
    ) -> CLIInvocation:
        command = self.build_command(
            prompt,
            workspace,
            prompt_file=prompt_file,
            image_paths=image_paths,
        )
        return CLIInvocation(command=command)

    @abstractmethod
    def parse_output(self, stdout: str, stderr: str) -> ParsedOutput:
        """Parse the provider's terminal event and accounting fields."""

    def run(
        self,
        *,
        prompt: str,
        workspace: Path,
        trajectory_dir: Path | None = None,
        image_paths: Sequence[Path] = (),
        timeout_s: int | None = None,
        env: Mapping[str, str] | None = None,
    ) -> AgentRunResult:
        workspace = workspace.expanduser().resolve()
        if not workspace.is_dir():
            raise NotADirectoryError(f"backend workspace does not exist: {workspace}")
        if not prompt.strip():
            raise ValueError("agent prompt must not be empty")

        images = tuple(path.expanduser().resolve() for path in image_paths)
        missing_images = [path for path in images if not path.is_file()]
        if missing_images:
            raise FileNotFoundError(f"agent image input not found: {missing_images[0]}")

        if trajectory_dir is None:
            trajectory_dir = workspace / "trajectories" / self.name
        trajectory_dir = trajectory_dir.expanduser().resolve()
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = trajectory_dir / "prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")

        invocation = self.build_invocation(
            prompt,
            workspace,
            prompt_file=prompt_path,
            image_paths=images,
        )
        before = _snapshot_workspace(workspace, excluded=trajectory_dir)
        effective_timeout = timeout_s if timeout_s is not None else self.default_timeout_s
        if effective_timeout <= 0:
            raise ValueError("timeout_s must be greater than zero")

        started = time.monotonic()
        try:
            process = run_process(
                invocation.command,
                cwd=workspace,
                timeout_s=effective_timeout,
                input_text=invocation.stdin,
                env=env,
            )
        except OSError as exc:
            process = ProcessResult(
                command=invocation.command,
                returncode=127,
                stdout="",
                stderr=str(exc),
                duration_s=time.monotonic() - started,
            )

        transcript_path = trajectory_dir / "transcript.jsonl"
        stderr_path = trajectory_dir / "stderr.log"
        transcript_path.write_text(process.stdout, encoding="utf-8")
        stderr_path.write_text(process.stderr, encoding="utf-8")

        parsed = self.parse_output(process.stdout, process.stderr)
        final_message_path = trajectory_dir / "final_message.txt"
        if parsed.final_message:
            final_message_path.write_text(parsed.final_message.rstrip() + "\n", encoding="utf-8")
        elif not final_message_path.exists():
            final_message_path.write_text("", encoding="utf-8")

        exit_reason = _classify_exit(process, parsed)
        success = exit_reason == "completed"
        files_modified = _changed_files(
            workspace,
            before,
            excluded=trajectory_dir,
        )
        result_path = trajectory_dir / "result.json"
        result_payload = {
            "backend": self.name,
            "model": self.model,
            "command": list(invocation.display_command or invocation.command),
            "success": success,
            "exit_reason": exit_reason,
            "returncode": process.returncode,
            "timed_out": process.timed_out,
            "duration_s": process.duration_s,
            "session_id": parsed.session_id,
            "usage": parsed.usage,
            "model_usage": parsed.model_usage,
            "cost_usd": parsed.cost_usd,
            "files_modified": [_relative_or_absolute(path, workspace) for path in files_modified],
            "error": parsed.error,
        }
        _write_json(result_path, result_payload)

        return AgentRunResult(
            backend=self.name,
            model=self.model,
            command=invocation.command,
            success=success,
            exit_reason=exit_reason,
            returncode=process.returncode,
            timed_out=process.timed_out,
            duration_s=process.duration_s,
            stdout=process.stdout,
            stderr=process.stderr,
            final_message=parsed.final_message,
            session_id=parsed.session_id,
            usage=dict(parsed.usage),
            model_usage=dict(parsed.model_usage),
            cost_usd=parsed.cost_usd,
            files_modified=files_modified,
            prompt_path=prompt_path,
            transcript_path=transcript_path,
            stderr_path=stderr_path,
            result_path=result_path,
            error=parsed.error,
        )


def json_objects(text: str) -> list[dict[str, Any]]:
    """Parse either one JSON document or newline-delimited JSON events."""

    stripped = text.strip()
    if not stripped:
        return []
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        values: list[dict[str, Any]] = []
        for line in stripped.splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                values.append(value)
        return values
    return [value] if isinstance(value, dict) else []


def _classify_exit(process: ProcessResult, parsed: ParsedOutput) -> ExitReason:
    if process.timed_out:
        return "timeout"
    if process.returncode != 0:
        return "quota" if _has_quota_marker(process.stderr + "\n" + process.stdout) else "error"
    if parsed.error:
        return "quota" if _has_quota_marker(process.stderr + "\n" + parsed.error) else "error"
    if not parsed.saw_terminal_event or parsed.terminal_success is not True:
        return "error"
    return "completed"


def _has_quota_marker(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _QUOTA_MARKERS)


def _snapshot_workspace(root: Path, *, excluded: Path) -> dict[Path, tuple[int, int]]:
    snapshot: dict[Path, tuple[int, int]] = {}
    for path in root.rglob("*"):
        if not path.is_file() or _is_relative_to(path, excluded):
            continue
        stat = path.stat()
        snapshot[path] = (stat.st_mtime_ns, stat.st_size)
    return snapshot


def _changed_files(
    root: Path,
    before: dict[Path, tuple[int, int]],
    *,
    excluded: Path,
) -> tuple[Path, ...]:
    after = _snapshot_workspace(root, excluded=excluded)
    paths = set(before) | set(after)
    return tuple(sorted(path for path in paths if before.get(path) != after.get(path)))


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)
