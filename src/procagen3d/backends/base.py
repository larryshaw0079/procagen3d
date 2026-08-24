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
from typing import Any, Callable, ClassVar, Literal, Mapping, Sequence

from procagen3d.process import ProcessResult, run_process


ExitReason = Literal["completed", "timeout", "quota", "error"]
ActivityCallback = Callable[[str], None]

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

    def activity_message(
        self,
        event: Mapping[str, Any],
        *,
        workspace: Path,
    ) -> str | None:
        """Map a streaming provider event to one bounded, safe status line.

        Provider implementations must not return raw prompts, reasoning,
        commands, assistant prose, tool arguments, or command output.
        """

        del event, workspace
        return None

    def heartbeat_message(
        self,
        *,
        elapsed_s: float,
        provider_event_count: int,
        last_output_age_s: float | None,
        workspace: Path,
    ) -> str:
        """Describe a quiet-period heartbeat without exposing provider payloads."""

        if last_output_age_s is None:
            output_state = "no CLI output yet"
        else:
            output_state = f"last CLI output {_format_duration(last_output_age_s)} ago"
        event_word = "event" if provider_event_count == 1 else "events"
        source_state = "; ".join(
            _source_file_state(workspace / "src" / name)
            for name in ("plan.json", "program.py")
        )
        return (
            f"{self.name.title()} process still running — "
            f"{_format_duration(elapsed_s)} elapsed; "
            f"{provider_event_count:,} provider {event_word}; "
            f"{output_state}; {source_state}"
        )

    def run(
        self,
        *,
        prompt: str,
        workspace: Path,
        trajectory_dir: Path | None = None,
        image_paths: Sequence[Path] = (),
        timeout_s: int | None = None,
        env: Mapping[str, str] | None = None,
        on_activity: ActivityCallback | None = None,
        heartbeat_interval_s: float | None = None,
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
        if on_activity is not None and heartbeat_interval_s is not None and heartbeat_interval_s <= 0:
            raise ValueError("heartbeat_interval_s must be greater than zero")
        heartbeat_enabled = on_activity is not None and heartbeat_interval_s is not None

        started = time.monotonic()
        provider_event_count = 0
        last_output_at: float | None = None
        last_activity_message: str | None = None

        def emit_activity(message: str) -> None:
            nonlocal last_activity_message
            if on_activity is None or not message or message == last_activity_message:
                return
            last_activity_message = message
            try:
                on_activity(message)
            except (Exception, SystemExit):
                # Activity rendering is observational.
                return

        def observe_stdout(line: str) -> None:
            nonlocal provider_event_count, last_output_at
            last_output_at = time.monotonic()
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                return
            if not isinstance(event, dict):
                return
            provider_event_count += 1
            try:
                message = self.activity_message(event, workspace=workspace)
            except (Exception, SystemExit):
                return
            if message:
                emit_activity(message)

        def observe_stderr(line: str) -> None:
            nonlocal last_output_at
            del line
            last_output_at = time.monotonic()

        def observe_heartbeat(elapsed_s: float) -> None:
            emit_activity(
                self.heartbeat_message(
                    elapsed_s=elapsed_s,
                    provider_event_count=provider_event_count,
                    last_output_age_s=(
                        None if last_output_at is None else time.monotonic() - last_output_at
                    ),
                    workspace=workspace,
                )
            )

        try:
            process = run_process(
                invocation.command,
                cwd=workspace,
                timeout_s=effective_timeout,
                input_text=invocation.stdin,
                env=env,
                on_stdout_line=observe_stdout if on_activity is not None else None,
                on_stderr_line=observe_stderr if on_activity is not None else None,
                on_heartbeat=observe_heartbeat if heartbeat_enabled else None,
                heartbeat_interval_s=heartbeat_interval_s or 30.0,
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


def source_change_paths(event: Mapping[str, Any], *, workspace: Path) -> tuple[str, ...]:
    """Extract only workspace-local ``src/`` paths from a file-change event."""

    item = event.get("item")
    if not isinstance(item, Mapping):
        return ()
    changes = item.get("changes")
    if not isinstance(changes, list):
        return ()
    root = workspace.resolve(strict=False)
    paths: list[str] = []
    for change in changes:
        if not isinstance(change, Mapping) or not isinstance(change.get("path"), str):
            continue
        candidate = Path(change["path"]).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        try:
            relative = candidate.resolve(strict=False).relative_to(root)
        except (OSError, RuntimeError, ValueError):
            continue
        if relative.parts and relative.parts[0] == "src":
            normalized = relative.as_posix()
            if normalized not in paths:
                paths.append(normalized)
    return tuple(paths)


def usage_activity(provider: str, usage: Mapping[str, Any]) -> str:
    """Format bounded terminal token accounting for an activity line."""

    output_tokens = _numeric_value(
        usage,
        "output_tokens",
        "outputTokens",
        "completion_tokens",
        "completionTokens",
    )
    reasoning_tokens = _numeric_value(
        usage,
        "reasoning_output_tokens",
        "reasoningTokens",
        "reasoning_tokens",
    )
    parts: list[str] = []
    if output_tokens is not None:
        parts.append(f"{output_tokens:,} output tokens")
    if reasoning_tokens is not None:
        parts.append(f"{reasoning_tokens:,} reasoning tokens")
    suffix = f" — {', '.join(parts)}" if parts else ""
    return f"{provider} model turn completed{suffix}"


def _numeric_value(value: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        candidate = value.get(key)
        if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
            return int(candidate)
    return None


def _format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _source_file_state(path: Path) -> str:
    try:
        size = path.stat().st_size
    except (FileNotFoundError, OSError):
        return f"{path.name} not created"
    formatted = f"{size} B" if size < 1024 else f"{size / 1024:.1f} KiB"
    return f"{path.name} {formatted}"


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
