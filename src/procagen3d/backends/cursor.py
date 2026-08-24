"""Cursor Agent CLI backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Sequence

from .base import CLIBackend, CLIInvocation, ParsedOutput, json_objects


@dataclass(frozen=True)
class CursorBackend(CLIBackend):
    """Invoke Cursor's Grok 4.6 Extra High Fast model headlessly."""

    name: ClassVar[str] = "cursor"
    cli: str = "cursor-agent"
    model: str = "cursor-grok-4.6-xhigh-fast"
    sandbox: str = "enabled"
    auto_review: bool = True
    force: bool = False
    trust_workspace: bool = True
    extra_dirs: tuple[Path, ...] = field(default_factory=tuple)
    extra_args: tuple[str, ...] = field(default_factory=tuple)
    default_timeout_s: int = 1800

    def __post_init__(self) -> None:
        if self.auto_review and self.force:
            raise ValueError("Cursor auto_review and force are mutually exclusive")

    def build_command(
        self,
        prompt: str,
        workspace: Path,
        *,
        prompt_file: Path | None = None,
        image_paths: Sequence[Path] = (),
    ) -> tuple[str, ...]:
        del prompt_file, image_paths
        workspace = workspace.expanduser().resolve()
        command = [
            self.cli,
            "--print",
            "--output-format",
            "stream-json",
            "--model",
            self.model,
            "--sandbox",
            self.sandbox,
            "--workspace",
            str(workspace),
        ]
        if self.trust_workspace:
            command.append("--trust")
        if self.auto_review:
            command.append("--auto-review")
        if self.force:
            command.append("--force")
        for path in self.extra_dirs:
            command.extend(("--add-dir", str(path.expanduser().resolve())))
        command.extend(self.extra_args)
        command.append(prompt)
        return tuple(command)

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
        display_command = (*command[:-1], "<prompt>")
        return CLIInvocation(command=command, display_command=display_command)

    def parse_output(self, stdout: str, stderr: str) -> ParsedOutput:
        del stderr
        saw_terminal = False
        terminal_success: bool | None = None
        final_message = ""
        session_id: str | None = None
        usage: dict[str, Any] = {}
        error: str | None = None

        for event in json_objects(stdout):
            event_type = event.get("type")
            if event_type == "system" and event.get("subtype") == "init":
                session_id = event.get("session_id") or session_id
            elif event_type == "result":
                saw_terminal = True
                is_error = bool(event.get("is_error"))
                terminal_success = event.get("subtype") == "success" and not is_error
                final_message = str(event.get("result") or "")
                session_id = event.get("session_id") or session_id
                if isinstance(event.get("usage"), dict):
                    usage = dict(event["usage"])
                if not terminal_success:
                    errors = event.get("errors")
                    error = str(errors or event.get("result") or "cursor error")
            elif event_type == "error":
                saw_terminal = True
                terminal_success = False
                error = str(event.get("message") or "cursor error")

        return ParsedOutput(
            saw_terminal_event=saw_terminal,
            terminal_success=terminal_success,
            final_message=final_message,
            session_id=session_id,
            usage=usage,
            error=error,
        )


CursorCLIBackend = CursorBackend
