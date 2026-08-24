"""Codex CLI backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Sequence

from .base import CLIBackend, CLIInvocation, ParsedOutput, json_objects


@dataclass(frozen=True)
class CodexBackend(CLIBackend):
    """Invoke Codex non-interactively with GPT-5.6 Sol at max effort."""

    name: ClassVar[str] = "codex"
    cli: str = "codex"
    model: str = "gpt-5.6-sol"
    reasoning_effort: str = "max"
    sandbox: str = "workspace-write"
    approval_policy: str = "never"
    ephemeral: bool = True
    ignore_user_config: bool = True
    skip_git_repo_check: bool = True
    extra_args: tuple[str, ...] = field(default_factory=tuple)
    default_timeout_s: int = 1800

    def build_command(
        self,
        prompt: str,
        workspace: Path,
        *,
        prompt_file: Path | None = None,
        image_paths: Sequence[Path] = (),
    ) -> tuple[str, ...]:
        del prompt
        workspace = workspace.expanduser().resolve()
        final_path = (prompt_file.parent if prompt_file else workspace) / "final_message.txt"
        command = [self.cli, "-a", self.approval_policy, "exec", "-C", str(workspace)]
        if self.skip_git_repo_check:
            command.append("--skip-git-repo-check")
        if self.ephemeral:
            command.append("--ephemeral")
        if self.ignore_user_config:
            command.append("--ignore-user-config")
        if image_paths:
            command.append("-i")
            command.extend(str(path.expanduser().resolve()) for path in image_paths)
        command.extend(("-m", self.model))
        command.extend(("-c", f"model_reasoning_effort={self.reasoning_effort}"))
        command.extend(("-s", self.sandbox, "--color", "never", "--json"))
        command.extend(("--output-last-message", str(final_path)))
        command.extend(self.extra_args)
        command.append("-")
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
        return CLIInvocation(command=command, stdin=prompt)

    def parse_output(self, stdout: str, stderr: str) -> ParsedOutput:
        del stderr
        final_message = ""
        session_id: str | None = None
        usage: dict = {}
        saw_terminal = False
        terminal_success: bool | None = None
        error: str | None = None

        for event in json_objects(stdout):
            event_type = event.get("type")
            if event_type == "thread.started":
                session_id = event.get("thread_id") or session_id
            elif event_type == "item.completed":
                item = event.get("item") or {}
                if item.get("type") == "agent_message" and isinstance(item.get("text"), str):
                    final_message = item["text"]
            elif event_type == "turn.completed":
                saw_terminal = True
                terminal_success = True
                if isinstance(event.get("usage"), dict):
                    usage = dict(event["usage"])
            elif event_type in {"turn.failed", "error"}:
                saw_terminal = True
                terminal_success = False
                candidate = event.get("error") or event.get("message")
                error = str(candidate) if candidate else event_type

        return ParsedOutput(
            saw_terminal_event=saw_terminal,
            terminal_success=terminal_success,
            final_message=final_message,
            session_id=session_id,
            usage=usage,
            error=error,
        )


# Descriptive alias for callers following OpenTopos's backend naming scheme.
CodexCLIBackend = CodexBackend
