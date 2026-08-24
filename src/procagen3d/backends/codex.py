"""Codex CLI backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Mapping, Sequence

from .base import (
    CLIBackend,
    CLIInvocation,
    ParsedOutput,
    json_objects,
    source_change_paths,
    usage_activity,
)


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

    def activity_message(
        self,
        event: Mapping[str, Any],
        *,
        workspace: Path,
    ) -> str | None:
        event_type = event.get("type")
        if event_type == "thread.started":
            return "Codex session started"
        if event_type == "turn.started":
            return "Codex began analyzing the reference and planning the asset"
        if event_type == "turn.completed":
            usage = event.get("usage")
            return usage_activity("Codex", usage if isinstance(usage, Mapping) else {})
        if event_type in {"turn.failed", "error"}:
            return "Codex reported a model-turn failure"
        if event_type not in {"item.started", "item.updated", "item.completed"}:
            return None

        item = event.get("item")
        if not isinstance(item, Mapping):
            return None
        item_type = item.get("type")
        completed = event_type == "item.completed"

        if item_type == "command_execution":
            action = _classify_command(str(item.get("command") or ""))
            if completed:
                exit_code = item.get("exit_code")
                if isinstance(exit_code, int) and exit_code != 0:
                    return f"Codex workspace check failed — exit code {exit_code}"
                return f"Codex finished {action}"
            return f"Codex is {action}"

        if item_type == "file_change":
            paths = source_change_paths(event, workspace=workspace)
            if not paths:
                return "Codex is updating generated source"
            shown = ", ".join(paths[:2])
            if len(paths) > 2:
                shown += f" (+{len(paths) - 2} more)"
            if not completed:
                return f"Codex is writing {shown}"
            changes = item.get("changes")
            kinds = (
                {
                    str(change.get("kind"))
                    for change in changes
                    if isinstance(change, Mapping)
                }
                if isinstance(changes, list)
                else set()
            )
            verb = "created" if kinds == {"add"} else "updated"
            return f"Codex {verb} {shown}"

        if item_type == "agent_message" and completed:
            return "Codex posted a progress update"
        if item_type == "reasoning" and completed:
            return "Codex completed a reasoning step"
        if item_type == "web_search":
            return "Codex is inspecting supporting information"
        if item_type == "mcp_tool_call":
            return "Codex is running a connected workspace tool"
        if item_type == "todo_list" and completed:
            return "Codex updated its implementation checklist"
        if item_type == "error":
            return "Codex reported an item-level failure"
        return None


def _classify_command(command: str) -> str:
    lowered = command.lower()
    if any(
        marker in lowered
        for marker in (
            "evidence/",
            "/evidence",
            "reference_scene",
            "reference_views",
            "camera_contract",
            "glb_probe",
            "inputs/reference",
        )
    ):
        return "inspecting reference evidence"
    if any(
        marker in lowered
        for marker in (
            "py_compile",
            "compileall",
            "ast.parse",
            "json.load",
            "jq -e",
            "src/program.py",
            "src/plan.json",
        )
    ):
        return "validating generated source"
    if "git status" in lowered or "git diff" in lowered:
        return "checking workspace changes"
    return "running a workspace check"


# Descriptive alias for callers following OpenTopos's backend naming scheme.
CodexCLIBackend = CodexBackend
