"""Grok Build CLI backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Sequence

from .base import CLIBackend, ParsedOutput, json_objects


_SUCCESS_STOP_REASONS = {"completed", "end_turn", "stop", "success"}


@dataclass(frozen=True)
class GrokBackend(CLIBackend):
    """Invoke Grok 4.6 at extra-high effort.

    Grok Build 1.0.5 exposes no separate ``fast`` flag or fast model ID.  The
    requested Extra High configuration is therefore represented exactly as
    model ``grok-4.6`` plus reasoning effort ``xhigh``.
    """

    name: ClassVar[str] = "grok"
    cli: str = "grok"
    model: str = "grok-4.6"
    reasoning_effort: str = "xhigh"
    sandbox: str = "workspace"
    always_approve: bool = True
    disable_web_search: bool = True
    no_subagents: bool = True
    max_turns: int = 24
    deny_rules: tuple[str, ...] = (
        "Bash(rm -rf *)",
        "Bash(sudo *)",
        "Bash(git push*)",
    )
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
        del prompt, image_paths
        workspace = workspace.expanduser().resolve()
        prompt_file = prompt_file or (workspace / ".procagen3d_prompt.txt")
        command = [
            self.cli,
            "--cwd",
            str(workspace),
            "--model",
            self.model,
            "--reasoning-effort",
            self.reasoning_effort,
            "--sandbox",
            self.sandbox,
            "--output-format",
            "streaming-json",
            "--max-turns",
            str(self.max_turns),
            "--verbatim",
        ]
        if self.always_approve:
            command.append("--always-approve")
        if self.disable_web_search:
            command.append("--disable-web-search")
        if self.no_subagents:
            command.append("--no-subagents")
        for rule in self.deny_rules:
            command.extend(("--deny", rule))
        command.extend(self.extra_args)
        command.extend(("--prompt-file", str(prompt_file.expanduser().resolve())))
        return tuple(command)

    def parse_output(self, stdout: str, stderr: str) -> ParsedOutput:
        del stderr
        text_chunks: list[str] = []
        session_id: str | None = None
        usage: dict[str, Any] = {}
        model_usage: dict[str, Any] = {}
        cost_usd: float | None = None
        saw_terminal = False
        terminal_success: bool | None = None
        error: str | None = None

        for event in json_objects(stdout):
            event_type = event.get("type")
            if event_type == "text" and isinstance(event.get("data"), str):
                text_chunks.append(event["data"])
            elif event_type == "end":
                saw_terminal = True
                stop_reason = str(event.get("stopReason") or "")
                terminal_success = stop_reason in _SUCCESS_STOP_REASONS
                session_id = event.get("sessionId") or session_id
                usage = _dict_value(event.get("usage"))
                model_usage = _dict_value(event.get("modelUsage"))
                cost_usd = _float_value(event.get("total_cost_usd"))
                if not terminal_success:
                    error = str(event.get("message") or stop_reason or "grok stopped early")
            elif event_type == "error":
                saw_terminal = True
                terminal_success = False
                session_id = event.get("sessionId") or session_id
                usage = _dict_value(event.get("usage")) or usage
                model_usage = _dict_value(event.get("modelUsage")) or model_usage
                cost_usd = _float_value(event.get("total_cost_usd"))
                error = str(event.get("message") or "grok error")
            elif "text" in event and "stopReason" in event:
                # Also accept Grok's non-streaming --output-format json shape.
                saw_terminal = True
                stop_reason = str(event.get("stopReason") or "")
                terminal_success = stop_reason in _SUCCESS_STOP_REASONS
                text_chunks = [str(event.get("text") or "")]
                session_id = event.get("sessionId") or session_id
                usage = _dict_value(event.get("usage"))
                model_usage = _dict_value(event.get("modelUsage"))
                cost_usd = _float_value(event.get("total_cost_usd"))
                if not terminal_success:
                    error = str(event.get("message") or stop_reason or "grok stopped early")

        return ParsedOutput(
            saw_terminal_event=saw_terminal,
            terminal_success=terminal_success,
            final_message="".join(text_chunks),
            session_id=session_id,
            usage=usage,
            model_usage=model_usage,
            cost_usd=cost_usd,
            error=error,
        )


def _dict_value(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _float_value(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


GrokCLIBackend = GrokBackend
