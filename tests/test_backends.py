from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping, Sequence

from procagen3d.backends import (
    CLIBackend,
    CodexBackend,
    CodexCLIBackend,
    CursorBackend,
    CursorCLIBackend,
    GrokBackend,
    GrokCLIBackend,
    ParsedOutput,
    create_backend,
)


class StreamingFixtureBackend(CLIBackend):
    name = "fixture"
    model = "fixture-model"
    default_timeout_s = 5

    def build_command(
        self,
        prompt: str,
        workspace: Path,
        *,
        prompt_file: Path | None = None,
        image_paths: Sequence[Path] = (),
    ) -> tuple[str, ...]:
        del prompt, prompt_file, image_paths
        plan_path = workspace / "src" / "plan.json"
        script = "\n".join(
            (
                "import json, pathlib, sys, time",
                "print(json.dumps({'type': 'started'}, ensure_ascii=False), flush=True)",
                f"plan = pathlib.Path({str(plan_path)!r})",
                "plan.parent.mkdir(parents=True, exist_ok=True)",
                "plan.write_text('1234', encoding='utf-8')",
                "time.sleep(0.04)",
                "print(json.dumps({'type': 'completed', 'text': '完成'}, ensure_ascii=False), flush=True)",
                "sys.stderr.write('fixture warning\\n')",
            )
        )
        return (sys.executable, "-u", "-c", script)

    def parse_output(self, stdout: str, stderr: str) -> ParsedOutput:
        del stderr
        events = [json.loads(line) for line in stdout.splitlines()]
        completed = any(event.get("type") == "completed" for event in events)
        return ParsedOutput(
            saw_terminal_event=completed,
            terminal_success=completed,
            final_message="complete" if completed else "",
        )

    def activity_message(
        self,
        event: Mapping[str, Any],
        *,
        workspace: Path,
    ) -> str | None:
        del workspace
        if event.get("type") == "started":
            return "Fixture provider started"
        if event.get("type") == "completed":
            return "Fixture provider completed"
        return None


class BackendCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name).resolve()
        self.prompt_file = self.workspace / "trajectories" / "iter_00" / "prompt.txt"
        self.image = self.workspace / "inputs" / "reference.png"

    def test_codex_command_pins_sol_xhigh_and_safe_headless_flags(self) -> None:
        backend = CodexBackend()
        command = backend.build_command(
            "write the program",
            self.workspace,
            prompt_file=self.prompt_file,
            image_paths=(self.image,),
        )

        self.assertEqual(
            command,
            (
                "codex",
                "-a",
                "never",
                "exec",
                "-C",
                str(self.workspace),
                "--skip-git-repo-check",
                "--ephemeral",
                "--ignore-user-config",
                "-i",
                str(self.image),
                "-m",
                "gpt-5.6-sol",
                "-c",
                "model_reasoning_effort=xhigh",
                "-s",
                "workspace-write",
                "--color",
                "never",
                "--json",
                "--output-last-message",
                str(self.prompt_file.parent / "final_message.txt"),
                "-",
            ),
        )
        invocation = backend.build_invocation(
            "large prompt stays off argv",
            self.workspace,
            prompt_file=self.prompt_file,
            image_paths=(),
        )
        self.assertEqual(invocation.stdin, "large prompt stays off argv")
        self.assertEqual(invocation.command[-1], "-")

    def test_grok_command_pins_46_xhigh_and_workspace_sandbox(self) -> None:
        backend = GrokBackend()
        command = backend.build_command(
            "write the program",
            self.workspace,
            prompt_file=self.prompt_file,
            image_paths=(self.image,),
        )

        self.assertEqual(
            command,
            (
                "grok",
                "--cwd",
                str(self.workspace),
                "--model",
                "grok-4.6",
                "--reasoning-effort",
                "xhigh",
                "--sandbox",
                "workspace",
                "--output-format",
                "streaming-json",
                "--max-turns",
                "24",
                "--verbatim",
                "--always-approve",
                "--disable-web-search",
                "--no-subagents",
                "--deny",
                "Bash(rm -rf *)",
                "--deny",
                "Bash(sudo *)",
                "--deny",
                "Bash(git push*)",
                "--prompt-file",
                str(self.prompt_file),
            ),
        )
        self.assertNotIn("fast", command)

    def test_cursor_command_uses_exact_xhigh_fast_model(self) -> None:
        backend = CursorBackend()
        prompt = "write the program"
        command = backend.build_command(prompt, self.workspace)

        self.assertEqual(
            command,
            (
                "cursor-agent",
                "--print",
                "--output-format",
                "stream-json",
                "--model",
                "cursor-grok-4.6-xhigh-fast",
                "--sandbox",
                "enabled",
                "--workspace",
                str(self.workspace),
                "--trust",
                "--auto-review",
                prompt,
            ),
        )
        invocation = backend.build_invocation(
            prompt,
            self.workspace,
            prompt_file=self.prompt_file,
            image_paths=(),
        )
        self.assertEqual(invocation.display_command[-1], "<prompt>")
        self.assertNotIn(prompt, invocation.display_command)


class BackendParserTests(unittest.TestCase):
    def test_codex_terminal_event_extracts_message_session_and_usage(self) -> None:
        stdout = "\n".join(
            (
                json.dumps({"type": "thread.started", "thread_id": "codex-session"}),
                json.dumps({"type": "turn.started"}),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {"type": "agent_message", "text": "files written"},
                    }
                ),
                json.dumps(
                    {
                        "type": "turn.completed",
                        "usage": {
                            "input_tokens": 101,
                            "cached_input_tokens": 50,
                            "output_tokens": 7,
                        },
                    }
                ),
            )
        )
        parsed = CodexBackend().parse_output(stdout, "")

        self.assertTrue(parsed.saw_terminal_event)
        self.assertTrue(parsed.terminal_success)
        self.assertEqual(parsed.final_message, "files written")
        self.assertEqual(parsed.session_id, "codex-session")
        self.assertEqual(parsed.usage["input_tokens"], 101)

    def test_grok_stream_terminal_event_extracts_cost_and_model_usage(self) -> None:
        stdout = "\n".join(
            (
                json.dumps({"type": "text", "data": "files "}),
                json.dumps({"type": "text", "data": "written"}),
                json.dumps(
                    {
                        "type": "end",
                        "stopReason": "end_turn",
                        "sessionId": "grok-session",
                        "usage": {"input_tokens": 103, "output_tokens": 9},
                        "modelUsage": {"grok-4.6": {"modelCalls": 2}},
                        "total_cost_usd": 0.123,
                    }
                ),
            )
        )
        parsed = GrokBackend().parse_output(stdout, "")

        self.assertTrue(parsed.saw_terminal_event)
        self.assertTrue(parsed.terminal_success)
        self.assertEqual(parsed.final_message, "files written")
        self.assertEqual(parsed.session_id, "grok-session")
        self.assertEqual(parsed.usage["output_tokens"], 9)
        self.assertEqual(parsed.model_usage["grok-4.6"]["modelCalls"], 2)
        self.assertEqual(parsed.cost_usd, 0.123)

    def test_cursor_terminal_result_is_authoritative(self) -> None:
        stdout = "\n".join(
            (
                json.dumps(
                    {
                        "type": "system",
                        "subtype": "init",
                        "session_id": "cursor-session",
                    }
                ),
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "duplicate"}],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "result",
                        "subtype": "success",
                        "is_error": False,
                        "result": "files written",
                        "session_id": "cursor-session",
                        "usage": {"inputTokens": 107, "outputTokens": 11},
                    }
                ),
            )
        )
        parsed = CursorBackend().parse_output(stdout, "")

        self.assertTrue(parsed.saw_terminal_event)
        self.assertTrue(parsed.terminal_success)
        self.assertEqual(parsed.final_message, "files written")
        self.assertEqual(parsed.session_id, "cursor-session")
        self.assertEqual(parsed.usage["inputTokens"], 107)

    def test_error_and_missing_terminal_events_do_not_report_success(self) -> None:
        codex_error = CodexBackend().parse_output(
            json.dumps({"type": "turn.failed", "error": "rate limit reached"}),
            "",
        )
        grok_incomplete = GrokBackend().parse_output(
            json.dumps({"type": "text", "data": "partial"}),
            "",
        )
        cursor_error = CursorBackend().parse_output(
            json.dumps({"type": "error", "message": "authentication failed"}),
            "",
        )

        self.assertFalse(codex_error.terminal_success)
        self.assertEqual(codex_error.error, "rate limit reached")
        self.assertFalse(grok_incomplete.saw_terminal_event)
        self.assertIsNone(grok_incomplete.terminal_success)
        self.assertFalse(cursor_error.terminal_success)
        self.assertEqual(cursor_error.error, "authentication failed")

    def test_provider_activity_is_bounded_and_suppresses_raw_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve()
            source = workspace / "src" / "program.py"
            source.parent.mkdir()
            source.write_text("pass\n", encoding="utf-8")
            secret = "DO-NOT-PRINT-RAW-PAYLOAD"

            codex = CodexBackend()
            command_message = codex.activity_message(
                {
                    "type": "item.started",
                    "item": {
                        "type": "command_execution",
                        "command": f"rg {secret} {workspace / 'evidence'}",
                        "aggregated_output": secret * 100,
                    },
                },
                workspace=workspace,
            )
            failed_command_message = codex.activity_message(
                {
                    "type": "item.completed",
                    "item": {
                        "type": "command_execution",
                        "command": f"rg {secret} {workspace / 'evidence'}",
                        "aggregated_output": secret * 100,
                        "exit_code": 2,
                    },
                },
                workspace=workspace,
            )
            file_message = codex.activity_message(
                {
                    "type": "item.completed",
                    "item": {
                        "type": "file_change",
                        "changes": [{"path": str(source), "kind": "add"}],
                    },
                },
                workspace=workspace,
            )
            chatter_messages = [
                codex.activity_message(
                    {"type": "item.completed", "item": {"type": item_type, "text": secret}},
                    workspace=workspace,
                )
                for item_type in ("agent_message", "reasoning", "todo_list")
            ]
            grok_message = GrokBackend().activity_message(
                {"type": "text", "data": secret},
                workspace=workspace,
            )
            cursor_message = CursorBackend().activity_message(
                {
                    "type": "assistant",
                    "message": {"content": [{"text": secret}]},
                },
                workspace=workspace,
            )

            self.assertIsNone(command_message)
            self.assertEqual(
                failed_command_message,
                "Codex workspace check failed — exit code 2",
            )
            self.assertEqual(file_message, "Codex created src/program.py")
            self.assertEqual(chatter_messages, [None, None, None])
            self.assertIn("drafting", grok_message or "")
            self.assertIn("drafting", cursor_message or "")
            for message in (
                command_message,
                failed_command_message,
                file_message,
                *chatter_messages,
                grok_message,
                cursor_message,
            ):
                self.assertNotIn(secret, message or "")
                self.assertNotIn(str(workspace), message or "")

    def test_codex_heartbeat_is_compact_and_source_focused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve()
            source = workspace / "src" / "program.py"
            source.parent.mkdir()
            source.write_text("pass\n", encoding="utf-8")

            message = CodexBackend().heartbeat_message(
                elapsed_s=245.9,
                provider_event_count=987,
                last_output_age_s=0.1,
                workspace=workspace,
            )

            self.assertEqual(
                message,
                "Codex still authoring — 4m 05s; "
                "plan.json not created; program.py 5 B",
            )
            self.assertNotIn("provider", message)
            self.assertNotIn("last CLI output", message)

    def test_terminal_activity_reports_only_numeric_usage(self) -> None:
        workspace = Path(tempfile.gettempdir()).resolve()
        message = CodexBackend().activity_message(
            {
                "type": "turn.completed",
                "usage": {
                    "output_tokens": 41313,
                    "reasoning_output_tokens": 13232,
                    "private": "not surfaced",
                },
            },
            workspace=workspace,
        )

        self.assertEqual(
            message,
            "Codex model turn completed — 41,313 output tokens, 13,232 reasoning tokens",
        )


class BackendFactoryAndRunTests(unittest.TestCase):
    def test_factory_names_and_opentopos_style_aliases(self) -> None:
        self.assertIs(CodexCLIBackend, CodexBackend)
        self.assertIs(GrokCLIBackend, GrokBackend)
        self.assertIs(CursorCLIBackend, CursorBackend)
        self.assertIsInstance(create_backend("codex"), CodexBackend)
        self.assertIsInstance(create_backend("grok-build"), GrokBackend)
        self.assertIsInstance(create_backend("cursor_agent"), CursorBackend)
        with self.assertRaisesRegex(ValueError, "unknown backend"):
            create_backend("missing")

    def test_missing_binary_returns_failure_and_preserves_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve()
            trajectory = workspace / "trajectories" / "iter_00"
            missing_cli = workspace / "does-not-exist" / "codex"
            result = CodexBackend(cli=str(missing_cli)).run(
                prompt="write both deliverables",
                workspace=workspace,
                trajectory_dir=trajectory,
                timeout_s=5,
            )

            self.assertFalse(result.success)
            self.assertFalse(result.ok)
            self.assertEqual(result.exit_reason, "error")
            self.assertEqual(result.returncode, 127)
            self.assertFalse(result.timed_out)
            self.assertEqual(result.files_modified, ())
            self.assertEqual(result.prompt_path.read_text(encoding="utf-8"), "write both deliverables")
            self.assertEqual(result.transcript_path.read_text(encoding="utf-8"), "")
            self.assertIn("does-not-exist", result.stderr_path.read_text(encoding="utf-8"))

            payload = json.loads(result.result_path.read_text(encoding="utf-8"))
            self.assertFalse(payload["success"])
            self.assertEqual(payload["exit_reason"], "error")
            self.assertEqual(payload["returncode"], 127)
            self.assertEqual(payload["command"][0], str(missing_cli))
            self.assertTrue((trajectory / "final_message.txt").is_file())

    def test_streamed_activity_preserves_exact_trajectory_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve()
            trajectory = workspace / "trajectories" / "iter_00"
            activity: list[str] = []
            result = StreamingFixtureBackend().run(
                prompt="write source",
                workspace=workspace,
                trajectory_dir=trajectory,
                timeout_s=3,
                on_activity=activity.append,
                heartbeat_interval_s=0.01,
            )

            expected = (
                '{"type": "started"}\n'
                '{"type": "completed", "text": "完成"}\n'
            )
            self.assertTrue(result.ok)
            self.assertEqual(result.stdout, expected)
            self.assertEqual(result.transcript_path.read_text(encoding="utf-8"), expected)
            self.assertEqual(result.stderr, "fixture warning\n")
            self.assertEqual(result.stderr_path.read_text(encoding="utf-8"), "fixture warning\n")
            self.assertIn("Fixture provider started", activity)
            self.assertIn("Fixture provider completed", activity)
            self.assertTrue(any("process still running" in message for message in activity))
            self.assertTrue(any("plan.json 4 B" in message for message in activity))
            self.assertTrue(any("program.py not created" in message for message in activity))

    def test_streamed_activity_has_no_heartbeat_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve()
            activity: list[str] = []

            result = StreamingFixtureBackend().run(
                prompt="write source",
                workspace=workspace,
                timeout_s=3,
                on_activity=activity.append,
            )

            self.assertTrue(result.ok)
            self.assertIn("Fixture provider started", activity)
            self.assertIn("Fixture provider completed", activity)
            self.assertFalse(any("still running" in message for message in activity))


if __name__ == "__main__":
    unittest.main()
