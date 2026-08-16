"""Tests for the resumable ProcAgen3D workflow state machine."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = SKILL_ROOT / "scripts"
CLI = SCRIPTS_DIR / "procagen3d.py"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from harness.workflow import (  # noqa: E402
    WorkflowStateError,
    audit_state,
    authorize_pipeline_command,
    complete_manual_step,
    current_step,
    load_state,
    new_state,
    record_pipeline_result,
    save_state,
    start_repair,
    validate_state,
)


def write(path: Path, content: str = "evidence\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def make_state(root: Path, **overrides):
    options = {
        "state_path": root / ".procagen3d/state.json",
        "out": root / "asset",
        "workspace": root,
        "cli": CLI,
        "python": sys.executable,
    }
    options.update(overrides)
    return new_state(**options)


def finish_manual(state: dict, step_id: str, path: Path) -> None:
    write(path)
    complete_manual_step(state, step_id, [str(path)], f"{step_id} passed")


def create_build_outputs(root: Path, *, form_diagnostics: bool = False) -> None:
    write(root / "program.py", "def build():\n    return None\n")
    write(root / "diagnostics.json", json.dumps({"build_ok": True}))
    write(root / "scene_graph.json", json.dumps({"objects": [], "totals": {}}))
    write(root / "model.glb", "glb")
    write(root / "scene.blend", "blend")
    write(root / "renders/sheet.png", "sheet")
    if form_diagnostics:
        write(root / "renders/form_sheet.png", "form-sheet")


class WorkflowShapeTests(unittest.TestCase):
    def test_rectilinear_text_state_has_only_applicable_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            state = make_state(Path(directory))
        ids = [entry["id"] for entry in state["steps"]]
        self.assertEqual(state["current_step"], "intake")
        self.assertNotIn("form-probe-build", ids)
        self.assertNotIn("fit", ids)
        self.assertNotIn("joints", ids)
        self.assertNotIn("score", ids)
        self.assertLess(ids.index("source-lint"), ids.index("build"))
        self.assertLess(ids.index("build"), ids.index("check"))

    def test_conditional_state_includes_form_fit_joints_and_score(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = make_state(
                root,
                form="mixed",
                references=[str(root / "reference.png")],
                spec=root / "spec.yaml",
                has_joints=True,
            )
        ids = [entry["id"] for entry in state["steps"]]
        design = next(entry for entry in state["steps"]
                      if entry["id"] == "design")
        self.assertLess(ids.index("design"), ids.index("form-probe-build"))
        self.assertIn("asset/form_probe.py", design["outputs"])
        self.assertLess(ids.index("form-probe-review"), ids.index("synthesize"))
        self.assertLess(ids.index("build"), ids.index("fit"))
        self.assertLess(ids.index("fit"), ids.index("check"))
        self.assertLess(ids.index("visual-review"), ids.index("joints"))
        self.assertLess(ids.index("joints"), ids.index("score"))
        self.assertLess(ids.index("score"), ids.index("deliver"))


class WorkflowEvidenceTests(unittest.TestCase):
    def test_manual_steps_require_order_evidence_and_note(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = make_state(root)
            evidence = write(root / "asset/intake.md")
            with self.assertRaisesRegex(WorkflowStateError, "out-of-order"):
                complete_manual_step(
                    state, "design", [str(evidence)], "design passed")
            with self.assertRaisesRegex(WorkflowStateError, "--note"):
                complete_manual_step(state, "intake", [str(evidence)], "")
            with self.assertRaisesRegex(WorkflowStateError, "does not exist"):
                complete_manual_step(
                    state, "intake", [str(root / "missing.md")], "passed")
            unrelated = write(root / "unrelated.md")
            with self.assertRaisesRegex(WorkflowStateError, "must include"):
                complete_manual_step(
                    state, "intake", [str(unrelated)], "passed")
            complete_manual_step(state, "intake", [str(evidence)], "passed")
            self.assertEqual(state["current_step"], "design")

    def test_changed_evidence_reopens_completed_suffix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = make_state(root)
            intake = write(root / "asset/intake.md", "v1\n")
            complete_manual_step(state, "intake", [str(intake)], "passed")
            finish_manual(state, "design", root / "asset/design.md")
            write(intake, "v2\n")
            self.assertTrue(audit_state(state))
            self.assertEqual(state["current_step"], "intake")
            self.assertTrue(all(step["status"] == "pending" for step in state["steps"]))
            self.assertEqual(state["events"][-1]["action"], "invalidated")

    def test_image_intake_requires_every_declared_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = write(root / "asset/reference_01.png", "png")
            priors = write(root / "asset/priors.md")
            fit_spec = write(root / "asset/fit_spec.json", "{}")
            state = make_state(root, references=[str(reference)])
            with self.assertRaisesRegex(WorkflowStateError, "fit_spec.json"):
                complete_manual_step(
                    state,
                    "intake",
                    [str(reference), str(priors)],
                    "intake ready",
                )
            complete_manual_step(
                state,
                "intake",
                [str(reference), str(priors), str(fit_spec)],
                "intake ready",
            )
            self.assertEqual(state["current_step"], "design")

    def test_malformed_progress_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            state = make_state(Path(directory))
            state["current_step"] = "build"
            with self.assertRaisesRegex(WorkflowStateError, "inconsistent"):
                validate_state(state)

    def test_missing_gate_fails_workflow_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            state = make_state(Path(directory))
            state["steps"] = [
                entry for entry in state["steps"] if entry["id"] != "build"
            ]
            with self.assertRaisesRegex(WorkflowStateError, "workflow contract"):
                validate_state(state)

    def test_probe_source_edit_reopens_probe_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_path = root / ".procagen3d/state.json"
            state = make_state(root, form="mixed")
            finish_manual(state, "intake", root / "asset/intake.md")
            design = write(root / "asset/design.md")
            probe = write(root / "asset/form_probe.py", "def build():\n    pass\n")
            complete_manual_step(
                state,
                "design",
                [str(design), str(probe)],
                "probe design ready",
            )
            save_state(state_path, state)

            probe_out = root / "asset/form_probe"
            create_build_outputs(probe_out, form_diagnostics=True)
            build_args = SimpleNamespace(
                cmd="build",
                state=str(state_path),
                program=str(probe),
                out=str(probe_out),
                size=512,
                engine="workbench",
                no_render=False,
                form_diagnostics=True,
            )
            record_pipeline_result(authorize_pipeline_command(build_args), 0)
            check_args = SimpleNamespace(
                cmd="check",
                state=str(state_path),
                dir=str(probe_out),
                tier="quick",
                form="mixed",
            )
            record_pipeline_result(authorize_pipeline_command(check_args), 0)
            state = load_state(state_path)
            self.assertEqual(state["current_step"], "form-probe-review")

            write(probe, "def build():\n    return 'revised'\n")
            self.assertTrue(audit_state(state))
            self.assertEqual(state["current_step"], "design")


class WorkflowCommandTests(unittest.TestCase):
    def test_managed_commands_are_blocked_until_the_exact_step(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_path = root / ".procagen3d/state.json"
            state = make_state(root)
            save_state(state_path, state)
            args = SimpleNamespace(
                cmd="lint",
                state=str(state_path),
                program=str(root / "asset/asset.py"),
            )
            with self.assertRaisesRegex(WorkflowStateError, "manual evidence"):
                authorize_pipeline_command(args)

    def test_successful_lint_advances_and_binds_source_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_path = root / ".procagen3d/state.json"
            state = make_state(root)
            finish_manual(state, "intake", root / "asset/intake.md")
            finish_manual(state, "design", root / "asset/design.md")
            program = write(root / "asset/asset.py", "def build():\n    return None\n")
            complete_manual_step(state, "synthesize", [str(program)], "source ready")
            save_state(state_path, state)

            wrong = SimpleNamespace(
                cmd="lint", state=str(state_path), program=str(root / "wrong.py"))
            with self.assertRaisesRegex(WorkflowStateError, "program must be"):
                authorize_pipeline_command(wrong)

            args = SimpleNamespace(
                cmd="lint", state=str(state_path), program=str(program))
            context = authorize_pipeline_command(args)
            record_pipeline_result(context, 0)
            advanced = load_state(state_path)
            self.assertEqual(advanced["current_step"], "build")
            lint_step = next(step for step in advanced["steps"]
                             if step["id"] == "source-lint")
            self.assertEqual(len(lint_step["evidence"]), 1)

    def test_failed_command_does_not_advance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_path = root / ".procagen3d/state.json"
            state = make_state(root)
            finish_manual(state, "intake", root / "asset/intake.md")
            finish_manual(state, "design", root / "asset/design.md")
            program = write(root / "asset/asset.py")
            complete_manual_step(state, "synthesize", [str(program)], "ready")
            save_state(state_path, state)
            args = SimpleNamespace(
                cmd="lint", state=str(state_path), program=str(program))
            context = authorize_pipeline_command(args)
            record_pipeline_result(context, 1)
            failed = load_state(state_path)
            self.assertEqual(failed["current_step"], "source-lint")
            self.assertEqual(failed["events"][-1]["action"], "command-failed")


class WorkflowRepairTests(unittest.TestCase):
    def _state_at_visual_review(self, root: Path) -> dict:
        state_path = root / ".procagen3d/state.json"
        state = make_state(root, max_repairs=1)
        finish_manual(state, "intake", root / "asset/intake.md")
        finish_manual(state, "design", root / "asset/design.md")
        program = write(root / "asset/asset.py", "def build():\n    return None\n")
        complete_manual_step(state, "synthesize", [str(program)], "ready")
        save_state(state_path, state)
        lint_args = SimpleNamespace(
            cmd="lint", state=str(state_path), program=str(program))
        record_pipeline_result(authorize_pipeline_command(lint_args), 0)

        create_build_outputs(root / "asset")
        build_args = SimpleNamespace(
            cmd="build",
            state=str(state_path),
            program=str(program),
            out=str(root / "asset"),
            size=512,
            engine="workbench",
            no_render=False,
            form_diagnostics=False,
        )
        record_pipeline_result(authorize_pipeline_command(build_args), 0)
        check_args = SimpleNamespace(
            cmd="check",
            state=str(state_path),
            dir=str(root / "asset"),
            tier="standard",
            form="rectilinear",
        )
        record_pipeline_result(authorize_pipeline_command(check_args), 0)
        return load_state(state_path)

    def test_repair_snapshots_source_and_resets_only_full_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = self._state_at_visual_review(root)
            review = root / "asset/renders/sheet.png"
            start_repair(
                state,
                evidence=[str(review)],
                reason="Silhouette too narrow; preserve height and hierarchy",
            )
            self.assertEqual(state["repairs"]["used"], 1)
            self.assertEqual(state["current_step"], "repair-edit")
            snapshot = state["repairs"]["history"][0]["snapshot"]["path"]
            self.assertTrue((root / snapshot).is_file())
            setup = [step for step in state["steps"] if step["scope"] == "setup"]
            self.assertTrue(all(step["status"] == "done" for step in setup))
            ids = [step["id"] for step in state["steps"]]
            self.assertLess(ids.index("repair-edit"), ids.index("repair-guard"))
            self.assertLess(ids.index("repair-guard"), ids.index("source-lint"))
            validate_state(state)

            start_repair(
                state,
                evidence=[str(review)],
                reason="A second repair would exceed the configured ceiling",
            )
            self.assertEqual(state["status"], "stopped")
            self.assertEqual(state["stop_reason"], "max-repair-iterations-reached")

    def test_repair_records_and_prints_guard_exceptions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = self._state_at_visual_review(root)
            start_repair(
                state,
                evidence=[str(root / "asset/renders/sheet.png")],
                reason="Rewrite the body representation; preserve the openings",
                allow_shrink=True,
                allow_drop=["LegacyTrim*"],
            )
            policy = state["repairs"]["history"][0]["guard_policy"]
            self.assertEqual(
                policy, {"allow_shrink": True, "allow_drop": ["LegacyTrim*"]})
            guard = next(step for step in state["steps"]
                         if step["id"] == "repair-guard")
            self.assertIn("--allow-shrink", guard["command"])
            self.assertIn("--allow-drop 'LegacyTrim*'", guard["command"])
            validate_state(state)


class WorkflowCliTests(unittest.TestCase):
    def run_cli(self, root: Path, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(CLI), *arguments],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    def test_cli_initializes_resumes_and_auto_advances_lint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initialized = self.run_cli(root, "next", "--init", "--out", "asset")
            self.assertEqual(initialized.returncode, 0, initialized.stdout)
            self.assertIn("step=intake", initialized.stdout)
            self.assertIn("--evidence asset/intake.md", initialized.stdout)
            self.assertNotIn("<file>", initialized.stdout)
            state_path = root / ".procagen3d/state.json"
            self.assertTrue(state_path.is_file())

            intake = write(root / "asset/intake.md")
            result = self.run_cli(
                root, "next", "--done", "intake", "--evidence", str(intake),
                "--note", "requirements recorded")
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertIn("step=design", result.stdout)

            design = write(root / "asset/design.md")
            result = self.run_cli(
                root, "next", "--done", "design", "--evidence", str(design),
                "--note", "design accepted")
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertIn("step=synthesize", result.stdout)

            program = write(
                root / "asset/asset.py",
                "import bpy\n\ndef build():\n    return None\n",
            )
            result = self.run_cli(
                root, "next", "--done", "synthesize", "--evidence", str(program),
                "--note", "source complete")
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertIn("step=source-lint", result.stdout)

            linted = self.run_cli(root, "lint", "asset/asset.py")
            self.assertEqual(linted.returncode, 0, linted.stdout)
            resumed = self.run_cli(root, "next", "--json")
            self.assertEqual(resumed.returncode, 0, resumed.stdout)
            payload = json.loads(resumed.stdout)
            self.assertEqual(payload["current_step"], "build")

    def test_cli_without_default_state_remains_legacy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            program = write(root / "program.py", "def build():\n    return None\n")
            result = self.run_cli(root, "lint", str(program))
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertIn("source lint passed", result.stdout)
            self.assertNotIn("workflow advanced", result.stdout)

    def test_cli_refuses_to_overwrite_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self.run_cli(root, "next", "--init", "--out", "asset")
            second = self.run_cli(root, "next", "--init", "--out", "other")
            self.assertEqual(first.returncode, 0, first.stdout)
            self.assertEqual(second.returncode, 2, second.stdout)
            self.assertIn("refusing to overwrite", second.stdout)

    def test_explicit_missing_state_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            program = write(root / "program.py", "def build():\n    return None\n")
            result = self.run_cli(
                root,
                "--state",
                ".procagen3d/missing.json",
                "lint",
                str(program),
            )
            self.assertEqual(result.returncode, 2, result.stdout)
            self.assertIn("state file does not exist", result.stdout)


if __name__ == "__main__":
    unittest.main()
