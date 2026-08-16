"""Resumable, evidence-backed ProcAgen3D workflow state.

The state file is deliberately local and stdlib-only. Existing CLI behavior is
unchanged when no state file exists. Once initialized, managed pipeline
commands must follow the recorded order; successful commands advance the
checklist and bind their output files by SHA-256.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shlex
import shutil
import tempfile
from pathlib import Path

from .tags import OK, fail


SCHEMA_VERSION = 1
DEFAULT_STATE_PATH = Path(".procagen3d/state.json")
STEP_STATUSES = {"pending", "done"}
STEP_KINDS = {"manual", "command"}
MANAGED_COMMANDS = {"lint", "build", "fit", "check", "joints", "score", "guard"}


class WorkflowStateError(ValueError):
    """Raised when state is invalid or a command violates workflow order."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _store_path(workspace: Path, value: str | Path) -> str:
    path = Path(value).expanduser()
    resolved = path.resolve() if path.is_absolute() else (workspace / path).resolve()
    try:
        return str(resolved.relative_to(workspace))
    except ValueError:
        return str(resolved)


def resolve_state_path(state: dict, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(state["workspace"]) / path).resolve()


def state_path_for_args(args) -> Path:
    raw = getattr(args, "state", None)
    if not raw:
        raw = os.environ.get("PROCAGEN3D_STATE") or DEFAULT_STATE_PATH
    return Path(raw).expanduser()


def _event(state: dict, action: str, **fields) -> None:
    events = state.setdefault("events", [])
    events.append({"sequence": len(events) + 1, "action": action, **fields})


def _manual_step(
    step_id: str,
    instruction: str,
    required_evidence: list[str],
    *,
    scope: str,
) -> dict:
    return {
        "id": step_id,
        "scope": scope,
        "kind": "manual",
        "status": "pending",
        "instruction": instruction,
        "command": "",
        "match": {},
        "outputs": list(required_evidence),
        "evidence": [],
        "note": "",
    }


def _command_step(
    step_id: str,
    command: str,
    match: dict,
    outputs: list[str],
    *,
    scope: str,
) -> dict:
    return {
        "id": step_id,
        "scope": scope,
        "kind": "command",
        "status": "pending",
        "instruction": "",
        "command": command,
        "match": match,
        "outputs": outputs,
        "evidence": [],
        "note": "",
    }


def _cli_prefix(config: dict) -> str:
    return " ".join((
        shlex.quote(str(config["python"])),
        shlex.quote(str(config["cli"])),
        "--state",
        shlex.quote(str(config["state_path"])),
    ))


def _full_pipeline_steps(config: dict, *, repair: dict | None = None) -> list[dict]:
    prefix = _cli_prefix(config)
    program = config["program"]
    out = config["out"]
    form = config["form"]
    tier = config["tier"]
    engine = config["engine"]
    size = config["size"]
    steps = []

    if repair is None:
        steps.append(_manual_step(
            "synthesize",
            f"Write the complete deterministic authoring program at {program}.",
            [program],
            scope="full",
        ))
    else:
        steps.extend((
            _manual_step(
                "repair-edit",
                f"Apply one minimal, evidence-directed edit to {program}; preserve "
                "the passing features recorded in the repair note.",
                [program],
                scope="full",
            ),
            _command_step(
                "repair-guard",
                f"{prefix} guard {shlex.quote(repair['snapshot'])} "
                f"{shlex.quote(program)}"
                + (" --allow-shrink" if repair["allow_shrink"] else "")
                + "".join(
                    f" --allow-drop {shlex.quote(pattern)}"
                    for pattern in repair["allow_drop"]
                ),
                {
                    "cmd": "guard",
                    "old": repair["snapshot"],
                    "new": program,
                    "allow_shrink": repair["allow_shrink"],
                    "allow_drop": repair["allow_drop"],
                },
                [repair["snapshot"], program],
                scope="full",
            ),
        ))

    steps.append(_command_step(
        "source-lint",
        f"{prefix} lint {shlex.quote(program)}",
        {"cmd": "lint", "program": program},
        [program],
        scope="full",
    ))
    build_command = (
        f"{prefix} build {shlex.quote(program)} --out {shlex.quote(out)} "
        f"--size {size} --engine {shlex.quote(engine)}"
    )
    if form in {"curved", "mixed"}:
        build_command += " --form-diagnostics"
    build_outputs = [
        f"{out}/program.py",
        f"{out}/diagnostics.json",
        f"{out}/scene_graph.json",
        f"{out}/model.glb",
        f"{out}/scene.blend",
        f"{out}/renders/sheet.png",
    ]
    if form in {"curved", "mixed"}:
        build_outputs.append(f"{out}/renders/form_sheet.png")
    steps.append(_command_step(
        "build",
        build_command,
        {
            "cmd": "build",
            "program": program,
            "out": out,
            "size": size,
            "engine": engine,
            "render": True,
            "form_diagnostics": form in {"curved", "mixed"},
        },
        build_outputs,
        scope="full",
    ))

    if config["image_conditioned"]:
        fit_spec = config["fit_spec"]
        steps.append(_command_step(
            "fit",
            f"{prefix} fit {shlex.quote(out)} --spec {shlex.quote(fit_spec)} "
            f"--engine {shlex.quote(engine)}",
            {"cmd": "fit", "dir": out, "spec": fit_spec, "engine": engine},
            [
                f"{out}/fit_report.json",
                f"{out}/renders/reference_match.png",
                f"{out}/renders/reference_overlay.png",
            ],
            scope="full",
        ))

    check_outputs = [f"{out}/scene_graph.json"]
    if config["image_conditioned"]:
        check_outputs.append(f"{out}/fit_report.json")
    steps.append(_command_step(
        "check",
        f"{prefix} check {shlex.quote(out)} --tier {tier} --form {form}",
        {"cmd": "check", "dir": out, "tier": tier, "form": form},
        check_outputs,
        scope="full",
    ))
    review_evidence = [f"{out}/renders/sheet.png"]
    review_views = review_evidence[0]
    if form in {"curved", "mixed"}:
        review_evidence.append(f"{out}/renders/form_sheet.png")
        review_views += f" and {review_evidence[-1]}"
    if config["image_conditioned"]:
        review_evidence.append(f"{out}/renders/reference_overlay.png")
        review_views += f", plus {review_evidence[-1]}"
    steps.append(_manual_step(
        "visual-review",
        f"Inspect {review_views}; record shape, form, scale, part coverage, and "
        "detail verdicts. Start a repair instead of marking done if any hard "
        "verdict fails.",
        review_evidence,
        scope="full",
    ))

    if config["has_joints"]:
        steps.append(_command_step(
            "joints",
            f"{prefix} joints {shlex.quote(out)}",
            {"cmd": "joints", "dir": out, "strict": False},
            [f"{out}/joints_report.json"],
            scope="final",
        ))
    if config["spec"]:
        steps.append(_command_step(
            "score",
            f"{prefix} score {shlex.quote(out)} --spec "
            f"{shlex.quote(config['spec'])}",
            {"cmd": "score", "dir": out, "spec": config["spec"]},
            [f"{out}/score_report.json", config["spec"]],
            scope="final",
        ))
    steps.append(_manual_step(
        "deliver",
        f"Deliver {out}/program.py and {out}/model.glb with verified counts, "
        "gate results, and named residual limitations.",
        [f"{out}/program.py", f"{out}/model.glb"],
        scope="final",
    ))
    return steps


def _initial_steps(config: dict) -> list[dict]:
    prefix = _cli_prefix(config)
    out = config["out"]
    form = config["form"]
    steps = [
        _manual_step(
            "intake",
            (f"Preserve every reference and write {out}/priors.md plus "
             f"{out}/fit_spec.json."
             if config["image_conditioned"] else
             f"Record requirements, assumptions, and measurable constraints in "
             f"{out}/intake.md."),
            (config["references"] + [f"{out}/priors.md", f"{out}/fit_spec.json"]
             if config["image_conditioned"] else [f"{out}/intake.md"]),
            scope="setup",
        ),
        _manual_step(
            "design",
            f"Write {out}/design.md with constants, part hierarchy, mandatory "
            "features, form methods, joints, and proof views"
            + (f"; also author the shape-only {out}/form_probe.py."
               if form in {"curved", "mixed"} else "."),
            ([f"{out}/design.md", f"{out}/form_probe.py"]
             if form in {"curved", "mixed"} else [f"{out}/design.md"]),
            scope="setup",
        ),
    ]
    if form in {"curved", "mixed"}:
        probe_program = config["form_probe_program"]
        probe_out = config["form_probe_out"]
        steps.extend((
            _command_step(
                "form-probe-build",
                f"{prefix} build {shlex.quote(probe_program)} --out "
                f"{shlex.quote(probe_out)} --size {config['size']} --engine "
                f"{shlex.quote(config['engine'])} --form-diagnostics",
                {
                    "cmd": "build",
                    "program": probe_program,
                    "out": probe_out,
                    "size": config["size"],
                    "engine": config["engine"],
                    "render": True,
                    "form_diagnostics": True,
                },
                [
                    f"{probe_out}/program.py",
                    f"{probe_out}/diagnostics.json",
                    f"{probe_out}/scene_graph.json",
                    f"{probe_out}/model.glb",
                    f"{probe_out}/scene.blend",
                    f"{probe_out}/renders/sheet.png",
                    f"{probe_out}/renders/form_sheet.png",
                ],
                scope="form",
            ),
            _command_step(
                "form-probe-check",
                f"{prefix} check {shlex.quote(probe_out)} --tier quick --form "
                f"{form}",
                {
                    "cmd": "check",
                    "dir": probe_out,
                    "tier": "quick",
                    "form": form,
                },
                [f"{probe_out}/scene_graph.json"],
                scope="form",
            ),
            _manual_step(
                "form-probe-review",
                f"Inspect {probe_out}/renders/sheet.png and "
                f"{probe_out}/renders/form_sheet.png; record silhouette, volume, "
                "negative-space, attachment, and continuity verdicts.",
                [
                    f"{probe_out}/renders/sheet.png",
                    f"{probe_out}/renders/form_sheet.png",
                ],
                scope="form",
            ),
        ))
    steps.extend(_full_pipeline_steps(config))
    return steps


def new_state(
    *,
    state_path: str | Path,
    out: str | Path,
    program: str | Path | None = None,
    tier: str = "standard",
    form: str = "rectilinear",
    references: list[str] | None = None,
    spec: str | Path | None = None,
    has_joints: bool = False,
    max_repairs: int = 3,
    size: int = 512,
    engine: str = "workbench",
    cli: str | Path = "procagen3d-skill/scripts/procagen3d.py",
    python: str = "python3",
    workspace: str | Path | None = None,
) -> dict:
    workspace_path = Path(workspace or Path.cwd()).expanduser().resolve()
    if tier not in {"quick", "standard", "showcase"}:
        raise WorkflowStateError("tier must be quick, standard, or showcase")
    if form not in {"rectilinear", "curved", "mixed"}:
        raise WorkflowStateError("form must be rectilinear, curved, or mixed")
    if engine not in {"workbench", "eevee", "cycles"}:
        raise WorkflowStateError("engine must be workbench, eevee, or cycles")
    if not isinstance(size, int) or size < 64:
        raise WorkflowStateError("size must be an integer >= 64")
    if not isinstance(max_repairs, int) or max_repairs < 1:
        raise WorkflowStateError("max-repairs must be a positive integer")

    out_value = _store_path(workspace_path, out)
    out_name = Path(out_value).name
    program_value = _store_path(
        workspace_path,
        program if program is not None else Path(out_value) / f"{out_name}.py",
    )
    state_value = _store_path(workspace_path, state_path)
    references = [_store_path(workspace_path, item) for item in references or []]
    spec_value = _store_path(workspace_path, spec) if spec else ""
    config = {
        "state_path": state_value,
        "cli": _store_path(workspace_path, cli),
        "python": str(python),
        "out": out_value,
        "program": program_value,
        "tier": tier,
        "form": form,
        "references": references,
        "image_conditioned": bool(references),
        "fit_spec": f"{out_value}/fit_spec.json",
        "spec": spec_value,
        "has_joints": bool(has_joints),
        "max_repairs": max_repairs,
        "size": size,
        "engine": engine,
        "form_probe_program": f"{out_value}/form_probe.py",
        "form_probe_out": f"{out_value}/form_probe",
    }
    state = {
        "schema_version": SCHEMA_VERSION,
        "workspace": str(workspace_path),
        "status": "active",
        "current_step": "",
        "stop_reason": "",
        "config": config,
        "repairs": {"used": 0, "max": max_repairs, "history": []},
        "steps": _initial_steps(config),
        "events": [],
    }
    _event(state, "initialized", out=out_value, program=program_value)
    recompute(state)
    return validate_state(state)


def validate_state(state) -> dict:
    if not isinstance(state, dict):
        raise WorkflowStateError("state must be a JSON object")
    if (type(state.get("schema_version")) is not int
            or state["schema_version"] != SCHEMA_VERSION):
        raise WorkflowStateError(
            f"unsupported state schema_version: {state.get('schema_version')!r}")
    workspace = state.get("workspace")
    if not isinstance(workspace, str) or not Path(workspace).is_absolute():
        raise WorkflowStateError("state workspace must be an absolute path")
    if state.get("status") not in {"active", "complete", "stopped"}:
        raise WorkflowStateError("state status is invalid")
    config = state.get("config")
    if not isinstance(config, dict):
        raise WorkflowStateError("state config must be an object")
    required_config = {
        "state_path", "cli", "python", "out", "program", "tier", "form",
        "references", "image_conditioned", "fit_spec", "spec", "has_joints",
        "max_repairs", "size", "engine", "form_probe_program", "form_probe_out",
    }
    missing = sorted(required_config - set(config))
    if missing:
        raise WorkflowStateError(f"state config is missing: {', '.join(missing)}")
    string_config = {
        "state_path", "cli", "python", "out", "program", "tier", "form",
        "fit_spec", "spec", "engine", "form_probe_program", "form_probe_out",
    }
    malformed_strings = sorted(
        name for name in string_config if not isinstance(config.get(name), str))
    if malformed_strings:
        raise WorkflowStateError(
            f"state config fields must be strings: {', '.join(malformed_strings)}")
    if config["tier"] not in {"quick", "standard", "showcase"}:
        raise WorkflowStateError("state config tier is invalid")
    if config["form"] not in {"rectilinear", "curved", "mixed"}:
        raise WorkflowStateError("state config form is invalid")
    if config["engine"] not in {"workbench", "eevee", "cycles"}:
        raise WorkflowStateError("state config engine is invalid")
    if not isinstance(config["references"], list) or not all(
            isinstance(item, str) for item in config["references"]):
        raise WorkflowStateError("state config references must be a string list")
    if type(config["image_conditioned"]) is not bool or type(
            config["has_joints"]) is not bool:
        raise WorkflowStateError(
            "state config image_conditioned and has_joints must be booleans")
    if config["image_conditioned"] != bool(config["references"]):
        raise WorkflowStateError(
            "state config image_conditioned conflicts with references")
    if type(config["size"]) is not int or config["size"] < 64:
        raise WorkflowStateError("state config size must be an integer >= 64")
    if type(config["max_repairs"]) is not int or config["max_repairs"] < 1:
        raise WorkflowStateError(
            "state config max_repairs must be a positive integer")
    steps = state.get("steps")
    if not isinstance(steps, list) or not steps:
        raise WorkflowStateError("state steps must be a non-empty list")
    seen = set()
    saw_pending = False
    for entry in steps:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            raise WorkflowStateError("every state step needs a string id")
        if entry["id"] in seen:
            raise WorkflowStateError(f"duplicate state step: {entry['id']}")
        seen.add(entry["id"])
        if entry.get("status") not in STEP_STATUSES:
            raise WorkflowStateError(f"invalid status for step {entry['id']}")
        if entry.get("kind") not in STEP_KINDS:
            raise WorkflowStateError(f"invalid kind for step {entry['id']}")
        if entry.get("scope") not in {"setup", "form", "full", "final"}:
            raise WorkflowStateError(f"invalid scope for step {entry['id']}")
        if not isinstance(entry.get("instruction"), str) or not isinstance(
                entry.get("command"), str) or not isinstance(
                entry.get("note"), str):
            raise WorkflowStateError(
                f"step {entry['id']} has malformed text fields")
        if not isinstance(entry.get("match"), dict):
            raise WorkflowStateError(f"step {entry['id']} match must be an object")
        outputs = entry.get("outputs")
        if (not isinstance(outputs, list) or not outputs
                or not all(isinstance(item, str) and item for item in outputs)):
            raise WorkflowStateError(
                f"step {entry['id']} outputs must be a non-empty string list")
        if entry["kind"] == "command":
            if not entry["command"] or entry["instruction"]:
                raise WorkflowStateError(
                    f"command step {entry['id']} has invalid command text")
            if entry["match"].get("cmd") not in MANAGED_COMMANDS:
                raise WorkflowStateError(
                    f"command step {entry['id']} has invalid command match")
        elif not entry["instruction"] or entry["command"] or entry["match"]:
            raise WorkflowStateError(
                f"manual step {entry['id']} has invalid instruction fields")
        if entry["status"] == "pending":
            saw_pending = True
        elif saw_pending:
            raise WorkflowStateError(
                f"done step {entry['id']} appears after a pending step")
        evidence = entry.get("evidence")
        if not isinstance(evidence, list):
            raise WorkflowStateError(f"step {entry['id']} evidence must be a list")
        for item in evidence:
            if (not isinstance(item, dict) or not isinstance(item.get("path"), str)
                    or not isinstance(item.get("sha256"), str)
                    or len(item["sha256"]) != 64
                    or any(char not in "0123456789abcdef" for char in item["sha256"])
                    or type(item.get("size")) is not int or item["size"] < 0
                    or type(item.get("mtime_ns")) is not int):
                raise WorkflowStateError(
                    f"step {entry['id']} has malformed evidence")
        if entry["status"] == "done" and not evidence:
            raise WorkflowStateError(f"done step {entry['id']} has no evidence")
        if entry["status"] == "done" and not entry["note"]:
            raise WorkflowStateError(f"done step {entry['id']} has no note")
        evidence_paths = {item["path"] for item in evidence}
        missing_outputs = [item for item in outputs if item not in evidence_paths]
        if entry["status"] == "done" and missing_outputs:
            raise WorkflowStateError(
                f"done step {entry['id']} lacks required evidence: "
                f"{', '.join(missing_outputs)}")
        if entry["status"] == "pending" and (evidence or entry["note"]):
            raise WorkflowStateError(
                f"pending step {entry['id']} retains completion evidence")
    repairs = state.get("repairs")
    if not isinstance(repairs, dict):
        raise WorkflowStateError("state repairs must be an object")
    if (type(repairs.get("used")) is not int
            or type(repairs.get("max")) is not int
            or repairs["used"] < 0 or repairs["max"] < 1
            or repairs["used"] > repairs["max"]):
        raise WorkflowStateError("state repair counters are invalid")
    if not isinstance(repairs.get("history"), list):
        raise WorkflowStateError("state repair history must be a list")
    if repairs["max"] != config["max_repairs"]:
        raise WorkflowStateError("state repair ceiling conflicts with config")
    if len(repairs["history"]) != repairs["used"]:
        raise WorkflowStateError("state repair history conflicts with counter")
    for iteration, repair in enumerate(repairs["history"], 1):
        if (not isinstance(repair, dict) or repair.get("iteration") != iteration
                or not isinstance(repair.get("reason"), str)
                or not repair["reason"]
                or not isinstance(repair.get("steps"), list)):
            raise WorkflowStateError("state repair history is malformed")
        snapshot = repair.get("snapshot")
        if (not isinstance(snapshot, dict)
                or not isinstance(snapshot.get("path"), str)
                or not isinstance(snapshot.get("sha256"), str)
                or len(snapshot["sha256"]) != 64):
            raise WorkflowStateError("state repair snapshot is malformed")
        policy = repair.get("guard_policy")
        if (not isinstance(policy, dict)
                or type(policy.get("allow_shrink")) is not bool
                or not isinstance(policy.get("allow_drop"), list)
                or not all(isinstance(item, str) for item in policy["allow_drop"])):
            raise WorkflowStateError("state repair guard policy is malformed")

    if repairs["used"]:
        last_repair = repairs["history"][-1]
        expected = [
            entry for entry in _initial_steps(config)
            if entry["scope"] in {"setup", "form"}
        ]
        expected.extend(_full_pipeline_steps(
            config,
            repair={
                "snapshot": last_repair["snapshot"]["path"],
                "allow_shrink": last_repair["guard_policy"]["allow_shrink"],
                "allow_drop": last_repair["guard_policy"]["allow_drop"],
            },
        ))
    else:
        expected = _initial_steps(config)
    contract_fields = (
        "id", "scope", "kind", "instruction", "command", "match", "outputs",
    )
    actual_contract = [
        {name: entry[name] for name in contract_fields} for entry in steps
    ]
    expected_contract = [
        {name: entry[name] for name in contract_fields} for entry in expected
    ]
    if actual_contract != expected_contract:
        raise WorkflowStateError(
            "state steps do not match the configured workflow contract")
    events = state.get("events")
    if not isinstance(events, list):
        raise WorkflowStateError("state events must be a list")
    for index, event in enumerate(events, 1):
        if (not isinstance(event, dict)
                or type(event.get("sequence")) is not int
                or event["sequence"] != index
                or not isinstance(event.get("action"), str)):
            raise WorkflowStateError("state event history is malformed")
    pending = next((entry for entry in steps if entry["status"] == "pending"), None)
    status = state["status"]
    if not isinstance(state.get("current_step"), str) or not isinstance(
            state.get("stop_reason"), str):
        raise WorkflowStateError("state progress fields must be strings")
    if status == "active" and (
            pending is None or state["current_step"] != pending["id"]
            or state["stop_reason"]):
        raise WorkflowStateError("active state progress is inconsistent")
    if status == "complete" and (
            pending is not None or state["current_step"] != "complete"
            or state["stop_reason"]):
        raise WorkflowStateError("complete state progress is inconsistent")
    if status == "stopped" and (
            state["current_step"] != "stopped" or not state["stop_reason"]):
        raise WorkflowStateError("stopped state progress is inconsistent")
    return state


def load_state(path: str | Path) -> dict:
    path = Path(path).expanduser()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise WorkflowStateError(f"state file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise WorkflowStateError(f"state file is not valid JSON: {path}: {exc}") from exc
    return validate_state(payload)


def save_state(path: str | Path, state: dict) -> None:
    validate_state(state)
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(state, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def current_step(state: dict) -> dict | None:
    return next((entry for entry in state["steps"]
                 if entry["status"] == "pending"), None)


def recompute(state: dict) -> None:
    if state.get("status") == "stopped":
        state["current_step"] = "stopped"
        return
    entry = current_step(state)
    if entry is None:
        state["status"] = "complete"
        state["current_step"] = "complete"
        state["stop_reason"] = ""
    else:
        state["status"] = "active"
        state["current_step"] = entry["id"]
        state["stop_reason"] = ""


def _evidence_records(state: dict, paths: list[str]) -> list[dict]:
    if not paths:
        raise WorkflowStateError("completing a step requires evidence")
    records = []
    seen = set()
    for value in paths:
        stored = _store_path(Path(state["workspace"]), value)
        if stored in seen:
            continue
        seen.add(stored)
        path = resolve_state_path(state, stored)
        if not path.is_file():
            raise WorkflowStateError(f"evidence file does not exist: {stored}")
        stat = path.stat()
        records.append({
            "path": stored,
            "sha256": sha256_file(path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        })
    return records


def audit_state(state: dict) -> bool:
    """Invalidate a completed suffix when any bound evidence becomes stale."""

    if state.get("status") == "stopped":
        return False
    metadata_changed = False
    for index, entry in enumerate(state["steps"]):
        if entry["status"] != "done":
            break
        stale = ""
        for evidence in entry["evidence"]:
            path = resolve_state_path(state, evidence["path"])
            if not path.is_file():
                stale = f"missing {evidence['path']}"
                break
            stat = path.stat()
            if (evidence.get("size") == stat.st_size
                    and evidence.get("mtime_ns") == stat.st_mtime_ns):
                continue
            if sha256_file(path) != evidence["sha256"]:
                stale = f"changed {evidence['path']}"
                break
            evidence["size"] = stat.st_size
            evidence["mtime_ns"] = stat.st_mtime_ns
            metadata_changed = True
        if not stale:
            continue
        invalidated = []
        for later in state["steps"][index:]:
            if later["status"] == "done":
                invalidated.append(later["id"])
            later["status"] = "pending"
            later["evidence"] = []
            later["note"] = ""
        _event(
            state,
            "invalidated",
            step=entry["id"],
            reason=stale,
            invalidated=invalidated,
        )
        recompute(state)
        return True
    recompute(state)
    return metadata_changed


def complete_manual_step(
    state: dict,
    step_id: str,
    evidence: list[str],
    note: str,
) -> None:
    audit_state(state)
    if state["status"] == "stopped":
        raise WorkflowStateError(f"workflow is stopped: {state['stop_reason']}")
    entry = current_step(state)
    expected = entry["id"] if entry else "complete"
    if entry is None or entry["id"] != step_id:
        raise WorkflowStateError(
            f"out-of-order completion: expected {expected}, received {step_id}")
    if entry["kind"] != "manual":
        raise WorkflowStateError(
            f"step {step_id} advances only after its exact command succeeds")
    if not note.strip():
        raise WorkflowStateError("manual completion requires --note")
    records = _evidence_records(state, evidence)
    received = {resolve_state_path(state, item["path"]) for item in records}
    missing = [
        required for required in entry["outputs"]
        if resolve_state_path(state, required) not in received
    ]
    if missing:
        raise WorkflowStateError(
            f"{step_id} evidence must include: {', '.join(missing)}")
    entry["status"] = "done"
    entry["evidence"] = records
    entry["note"] = note.strip()
    _event(state, "completed", step=step_id, evidence=records)
    recompute(state)


def _same_path(state: dict, actual, expected: str) -> bool:
    if actual is None:
        return False
    return resolve_state_path(state, actual) == resolve_state_path(state, expected)


def _command_mismatch(state: dict, entry: dict, args) -> str | None:
    match = entry["match"]
    if args.cmd != match.get("cmd"):
        return f"expected command {match.get('cmd')!r}, received {args.cmd!r}"
    for name in ("program", "out", "dir", "spec", "old", "new"):
        if name in match and not _same_path(state, getattr(args, name, None), match[name]):
            return f"{name} must be {match[name]!r}"
    if "tier" in match and getattr(args, "tier", None) != match["tier"]:
        return f"--tier must be {match['tier']}"
    if "form" in match and getattr(args, "form", None) != match["form"]:
        return f"--form must be {match['form']}"
    if "size" in match and getattr(args, "size", None) != match["size"]:
        return f"--size must be {match['size']}"
    if "engine" in match and getattr(args, "engine", None) != match["engine"]:
        return f"--engine must be {match['engine']}"
    if match.get("render") and getattr(args, "no_render", False):
        return "the state workflow requires rendered evidence; remove --no-render"
    if ("form_diagnostics" in match
            and bool(getattr(args, "form_diagnostics", False))
            != bool(match["form_diagnostics"])):
        expected = "include" if match["form_diagnostics"] else "omit"
        return f"{expected} --form-diagnostics"
    for name in ("strict", "allow_shrink"):
        if name in match and bool(getattr(args, name, False)) != match[name]:
            expected = "include" if match[name] else "omit"
            return f"{expected} --{name.replace('_', '-')}"
    if "allow_drop" in match:
        actual = list(getattr(args, "allow_drop", None) or [])
        if actual != match["allow_drop"]:
            return f"--allow-drop must be {match['allow_drop']!r}"
    return None


def authorize_pipeline_command(args):
    """Return loaded state for a managed command, or ``None`` when inactive."""

    path = state_path_for_args(args)
    explicit = bool(
        getattr(args, "state", None) or os.environ.get("PROCAGEN3D_STATE"))
    if not os.path.lexists(path):
        if explicit:
            raise WorkflowStateError(f"state file does not exist: {path}")
        return None
    if not path.is_file():
        raise WorkflowStateError(f"state path is not a regular file: {path}")
    state = load_state(path)
    changed = audit_state(state)
    if changed:
        save_state(path, state)
    if state["status"] == "stopped":
        raise WorkflowStateError(f"workflow is stopped: {state['stop_reason']}")
    entry = current_step(state)
    if args.cmd not in MANAGED_COMMANDS:
        return {"path": path, "state": state, "entry": None}
    if entry is None:
        raise WorkflowStateError("workflow is complete; no managed command is pending")
    if entry["kind"] != "command":
        raise WorkflowStateError(
            f"current step {entry['id']!r} requires manual evidence first; run `next`")
    mismatch = _command_mismatch(state, entry, args)
    if mismatch:
        raise WorkflowStateError(
            f"command does not match current step {entry['id']!r}: {mismatch}. "
            f"Run `next` for the exact command")
    return {"path": path, "state": state, "entry": entry}


def _validate_json_output(path: Path, step_id: str) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowStateError(f"invalid {path.name} for {step_id}: {exc}") from exc
    if step_id in {"build", "form-probe-build"} and payload.get("build_ok") is not True:
        raise WorkflowStateError(f"{path.name} does not record build_ok=true")
    if step_id == "fit" and payload.get("passed") is not True:
        raise WorkflowStateError(f"{path.name} does not record a passing fit")
    if step_id == "joints" and payload.get("failures") != 0:
        raise WorkflowStateError(f"{path.name} does not record zero failures")
    if step_id == "score" and (
            type(payload.get("passed")) is not int
            or type(payload.get("total")) is not int
            or payload["total"] < 1
            or payload["passed"] != payload["total"]):
        raise WorkflowStateError(
            f"{path.name} does not record a non-empty passing constraint set")


def record_pipeline_result(context, exit_code: int) -> None:
    if context is None or context["entry"] is None:
        return
    path = context["path"]
    state = load_state(path)
    audit_state(state)
    entry = current_step(state)
    expected_id = context["entry"]["id"]
    if entry is None or entry["id"] != expected_id:
        received = entry["id"] if entry else state["current_step"]
        raise WorkflowStateError(
            f"workflow changed while {expected_id!r} was running; current step "
            f"is {received!r}")
    if exit_code != 0:
        _event(state, "command-failed", step=entry["id"], exit_code=exit_code)
        save_state(path, state)
        return
    outputs = [resolve_state_path(state, item) for item in entry["outputs"]]
    missing = [entry["outputs"][index] for index, item in enumerate(outputs)
               if not item.is_file()]
    if missing:
        raise WorkflowStateError(
            f"command succeeded but required evidence is missing: {', '.join(missing)}")
    for output in outputs:
        if output.name in {
            "diagnostics.json", "fit_report.json", "joints_report.json",
            "score_report.json",
        }:
            _validate_json_output(output, entry["id"])
    if entry["id"] == "build":
        authored = resolve_state_path(state, state["config"]["program"])
        retained = resolve_state_path(
            state, f"{state['config']['out']}/program.py")
        if authored == retained:
            refreshed = sha256_file(authored)
            for previous in state["steps"]:
                if previous is entry:
                    break
                for evidence in previous["evidence"]:
                    if resolve_state_path(state, evidence["path"]) == authored:
                        evidence["sha256"] = refreshed
                        stat = authored.stat()
                        evidence["size"] = stat.st_size
                        evidence["mtime_ns"] = stat.st_mtime_ns
            _event(state, "source-frozen-in-place", path=state["config"]["program"])
    records = _evidence_records(state, entry["outputs"])
    entry["status"] = "done"
    entry["evidence"] = records
    entry["note"] = "command exited 0"
    _event(state, "completed", step=entry["id"], evidence=records)
    recompute(state)
    save_state(path, state)
    print(f"{OK} workflow advanced {entry['id']} -> {state['current_step']}")


def start_repair(
    state: dict,
    *,
    evidence: list[str],
    reason: str,
    allow_shrink: bool = False,
    allow_drop: list[str] | None = None,
) -> None:
    audit_state(state)
    if state["status"] == "stopped":
        raise WorkflowStateError(f"workflow is stopped: {state['stop_reason']}")
    if not reason.strip():
        raise WorkflowStateError("starting a repair requires --reason")
    review_evidence = _evidence_records(state, evidence)
    repairs = state["repairs"]
    if repairs["used"] >= repairs["max"]:
        state["status"] = "stopped"
        state["current_step"] = "stopped"
        state["stop_reason"] = "max-repair-iterations-reached"
        _event(state, "stopped", reason=state["stop_reason"])
        return
    entry = current_step(state)
    if entry is None or entry.get("scope") not in {"full", "final"}:
        raise WorkflowStateError(
            "a repair can start only after entering the full-program pipeline")
    program = resolve_state_path(state, state["config"]["program"])
    if not program.is_file():
        raise WorkflowStateError(f"cannot snapshot missing program: {program}")
    out = resolve_state_path(state, state["config"]["out"])
    snapshot_index = 1
    while (out / f"program.iter{snapshot_index}.py").exists():
        snapshot_index += 1
    snapshot = out / f"program.iter{snapshot_index}.py"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(program, snapshot)
    snapshot_value = _store_path(Path(state["workspace"]), snapshot)

    repairs["used"] += 1
    repairs["history"].append({
        "iteration": repairs["used"],
        "reason": reason.strip(),
        "review_evidence": review_evidence,
        "snapshot": {
            "path": snapshot_value,
            "sha256": sha256_file(snapshot),
        },
        "guard_policy": {
            "allow_shrink": bool(allow_shrink),
            "allow_drop": list(allow_drop or []),
        },
        "steps": copy.deepcopy(state["steps"]),
    })
    persistent = [copy.deepcopy(step) for step in state["steps"]
                  if step["scope"] in {"setup", "form"}]
    repair = {
        "snapshot": snapshot_value,
        "allow_shrink": bool(allow_shrink),
        "allow_drop": list(allow_drop or []),
    }
    state["steps"] = persistent + _full_pipeline_steps(state["config"], repair=repair)
    state["status"] = "active"
    state["stop_reason"] = ""
    _event(
        state,
        "repair-started",
        iteration=repairs["used"],
        reason=reason.strip(),
        snapshot=snapshot_value,
    )
    recompute(state)


def _manual_completion_command(state: dict, entry: dict) -> str:
    evidence = "".join(
        f" --evidence {shlex.quote(path)}" for path in entry["outputs"])
    return (
        f"{_cli_prefix(state['config'])} next --done {shlex.quote(entry['id'])}"
        f"{evidence} --note '<verdict>'"
    )


def _status_payload(state: dict) -> dict:
    entry = current_step(state) if state["status"] == "active" else None
    next_command = None
    if entry:
        next_command = (
            entry["command"] if entry["kind"] == "command"
            else _manual_completion_command(state, entry)
        )
    return {
        "schema_version": state["schema_version"],
        "status": state["status"],
        "current_step": state["current_step"],
        "stop_reason": state["stop_reason"],
        "next_command": next_command,
        "instruction": entry["instruction"] if entry and entry["kind"] == "manual" else None,
        "required_evidence": entry["outputs"] if entry else [],
        "pending": [step["id"] for step in state["steps"]
                    if step["status"] == "pending"],
        "repairs": {
            "used": state["repairs"]["used"],
            "max": state["repairs"]["max"],
        },
        "config": state["config"],
    }


def print_status(state: dict, *, as_json: bool = False) -> None:
    payload = _status_payload(state)
    if as_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    repairs = payload["repairs"]
    print(
        f"{OK} workflow status={payload['status']} "
        f"step={payload['current_step']} repairs={repairs['used']}/{repairs['max']}"
    )
    if payload["stop_reason"]:
        print(f"STOP: {payload['stop_reason']}")
    elif payload["instruction"]:
        print(f"instruction: {payload['instruction']}")
        print(f"next command: {payload['next_command']}")
    elif payload["next_command"]:
        print(f"next command: {payload['next_command']}")
    if payload["pending"]:
        print("pending: " + " -> ".join(payload["pending"]))


def cmd_next(args) -> int:
    path = Path(getattr(args, "next_state", None) or state_path_for_args(args))
    try:
        if args.init:
            if not args.out:
                raise WorkflowStateError("--out is required with --init")
            if os.path.lexists(path):
                raise WorkflowStateError(f"refusing to overwrite existing state: {path}")
            state = new_state(
                state_path=path,
                out=args.out,
                program=args.program,
                tier=args.tier,
                form=args.form,
                references=args.reference,
                spec=args.spec,
                has_joints=args.joints,
                max_repairs=args.max_repairs,
                size=args.size,
                engine=args.engine,
                cli=args.cli_path,
                python=args.python,
            )
            resolve_state_path(state, state["config"]["out"]).mkdir(
                parents=True, exist_ok=True)
            save_state(path, state)
            print_status(state, as_json=args.json)
            return 0

        state = load_state(path)
        changed = audit_state(state)
        if args.done:
            complete_manual_step(state, args.done, args.evidence, args.note)
            changed = True
        elif args.repair:
            start_repair(
                state,
                evidence=args.evidence,
                reason=args.reason,
                allow_shrink=args.allow_shrink,
                allow_drop=args.allow_drop,
            )
            changed = True
        if changed:
            save_state(path, state)
        print_status(state, as_json=args.json)
        return 3 if state["status"] == "stopped" else 0
    except (OSError, WorkflowStateError) as exc:
        fail("STATE", str(exc))
        return 2
