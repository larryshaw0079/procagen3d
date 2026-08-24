"""Rich terminal presentation for CLI progress and summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console, Group
from rich.panel import Panel
from rich.status import Status
from rich.table import Table
from rich.text import Text

from .progress import ProgressEvent


class ProgressConsole(Console):
    """Rich console whose broken stderr pipe never rewires stdout."""

    def on_broken_pipe(self) -> None:
        # Rich's default handler dup2s /dev/null over sys.stdout before exiting.
        # That is appropriate for a primary output pipe, but progress lives on
        # stderr and must not suppress the eventual stdout result.
        self.quiet = True
        raise BrokenPipeError


class RichProgressReporter:
    """Render one animated status for the currently blocking pipeline stage."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or ProgressConsole(stderr=True)
        self._status: Status | None = None
        self._active_stage: str | None = None

    def __enter__(self) -> "RichProgressReporter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def close(self) -> None:
        status = self._status
        self._status = None
        self._active_stage = None
        if status is not None:
            try:
                status.stop()
            except (Exception, SystemExit):
                # Terminal presentation is observational: Live/console cleanup
                # must not mask a pipeline error or invalidate a completed build.
                return

    def __call__(self, event: ProgressEvent) -> None:
        if event.kind == "start":
            self.close()
            label = Text(event.message, style="bold cyan")
            label.append(f"  [{event.stage}]", style="dim")
            self._active_stage = event.stage
            if not self.console.is_terminal:
                line = Text("→ ", style="bold cyan")
                line.append_text(label)
                self.console.print(line)
                return
            self._status = Status(label, console=self.console, spinner="dots")
            self._status.start()
            return

        completes_active_step = (
            event.elapsed_s is not None and event.stage == self._active_stage
        )
        if event.kind in {"success", "failure"} or completes_active_step:
            self.close()

        icon, style = {
            "success": ("✓", "bold green"),
            "failure": ("✗", "bold red"),
            "warning": ("!", "bold yellow"),
            "info": ("•", "bold blue"),
        }[event.kind]
        line = Text()
        line.append(f"{icon} ", style=style)
        line.append(event.message)
        if event.elapsed_s is not None:
            line.append(f"  {event.elapsed_s:.1f}s", style="dim")
        error = event.details.get("error")
        if error:
            first_line = str(error).strip().splitlines()[0][:240]
            if first_line:
                line.append(f" — {first_line}", style="dim red")
        self.console.print(line)


def print_json(console: Console, value: Any) -> None:
    """Pretty-print JSON while remaining plain JSON when output is redirected."""

    console.print_json(json.dumps(value, ensure_ascii=False))


def print_workspace_summary(
    console: Console,
    *,
    workspace: Path,
    report: dict[str, Any],
) -> None:
    status = str(report["status"])
    status_style = {
        "complete": "bold green",
        "prepared": "bold cyan",
        "needs-review": "bold yellow",
        "failed": "bold red",
    }.get(status, "bold")
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold")
    table.add_column(overflow="fold")
    table.add_row("Status", Text(status, style=status_style))
    table.add_row("Workspace", str(workspace))
    if report.get("score") is not None:
        table.add_row("Fidelity", f"{float(report['score']):.4f}")
    deliverables = report.get("deliverables")
    if deliverables:
        table.add_row("Blender source", str(workspace / str(deliverables["program"])))
        table.add_row("Compiled GLB", str(workspace / str(deliverables["glb"])))
        table.add_row("Blender scene", str(workspace / str(deliverables["blend"])))
    content: list[Any] = [table]
    if report.get("warning"):
        warning = Text(str(report["warning"]), style="yellow")
        content.extend((Text(), warning))
    console.print(Panel(Group(*content), title="ProcAgen3D", border_style=status_style))


def print_comparison(console: Console, report: dict[str, Any]) -> None:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    table = Table(title="Compiled GLB fidelity", show_header=True, header_style="bold cyan")
    table.add_column("Metric")
    table.add_column("Score", justify="right")
    rows = (
        ("Overall", report.get("score")),
        ("Silhouette IoU", summary.get("mean_silhouette_iou")),
        ("Area", summary.get("mean_area_similarity")),
        ("Spatial RGB", summary.get("mean_spatial_rgb_similarity")),
        ("Dimensions", summary.get("dimension_similarity")),
        ("Center", summary.get("center_similarity")),
    )
    for name, value in rows:
        table.add_row(name, f"{float(value):.4f}" if isinstance(value, (int, float)) else "—")
    passed = bool(report.get("passed"))
    table.caption = "PASS" if passed else "NEEDS REVIEW"
    table.caption_style = "bold green" if passed else "bold yellow"
    console.print(table)
