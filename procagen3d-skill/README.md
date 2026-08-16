# procagen3d-skill

**Version:** 0.1.1 — see the repository [update log](../README.md#update-log).

An agent skill implementing **ProcAgen3D: Code-Native Generation of Programmable
3D Assets** ([arXiv:2607.22738](https://arxiv.org/abs/2607.22738)) for
Claude Code, Codex, and compatible agent runtimes.

The agent plays the paper's generator ℳθ: it writes an executable Blender
Python program (named parts, real transform tree, joints with limits,
dimensions as constants); headless Blender compiles it to a GLB. A
deterministic harness enforces the paper's pipeline — build + canonical
renders → registered image-fit gates → deterministic checks → agent-vision
inspection → guarded repair loop (≤3) →
articulation validation → constraint scoring — while the agent's judgment
does design and visual review.

## Layout

```
procagen3d-skill/
├── SKILL.md                  # entry point — the staged loop and gates
├── runtime/
│   └── procagen3d_runtime.py # versioned API vendored into program.py
├── scripts/
│   ├── procagen3d.py         # CLI entry: argparse + dispatch
│   ├── blender_stages.py     # Blender entry: invoked by the driver
│   ├── harness/              # stdlib gates (lint, check, guard, score, …)
│   └── bpy_stages/           # bpy-side stages (build, render, fit, joints)
├── references/               # routed depth, read on demand from SKILL.md
│   ├── doctrine.md           # representation doctrine + canonical runtime
│   ├── image-analysis.md     # agent-vision perception stage
│   ├── image-fit.md          # registered camera/mask/landmark/layout contract
│   ├── complex-forms.md      # loft/sweep/shell routing + shape-first probe
│   ├── detail.md             # tier floors + decomposition recipes
│   ├── articulation.md       # joint schema + validator semantics
│   ├── constraints.md        # spec.yaml format + scorer
│   ├── local-edits.md        # source-level edit protocol + gates
│   └── blender-pitfalls.md   # headless bpy traps
└── examples/                 # ProcAgen3D-Bench-style items (manifest + spec)
    ├── L1_stool/  L2_bicycle/  L3_robot_arm/
```

## Requirements

- **Blender 4.5 LTS or 5.2**. Discovery order: `--blender` flag →
  `$PROCAGEN3D_BLENDER` → `blender` on PATH → `~/.cache/procagen3d/*/blender`.
- **Python 3.10+** for the driver. No pip packages — scripts are stdlib
  only; Blender stages use Blender's bundled Python.

## Install

Keep one checkout and symlink it into each runtime's skill directory so
they never drift apart:

```sh
# Claude Code
ln -s "$(pwd)" ~/.claude/skills/procagen3d
# Codex CLI
ln -s "$(pwd)" ~/.codex/skills/procagen3d
```

Then invoke naturally ("generate a GLB of a wheelbarrow with a rotating
wheel") or explicitly via `/procagen3d` in Claude Code. Any other runtime:
point the agent at `SKILL.md`.

## CLI (used by the agent, usable by hand)

```sh
python3 scripts/procagen3d.py lint program.py   # source safety + runtime import gate
python3 scripts/procagen3d.py build program.py --out out/ --form-diagnostics  # curved/mixed
python3 scripts/procagen3d.py fit out/ --spec out/fit_spec.json               # image-conditioned
python3 scripts/procagen3d.py check out/ --tier showcase --form auto
python3 scripts/procagen3d.py joints out/                   # articulation validation
python3 scripts/procagen3d.py score out/ --spec spec.yaml   # constraint scoring
python3 scripts/procagen3d.py guard old.py new.py           # repair doctrine guard
python3 scripts/procagen3d.py edit-gates base/ edited/ --target "Handle"
python3 scripts/procagen3d.py render out/ --engine eevee    # re-render (beauty pass)
```

Exit 0 = pass, 1 = failure with printed `[PROCAGEN3D:FAIL:*]` reasons.
Authoring programs can import selected helpers from `procagen3d_runtime`.
`build` freezes the tested runtime source into the retained `out/program.py`,
so the deliverable has no external module dependency. Direct
`bpy.ops.object.transform_apply` calls fail source validation unless all of
`location=`, `rotation=`, and `scale=` are explicit.
Image-conditioned runs also retain each used input at the output root as
`reference_01.<ext>`, `reference_02.<ext>`, etc., with provenance recorded in
`priors.md`. They also retain `fit_spec.json`, a hash-bound `fit_report.json`,
registered reference render/overlay, and scored masks. Curved/mixed runs add a
neutral clay `form_sheet.png` and use a shape-only probe before the full build.

## Design notes

Structural patterns adapted from prior agent projects:
[img2threejs](https://github.com/img2threejs/img2threejs) ("scripts do
enforcement, agent vision does judgment"; bounded correction loops; one
contact sheet per review) and
[opentopos](https://github.com/gaoypeng/opentopos) (failure-mode-first
reference docs with drop-in code; grep-able WARN tags; dual-runtime symlink
install). The curved-form route also adapts img2threejs's topology-before-
primitive and anti-cardboard lessons plus
[build-web-3d-models](https://github.com/giraffe-tree/build-web-3d-models)'
form-first loft/sweep/surface practice. The implementation is original Blender
Python rather than copied project code. Semantic perception remains agent
vision, while visible projection is enforced by the dependency-free registered
camera, mask, landmark, ratio, and layout gates in `references/image-fit.md`.
