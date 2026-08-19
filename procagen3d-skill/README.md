# procagen3d-skill

**Version:** 0.2.0 — see the repository [update log](../README.md#update-log).

An agent skill implementing **ProcAgen3D: Code-Native Generation of Programmable
3D Assets** ([arXiv:2607.22738](https://arxiv.org/abs/2607.22738)) for
Claude Code, Codex, and compatible agent runtimes.

The agent plays the paper's generator ℳθ: it writes an executable Blender
Python program (named parts, real transform tree, joints with limits,
dimensions as constants); headless Blender compiles it to a GLB. A
deterministic harness enforces the paper's pipeline — reconstruction plan/probe
→ build + canonical renders → local-silhouette/pose-aware registered fit → deterministic checks → agent-vision
inspection → guarded repair loop (≤3) →
articulation validation → constraint scoring — while the agent's judgment
does design and visual review.

## Layout

```
procagen3d-skill/
├── SKILL.md                  # entry point — the staged loop and gates
├── scripts/
│   ├── procagen3d.py         # driver + stdlib gates (fit/check/score/guard/edit-gates)
│   └── blender_stages.py     # bpy-side: build, render, fit, joints (runs inside Blender)
├── references/               # routed depth, read on demand from SKILL.md
│   ├── doctrine.md           # representation doctrine + canonical helpers
│   ├── image-analysis.md     # agent-vision perception stage
│   ├── reconstruction-planning.md # shape family, pose, complexity contract
│   ├── image-fit.md          # registered camera/local-mask/pose/layout contract
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

- **Blender 4.x or 5.x** (tested on 4.5 LTS and 5.2 LTS). Discovery order:
  `--blender` flag → `$PROCAGEN3D_BLENDER` → `blender` on PATH →
  `~/.cache/procagen3d/*/blender`. On macOS, Blender 5.x headless needs real
  Metal GPU access; an agent sandbox makes it SIGSEGV at startup.
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
Image-conditioned runs also retain each used input at the output root as
`reference_01.<ext>`, `reference_02.<ext>`, etc., with provenance recorded in
`priors.md`. They also retain `reconstruction_plan.json`, version-2
`fit_spec.json`, a hash-bound `fit_report.json`, registered reference
render/overlay, and scored masks. Image-conditioned runs use a neutral
reconstruction probe before the full build; curved/mixed runs also add
`form_sheet.png`.

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
camera, local silhouette, landmark, pose-chain, ratio, and layout gates in
`references/image-fit.md`.
