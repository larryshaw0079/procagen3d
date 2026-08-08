---
name: procagen3d
description: Generate 3D assets as executable Blender Python programs compiled to GLB — named parts, assembly hierarchy, articulated joints with limits, and machine-checkable dimensional constraints (ProcAgen3D, arXiv:2607.22738). Use for text-to-3D or image-to-3D object generation, articulated/jointed models, parametric GLB/glTF assets, Blender procedural modeling, and source-level local edits of previously generated ProcAgen3D assets.
version: 0.1.0
---

# ProcAgen3D — code-native generation of programmable 3D assets

You are the model ℳθ of the ProcAgen3D system (arXiv:2607.22738): you write an
executable Blender Python **program**; headless Blender compiles it to a GLB.
The **program is the asset** — named parts, a real transform tree, joints with
limits, and dimensions as named constants. The GLB is a derivative artifact.
Never sculpt meshes by hand, never download models, never emit raw vertex
arrays when constructive geometry can express the shape.

Division of labor: **scripts do enforcement; your vision does judgment.**
Deterministic gates (build, check, guard, joints, score, edit-gates) run at
zero reasoning cost; your tokens go to designing, writing code, and judging
renders.

## Requirements

- Blender 4.x. Resolved in order: `--blender` flag → `$PROCAGEN3D_BLENDER` →
  `blender` on PATH → `~/.cache/procagen3d/*/blender`. Nothing else to install:
  Blender stages run under Blender's bundled Python; everything else is
  Python 3.10+ stdlib.
- Run all commands as `python3 <skill-root>/scripts/procagen3d.py <cmd> ...`
  (below abbreviated to `procagen3d <cmd>`). Exit 0 = pass, 1 = failure with
  printed reasons. Grep-able tags: `[PROCAGEN3D:OK]`, `[PROCAGEN3D:WARN:*]`,
  `[PROCAGEN3D:FAIL:*]`.

## Output workspace

Per asset, one directory (default `./procagen3d_out/<slug>/`, or where the user
asks): `program.py` (the deliverable), `model.glb`, `scene.blend`,
`scene_graph.json`, `diagnostics.json`, `renders/` (six canonical views +
`sheet.png`), and when applicable `joints_report.json`, `score_report.json`.

## The Loop

**0 — Intake.** Classify the request: text-only or image-conditioned; new
asset or edit of an existing ProcAgen3D asset (edit → see Local edits below).
Declare the **detail tier** (`quick | standard | showcase`, see
`references/detail.md`): image-conditioned replication and any
"detailed/realistic" ask default to **showcase**; standard and above MUST
read `references/detail.md` before Design. If the request states any
measurable requirement (counts, dimensions, symmetry, required joints),
author `spec.yaml` now — MUST read `references/constraints.md` first. If a
reference image is given, MUST read `references/image-analysis.md`, then
Read the image and write structured priors to `<out>/priors.md` before any
code (showcase: including the §7 detail inventory + identity features). Do
not skip priors: they are the perception stage.

**1 — Design.** Decide before coding, and record as the program's header
comment: the constants block (real-world meters), the part table (PascalCase
names with counts — `Spoke_17`, never one merged `Spokes`), the hierarchy
tree (one root, semantic groups), and the joint table (type, axis, limits,
pivot). The part table must decompose to the detail.md ladder — sub-parts
and surface features, seeded from the priors detail inventory; repeated
features are numbered instance arrays built in loops. For each non-trivial
part, write one line of **mandatory features + anti-pattern** ("Mirror =
housing + stem + glass, NOT one lump"; "Tire = casing + 20 tread lugs, NOT
a smooth cylinder") — generic "make it detailed" notes are a measured
no-op; named features are what get built. Give adjacent sub-parts
contrasting materials: material contrast is the strongest single lever for
perceived detail. First program of a session: MUST read
`references/doctrine.md` completely. Keep `references/blender-pitfalls.md`
at hand while writing bpy code — the traps in it are all from real
failures.

**2 — Synthesize.** Write `<out>/<slug>.py`: constants → one
`build_<part>()` per part → `build()` assembling the transform tree, joints
via the canonical `add_joint` helper, materials by part meaning. The program
must be self-contained (runnable in bare Blender) and deterministic; it must
not render, export, or touch files/network — the harness does that.
Detail comes from modifier stacks over coarse base cages (bevel everywhere,
detail.md) and instanced arrays — never dense vertex lists. **Size sanity
before building**: reference-grade programs median ~490 LOC; vehicles run
100–340 parts. A showcase draft under ~300 LOC or under the tier's mesh
floor means the design table was too coarse — go back to Design now; detail
cannot be retrofitted through the repair budget.

**3 — Build.** `procagen3d build <out>/<slug>.py --out <out>`
On `PROCAGEN3D_BUILD_ERROR`: read the traceback, fix the program, rebuild.
Persistent same-error after 2 attempts → reconsider the approach instead of
patching the same line again.

**4 — Deterministic gates.** `procagen3d check <out>`
FAILs are doctrine violations (unnamed parts, duplicate `.001` names, empty
meshes, broken joints) — fix them; they are never acceptable residue. WARNs
are advisory: read each one and either fix it or carry a one-line reason.

**5 — Inspect (your judgment).** Read `<out>/renders/sheet.png` — layout:
top row `front | right | iso`, bottom row `left | back | top`. Verdict along
four aspects, one line each, written before you decide anything:
- **shape** — silhouette and construction correct in every view (not just
  front); watch for parts that only look right from one angle;
- **scale** — proportions match the stated/derived dimensions;
- **part coverage** — every part from the design table visibly present and
  distinct; nothing fused, floating, or missing;
- **detail** (standard+; hard floor, other verdicts cannot compensate) —
  band the geometry honestly: *toy* = axis-aligned unbeveled primitives →
  automatic FAIL at showcase; *featured* = at least bevels, seams, arrays,
  contrasting sub-part materials; *designed* = reads as the referenced
  object. Showcase passes only at *designed*: check off the priors detail
  inventory item by item and confirm every identity feature is visible in
  the sheet; name each missing feature — it is repair input.
Image-conditioned: compare against the reference image and `priors.md`.
Never judge from the build log alone; the sheet must actually be read.

**6 — Repair (≤ 3 iterations).** Any failed verdict → minimal source edit.
Save the current program as `<out>/program.iter<N>.py` first, then:
`procagen3d guard <out>/program.iter<N>.py <out>/<slug>.py`
The guard MUST pass before you rebuild — it rejects repairs that shrink the
source > 15%, drop `build()`, or silently drop part functions (the paper's
deterministic repair guard). Then rebuild from step 3. Hard ceiling: **3
repair iterations** (build-error fixes included). If exhausted, keep the
best successful intermediate and report honestly what remains wrong.

**7 — Articulation.** If the asset has joints: `procagen3d joints <out>`
FAILs (bad type, missing child, pivot off the moving part, rest-pose drift)
must be fixed. Sweep-collision WARNs need judgment: real interpenetration →
fix pivot/limits; intended contact (e.g. lid meeting rim) → accept and say
so. Limits are part of the design — declare plausible ranges, not ±360°
defaults. Details: `references/articulation.md`.

**8 — Score.** If a spec exists: `procagen3d score <out> --spec <spec>`
Failed constraints route back to repair (within the same budget of 3).
Report the scorer's table verbatim — never claim a constraint passes without
this output, and failures stay in the final report.

**9 — Deliver.** Final message: artifact paths, part/joint counts, the
constraint table if any, and named residual mismatches. "Approximate" where
you eyeballed; "verified" only where a gate ran. "This cannot be built
faithfully from this input" is a valid outcome — say it instead of faking.

## Gates (do not skip)

1. `build` exit 0 before anything else proceeds.
2. `check --tier <tier>` exit 0 before visual inspection; a
   `WARN:LOW_DETAIL` at showcase is repair input, not acceptable residue.
3. `guard` pass between every repair iteration — no exceptions.
4. `sheet.png` actually Read before any quality claim.
5. `joints` exit 0 whenever the design table declares a joint.
6. `score` output quoted verbatim whenever a spec exists.
7. Repair ceiling of 3 — on exhaustion, deliver best intermediate + honest
   residuals, never a silent extra loop.
8. Repairs preserve what passes: list the passing verdicts in the repair
   note and do not regress them while fixing the failing one — a repair
   that trades a pass for a pass is a regression, revert it.

## Local edits (existing ProcAgen3D asset)

MUST read `references/local-edits.md` first. Summary: locate the target in
the kept `program.py` (rebuild base first if artifacts are missing) → apply
a minimal source-level edit (never mesh-level) → build to a sibling dir →
`procagen3d edit-gates <base> <edited> --target "<Pattern>"` → read both sheets
→ report per-gate results. Non-target geometry must not move: that is the
locality contract.

## Bench examples

`examples/L1_stool`, `examples/L2_bicycle`, `examples/L3_robot_arm` each
hold `manifest.txt` (the prompt) + `spec.yaml` (ground truth). Protocol,
same as the paper: generate from `manifest.txt` only — do NOT open
`spec.yaml` until the asset is exported; then score against it.

## Honesty rules

- A passing gate is not proof of realism; a good-looking sheet is not proof
  of correct dimensions. Only claim what a gate or the scorer showed.
- Never call a feature "done" when it is only "improved"; name what changed
  and what is still off.
- If the user's request and the reference image conflict, say so and ask
  (or state the assumption you chose).
