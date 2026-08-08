---
name: procagen3d
description: Generate 3D assets as executable Blender Python programs compiled to GLB — named parts, assembly hierarchy, articulated joints with limits, curved/streamlined and irregular mecha form workflows, and machine-checkable dimensional constraints (ProcAgen3D, arXiv:2607.22738). Use for text-to-3D or image-to-3D object generation, articulated/jointed models, parametric GLB/glTF assets, Blender procedural modeling, and source-level local edits of previously generated ProcAgen3D assets.
---

# ProcAgen3D — code-native generation of programmable 3D assets

You are the model ℳθ of the ProcAgen3D system (arXiv:2607.22738): you write an
executable Blender Python **program**; headless Blender compiles it to a GLB.
The **program is the asset** — named parts, a real transform tree, joints with
limits, and dimensions as named constants. The GLB is a derivative artifact.
Never sculpt meshes by hand or download models. Never dump arbitrary vertex
clouds; compact semantic control arrays (profiles, section rings, spines,
surface grids) are constructive geometry and are required when primitives
cannot express the form.

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
`sheet.png`; curved/mixed also `form_sheet.png` and optional
`reference_match.png`), and when applicable `form_probe/`, `spec.yaml`, `joints_report.json`,
`score_report.json`. Image-conditioned assets also contain `priors.md` and an
unaltered copy of every reference image used, named `reference_01.<ext>`,
`reference_02.<ext>`, etc. in input order.

## The Loop

**0 — Intake.** Classify the request: text-only or image-conditioned; new
asset or edit of an existing ProcAgen3D asset (edit → see Local edits below).
Declare the **detail tier** (`quick | standard | showcase`, see
`references/detail.md`): image-conditioned replication and any
"detailed/realistic" ask default to **showcase**; standard and above MUST
read `references/detail.md` before Design. Declare the **form profile**
(`rectilinear | curved | mixed`): use `curved` for streamlined/compound
surfaces and `mixed` for biomechanical mecha or machinery combining sculpted
and block-built masses. Curved/mixed MUST read
`references/complex-forms.md`; do not classify by object category alone—a
hard-surface object may still be form-dominant. If the request states any
measurable requirement (counts, dimensions, symmetry, required joints),
author `spec.yaml` now — MUST read `references/constraints.md` first. If a
reference image is used, MUST read `references/image-analysis.md`. Before
analysis or code, create `<out>` and copy every used input image byte-for-byte
to `<out>/reference_01.<ext>`, `<out>/reference_02.<ext>`, etc. in the order
supplied, preserving each format/extension. Record the mapping from saved path
to original filename, URL, or attachment label in `<out>/priors.md`; then Read
the saved copies and complete the structured priors (showcase: including the
§7 detail inventory + identity features; curved/mixed: the form blueprint and
macro identity forms). If the runtime cannot expose the
original bytes, save the highest-resolution representation available and mark
that substitution in `priors.md`. Do not skip the copies or priors: together
they are the reproducible perception input and stage.

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
failures. For curved/mixed, add a structural-form table with
`part | role | topology | method | guide stations/outline | evidence`; tag
every macro/meso structural mesh with the same contract (`primary` for
identity/silhouette masses, `secondary` for supports and deliberately
assembled structure), and set
`root["procagen3d_form_profile"]`. A `continuous` mass may not route to a box,
deformed box, or straight extrusion.

**2 — Form gate (curved/mixed only).** Write `<out>/form_probe.py` with only
primary masses, ground/contact parts, joint-center markers, and negative-space
openings in one neutral material. Build and gate it before any detail work:

`procagen3d build <out>/form_probe.py --out <out>/form_probe --form-diagnostics`

`procagen3d check <out>/form_probe --tier quick --form <profile>`

Read both probe `renders/sheet.png` and `renders/form_sheet.png`; compare the
reference-matched view when emitted. Pass silhouette, cross-section/volume,
negative space, attachments, and continuity in all useful views. Bad body
family → rewrite its primary builder, not its dimensions or decoration. Allow
at most two probe corrections. If it still fails, stop/request better views
or report the limitation; never advance to detail. Otherwise transfer the
accepted form constants and builders into the full program. Details:
`references/complex-forms.md`.

**3 — Synthesize.** Write `<out>/<slug>.py`: constants → one
`build_<part>()` per part → `build()` assembling the transform tree, joints
via the canonical `add_joint` helper, materials by part meaning. The program
must be self-contained (runnable in bare Blender) and deterministic; it must
not render, export, or touch files/network — the harness does that. Choose
geometry by the form table: primitive CSG for assembled solids; profile
extrusion for intentional facets; loft/sweep/revolve/subdivision/grid for
continuous or shell forms; modifiers for edge treatment and instanced arrays
for repetition. **Size sanity before building**: reference-grade programs
median ~490 LOC; vehicles often run 100–340 parts. A showcase draft under
~300 LOC or under the tier's secondary-detail floor means the design table was
too coarse, but never fragment an accepted continuous surface to inflate the
count. Detail cannot be retrofitted through the repair budget.

**4 — Build.** Run `procagen3d build <out>/<slug>.py --out <out>`; add
`--form-diagnostics` for curved/mixed. On `PROCAGEN3D_BUILD_ERROR`, read the
traceback, fix the program, and rebuild. Persistent same-error after two
attempts → reconsider the approach instead of patching the same line again.

**5 — Deterministic gates.** Run
`procagen3d check <out> --tier <tier> --form <profile>`. FAILs are doctrine or
form-contract violations (unnamed parts, duplicate `.001` names, incompatible
continuous/box methods, empty meshes, broken joints) — fix them; they are
never acceptable residue. WARNs are advisory: read each one and either fix it
or carry a one-line reason. `LOW_DETAIL`, `FORM_SECTIONS`, and
`FORM_PRIMITIVES` are repair input at showcase and during the form probe;
`FORM_MACRO_COVERAGE` is the hard gate against a token loft hiding a box-built
body.

**6 — Inspect (your judgment).** Read `<out>/renders/sheet.png` — top row
`front | right | iso`, bottom row `left | back | top`. Curved/mixed MUST also
read `form_sheet.png`; image-conditioned runs with a camera contract MUST read
`reference_match.png`. Write one line per aspect before deciding:

- **shape** — silhouette and construction correct in every useful view (not
  just front); watch for parts that only look right from one angle;
- **form** (curved/mixed; hard floor) — named swells, pinches, taper/twist,
  cross-sections, negative spaces, and attachment transitions match the form
  blueprint; no slab collapse, box stacking, or ellipsoid collage;
- **scale** — proportions match the stated/derived dimensions;
- **part coverage** — every part from the design table is visibly present and
  distinct; nothing fused, floating, or missing;
- **detail** (standard+; hard floor) — band honestly: *toy* = raw primitives;
  *featured* = bevels, seams, arrays, contrasting materials; *designed* = reads
  as the referenced object. Showcase passes only at *designed*: check the
  priors detail inventory and identity features item by item.

Image-conditioned: compare against every saved reference and `priors.md`.
An admitted residual on an identity-bearing silhouette, compound transition,
or panel language keeps shape/form at FAIL; detail and materials cannot
compensate. Never judge from the build log alone.

**7 — Repair (≤ 3 full-program iterations).** For each failed verdict, name
the single highest-impact defect, its evidence view, and an explicit PRESERVE
list of passing dimensions/features/views. Decide whether the blueprint is
wrong (`refine-priors`) or its implementation is wrong (`refine-code`). Apply
a minimal source edit; if the representation family is wrong, rewrite only
that primary builder rather than adding cosmetic parts. Save the current
program as `<out>/program.iter<N>.py`, then run
`procagen3d guard <out>/program.iter<N>.py <out>/<slug>.py`. Use
`--allow-shrink` only for a declared representation rewrite and record why.
The guard MUST pass before rebuilding from step 4. Hard ceiling: **3 repair
iterations** (build-error fixes included). If exhausted, keep the best
successful intermediate and report honestly what remains wrong.

**8 — Articulation.** If the asset has joints: `procagen3d joints <out>`
FAILs (bad type, missing child, pivot off the moving part, rest-pose drift)
must be fixed. Sweep-collision WARNs need judgment: real interpenetration →
fix pivot/limits; intended contact (e.g. lid meeting rim) → accept and say
so. Limits are part of the design — declare plausible ranges, not ±360°
defaults. Details: `references/articulation.md`.

**9 — Score.** If a spec exists: `procagen3d score <out> --spec <spec>`
Failed constraints route back to repair (within the same budget of 3).
Report the scorer's table verbatim — never claim a constraint passes without
this output, and failures stay in the final report.

**10 — Deliver.** Final message: artifact paths, part/joint counts, the
constraint table if any, and named residual mismatches. "Approximate" where
you eyeballed; "verified" only where a gate ran. "This cannot be built
faithfully from this input" is a valid outcome — say it instead of faking.

## Gates (do not skip)

1. Image-conditioned: every reference image actually used is present at the
   output root as `reference_NN.<ext>` and mapped in `priors.md` before code.
2. Curved/mixed: tagged form probe passes `check --form <profile>` and both
   probe sheets are Read before full synthesis; resolve `FORM_*` warnings and
   add no detail before form pass.
3. Full `build` exit 0 before anything else proceeds.
4. `check --tier <tier> --form <profile>` exit 0 before visual inspection;
   form/detail WARNs at showcase are repair input, not acceptable residue.
5. `guard` pass between every full-program repair iteration — no exceptions.
6. `sheet.png` actually Read; curved/mixed also `form_sheet.png`, and use
   `reference_match.png` whenever emitted.
7. `joints` exit 0 whenever the design table declares a joint.
8. `score` output quoted verbatim whenever a spec exists.
9. Repair ceiling of 3 — on exhaustion, deliver best intermediate + honest
   residuals, never a silent extra loop.
10. Repairs preserve what passes: list the passing verdicts in the repair
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
