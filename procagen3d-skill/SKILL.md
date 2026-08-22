---
name: procagen3d
description: Generate 3D assets as executable Blender Python programs compiled to GLB — named parts, assembly hierarchy, articulated joints, pose-aware image reconstruction, evidence-backed primitive/form priors, complexity-adaptive detail, and machine-checkable constraints (ProcAgen3D, arXiv:2607.22738). Use for text-to-3D or image-to-3D object generation, articulated/jointed models, parametric GLB/glTF assets, Blender procedural modeling, and source-level local edits of previously generated ProcAgen3D assets.
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

- Blender 4.x or 5.x (tested on 4.5 LTS and 5.2 LTS). Resolved in order:
  `--blender` flag → `$PROCAGEN3D_BLENDER` → `blender` on PATH →
  `~/.cache/procagen3d/*/blender`. Nothing else to install: Blender stages
  run under Blender's bundled Python; everything else is Python 3.10+ stdlib.
  **macOS + Blender 5.x:** every `build`/`render`/`fit`/`joints` invocation
  must run unsandboxed with Metal GPU access. A sandboxed launch SIGSEGVs in
  `supports_barycentric_whitelist` during `WM_init` — Python never starts.
  That is a Blender Metal-detection bug, not a 4.x-vs-5.x bpy API mismatch.
- Run all commands as `python3 <skill-root>/scripts/procagen3d.py <cmd> ...`
  (below abbreviated to `procagen3d <cmd>`). Exit 0 = pass, 1 = failure with
  printed reasons. Grep-able tags: `[PROCAGEN3D:OK]`, `[PROCAGEN3D:WARN:*]`,
  `[PROCAGEN3D:FAIL:*]`.

## Output workspace

Per asset, one directory (default `./procagen3d_out/<slug>/`, or where the user
asks): `program.py` (the deliverable), `model.glb`, `scene.blend`,
`scene_graph.json` (per-mesh shape signatures and canonical-view silhouette
areas), `diagnostics.json`, `renders/` (six canonical views +
`sheet.png`; curved/mixed also `form_sheet.png`), and when applicable
`form_probe/`, `spec.yaml`, `joints_report.json`,
`score_report.json`, and `limitations.md` whenever a single-view reconstruction
is delivered as approximate. Image-conditioned assets also contain `priors.md`,
`reconstruction_plan.json`, version-2 `fit_spec.json`, `fit_report.json`,
registered `reference_match.png` and `reference_overlay.png`, scored
reference/render masks, and an
unaltered copy of every reference image used, named `reference_01.<ext>`,
`reference_02.<ext>`, etc. in input order. Character-route assets additionally
contain `character_plan.json`.

## The Loop

**0 — Intake.** Classify the request: text-only or image-conditioned; new
asset or edit of an existing ProcAgen3D asset (edit → see Local edits below).
Classify the **subject domain** as `object | scene | character`. Humans,
humanoids, anthropomorphic figures, and organic creatures use `character`;
rigid robots/mecha remain `object` unless visible anatomy genuinely deforms.
Character work MUST read `references/character-reconstruction.md`, set
`reconstruction_plan.subject_domain = "character"`, and author
`character_plan.json` before code. The route is opt-in; object/scene behavior
and thresholds do not change.
Declare the **detail tier** (`quick | standard | showcase`, see
`references/detail.md`): image-conditioned replication and any
"detailed/realistic" ask default to **showcase**; standard and above MUST
read `references/detail.md` before Design. Declare the **form profile**
(`rectilinear | curved | mixed`): use `curved` for streamlined/compound
surfaces and `mixed` only when evidence shows both continuous/shell and
assembled masses. Mixed is not a continuous-geometry quota. Curved/mixed MUST
read `references/complex-forms.md`; do not classify by object category alone—a
hard-surface object may still be form-dominant. If the request states any
measurable requirement (counts, dimensions, symmetry, required joints),
author `spec.yaml` now — MUST read `references/constraints.md` first. If a
reference image is used, MUST read `references/image-analysis.md`,
`references/reconstruction-planning.md`, and `references/image-fit.md`. Before
analysis or code, create `<out>` and copy every used input image byte-for-byte
to `<out>/reference_01.<ext>`, `<out>/reference_02.<ext>`, etc. in the order
supplied, preserving each format/extension. Record the mapping from saved path
to original filename, URL, or attachment label in `<out>/priors.md`; then Read
the saved copies and complete the structured priors (showcase: including the
§7 detail inventory + identity features; curved/mixed: the form blueprint and
macro identity forms). Solve canonical object frame, registered camera, rigid
pose, and every visible articulated chain before dimensions. Author
`<out>/reconstruction_plan.json` with a locked `camera_solve` (at least two
compared viewpoints and why the loser lost), candidate-tested primitive
families, rejected alternatives, perceptual complexity, and required feature
groups. Then author version-2 `<out>/fit_spec.json` with the locked camera,
whole-frame mask, one local silhouette region per two primary masses (minimum
three), geometry-bound landmarks/ratios, frame axes, and pose chains; add
per-instance boxes/relations for multi-object references. **Do not write
threshold values**: pass marks live in the harness and a spec that asks for a
looser one is clamped and failed. Version 2 and the reconstruction plan are
both mandatory for image-conditioned work; version 1 no longer validates. If the runtime cannot expose the original bytes, save the
highest-resolution representation available and mark
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
perceived detail. For image-conditioned work, the part table must implement
every required `detail_features` pattern/count; use the plan's adaptive
complexity class, not object category, LOC, or repeated-instance totals. First
program of a session: MUST read
`references/doctrine.md` completely. Keep `references/blender-pitfalls.md`
at hand while writing bpy code — the traps in it are all from real
failures. Character plans decompose geometry by deformation layer
(`core_volume`, `deformable_appendage`, `cross_joint_shell`,
`rigid_attachment`, `surface_detail`) and explicitly plan neck/shoulder/elbow/
hip/knee transitions; the body envelope is not decomposed by the generic
independent-part ladder. For every image-conditioned macro/meso mass, add a table with
`part | role | topology | method | shape family | guide/outline | evidence |
rejected alternative`; choose the simplest family supported by visible
planar/edge/section evidence. Rounded edges do not make a box/prism an
ellipsoid or loft. Tag every structural mesh with the same contract (`primary`
for identity/silhouette masses, `secondary` for supports and deliberately
assembled structure), and set
`root["procagen3d_form_profile"]`, including `procagen3d_shape_family`. A
`continuous` mass may not route to a box, deformed box, or straight extrusion;
a planned box/prism may not route to a loft/ellipsoid merely to soften edges.
Image-conditioned programs must also create semantic empty markers for
identity landmarks whose part-bbox anchors are insufficient; marker names must
match `fit_spec.json`, derive from the same named dimension constants as the
geometry, and remain parented to the relevant assembly.

**2 — Reconstruction probe.** Required for every image-conditioned asset and
every curved/mixed target. Write `<out>/form_probe.py` with every structural
mass in the shape-prior plan, ground/contact parts, joint/pose markers, and
negative-space openings in one neutral material; omit detail. Build and gate it
before detail:

`procagen3d build <out>/form_probe.py --out <out>/form_probe --form-diagnostics`

Image-conditioned: solve the camera before scoring anything against it —

`procagen3d solve-camera <out>/form_probe --spec <out>/fit_spec.json [--solve-root]`

resects it from six or more image-read landmarks and writes
`camera_solution.json`; paste the result into `camera_solve.camera` and
`fit_spec.camera`. Never hand-tune the viewpoint. Read the residual: a low RMS
means the camera was the fault and is now fixed; an RMS that stays high under
every viewpoint means the proportions or per-limb pose are wrong and no camera
will rescue them — the worst individual landmarks name the parts. `--solve-root`
estimates a rigid lean, but root pitch and camera elevation are
gauge-equivalent, so treat an unchanged residual as evidence that the subject is
*not* leaning.

`procagen3d fit <out>/form_probe --spec <out>/fit_spec.json`

`procagen3d check <out>/form_probe --tier quick --form <profile> [--subject character]`

Read the probe sheet; curved/mixed also read `form_sheet.png`; image-conditioned
also read the registered render/overlay/report. Pass primitive family, local
silhouette, camera/frame, rigid/articulated pose, cross-section, negative space,
and attachments in all useful views. Wrong family → revise the prior and rewrite
that builder in either direction; wrong pose → repair camera/root/joint
transforms, not dimensions. Probe correction budget scales with the plan's
complexity class: **2** for simple/moderate, **3** for complex, **4** for
extreme.

Characters use the ordered probe in `references/character-reconstruction.md`:
landmark/chain scaffold → coherent body envelope and all visible joint
transitions → major hair/clothing/equipment shells. Do not add facial
microdetail, rivets, stitches, or strand arrays until those three layers pass.

On exhaustion, what happens next depends on how much evidence the input
actually contained:

- **Two or more reference views.** Stop. The evidence was sufficient, so a
  failing probe means the reconstruction is wrong. Report the limitation and
  never decorate a rejected reconstruction.
- **One reference view, near miss.** Depth is unobservable from one image, so
  some residual is the input's fault rather than the model's. Continue to
  synthesis on the best probe, but the asset is **approximate**: write
  `<out>/limitations.md` naming every failing gate with its measured value,
  carry those residuals verbatim into the final report, and ask for the
  specific views that would resolve them.
- **One reference view, wide miss.** Stop. The approximation escape is bounded:
  at most **25% of gates** may fail, no IoU gate by more than **0.08**, and no
  error gate by more than **2×** its ceiling. Past that the model is not
  approximate, it is wrong, and writing the failures down does not change that.

`check` enforces the bounds and the documentation, and never accepts a
`threshold_policy` or `landmark_provenance` failure at all — those are
integrity faults, not evidence limits.

Delivering an approximate model with an honest account of what is unverified is
more useful than delivering nothing. Claiming it is faithful is not.

**`complex` and `extreme` plans probe one assembly at a time.** Split the object
into five to nine named assemblies, give each its own silhouette region, and
register them in identity order. One aggregate score over a thirty-mass object
cannot tell you which mass is wrong; a per-region row can. Details:
`references/reconstruction-planning.md` §6 and `references/complex-forms.md`.

**3 — Synthesize.** Write `<out>/<slug>.py`: constants → one
`build_<part>()` per part → `build()` assembling the transform tree, joints
via the canonical `add_joint` helper, materials by part meaning. The program
must be self-contained (runnable in bare Blender) and deterministic; it must
not render, export, or touch files/network — the harness does that. Choose
geometry by the form table: primitive CSG for assembled solids; profile
extrusion for intentional facets; loft/sweep/revolve/subdivision/grid for
continuous or shell forms; modifiers for edge treatment and instanced arrays
for repetition. Implement feature groups region by region in priority order:
identity/structural → secondary → micro → inferred. Before building, verify
every required plan pattern/count exists in source, every declared occupied
object region has visible structure/detail, and the adaptive complexity floor
is plausible. LOC and raw part count are not quality targets: repeated
spokes can make a simple object large, while a complex car/mecha can remain
underdesigned. Never fragment an accepted surface to inflate counts. Detail
cannot be retrofitted through the repair budget.

On the character route, build the anatomy scaffold and `core_volume` first;
then deformable appendages and joint-crossing shells; then single-link rigid
attachments and surface detail. Tag the root with
`procagen3d_subject_domain="character"` and
`procagen3d_character_routine="organic-v1"`, and tag every planned character
mesh with its `procagen3d_character_layer` and
`procagen3d_character_construction`; tag transitions exactly as
`character_plan.json` declares.

**4 — Build.** Run `procagen3d build <out>/<slug>.py --out <out>`; add
`--form-diagnostics` for curved/mixed. On `PROCAGEN3D_BUILD_ERROR`, read the
traceback, fix the program, and rebuild. Persistent same-error after two
attempts → reconsider the approach instead of patching the same line again.

**4a — Registered image fit (image-conditioned).** Run
`procagen3d fit <out> --spec <out>/fit_spec.json`. It renders at the reference's
exact resolution/aspect and the locked camera, then scores mask IoU,
bbox/centroid alignment, local silhouettes, geometry-bound landmarks/ratios,
frame axes, pose chains, and instance relations. Every gate MUST pass. Read the
overlay, masks, and report; fix camera/frame → rigid pose → articulated pose →
shape/dimensions in that order, locking each passing layer. Never proceed on the
legacy auto-centered preview alone.

Thresholds are the harness's, not yours. Mask IoU floors run 0.88/0.84/0.80/0.76
for simple/moderate/complex/extreme, then move with the evidence: **−0.08 for a
single view**, unchanged at two, **+0.03 at three or more**. One image cannot
determine depth, so demanding multi-view registration from it asks for
something the input does not contain; more views make the problem better posed
and raise the bar. A further allowance of up to 0.15 applies to genuinely wiry
subjects, measured by eroding the reference mask. The `threshold_policy` gate
fails and lists every value the spec tried to loosen.

Targets are the image's, not yours either. Every `reference_uv` must be read
off the saved reference; deriving one by projecting your own model makes that
landmark and every ratio, frame axis, and pose chain built on it a comparison
of the model with itself, which passes unconditionally. `landmark_provenance`
fails when the median landmark error drops below 0.001, because estimating a
point from an image is worth about a pixel and cannot be that good. **And note what this stage cannot see** — every metric here
is one camera's image plane, so a passing fit says "registers from this view",
never "is correct in 3D". Depth is gated in step 5.

**5 — Deterministic gates.** Run
`procagen3d check <out> --tier <tier> --form <profile> [--subject character]`.
FAILs are doctrine or
form/reconstruction violations (unnamed parts, duplicate `.001` names, shape
family/method mismatch, missing plan features or occupied regions, empty
meshes, broken joints) — fix them; they are never acceptable residue.
Version-2 showcase detail floors and feature/region coverage are hard gates.
WARNs remain advisory only where explicitly printed.
For image-conditioned outputs, `check` also fails when fit evidence is missing,
failed, or stale relative to the reference, fit spec, or scene graph.
Character-route checks additionally enforce the anatomy scaffold, deformation
layers, paired face regions, major joint-transition hosts, and requested
armature. Character mesh floors are intentionally lower than object floors;
coherent topology, not hundreds of separate beads, is the completion signal.

Six of these gates measure the built geometry rather than cross-checking one
declaration against another, and they are the ones that catch what renders
reveal:

- `SHAPE_FAMILY_MEASURED` — declared family versus measured `fill_ratio` and
  `planar_area_fraction`. An ellipsoid cannot ship under a `box` tag, and a
  flat-sided block cannot ship as a `loft`. Bevelling a box leaves it at 82%
  planar; an ellipsoid is at 8%. Rounded corners are not curvature.
- `PRIMARY_DEPTH` — a primary mass whose thinnest dimension is under 8% of its
  longest is a cut-out, not a volume.
- `VIEW_COLLAPSE` — the canonical orthographic views share one scale, so their
  silhouette areas are comparable. A bas-relief that fits one camera and
  nothing else shows up here and nowhere else.
- `INSTANCE_CONGRUENCE` — members of a `Name_NN` array must share
  rotation-invariant size unless the plan says why they differ.
- `REGION_DENSITY` — every region the plan calls occupied must actually contain
  meshes; naming a cube `Vent_03` no longer counts as a vent.
- `FORM_PROMISED_CURVATURE` — masses declared `continuous`/`shell` must be
  measurably curved.
- `SCENE_INTERPENETRATION` — objects the fit spec declares as separate
  instances may touch but must not occupy the same space. Depth is measured by
  containment, so a lamp standing *on* a table reads ~0% while a lamp sunk into
  a sofa reads its true burial depth. Scoped to declared instances, because the
  sub-assemblies of one articulated body are supposed to overlap at every joint.
- `CAMERA_SOLVE` — a resection that did not converge is a verdict, not a
  formality: no viewpoint explains your landmarks, so proportions or instance
  layout are wrong. Re-solve; never carry on with the rejected seed.
- `DETACHED_PARTS` — an assembled object is one connected solid. A part that
  touches nothing is floating, and from the one camera you fitted it can look
  perfectly attached. Build the piece that joins them (a neck, a stem, a
  mount); list genuinely separate pieces in `detached_groups`.
- `SYMMETRY` — left/right pairs must be mirror images in the canonical frame.
  One view cannot see how far forward a shoulder sits, but it can insist the
  two shoulders agree, which is where per-part depth discipline comes from.
  Author both sides from one builder with the side as a sign; never type two
  independent sets of coordinates.
- `RIGID_AXIS` — every long assembly whose parts point more than 8° apart must
  be classified in the plan as rigid (then held collinear) or articulated (then
  given the joint that bends). Undeclared is a failure; a rifle, mast, or axle
  has no true articulated answer. Author a rigid assembly as one origin plus
  one direction and derive every station along it — placing each endpoint
  separately to match the image, with depth free, is how a straight weapon
  comes out bent.

Nothing here asks for curved mass. `FORM_PROFILE_EVIDENCE` fires only when a
`curved` profile shows almost no measurable curvature, and its remedy is to
declare `rectilinear` or `mixed`. Faceted armour, receivers, and camera bodies
are boxes; keep them boxes and bevel the edges.

**6 — Inspect (your judgment).** Read `<out>/renders/sheet.png` — top row
`front | right | iso`, bottom row `left | back | top`. Curved/mixed MUST also
read `form_sheet.png`; image-conditioned runs with a camera contract MUST read
`reference_match.png`, `reference_overlay.png`, and `fit_report.json`. Write one
line per aspect before deciding:

- **shape** — silhouette and construction correct in every useful view (not
  just front); watch for parts that only look right from one angle. Look at
  the side and top tiles specifically and ask whether the object has depth;
  a model tuned against one reference camera collapses there first;
- **family** (image-conditioned; hard floor) — every planned box/prism retains
  planar fields/constant sections and every planned curved mass has evidenced
  curvature; bevels are not ellipsoids and lofts are not default blockouts;
- **form** (curved/mixed; hard floor) — named swells, pinches, taper/twist,
  cross-sections, negative spaces, and attachment transitions match the form
  blueprint; no wrong-family collage;
- **pose** (image-conditioned; hard floor) — camera-relative frame, root lean,
  joint chains, segment directions, bends, contacts, and overlaps match the
  reference without deforming part ratios;
- **scale** — proportions match the stated/derived dimensions;
- **part coverage** — every part from the design table is visibly present and
  distinct; nothing fused, floating, or missing;
- **detail** (standard+; hard floor) — band honestly: *toy* = raw primitives;
  *featured* = bevels, seams, arrays, contrasting materials; *designed* = reads
  as the referenced object. Showcase passes only at *designed*: check the
  plan's required feature groups region by region and identity features item by
  item; dense repetition cannot compensate for an empty complex region.

Image-conditioned: compare against every saved reference and `priors.md`.
An admitted residual on an identity-bearing silhouette, compound transition,
or panel language keeps shape/form at FAIL; detail and materials cannot
compensate. Never judge from the build log alone.

**7 — Repair (budget scales with complexity).** For each failed verdict, name
the single highest-impact defect, its evidence view, and an explicit PRESERVE
list of passing dimensions/features/views. Decide whether the blueprint is
wrong (`refine-priors`: camera/pose/family/complexity) or its implementation is
wrong (`refine-code`). Apply a minimal source edit; if the representation
family is wrong, rewrite only
that primary builder rather than adding cosmetic parts. Save the current
program as `<out>/program.iter<N>.py`, then run
`procagen3d guard <out>/program.iter<N>.py <out>/<slug>.py`. Use
`--allow-shrink` only for a declared representation rewrite and record why.
The guard MUST pass before rebuilding from step 4. Ceiling by complexity class
(build-error fixes included): **3** for text-only, `simple`, or `moderate`
work, **6** for `complex`, **8** for `extreme`. A car, mecha, or character
legitimately needs more passes than a stool; what it never gets is a lower bar.
If exhausted, keep the best successful intermediate and report honestly what
remains wrong.
Image-conditioned repairs must rerun `fit` after every rebuild; changes to
shape/feature plans must also rerun `check`. Older evidence is invalidated by
the new scene graph/spec.

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
you eyeballed; "verified" only where a gate ran. For a single-view
reconstruction, say so, quote `limitations.md`, and name the specific extra
views that would resolve each residual — "a side or rear view would fix the
shield depth" is actionable; "please provide better images" is not. "This
cannot be built faithfully from this input" is a valid outcome, but on a single
view prefer an approximate model plus an honest account of what is unverified:
that is more useful than nothing, and it is what the input supports.

## Gates (do not skip)

1. Image-conditioned: every reference image actually used is present at the
   output root as `reference_NN.<ext>`, mapped in `priors.md`; version-2
   `fit_spec.json` and `reconstruction_plan.json` (with a locked `camera_solve`)
   exist before code. Version 1 is rejected.
2. Image-conditioned or curved/mixed: reconstruction probe passes `check`; an
   image-conditioned probe also passes registered `fit`. Read its canonical and
   registered evidence before synthesis; add no detail before family/pose/form
   pass.
3. Character route: `character_plan.json` exists before code; landmark-chain,
   body-envelope, joint-transition, face-region, and rig contracts pass
   `check --subject character`. Never substitute generic region mesh density.
4. Full `build` exit 0 before anything else proceeds.
5. Image-conditioned: registered `fit` exit 0 after every full build, including
   local silhouette and pose gates; Read its overlay, masks, and report. On a
   single-view input a shortfall may instead be carried as an approximate
   delivery, but only with `limitations.md` naming every failing gate — never
   for a `threshold_policy` or `landmark_provenance` failure.
6. `check --tier <tier> --form <profile>` exit 0 before visual inspection;
   shape-prior, feature-coverage, adaptive showcase detail, and form failures
   are repair input, never acceptable residue.
7. `guard` pass between every full-program repair iteration — no exceptions.
8. `sheet.png` actually Read; curved/mixed also `form_sheet.png`, and use
   `reference_match.png` whenever emitted.
9. `joints` exit 0 whenever the design table declares a joint.
10. `score` output quoted verbatim whenever a spec exists.
11. Repair ceiling of 3 / 6 / 8 by complexity — on exhaustion, deliver best
   intermediate + honest residuals, never a silent extra loop.
12. Repairs preserve what passes: list the passing verdicts in the repair
   note and do not regress them while fixing the failing one — a repair
   that trades a pass for a pass is a regression, revert it.
13. Never author a fit threshold to make a gate pass. Thresholds belong to the
   harness; the spec may only tighten them. Loosening is itself a failure.

## Local edits (existing ProcAgen3D asset)

MUST read `references/local-edits.md` first. Summary: locate the target in
the kept `program.py` (rebuild base first if artifacts are missing) → apply
a minimal source-level edit (never mesh-level) → build to a sibling dir →
`procagen3d edit-gates <base> <edited> --target "<Pattern>"` → read both sheets
→ for image-conditioned bases retain the reference/plan/spec and rerun fit +
check → report per-gate results. Non-target geometry must not move: that is the
locality contract. Global detail remediation is regeneration, not a local edit.

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
