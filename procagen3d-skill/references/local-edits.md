# Local edits — source-level editing of an existing ProcAgen3D asset

Because the program is the asset, edits happen in the program text — never
in the mesh, never by post-processing the GLB. The contract is **locality**:
the target changes; every non-target part keeps its exact geometry,
transform, and place in the hierarchy.

## Protocol

1. **Base.** Locate the asset's directory with its kept `program.py`. If
   artifacts are stale or missing, rebuild the base first:
   `procagen3d build <base>/program.py --out <base>` — gates must pass before
   editing on top. If the base is image-conditioned, retain its references,
   `priors.md`, `reconstruction_plan.json`, and `fit_spec.json`; a local edit
   does not erase the reference-fidelity contract.
2. **Classify the instruction.** *Additive* (new part: "add a bell",
   "mount a rear rack") or *modify-existing* ("make the seat wider",
   "recolor the doors"). This picks the edit-gate mode and where the edit
   lands (new `build_<part>()` + one assembly line, vs. constant/builder
   change).
3. **Edit at source level, minimally.**
   - Dimension change → change the named constant, nothing else.
   - Shape/color change → edit inside that part's `build_<part>()` or its
     material definition.
   - Additive → new builder function + `reparent_keep_world` line in
     `build()`, attached to the semantically correct parent (a bell mounts
     on `Handlebar`, not on the root).
   - Do not reorder functions, rename unrelated parts, reformat untouched
     code, or "improve" things not asked for — every collateral diff line
     risks the locality gate and reviewer trust.
4. **Build to a sibling dir**, keeping the base intact:
   `cp <base>/program.py <edited>.py` → apply the edit →
   `procagen3d build <edited>.py --out <base>_edit1`
5. **Deterministic gates.**
   `procagen3d edit-gates <base> <base>_edit1 --target "<Pattern>"`
   (`--mode add|modify` if auto-detection guesses wrong; `--tol` defaults
   to 0.1 mm.) For an image-conditioned base, copy the retained perception
   files to the sibling, rerun `fit` and `check`, and read the new overlay. A
   geometry-local change can still break silhouette, pose, shape-family, or
   required-feature gates.
6. **Visual check.** Read both `renders/sheet.png` files; confirm the edit
   reads correctly *and* nothing else visibly changed. A shared-material
   recolor can pass geometric gates while being wrong (or vice versa) —
   the sheets are the semantic check the gates cannot do.
7. **Report** the five gate verdicts, the diff summary, and both sheet
   judgments. The edited directory becomes the new base for further edits.

## Gate semantics

| gate | FAIL means |
|------|-----------|
| artifact_validity | edited build produced no `model.glb` |
| target_addressability | the `--target` glob matches nothing (base for modify, edited for add) — wrong pattern or the part was never separately named |
| source_and_glb_change | program text unchanged, or target geometry identical — the "edit" didn't land |
| hierarchy_preservation | a non-target node vanished or was re-parented |
| non_target_locality | a non-target mesh moved/changed topology beyond tolerance (offenders listed) |

`hierarchy_preservation` tolerates children of the *target* moving with it
(that's the subtree working as designed). Material-only edits legitimately
show `target_geometry_changed=False`; if the source changed and the sheets
confirm the recolor, report that gate as N/A-material with the sheet as
evidence.

## Failure handling

A failed gate routes back to step 3 with a smaller diff — the same ≤3
iteration budget as generation. If the instruction is genuinely non-local
("make everything 20% bigger", "turn the chair into a bench"), say so and
treat it as regeneration with the old program as the starting point, not as
a local edit; don't fight the locality gate. Likewise, “make this complex
car/mecha substantially more detailed” is a reconstruction-plan revision
across many regions, not a neck/part-local edit.
