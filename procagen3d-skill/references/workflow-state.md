# Resumable workflow state

`.procagen3d/state.json` is the local authority for one active generation
workflow. It records the configuration, ordered checklist, evidence hashes,
repair count, and an append-only event history. Conversation memory is not a
substitute for it.

## Initialize once

Run from the workspace root before intake:

```sh
procagen3d next --init \
  --out procagen3d_out/<slug> \
  --program procagen3d_out/<slug>/<slug>.py \
  --tier standard \
  --form rectilinear
```

Add applicable conditions at initialization:

- `--reference <path>` for every image used; this inserts the registered
  `fit` stage;
- `--form curved|mixed` to insert the shape-probe build, check, and visual
  review;
- `--joints` to insert articulation validation;
- `--spec <spec.yaml>` to insert constraint scoring;
- `--max-repairs N` to change the default hard ceiling of three.

Initialization creates the output directory and refuses to overwrite an
existing state. The default authoring program is `<out>/<out-name>.py`.
Generated state is local and ignored by Git.

## Resume and advance

Run this at every start, resume, and transition:

```sh
procagen3d next
```

`next` always prints a concrete next command. For a deterministic stage, the
printed command is the only accepted pipeline action. When it exits
successfully, the CLI advances the state automatically and binds its required
outputs by SHA-256. A different build path, tier, form profile, render mode,
guard policy, or command is rejected while state is active.

Manual steps—intake, design, form review, synthesis, visual review, repair
editing, and delivery—print a completion command containing every required
evidence path. All those files must exist, and the command also requires a
concise verdict:

```sh
procagen3d next --done intake \
  --evidence procagen3d_out/<slug>/intake.md \
  --note "requirements and assumptions recorded"
```

Use extra repeated `--evidence` arguments for supporting files beyond the
required set. An image-conditioned intake requires every declared preserved
reference, `priors.md`, and `fit_spec.json`; synthesis and repair editing
require the configured authoring program; visual reviews require all printed
canonical sheets; delivery requires the retained `program.py` and `model.glb`.
For curved or mixed work, design also binds `form_probe.py`; editing that
source after a failed probe automatically reopens design, probe build, and
probe check instead of leaving the workflow stranded at review.

`procagen3d next --json` emits a machine-readable status payload, including
`next_command` and `required_evidence`. To keep a different active state, pass
`--state <path>` to `next` and pass the same global option before other
commands, or set `PROCAGEN3D_STATE`. An explicitly selected state that is
missing or not a regular file fails closed; only an absent default state keeps
legacy behavior.

## Freshness and failure behavior

Every completed step has at least one `{path, sha256, size, mtime_ns}` evidence
record. On each `next` or managed command, the driver checks the cheap file
metadata first and re-hashes only files whose metadata changed. A missing or
content-changed file reopens that step and every later step; stale renders,
fit reports, scenes, or programs therefore cannot remain accepted without
repeatedly hashing unchanged GLB and Blend files.

A failing deterministic command stays at the same step and records the failed
exit code. State tracking is opt-in: if no state file exists, all existing CLI
commands retain their previous behavior.

## Bounded repairs

When a full-program gate or visual review fails, start a repair with the
evidence and one highest-impact reason:

```sh
procagen3d next --repair \
  --evidence procagen3d_out/<slug>/renders/sheet.png \
  --reason "roof silhouette is too shallow; preserve footprint and openings"
```

For a declared representation rewrite, add `--allow-shrink`; use repeated
`--allow-drop <pattern>` only for parts the repair plan explicitly permits the
guard to remove. These exceptions are archived in state and included in the
exact guard command printed later.

The command copies the current authoring source to the next unused
`program.iterN.py`, archives the completed cycle in state, and resets only the
full-program suffix. The next ordered steps are `repair-edit`, `repair-guard`,
`source-lint`, `build`, conditional `fit`, `check`, visual review, and final
gates. Attempting to start another repair after the configured ceiling sets a
hard-stopped state and `next` exits 3.

The state file can be inspected, backed up, or diffed, but do not hand-edit it
to bypass a gate. Invalid schemas, duplicate/out-of-order steps, missing
evidence, and unsupported versions fail closed.
