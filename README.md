# ProcAgen3D

ProcAgen3D is a standalone Python application that turns an **image plus an
offline-generated reference GLB** into two linked deliverables:

- editable, standalone Blender Python in `src/program.py`; and
- `artifacts/model.glb`, compiled by running that program in headless Blender.

The Blender program is the source of truth. The supplied GLB is used only as
measurement and visual evidence; generated programs are statically rejected if
they try to load it at runtime.

![ProcAgen3D teaser](assets/teaser.png)

## Install with uv

Python 3.11+ and Blender 4.5/5.x are required. The coding-agent CLI you intend
to use must already be authenticated.

```sh
uv sync --group dev
uv run procagen3d doctor
```

The repository includes `uv.lock` for reproducible environments. The runtime
uses Rich for terminal progress; the dev group additionally installs pytest.

## Generate an asset

This source checkout includes eight image/GLB pairs under `assets/3d_glb`.

```sh
uv run procagen3d make \
  assets/3d_glb/mystic_mouse_wanderer/reference_01.png \
  assets/3d_glb/mystic_mouse_wanderer/object_0.glb \
  --backend codex \
  --prompt "Build the complete character with named, editable parts"
```

Use `--backend grok` or `--backend cursor` to switch coding agents. Workspaces
are written to `outputs/<name>` by default. Use `--name` or `--output` to
change that location.

To build evidence without spending an agent invocation:

```sh
uv run procagen3d make IMAGE GLB --prepare-only --name my-asset
```

After adding `outputs/my-asset/src/plan.json` and `program.py`, compile the
current source in a clean build without invoking an LLM:

```sh
uv run procagen3d build outputs/my-asset
```

Resume an incomplete run or let the configured agent repair it:

```sh
uv run procagen3d run outputs/my-asset --max-repairs 2
```

Long operations now report their intermediate stages: reference inspection,
canonical rendering, each agent author/repair pass, the clean Blender build,
GLB re-import, and fidelity scoring. Interactive terminals use a Rich spinner;
redirected or CI logs receive explicit start and completion lines. Progress is
written to stderr. During an agent pass, sanitized provider milestones and a
30-second heartbeat show elapsed time, structured-event count, last CLI-output
age, and the current sizes of `plan.json` and `program.py`. Raw reasoning,
commands, command output, and provider JSON remain only in the workspace logs.
Add `--no-progress` to `make`, `run`, or `build` for a quiet invocation.

## Agent defaults

| Backend | Executable | Default model/mode |
| --- | --- | --- |
| Codex | `codex` | `gpt-5.6-sol`, reasoning effort `max` |
| Grok Build | `grok` | `grok-4.6`, reasoning effort `xhigh` |
| Cursor | `cursor-agent` | `cursor-grok-4.6-xhigh-fast` |

The installed Grok Build 1.0.5 CLI does not expose a separate Fast selector.
ProcAgen3D therefore uses the closest truthful configuration—Grok 4.6 at
Extra High—and `doctor` reports the limitation. Cursor exposes an exact Extra
High Fast model ID.

Each backend is a shell-free subprocess adapter. It records the prompt, JSONL
transcript, stderr, terminal result, usage, model, and modified-file list in
`trajectories/iter_XX`. After a candidate passes GLB validation and clean
re-import, its executed result is also retained as
`trajectories/iter_XX/model.glb`; later repairs cannot overwrite an earlier
iteration's GLB. Agents may change only `src/`.

## Pipeline

```text
image + offline GLB
        │
        ├─ copy + SHA-256 provenance
        ├─ pure-Python GLB container/accessor probe
        └─ isolated Blender reference probe
             ├─ canonical normalization (Z-up, grounded, longest side = 2)
             ├─ evaluated bounds and recomputed mesh geometry
             ├─ welded-component hypotheses and Z cross-sections
             └─ canonical renders + packed silhouette/RGB evidence
                         │
                         ▼
               Codex / Grok / Cursor CLI
                         │
                 plan.json + program.py
                         │
             AST source contract and clean build
                         │
        Blender calls build() → editable .blend → exported GLB
                         │
             re-import exact GLB + same-camera renders
                         │
        silhouette/RGB/bounds scoring → bounded source repair loop
```

The sample GLBs are deliberately handled as semantically weak evidence: most
contain one anonymous merged mesh, one material, no normals, no rig, and no
animation. Node names are not trusted as part labels. Blender computes the
evaluated geometry, while the coding agent infers meaningful construction from
the original image, canonical views, bounds, connected components, and sparse
cross-sections.

For character and hybrid subjects, the validated plan adds an anatomy-aware track:
posture, head-to-body and limb proportions, facial landmarks, hair/headwear,
clothing layers, attachments, held props, and character-relative left/right.
This release verifies the evaluated rest-pose GLB. A generated program may build
an armature, but the pipeline does not yet certify skin weights, animation clips,
IK, or walk cycles; those remain explicit limitations rather than implied output.

The generated `program.py` must define a synchronous, zero-argument `build()`.
It may use `bpy`, `bmesh`, `mathutils`, `math`, and `random`, but it may not read
files, import/reference the source GLB, use the network, spawn processes, load
Blender libraries, render, save, or export. The application performs those
side effects after running the source in a clean temporary directory.

### Trust boundary

The AST guard, disposable agent workspace, clean build directory, and stripped
Blender environment prevent common accidental violations; they are not a
security sandbox for deliberately hostile Python. A generated Blender program
has the authority of the local Blender process. Use authenticated coding-agent
CLIs that you trust. For untrusted generation, run `make --prepare-only`, review
`src/program.py`, and execute Blender inside your own container/OS sandbox.

## Workspace

```text
outputs/<name>/
├── manifest.json
├── inputs/
│   ├── reference.<image-extension>
│   └── reference.glb
├── evidence/
│   ├── glb_probe.json
│   ├── reference_scene.json
│   ├── camera_contract.json
│   └── reference_views/{front,back,left,right,top,iso}.png
├── src/
│   ├── plan.json
│   └── program.py
├── artifacts/
│   ├── scene.blend
│   ├── model.glb
│   ├── scene_report.json
│   ├── model_probe.json
│   ├── build_manifest.json
│   ├── comparison.json
│   └── renders/
├── trajectories/iter_XX/
└── run_report.json
```

`complete` means the compiled artifact reached `--min-score` (default 0.35).
`needs-review` still contains a valid program, BLEND, and GLB, but exhausted the
bounded repair budget below that fidelity floor. Build/static failures never
ship as valid artifacts. Commands return exit code 2 for `needs-review`, so CI
can distinguish it from a passing result while retaining the last matching
source/artifact pair.

Reference evidence and build artifacts are assembled in same-filesystem staging
directories and promoted as complete directory transactions. A failed probe or
repair therefore leaves the previous valid evidence/artifact set intact. The
candidate report and renders are produced from a fresh re-import of the exported
GLB, so unsupported pre-export Blender state cannot improve its fidelity score.

## Commands

```text
procagen3d make IMAGE GLB     create, author, compile, and verify a workspace
procagen3d run WORKSPACE      resume generation or bounded repair
procagen3d build WORKSPACE    compile existing source without invoking an LLM
procagen3d inspect WORKSPACE  print manifest, artifacts, and comparison report
procagen3d probe MODEL.glb    inspect GLB structure with the Python stdlib
procagen3d examples           list image/GLB pairs under a local sample root
procagen3d doctor             check uv, Blender, CLIs, and model defaults
```

Run `uv run procagen3d COMMAND --help` for all options.
`examples` defaults to `assets/3d_glb` relative to the current directory; when
running an installed wheel elsewhere, pass `--root /path/to/your/examples`.

## Development

```sh
uv sync --group dev
uv run pytest -q
PROCAGEN3D_RUN_BLENDER_TESTS=1 uv run pytest -q tests/test_blender_integration.py
```

Tests cover GLB parsing/accessor transforms, semantic-boundary detection,
generated-source policy, backend commands/parsers, workspace provenance, mask
and RGB metrics, finite scene validation, atomic host-pipeline behavior, and
portable build provenance. The opt-in Blender integration test saves the
editable scene, exports a self-contained GLB, starts a second factory process,
re-imports it, and renders all six canonical views. It also verifies that a
generated render handler cannot cross that process boundary and inflate score.

## Design lineage

The application adopts the code-as-truth workspace and stateless Blender ideas
from [OpenTopos](https://github.com/gaoypeng/opentopos), and the principle of
using a GLB as measured multi-view evidence from
[img2threejs](https://github.com/img2threejs/img2threejs). Neither project nor the
retired ProcAgen3D skill is a runtime dependency.
