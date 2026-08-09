# ProcAgen3D

**Version:** 0.0.1

Code-native generation of programmable 3D assets, implemented as an agent
skill for Claude Code, Codex, and compatible runtimes.

![ProcAgen3D teaser](assets/teaser.png)

Instead of sculpting meshes or emitting vertex soup, the agent writes an
executable **Blender Python program** — named parts, a real transform tree,
articulated joints with limits, dimensions as named constants — and headless
Blender compiles it to a GLB. The program is the asset; the GLB is a
derivative artifact. Because the asset is source code, it stays editable:
"make the handlebar wider" is a minimal, gated source edit, not a remodel.

## Install

Install the skill for Codex:

```sh
git clone https://github.com/larryshaw0079/procagen3d.git ~/.codex/skills/procagen3d
```

Install the skill for Claude Code:

```sh
git clone https://github.com/larryshaw0079/procagen3d.git ~/.claude/skills/procagen3d
```

## Showcase

Representative reference-conditioned results selected from successful
`procagen3d_out/` runs. Each pair shows the closest available source-image
angle on the left and a studio-lit Eevee render of the exported `model.glb` on
the right. Every model starts as an executable Blender Python program and is
compiled to a portable GLB by the ProcAgen3D pipeline.

<table>
  <tr>
    <td align="center">
      <a href="assets/showcase/audi-sport-quattro-s1-reference.png">
        <img src="assets/showcase/audi-sport-quattro-s1-reference.png" width="205" alt="Reference image of an Audi Sport Quattro S1 rally car">
      </a>
      <a href="assets/showcase/audi-sport-quattro-s1.png">
        <img src="assets/showcase/audi-sport-quattro-s1.png" width="205" alt="ProcAgen3D GLB render of an Audi Sport Quattro S1 rally car">
      </a><br>
      <sub>Reference → ProcAgen3D GLB</sub><br>
      <strong>Audi Sport Quattro S1</strong>
    </td>
    <td align="center">
      <a href="assets/showcase/china-pavilion-reference.png">
        <img src="assets/showcase/china-pavilion-reference.png" width="205" alt="Reference image of a traditional Chinese pavilion">
      </a>
      <a href="assets/showcase/china-pavilion.png">
        <img src="assets/showcase/china-pavilion.png" width="205" alt="ProcAgen3D GLB render of a traditional Chinese pavilion">
      </a><br>
      <sub>Reference → ProcAgen3D GLB</sub><br>
      <strong>Traditional Chinese Pavilion</strong>
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="assets/showcase/benben-robot-reference.png">
        <img src="assets/showcase/benben-robot-reference.png" width="205" alt="Reference image of the Benben robot">
      </a>
      <a href="assets/showcase/benben-robot.png">
        <img src="assets/showcase/benben-robot.png" width="205" alt="ProcAgen3D GLB render of the Benben robot">
      </a><br>
      <sub>Reference → ProcAgen3D GLB</sub><br>
      <strong>Benben Robot</strong>
    </td>
    <td align="center">
      <a href="assets/showcase/nissan-skyline-super-silhouette-reference.png">
        <img src="assets/showcase/nissan-skyline-super-silhouette-reference.png" width="205" alt="Reference image of a Nissan Skyline Super Silhouette race car">
      </a>
      <a href="assets/showcase/nissan-skyline-super-silhouette.png">
        <img src="assets/showcase/nissan-skyline-super-silhouette.png" width="205" alt="ProcAgen3D GLB render of a Nissan Skyline Super Silhouette race car">
      </a><br>
      <sub>Reference → ProcAgen3D GLB</sub><br>
      <strong>Nissan Skyline Super Silhouette</strong>
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="assets/showcase/pagani-huayra-r-reference.png">
        <img src="assets/showcase/pagani-huayra-r-reference.png" width="205" alt="Reference image of a Pagani Huayra R">
      </a>
      <a href="assets/showcase/pagani-huayra-r.png">
        <img src="assets/showcase/pagani-huayra-r.png" width="205" alt="ProcAgen3D GLB render of a Pagani Huayra R">
      </a><br>
      <sub>Reference → ProcAgen3D GLB</sub><br>
      <strong>Pagani Huayra R</strong>
    </td>
    <td align="center">
      <a href="assets/showcase/toyota-sr5-reference.png">
        <img src="assets/showcase/toyota-sr5-reference.png" width="205" alt="Reference image of a Toyota SR5 pickup with its hood open">
      </a>
      <a href="assets/showcase/toyota-sr5.png">
        <img src="assets/showcase/toyota-sr5.png" width="205" alt="ProcAgen3D GLB render of a Toyota SR5 pickup with its hood open">
      </a><br>
      <sub>Reference → ProcAgen3D GLB</sub><br>
      <strong>Toyota SR5</strong>
    </td>
  </tr>
</table>

## How it works

The agent supplies design judgment and visual review; a deterministic,
stdlib-only harness enforces the paper's pipeline around it:

```
design → curved/mixed shape probe → synthesize program
→ build (headless Blender) → deterministic gates
→ canonical renders → agent-vision inspection → guarded repair loop (≤3)
→ articulation validation → constraint scoring → deliver
```

Every stage is a CLI command with exit codes and grep-able
`[PROCAGEN3D:OK|WARN|FAIL]` tags, so nothing depends on the agent's honesty:
build errors, doctrine violations, broken joints, and failed dimensional
constraints are all caught mechanically. Perception is agent vision by
design — no learned depth/normal/edge models — which keeps the skill fully
offline and dependency-free.

## Repository layout

```
procagen3d/
├── procagen3d-skill/     # the skill — see its README for full docs
│   ├── SKILL.md          # entry point: the staged loop and gates
│   ├── scripts/          # driver + Blender-side stages
│   ├── references/       # routed depth docs (doctrine, joints, edits, …)
│   └── examples/         # bench-style items: L1_stool, L2_bicycle, L3_robot_arm
└── assets/               # tracked teaser/showcase + gitignored test material
```

## Requirements

- **Blender 4.x** (tested on 4.5 LTS), found via `--blender` flag →
  `$PROCAGEN3D_BLENDER` → PATH → `~/.cache/procagen3d/*/blender`.
- **Python 3.10+**. No pip packages — the harness is stdlib only, and
  Blender stages run under Blender's bundled Python.

## Quick start

Then ask naturally ("generate a GLB of a wheelbarrow with a rotating
wheel", "rebuild this car from these photos") or invoke `$procagen3d`
explicitly in Codex. Outputs land in `./procagen3d_out/<slug>/`:
`program.py` (the deliverable), `model.glb`, `scene.blend`, diagnostics,
and a six-view render sheet. Image-conditioned runs also save unchanged copies
of the used inputs there as `reference_01.<ext>`, `reference_02.<ext>`, etc.,
with their provenance recorded in `priors.md`.

The harness is also usable by hand:

```sh
python3 procagen3d-skill/scripts/procagen3d.py build program.py --out out/
python3 procagen3d-skill/scripts/procagen3d.py check out/ --tier showcase --form auto
python3 procagen3d-skill/scripts/procagen3d.py joints out/
python3 procagen3d-skill/scripts/procagen3d.py score out/ --spec spec.yaml
```

See [`procagen3d-skill/README.md`](procagen3d-skill/README.md) for the full
CLI, install notes, and design rationale, and
[`procagen3d-skill/SKILL.md`](procagen3d-skill/SKILL.md) for the pipeline
the agent actually follows.

## Bench examples

`procagen3d-skill/examples/` holds three ProcAgen3D-Bench-style items
(`L1_stool`, `L2_bicycle`, `L3_robot_arm`), each with a `manifest.txt`
prompt and a `spec.yaml` ground truth. Protocol, same as the paper:
generate from the manifest only, then score the exported asset against the
spec — the spec is never opened during generation.
