# Articulation — joints, limits, validation

ProcAgen3D assets carry articulation as first-class structure: joint nodes at
pivots, existing meshes re-parented without moving a vertex, and
machine-readable type/axis/limits that survive into the GLB (as glTF node
`extras`, readable by any consumer).

## Joint model

A joint is an empty created by the canonical `add_joint` helper (see
`references/doctrine.md` — use it verbatim):

- chain: `parent ← Joint_X ← child` — rotating/translating the joint empty
  moves the child subtree and nothing else;
- `procagen3d_joint_type`: `revolute` | `prismatic` | `fixed`;
- `procagen3d_joint_axis`: world-space unit vector at rest pose;
- `procagen3d_joint_limits`: `[lo, hi]` — degrees for revolute, meters for
  prismatic; 0 is the rest pose, so limits must bracket 0;
- joint empty sits at the physical pivot; the child's own origin should sit
  there too (doctrine: pivot ownership).

Anything that rides on a moving part (a handle on a lid, a caliper on a
fork) is parented under the child, so it sweeps with it automatically.

## Naming and design

`Joint_<What>` (`Joint_Lid`, `Joint_Steer`, `Joint_Elbow`). Declare joints
in the program header's joint table before coding. Limits are design
decisions, not defaults: a door ≈ [0, 110]°, a jaw ≈ [0, 25]°, a steering
column ≈ [-50, 50]°. The paper's audit flagged exactly this failure — 29/56
generated revolute joints declared ≥300° because the model wrote generic
defaults. The validator warns at ≥300°; have a physical reason or narrow it.

## Common patterns

- **Hinge (door/lid):** axis along the hinge line, pivot on that edge,
  limits open one way from 0.
- **Axle (wheel/fan):** axis through the hub, continuous rotation is fine —
  declare [-180, 180] and accept the ≥300° warning with a stated reason, or
  keep [-179, 179] to stay under it.
- **Prismatic (drawer/telescope):** axis along travel, limits [0,
  travel_m]; make the travel shorter than the cavity depth.
- **Arm chains:** each link's joint parents the next link
  (`Base ← Joint_Shoulder ← Upper_Arm ← Joint_Elbow ← Forearm`); pivots at
  the physical joint constants (`SHOULDER`, `ELBOW`).

## Validator semantics — `procagen3d joints <out>`

Reads `scene.blend`, writes `joints_report.json`. FAIL ⇒ exit 1 ⇒ must fix:

| check | FAIL means |
|-------|-----------|
| type_valid | type not revolute/prismatic/fixed |
| child_exists | `procagen3d_joint_child` names a missing object |
| axis_nonzero | zero-length axis |
| pivot_on_moving_part | pivot lies outside the child subtree's bbox (+10% pad) — the joint would swing geometry it doesn't touch |
| rest_pose | validator restore drifted ⇒ the scene mutates under evaluation; usually a driver/constraint in the program |

WARNs need your judgment, in the render context:

- `JOINT_LIMITS` missing or ≥300° — declare a plausible range (above).
- `JOINT_SWEEP` collision at a sampled position (endpoints, quarters). The
  child subtree was posed there and overlap-tested (BVH) against all
  non-moving meshes except the joint's parent part. Decide per warning:
  real interpenetration (lid sweeping through a wall) → shrink limits or
  move the pivot; legitimate contact (a hinge knuckle grazing its mate) →
  accept and record the reason. `--strict` includes the parent part too.

The sweep restores the scene exactly; `rest_pose` failures prove it didn't.

## In the GLB

`Joint_*` nodes appear in the exported hierarchy with all `procagen3d_joint_*`
keys under `extras`. A downstream engine articulates the asset by rotating/
translating the joint node within its limits — no re-rigging needed. This is
the deliverable the paper measures as "joint geometric validity"; keep it
true by only ever creating joints through `add_joint`.
