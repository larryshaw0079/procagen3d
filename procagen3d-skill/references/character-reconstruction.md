# Character reconstruction — organic-v1 dedicated routine

Use this route for humans, humanoids, anthropomorphic characters, and creatures
whose identity depends on anatomy, pose flow, facial structure, cloth, fur, or
other deformable-looking surfaces. Do not use it for a rigid robot merely
because the robot has two arms and two legs; a visibly mechanical figure stays
on the object/mixed-form route.

This route is opt-in so established object and scene behavior does not change:

```python
root["procagen3d_subject_domain"] = "character"
root["procagen3d_character_routine"] = "organic-v1"
```

Also set `reconstruction_plan.json.subject_domain` to `"character"`, author
`character_plan.json`, and run `check --subject character`. The reusable schema
example is [`character-plan.example.json`](character-plan.example.json).

## Why the generic route fails on characters

A part inventory is not an anatomy model. A character can match the whole-frame
mask, contain every named accessory, and exceed a 400-mesh showcase floor while
still reading as a pile of beads. The characteristic failures are:

- sphere head + muzzle bead + eye beads instead of one designed cranial/face
  volume;
- upper-arm, forearm, thigh, and shin solids that terminate visibly at joints;
- shoulders attached to the side of the torso without a deltoid/clavicle
  transition;
- separate pelvis and torso blobs with no rib-to-waist-to-hip flow;
- clothing modeled as rigid plaques instead of shells that follow the body;
- hundreds of rivets, folds, hair strands, or scars used to satisfy density
  floors while the body envelope remains wrong;
- a fitted 2D pose whose limb depths and bilateral anatomy disagree in 3D.

The remedy is not soft-body physics. The soft appearance comes from coherent
organic topology, optional skeletal skinning, and small secondary deformations.

## The routine

### C0 — classify and lock anatomy evidence

Before choosing dimensions, record:

1. archetype: `humanoid`, `anthropomorphic`, or `creature`;
2. coverage: `full-body`, `upper-body`, or `head-only`;
3. body strategy: `continuous-skin` or `hybrid-skin`;
4. rig strategy: `skinned` or `segmented`;
5. `proportion_system`: body height in head units, whether it comes from the
   reference/canonical prior/a hybrid, and the two landmarks defining one head;
6. every articulated chain, including non-human tails, ears, wings, or
   digitigrade links;
7. the reference camera and root pose.

For a full humanoid, the scaffold normally includes head top, chin, neck,
pelvis, bilateral shoulders, elbows, wrists, hips, knees, ankles, and toes.
Read these from the reference and use the same ids in `fit_spec.json` and
`character_plan.json`. Build both sides from shared constants even when their
posed transforms differ.

Do not infer body dimensions from the outer silhouette before separating skin,
hair, clothing, armor, and carried equipment. A coat shoulder is not the
deltoid; baggy trousers are not the thigh radius.

### C1 — decompose by deformation behavior

Every visible part belongs to one primary layer:

| layer | purpose | preferred construction | behavior |
|---|---|---|---|
| `core_volume` | head, neck, torso, pelvis, exposed limbs | implicit union, connected loft, subdivision cage, or deliberately sculpted procedural surface | crosses major joints; preserve organic flow |
| `deformable_appendage` | tail, long ear, wing flesh, tentacle | skinned sweep, connected loft, or subdivision cage | bends along a chain |
| `cross_joint_shell` | shirt, trousers, coat, cape, hair mass spanning neck/shoulders | offset/loft/cloth shell | follows more than one joint |
| `rigid_attachment` | horn, buckle, weapon, cuff entirely owned by one link | single-bone or rigid parent | must not receive blended organic weights |
| `surface_detail` | brows, lips, scars, seams, markings | relief, decal, or restrained geometry | follows its host surface |

The classification question is: **does this part cross a joint?** Material
hardness is secondary. A rigid-looking chest plate spanning the shoulder is a
cross-joint shell; a soft wrist band contained entirely on the forearm can be a
rigid attachment.

Tag each planned mesh:

```python
body["procagen3d_character_layer"] = "core_volume"
body["procagen3d_character_construction"] = "implicit-union"
shirt["procagen3d_character_layer"] = "cross_joint_shell"
shirt["procagen3d_character_construction"] = "offset-shell"
buckle["procagen3d_character_layer"] = "rigid_attachment"
buckle["procagen3d_character_construction"] = "rigid-parent"
```

### C2 — build the anatomy probe in three passes

The character probe is not a miniature final model. Build and review it in this
order:

1. **Landmark scaffold.** Emit only semantic empties and thin chain guides.
   Register the camera and pass landmark, ratio, frame-axis, and pose-chain
   gates. A crouch, contrapposto, digitigrade stance, or foreshortened arm must
   already read correctly.
2. **Body envelope.** Replace guides with the core volume, deformable
   appendages, feet/hands, and every visible joint transition. Use one neutral
   material. Pass front, side, top, iso, and tight per-region silhouettes.
3. **Major shells.** Add hair masses, clothing, armor, and equipment that alter
   the silhouette. Fit them locally without changing the accepted anatomy.

Do not add facial microdetail, rivets, stitches, loose hair strands, surface
marks, or wear until all three passes succeed.

For complex/extreme characters, probe in identity order:

`pose scaffold → torso/pelvis → head/face envelope → limbs/hands/feet → major
garment shells → appendages/equipment`.

### C3 — construct a coherent body envelope

Choose the simplest method supported by the reference:

- **Implicit union + remesh:** best for a stylized body whose torso, shoulders,
  neck, and limb roots need one continuous surface. Author named capsule,
  ellipsoid, rounded-plate, add, and subtract controls; union them, then voxel
  remesh once. Keep the controls in code. Do not store a baked vertex dump.
- **Connected loft:** best when changing sections are clear. A torso loft needs
  stations at pelvis, waist, lower ribs, chest, clavicle, and neck. Shoulder and
  hip branches need explicit shared/bridged sections rather than endpoint caps.
- **Subdivision cage:** best for realistic or strongly asymmetric anatomy,
  especially a face or torso. Control loops belong at brow, cheek, jaw, mouth,
  clavicle, rib margin, waist, glute fold, and joint transitions—not at uniform
  intervals.
- **Procedurally sculpted surface:** acceptable for a mascot-like continuous
  capsule when radial bulges, pinches, and creases are explicit named functions.

Independent ellipsoids are allowed for an initial landmark proxy. They are not
an acceptable final representation for adjacent exposed body masses.

### C4 — make every major joint transition explicit

For full humanoids, the plan records neck, both shoulders, both elbows, both
hips, and both knees. Each transition uses one of four modes:

- `continuous`: both sides are already one connected host surface;
- `bridge`: a dedicated organic bridge overlaps/touches both body sections;
- `covered`: a cross-joint garment/hair/armor shell owns the visible seam;
- `mechanical`: an intentionally rigid socket, valid only for `hybrid-skin`.

Each transition entry names the host and the two connected patterns. Tag the
host with every transition it owns:

```python
body["procagen3d_character_transitions"] = ",".join([
    "neck_center", "shoulder_left", "shoulder_right",
    "hip_left", "hip_right",
])
```

Use a comma-separated string because Blender custom-property arrays do not
portably store strings. The checker also accepts a list in hand-authored scene
graphs, but executable Blender programs should use the string form above.

The checker verifies that the host exists, carries the semantic tag, and that
its world bounds reach both connected sides within 1% of character size. This
catches obvious gaps but is not proof of a manifold weld; the neutral form
sheet must still confirm the surface transition. It is deliberately stricter
than matching a front silhouette: a shoulder floating behind the torso can
look correct from one camera.

### C5 — solve the head as regions, not decorations

At minimum plan and build:

- cranial mass and jaw/chin flow;
- the planned `expected_eye_count` (normally two) with lids/socket depth, not
  discs pasted on a sphere;
- nose or muzzle transition into the mid-face;
- mouth/jaw region with an intentional lip or muzzle plane.

Stylized characters may exaggerate these relationships, but must still author
them. Hair is a separate mass system: scalp/base volume first, then locks or
strands only where they change silhouette, flow, or identity. Human likeness
requires a tight registered face crop; a whole-character mask cannot judge
inter-eye spacing, jaw width, nose projection, or mouth placement.

### C6 — add shells, rigid parts, and detail

Add in this order:

1. cross-joint clothing/hair shells;
2. single-link rigid attachments and equipment;
3. identity-bearing face, hand, footwear, and costume features;
4. secondary seams, folds, fasteners, and wear;
5. microdetail.

Do not fragment a surface to raise the mesh count. Character showcase floors
are intentionally lower than object floors because one high-quality connected
body can replace dozens of primitive parts. Triangle count is spent on smooth
silhouette and deformation; materials distinguish skin, hair, cloth, leather,
metal, eye, and emissive finishes.

### C7 — rigging

Use `skinned` when the deliverable needs deforming joints. Emit a Blender
armature, bind all `core_volume`, `deformable_appendage`, and
`cross_joint_shell` meshes that cross joints, and keep rigid attachments on a
single bone. Limit each vertex to the smallest useful influence set and test
shoulder, elbow, hip, and knee bends for collapse.

Use `segmented` only for a static reconstruction or a deliberately stylized
rigid hierarchy. The visible rest surface must still have continuous, bridged,
or covered joint transitions; `segmented` is not permission to leave gaps.

Secondary breathing, hair lag, ear spring, or cloth flutter is a later authored
deformation layer. It is not a substitute for correct rest topology and should
not be described as soft-body simulation.

## Character-specific validation

Run:

```sh
python3 scripts/procagen3d.py check <out> \
  --tier showcase --form auto --subject character
```

The route adds these hard contracts:

- `CHARACTER_ROUTE`: root selected `organic-v1`;
- `CHARACTER_PLAN`: valid archetype, coverage, landmark/chains, layers,
  transitions, and facial-region schema;
- `CHARACTER_ANATOMY`: planned markers and pose chains exist in the scene and
  registered fit;
- `CHARACTER_LAYERS`: planned meshes exist and carry the correct deformation
  layer tags; the core is not fragmented into more than 24 meshes;
- `CHARACTER_TRANSITIONS`: every planned host is tagged and its world bounds
  reach both connected sides;
- `CHARACTER_FACE`: cranium, paired eyes, nose/muzzle, and mouth/jaw regions
  are present;
- `CHARACTER_RIG`: a `skinned` plan actually emits an armature.

The generic object/scene route retains its original adaptive floors and region
density gate. The character route replaces mesh-density pressure with the
contracts above and emits `CHARACTER_FRAGMENTATION` when independent object
count is suspiciously high relative to visible feature groups.

## Visual acceptance checklist

- **pose:** all body and appendage chains match before detail;
- **proportion:** head units, shoulder/pelvis relationship, limb lengths, hand
  and foot scale, and bilateral depth are coherent;
- **flow:** neck-to-torso, shoulder-to-arm, rib-to-waist, pelvis-to-thigh, and
  exposed elbow/knee transitions read as designed surfaces;
- **face:** cranial, eye socket/lid, nose/muzzle, cheek, mouth, and jaw planes
  create the reference identity at face-crop scale;
- **shells:** clothing and hair follow the underlying form and cross joints
  only when planned;
- **negative space:** armpits, crotch, limb gaps, fingers, ears, wings, and
  carried-object gaps remain open where the reference shows them;
- **multiview:** side and top views contain credible depth rather than a fitted
  front-view cut-out;
- **detail:** features reinforce accepted masses; they never compensate for a
  failed body envelope.
