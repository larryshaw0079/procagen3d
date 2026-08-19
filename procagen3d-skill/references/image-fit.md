# Registered image fit

Use this reference for every image-conditioned asset. Convert visible evidence
into `fit_spec.json`, render through the declared camera, and make every numeric
gate pass after every full build. Fit proves visible projection, not hidden
geometry.

Schema `version: 2` is required, along with `reconstruction_plan.json`. Version
1 is rejected for image-conditioned assets: it had no local-silhouette, pose, or
shape-prior contract, which made declaring it a way to opt out of every
reconstruction gate at once.

## Contents

1. Author the contract
1a. You do not set the pass mark
2. Register the camera and mask
3. Gate local silhouette regions
4. Declare landmarks and ratios
5. Gate rigid and articulated pose
6. Declare scene instances and relations
7. Run and interpret the gate
8. A passing fit is not a correct model

## 1. Author the contract

Write `<out>/fit_spec.json` before geometry code. Use normalized image
coordinates `[u, v]` with top-left `(0, 0)` and bottom-right `(1, 1)`. Keep
image-derived targets here; keep explicit user requirements in `spec.yaml`.

```json
{
  "version": 2,
  "reference_image": "reference_01.png",
  "camera": {},
  "mask": {},
  "silhouette_regions": [],
  "landmarks": [],
  "pose": {"mode": "rigid", "frame_axes": [], "chains": []},
  "ratios": [],
  "instances": [],
  "relations": []
}
```

Measure against the preserved reference copy, not a screenshot or resized
preview. Record ambiguous evidence in `priors.md`. Version 2 and
`reconstruction_plan.json` are both required for any image-conditioned asset;
see `reconstruction-planning.md`.

## 1a. You do not set the pass mark

Pass thresholds live in `blender_stages.py`, not in your spec. A spec value is
honoured only when it is **stricter** than policy; anything looser is clamped
back and the run fails on the `threshold_policy` gate, which lists every value
you tried to relax. Omit thresholds entirely unless you are tightening one.

Whole-frame mask IoU floors, by the plan's complexity class:

| complexity | mask IoU | region IoU |
|---|---:|---:|
| `simple` | 0.88 | 0.85 |
| `moderate` | 0.84 | 0.82 |
| `complex` | 0.80 | 0.78 |
| `extreme` | 0.76 | 0.74 |

Error ceilings are fixed: bbox 0.05, centroid 0.04, whole-frame area ratio
0.20, region area ratio 0.18, landmark 0.04, ratio 0.10, pose axis 4°, chain
segment 5°, chain joint 7°, chain length fraction 0.04, instance bbox 0.05,
instance centroid 0.04, relation 0.05.

Wiry subjects get an automatic allowance of up to 0.15 IoU, derived by eroding
the reference mask and measuring how much of the foreground disappears. A
bicycle earns it; a camera body does not. It is measured, so there is nothing
to claim. The report records `threshold_policy.reference_thinness` and the
floors actually applied.

A floor you cannot reach is information. Fix the geometry, or report honestly
that the reference does not support the reconstruction — never negotiate the
bar.

## 2. Register the camera and mask

Declare a perspective camera with `fov_y_deg`, or an orthographic camera with
`ortho_scale_m`. Always declare real placement: `location_m` plus `target_m`,
or `azimuth_deg`, `elevation_deg`, `distance_m`, and `target_m`. Add
`roll_deg`, `shift_x`, and `shift_y` when nonzero. The reference supplies the
render resolution and aspect ratio.

```json
"camera": {
  "projection": "perspective",
  "azimuth_deg": 28.0,
  "elevation_deg": 12.0,
  "roll_deg": -3.0,
  "fov_y_deg": 34.0,
  "distance_m": 2.8,
  "target_m": [0.0, 0.0, 0.45],
  "shift_x": 0.0,
  "shift_y": 0.0
}
```

Do not let the harness auto-center or auto-fit the model: changing geometry
must not silently change the proof camera. Solve view orientation before
proportions. Use long parallel object edges, visible top/side-face amounts,
projected circles, and symmetry axes—not the whole-object bbox alone. Keep
canonical dimensions unchanged while tuning camera azimuth/elevation/roll.
Record a real root lean separately only when ground/contact or articulation
evidence proves it; a single floating product view cannot distinguish camera
rotation from rigid object rotation.

The camera is solved once, at probe time, and then **locked** into
`reconstruction_plan.json` under `camera_solve`. `check` compares the fit
spec's camera against that lock and fails on `CAMERA_LOCK` if they drift.
This exists because a wrong camera and a wrong shape produce the same
silhouette score, and when the camera is free to move it is always the shape
that ends up deformed to compensate. Changing the camera later is a
`refine-priors` decision: re-solve it against the reference, say why, and
update the lock — never nudge it to rescue a failing fit.

Choose one foreground-mask source:

- `alpha`: threshold reference alpha;
- `border`: estimate the background from the image border;
- `file`: read a black/white mask at `path` with identical dimensions;
- `auto`: use alpha when useful, otherwise border estimation.

```json
"mask": {
  "source": "auto",
  "color_threshold": 0.08,
  "min_iou": 0.75,
  "max_bbox_error": 0.04,
  "max_centroid_error": 0.03,
  "max_area_ratio_error": 0.20
}
```

Use a supplied mask for textured/nonuniform backgrounds. Border estimation is
appropriate only for clean product renders. Whole-frame evidence is the weakest
signal in the spec: a whole-object mask can pass while every local part is
wrong, which is what local regions exist to prevent.

## 3. Gate local silhouette regions

Declare one tight crop for every two primary masses — at least three, up to
eight. `check` enforces this as `FIT_REGION_COVERAGE`, because three regions on
a thirty-mass mecha leaves nearly every family and pose decision untested.
Choose regions around primitive decisions (box corner versus ellipsoid arc),
pose-sensitive limbs, tapers, openings, or attachments. Do not use a crop that
is nearly all foreground/background or simply repeats the whole frame.

```json
"silhouette_regions": [
  {"id": "body_upper_left_corner",
   "bbox_uv": [0.10, 0.18, 0.35, 0.55],
   "min_iou": 0.82, "max_area_ratio_error": 0.12},
  {"id": "right_leg_chain",
   "bbox_uv": [0.56, 0.48, 0.82, 0.96],
   "min_iou": 0.78, "max_area_ratio_error": 0.15},
  {"id": "receiver_fore_end",
   "bbox_uv": [0.12, 0.25, 0.48, 0.55],
   "min_iou": 0.84, "max_area_ratio_error": 0.12}
]
```

The gate computes crop-local mask IoU and foreground-area error. The overlay
draws magenta crop boxes. A region must contain a real contour; the tool rejects
reference occupancy outside 2–98 percent and duplicate crops. Per-region
`min_iou` and `max_area_ratio_error` are subject to §1a: state them only to go
stricter than the policy floor for that complexity class.

## 4. Declare landmarks and ratios

`reference_uv` is **what you saw in the image**. Read each point off the saved
reference and write down that coordinate. Never obtain it by projecting your
own model through the camera and copying the result: that makes the landmark,
every ratio built on it, and every frame axis and pose chain that uses it a
comparison of the model against itself, and they will all pass no matter how
wrong the model is.

The `landmark_provenance` gate detects this. Estimating a point from an image
is worth about a pixel; if the median landmark error across five or more
landmarks falls below 0.001, the targets were back-filled and the fit fails.
Real agreement on a good model lands around 0.005–0.03. A landmark you cannot
locate confidently in the image is a landmark you should not gate — set
`"gate": false` and say so in `priors.md`.

Place semantic empty objects at actual joints/control stations when necessary,
then match their names with `anchor: "origin"`. Otherwise anchor to a matched
part's projected bbox: `bbox_center`, `bbox_left`, `bbox_right`, `bbox_top`,
`bbox_bottom`, or a bbox corner. Use `world_point_m` only for an intentional
fixed world-space marker.

Use part-bbox anchors for silhouette extrema whenever possible. Empty markers
must derive from the same named constants/transforms as geometry; a free marker
placed only to pass fit makes the gate meaningless. Pose markers use actual
joint constants.

```json
"landmarks": [
  {"id": "muzzle", "reference_uv": [0.12, 0.45],
   "pattern": "Fit_MuzzleTip", "anchor": "origin", "max_error": 0.025},
  {"id": "butt", "reference_uv": [0.87, 0.48],
   "pattern": "Fit_ButtEnd", "anchor": "origin", "max_error": 0.025},
  {"id": "receiver_top", "reference_uv": [0.58, 0.34],
   "pattern": "Receiver_Core", "anchor": "bbox_top", "gate": false},
  {"id": "receiver_bottom", "reference_uv": [0.58, 0.60],
   "pattern": "Receiver_Core", "anchor": "bbox_bottom", "gate": false}
],
"ratios": [
  {"id": "length_over_receiver_height",
   "numerator": ["muzzle", "butt"],
   "denominator": ["receiver_top", "receiver_bottom"],
   "axis": "distance", "max_relative_error": 0.08}
]
```

Use `axis: "x"` or `"y"` for an axis-aligned projected ratio and
`"distance"` otherwise. A ratio derives its target from reference landmarks
and remains scale-independent.

## 5. Gate rigid and articulated pose

Declare `pose.mode` as `rigid`, `articulated`, or `unobservable`.

- `rigid`: add at least one directed frame axis between geometry-bound
  landmarks. Use two or three when longitudinal and up directions are visible.
- `articulated`: add every visible chain with at least three ordered joint/end
  landmarks; left/right chains are separate. Add frame axes when visible.
- `unobservable`: use only for rotationally symmetric/isotropic targets and
  provide a non-empty `reason`.

```json
"pose": {
  "mode": "articulated",
  "frame_axes": [
    {"id": "torso_up", "landmarks": ["pelvis", "neck"],
     "max_angle_error_deg": 3.0}
  ],
  "chains": [
    {"id": "left_leg",
     "landmarks": ["hip_l", "knee_l", "ankle_l", "toe_l"],
     "max_segment_angle_error_deg": 4.0,
     "max_joint_angle_error_deg": 6.0,
     "max_length_fraction_error": 0.035}
  ]
}
```

Frame axes compare directed projected angles. Chains compare every segment
direction, internal bend angle, and each segment's fraction of total chain
length. These catch a neutral/front-facing rebuild of a tilted or splayed
reference even when endpoints and the whole mask fall within loose tolerances.
Version 2 caps declared tolerances at 12° for frame axes, 15° for segment
directions, 20° for joint bends, and 0.15 for normalized segment-length error;
larger values are rejected rather than treated as meaningful gates.

## 6. Declare scene instances and relations

For a multi-object reference, add one semantic group empty per independent
piece. Measure each image envelope as
`reference_bbox_uv: [left, top, right, bottom]`.

```json
"instances": [
  {"id": "sofa", "pattern": "SofaAssembly",
   "reference_bbox_uv": [0.10, 0.29, 0.48, 0.78]},
  {"id": "table", "pattern": "CoffeeTableAssembly",
   "reference_bbox_uv": [0.40, 0.51, 0.73, 0.86]},
  {"id": "chair", "pattern": "ArmchairAssembly",
   "reference_bbox_uv": [0.64, 0.25, 0.91, 0.69]}
],
"relations": [
  {"id": "table_from_sofa", "type": "relative_position",
   "a": "sofa", "b": "table", "max_error": 0.04},
  {"id": "table_sofa_overlap", "type": "bbox_iou",
   "a": "sofa", "b": "table", "max_error": 0.05},
  {"id": "table_in_front", "type": "depth_order",
   "front": "table", "behind": "sofa", "min_margin_m": 0.02}
]
```

`relative_position` compares projected centroid vectors; `bbox_iou` compares
overlap; `depth_order` compares camera-space depth. Use per-instance and
relation gates together.

## 7. Run and interpret the gate

Run after every full build and on the reconstruction probe while tuning
camera, pose, primitive family, or macro dimensions:

```sh
procagen3d fit <out> --spec <out>/fit_spec.json
```

Read all evidence:

- `renders/reference_match.png` — registered model render;
- `renders/reference_overlay.png` — green overlap, red reference-only, blue
  render-only, yellow reference landmarks, cyan render landmarks, and magenta
  local-region boxes;
- `renders/reference_mask.png` and `render_mask.png` — scored masks;
- `fit_report.json` — mask, region, landmark, ratio, pose, instance, relation,
  and input-hash records.

Fix in this order: camera/frame → rigid root pose → articulated chains → macro
dimensions/families → detail. Lock each passing layer before continuing. Never
deform geometry to compensate for a knowingly wrong camera or joint pose.

## 8. A passing fit is not a correct model

Every metric on this page is measured in the image plane of one camera. Nothing
here can see depth. A model that is a flattened bas-relief — correct from the
reference viewpoint, paper-thin from the side — scores as well as a solid one,
and optimising against these numbers alone actively produces that failure.

Depth is gated separately, from the canonical orthographic renders, which all
share one scale and are therefore directly comparable:

- `PRIMARY_DEPTH` fails a primary mass whose smallest dimension drops below 8%
  of its largest, unless it is genuinely a shell, plate, or strand;
- `VIEW_COLLAPSE` fails when the smallest canonical silhouette falls below 10%
  of the largest, and warns below 22%.

So `fit` passing means "registers correctly from this camera," never "is
correct." Read `sheet.png` and judge the side and top views yourself before
calling shape passed.
`check` rejects missing, failed, or stale fit evidence whenever preserved
`reference_NN.*` inputs exist.
