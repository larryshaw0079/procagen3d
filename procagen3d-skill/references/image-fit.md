# Registered image fit

Use this reference for every image-conditioned asset. Convert visible evidence
into `fit_spec.json`, render through the declared camera, and make every numeric
gate pass after every full build. Fit proves visible projection, not hidden
geometry.

Author new assets with schema `version: 2`. Version 1 remains supported only
for rebuilding legacy assets. Version 2 adds mandatory local-silhouette and pose
contracts so a loose whole-frame fit cannot hide wrong primitive families or
an incorrect articulated stance.

## Contents

1. Author the contract
2. Register the camera and mask
3. Gate local silhouette regions
4. Declare landmarks and ratios
5. Gate rigid and articulated pose
6. Declare scene instances and relations
7. Run and interpret the gate

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
preview. Record ambiguous evidence in `priors.md`; do not loosen or tighten a
threshold to pretend uncertain evidence is exact. Version 2 also requires
`reconstruction_plan.json`; see `reconstruction-planning.md`.

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
appropriate only for clean product renders. Defaults are deliberately lenient
for whole-frame evidence; version 2's local regions provide the stricter shape
checks.

## 3. Gate local silhouette regions

Declare at least three tight crops that cross informative contours. Choose
regions around primitive decisions (box corner versus ellipsoid arc),
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
reference occupancy outside 2–98 percent, duplicate crops, v2 IoU thresholds
below 0.60, and v2 area-error thresholds above 0.35.

## 4. Declare landmarks and ratios

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
`check` rejects missing, failed, or stale fit evidence whenever preserved
`reference_NN.*` inputs exist.
