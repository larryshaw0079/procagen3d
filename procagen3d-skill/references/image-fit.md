# Registered image fit

Use this reference for every image-conditioned asset. Convert visible evidence
into `fit_spec.json`, render through the declared camera, and make the numeric
fit pass after every full build and before accepting the asset. Treat the fit
as proof of visible projection only; it cannot verify hidden geometry.

## Contents

1. Author the contract
2. Register the camera and mask
3. Declare landmarks and ratios
4. Declare scene instances and relations
5. Run and interpret the gate

## 1. Author the contract

Write `<out>/fit_spec.json` before geometry code. Use JSON `version: 1` and
paths relative to `<out>`. Use normalized image coordinates `[u, v]` with
top-left `(0, 0)` and bottom-right `(1, 1)`. Keep image-derived targets here;
keep explicit user requirements in `spec.yaml`.

```json
{
  "version": 1,
  "reference_image": "reference_01.png",
  "camera": {},
  "mask": {},
  "landmarks": [],
  "ratios": [],
  "instances": [],
  "relations": []
}
```

Measure coordinates against the preserved reference copy, not a screenshot or
resized preview. Record ambiguous evidence in `priors.md`; do not tighten a
threshold to pretend uncertain evidence is exact.

## 2. Register the camera and mask

Declare a perspective camera with `fov_y_deg`, or an orthographic camera with
`ortho_scale_m`. Always declare real placement: `location_m` plus `target_m`,
or `azimuth_deg`, `elevation_deg`, `distance_m`, and `target_m`. Add
`roll_deg`, `shift_x`, and `shift_y` when nonzero. The reference image supplies
the render resolution and aspect ratio.

```json
"camera": {
  "projection": "perspective",
  "azimuth_deg": 28.0,
  "elevation_deg": 12.0,
  "roll_deg": 0.0,
  "fov_y_deg": 34.0,
  "distance_m": 2.8,
  "target_m": [0.0, 0.0, 0.45],
  "shift_x": 0.0,
  "shift_y": 0.0
}
```

Do not let the harness auto-center or auto-fit the model: changing geometry
must not silently change the proof camera.

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

Use a supplied mask for textured or nonuniform backgrounds. Border estimation
is appropriate only for clean product renders. Defaults are deliberately
lenient (`IoU 0.70`, bbox `0.05`, centroid `0.04`, area-ratio error `0.25`);
calibrate stricter values from representative references.

## 3. Declare landmarks and ratios

Place semantic empty objects at identity-bearing stations when possible, then
match their names with `anchor: "origin"`. Alternatively anchor to a matched
part's projected bbox: `bbox_center`, `bbox_left`, `bbox_right`, `bbox_top`,
`bbox_bottom`, or a bbox corner. Use `world_point_m` only for an intentional
fixed world-space marker. Derive marker positions from the same named constants
that drive geometry; a marker that stays behind while its part changes makes a
ratio gate meaningless.

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

Use `axis: "x"` or `"y"` for an axis-aligned projected ratio. Use
`"distance"` otherwise. A ratio derives its target from the reference
landmarks and therefore remains scale-independent.

## 4. Declare scene instances and relations

For a multi-object reference, add one semantic group empty per independent
piece. Measure each visible/inferred image envelope and declare it as
`reference_bbox_uv: [left, top, right, bottom]`. The gate projects the matched
group's complete mesh envelope through the registered camera.

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

`relative_position` compares the pairwise projected centroid vector;
`bbox_iou` compares projected overlap; `depth_order` compares camera-space
depth. Use both per-instance and relation gates: a correct pairwise vector does
not prove correct absolute framing.

## 5. Run and interpret the gate

Run after every full build. You may also run it on a shape-only probe when
tuning primary dimensions, layout, or the camera:

```sh
procagen3d fit <out> --spec <out>/fit_spec.json
```

Read all emitted evidence:

- `renders/reference_match.png` — registered model render;
- `renders/reference_overlay.png` — green overlap, red reference-only, blue
  render-only; yellow crosses are reference landmarks and cyan are renders;
- `renders/reference_mask.png` and `render_mask.png` — exact scored masks;
- `fit_report.json` — target, measured value, verdict, and input hashes.

Fix the highest-impact camera error before geometry. Once the camera is
credible, lock it and repair macro dimensions/layout. Never deform the asset
to compensate for a knowingly wrong camera. `check` rejects missing, failed,
or stale fit evidence whenever preserved `reference_NN.*` inputs exist.
