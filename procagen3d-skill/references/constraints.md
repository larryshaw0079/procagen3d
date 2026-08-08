# Constraint specs and the scorer

Any measurable requirement in the request — counts ("8 bolts"), dimensions
("66 cm wheels", "seat at 45 cm"), symmetry, required joints — becomes a
machine-checkable `spec.yaml` authored at intake, **before** generation.
After export, `procagen3d score <out> --spec spec.yaml` measures the actual
geometry (from `scene_graph.json`, derived at export time) and prints a
target/measured/verdict table plus `score_report.json`. Never hand-verify a
number the scorer can measure; never claim a pass without its output.

## File format

YAML (a strict subset — see below) or JSON with the same shape:

```yaml
object: city_bicycle
units: meters
constraints:
  - id: wheel_count
    measure: count
    pattern: "Wheel_*"
    equals: 2
  - id: wheel_diameter
    measure: dimension
    pattern: "Wheel_Front"
    metric: extent_max
    value: 0.66
    tolerance: 10%
  - id: saddle_height
    measure: dimension
    pattern: "Saddle"
    metric: top_z
    range: [0.85, 1.05]
  - id: grip_symmetry
    measure: symmetry
    pair: ["Grip_L", "Grip_R"]
    plane: x
    tolerance: 0.01
  - id: steering
    measure: joint
    pattern: "Joint_Steer*"
    type: revolute
  - id: axle_span
    measure: distance
    between: ["Axle_Front", "Axle_Rear"]
    value: 1.05
    tolerance: 8%
  - id: has_kickstand
    measure: exists
    pattern: "Kickstand*"
```

## Measures

| measure | fields | semantics |
|---------|--------|-----------|
| `count` | `pattern`, `equals` | number of **mesh** objects whose name matches the glob |
| `exists` | `pattern` | ≥1 object of any type matches |
| `dimension` | `pattern`, `metric`, target | metric of the union world-bbox of matched geometry |
| `distance` | `between: [a, b]`, optional `axis`, target | origin-to-origin distance (or single-axis) |
| `symmetry` | `pair: [a, b]`, `plane` (x/y/z), `tolerance` | mirror a's center across plane=0, distance to b's center |
| `joint` | `pattern`, optional `type`, `min_count`/`equals` | declared joints matching name (+type) |

Metrics for `dimension`: `extent_x/y/z`, `extent_max`, `extent_min`,
`height` (= extent_z), `diameter_xy` (max of x/y extents), `top_z`,
`bottom_z`, `origin_z` (single object's origin height, works for empties).
A pattern that matches a group empty measures the union of its mesh
descendants — so `pattern: "Stool"` measures the whole assembly.

Targets: `equals: N` (exact, counts); `value` + `tolerance` (`10%` relative
or `0.05` absolute meters); or `range: [lo, hi]`. Patterns are
case-sensitive `fnmatch` globs against object names — spec patterns and the
program's part names must be designed together at intake.

An unmatched pattern scores **unmeasurable = FAIL**, which usually means the
program renamed a part the spec expected: fix the name, not the spec.

## YAML subset

Parsed with a built-in strict-subset parser (PyYAML not required): 2-space
indentation, block maps and `- ` lists, plain/quoted scalars, flow lists of
scalars like `[0.85, 1.05]`, `#` comments. No anchors, no multi-line
strings, no flow maps (`{...}`). JSON specs (`.json`) are always accepted.

## Bench protocol

`examples/L*/` items ship `manifest.txt` + `spec.yaml`. Faithful to the
paper's evaluation: generation sees **only** the manifest (and reference
image where present) — do not open `spec.yaml` until `model.glb` exists;
then score, and failures remain in the report. Reading the spec first is
benchmark contamination, not diligence.
