# Detail doctrine — fine-grained geometry

Coarse output is the #1 failure mode of this skill: correct part lists built
as unbeveled boxes and 16-segment cylinders read as toys. Detail is
**scheduled work with its own pass and its own gate**, never leftover budget.
Reference forensics (real Nova3D vehicle, same input image): 351 meshes,
~99k tris, 17 materials — one wheel alone is 34 named parts.

## Detail tiers

Declare the tier in the program header at Design time; it sets the floors
the `check` gate warns against (`--tier`).

| tier | when | mesh floor | tri floor | materials |
|------|------|-----------|-----------|-----------|
| quick | throwaway drafts, explicit "rough" requests | — | — | — |
| standard | text-only prompts, functional assets | 40 | 8 000 | 6+ |
| showcase | image-conditioned replication (DEFAULT), "detailed/realistic" asks | 150 | 25 000 | 12+ |

Tris up to ~150k build and render fine headless — never "optimize" below the
tier floor. A showcase asset that comes back under floor is unfinished, not
efficient.

## The decomposition ladder

Decompose until the smallest separate part is ~1% of the object's major
dimension (a door handle on a car, a knob on an amp). Four levels, all named:

    assembly (Wheel_FL) → part (Rim) → sub-part (Rim_Lip, Lug_1..5)
    → surface feature (Tread_00..19, Door_Seam, Badge letters)

If you can name a sub-feature, it gets its own `build_*` call and mesh.
Repetition is always an **array of numbered instances** (tread lugs, grille
slats, bed grooves, coil rings, spokes, vents) — count them in the reference,
place them with a loop.

## Recipes (patterns, not just vehicles)

- **Wheel/roller**: tire casing (48–64 seg), 16–24 tread lugs radially
  arrayed, rim dish, rim lip torus, 4–6 rim holes (dark inset cylinders are
  cheaper than booleans and read the same), hub, lug nuts. ≈ 30+ parts.
- **Lamp/light**: housing + bezel + lens, three materials (paint / chrome or
  satin / emissive-ish glass). Lens sits 1–2 mm proud.
- **Mirror / small appendage**: housing + stem + glass. Never one lump.
- **Grille / vent / radiator**: surround housing + N slats from the image.
- **Panel seams**: doors, hoods, hatches get 1–2 mm dark recessed strips (or
  proud trim strips) along their edges — this single trick makes flat sides
  read as assembled bodywork.
- **Graphics, stripes, badges**: thin plates (1–3 mm) lying 0.5–1 mm proud
  of the surface, own material. Lettering: per-letter blocks or extruded
  text-to-mesh, shallow depth — see the reference asset's tailgate.
- **Undercarriage / underside** (anything with ground clearance): chassis
  rails + crossmembers, axles, differentials, driveshaft, exhaust with tip.
  They are visible in side view at stance height; their absence is why
  models "float".
- **Interior behind glass**: if glazing is transparent in the reference,
  block in seat masses, dashboard, steering wheel — silhouettes through
  glass carry realism cheaply.
- **Springs/coils**: stack of numbered torus rings beats a curve-screw for
  editability and reads identically at render scale.

## Curvature and form

Constructive modeling handles curved masses fine — build them, don't hedge:

- **Segment floors**: silhouette-defining radii ≥ 48 segments; mid-size
  (fists to plates) ≥ 24; bolt-scale ≥ 12. Never leave a default 32-seg
  cylinder on a large visible radius; never spend 64 on a lug nut.
- **Bevel rule**: every visible hard edge gets a bevel — modifier, width
  0.5–2 % of the part's major dimension, 2–3 segments. A raw
  `primitive_cube_add` edge is the single strongest "toy" tell. (Remember
  the transform_apply trap: bake modifiers via the harness, not by hand.)
- **Body masses**: box → bevel → taper/shear via direct vertex transforms or
  Simple Deform; wedge cuts and window openings via boolean; wheel-arch
  openings: boolean cylinder cut, then a **flare** (torus segment or swept
  arc) proud around the opening.
- **Revolved forms** (bottles, hubs, bells): spin/lathe a profile or stack
  primitives; **lofted panels** (fenders, hulls): bezier profile + skin, or
  subdivided box with proportional-edit displacement. Say "stylized" only
  when the request is genuinely organic (faces, animals, cloth).

## Materials

One material per distinct real-world finish, named by finish, not by color
slot: `Off-road Rubber`, `Machined Silver`, `Smoked Glass`, `Signal Amber`,
`Satin Black Trim`. Showcase floor is 12; a two-tone paint job is two
materials; lens colors (amber / red / clear) are separate; chrome vs satin
black vs body paint always split. Generic `Black`/`Glass`/`Paint` palettes
cap perceived quality regardless of geometry.

## Detail inventory discipline

The detail pass builds exactly what `priors.md`'s detail inventory lists
(see image-analysis.md §7): sweep the reference region by region, enumerate
sub-features with sizes, then check each one off in the render. A feature
visible in the reference and absent in the sheet is a **detail verdict
failure** even when silhouette, scale, and part coverage all pass.
