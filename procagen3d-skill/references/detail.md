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
efficient. These floors measure secondary construction/detail coverage, not
the quality of a primary surface: never fragment one coherent body shell into
box/ellipsoid patches just to raise mesh count.

## The decomposition ladder

Decompose until the smallest separate part is ~1% of the object's major
dimension (a door handle on a car, a knob on an amp). Four levels, all named:

    assembly (Wheel_FL) → part (Rim) → sub-part (Rim_Lip, Lug_1..5)
    → surface feature (Tread_00..19, Door_Seam, Badge letters)

If you can name a sub-feature, it gets its own `build_*` call and mesh.
Repetition is always an **array of numbered instances** (tread lugs, grille
slats, bed grooves, coil rings, spokes, vents) — count them in the reference,
place them with a loop. Keep a continuous primary mass continuous; panel gaps,
trim, and attached shells may be separate, but the surface beneath them must
not become a collage of independently beveled primitives.

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

For compound curvature, changing cross-sections, streamlined bodies, or
irregular armor, MUST read `complex-forms.md` and pass its shape-first probe.
Choose topology before helper: `continuous` → loft/sweep/revolve/subdivision;
`shell` → surface patch/loft + thickness; deliberately faceted `assembled`
parts → profile extrusion/CSG. A Bevel modifier changes edge treatment, not
the underlying form family.

- **Segment floors**: silhouette-defining radii ≥ 48 segments; mid-size
  (fists to plates) ≥ 24; bolt-scale ≥ 12. Never leave a default 32-seg
  cylinder on a large visible radius; never spend 64 on a lug nut.
- **Bevel rule**: every visible hard edge gets a bevel — modifier, width
  0.5–2 % of the part's major dimension, 2–3 segments. A raw
  `primitive_cube_add` edge is the single strongest "toy" tell. (Remember
  the transform_apply trap: bake modifiers via the harness, not by hand.)
- **Body masses**: author measured section/spine/grid controls; use booleans
  only after the envelope passes. Cut a real wheel-arch opening, then add a
  conforming flare or swept lip around it.
- **Revolved forms** (bottles, hubs, bells): spin/lathe an explicit profile;
  do not approximate a visible S-curve with stacked cylinders.
- **Topology budget**: spend control loops at silhouette extrema, changing
  curvature, hard creases, openings, and attachment roots. Uniform density
  and high evaluated triangle count are not evidence of designed form.

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
