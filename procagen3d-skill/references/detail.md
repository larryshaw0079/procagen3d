# Detail doctrine — fine-grained geometry

Coarse output is a primary failure mode: correct part lists can still read as
toys, while dense repeated arrays can make a simple object exceed numeric
floors without adding identity. Detail is **scheduled work with a feature plan
and its own gate**, never leftover budget. Mesh/triangle/material floors are
only sanity checks; `reconstruction_plan.json` feature coverage is the real
completion contract for image-conditioned work.

## Contents

1. Detail tiers and adaptive floors
2. Decomposition ladder
3. Reusable recipes
4. Curvature and form
5. Materials
6. Inventory discipline

## Detail tiers

Declare tier and perceptual complexity in the program header at Design time.
Image-conditioned replication and “detailed/realistic” asks default to
`showcase`; `quick` has no floors. Count visible feature groups using
`reconstruction-planning.md`, then use the adaptive floors enforced by `check`:

| tier | complexity | mesh floor | tri floor | materials |
|------|------------|-----------:|----------:|----------:|
| standard | simple / moderate | 24 / 50 | 5k / 10k | 4 / 6 |
| standard | complex / extreme | 100 / 160 | 20k / 32k | 8 / 10 |
| showcase | simple / moderate | 60 / 140 | 12k / 28k | 8 / 10 |
| showcase | complex / extreme | 260 / 420 | 55k / 90k | 14 / 16 |

Legacy assets without a reconstruction plan keep fallback floors (standard
40/8k/6; showcase 150/25k/12). A version-2 showcase miss is a hard failure.
Never fragment a coherent surface or inflate a repeated array merely to reach a
number; fix missing feature groups first.

Global counts are the weakest possible detail signal — 260 meshes says nothing
about *where* they are, and a complex object hits that number easily while its
identity regions stay empty. The binding constraint is `REGION_DENSITY`, which
bins every mesh by where its centre actually sits in the object's 3×3 grid and
requires 2/4/8/12 meshes in each declared occupied region. A blob torso fails
it no matter how many panel slivers sit on the legs.

## The decomposition ladder

Decompose until the smallest separate part is ~1% of the object's major
dimension (a door handle on a car, a knob on an amp). Four levels, all named:

    assembly (Wheel_FL) → part (Rim) → sub-part (Rim_Lip, Lug_1..5)
    → surface feature (Tread_00..19, Door_Seam, Badge letters)

If you can name a sub-feature, it gets its own `build_*` call and mesh.
Repetition is always an **array of numbered instances** (tread lugs, grille
slats, bed grooves, coil rings, spokes, vents) — count them in the reference,
place them with a loop. One builder, shared dimension constants, and only the
transform varies per instance; `INSTANCE_CONGRUENCE` measures rotation-invariant
size across each array and fails a six-plus-member array that drifts past 25%
without a `reconstruction_plan.instance_arrays` entry explaining why. Sizing each
member by eye against the reference silhouette is how eighteen identical swords
become eighteen different swords. Keep a continuous primary mass continuous; panel gaps,
trim, and attached shells may be separate, but the surface beneath them must
not become a collage of independently beveled primitives.

For image-conditioned work, every visible group in the ladder is also a
`detail_features` entry with a semantic `pattern`, `min_count`, priority, and
object-centric 3×3 region. Repetition is one group with an honest count.
Complete identity and structural groups before microdetail. Declare the
actually occupied object regions in `complexity.occupied_regions`; the checker
requires coverage in at least 1/3/5/6 regions for
simple/moderate/complex/extreme work and rejects a declared occupied region
that has no required visible group. This prevents a dense wheel, torso, or
receiver from numerically excusing blank regions elsewhere.

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
`Satin Black Trim`. Use the adaptive tier floor; a two-tone paint job is two
materials, lens colors (amber / red / clear) are separate, and chrome vs satin
black vs body paint always split. Generic `Black`/`Glass`/`Paint` palettes cap
perceived quality regardless of geometry.

## Detail inventory discipline

The detail pass builds exactly what `priors.md` and
`reconstruction_plan.json` list (see image-analysis.md §7): sweep the reference
region by region, enumerate feature groups with sizes/counts, then check each
one in the render. The checker verifies semantic pattern/count coverage; vision
verifies that the geometry actually reads correctly. A visible feature absent
from the sheet is a **detail verdict failure** even when numeric floors,
silhouette, scale, and part coverage pass.
