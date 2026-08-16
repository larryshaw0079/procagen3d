"""Exercise every public geometry path in the canonical modeling runtime."""

import math

from procagen3d_runtime import (
    add_joint,
    box,
    cylinder_between,
    ellipsoid,
    loft_rings,
    make_material,
    new_group,
    reparent_keep_world,
    revolve_profile,
    sweep_profile,
)


def build():
    root = new_group("RuntimeSmoke")
    root["procagen3d_form_profile"] = "mixed"
    red = make_material("RuntimeRed", (0.65, 0.08, 0.04), roughness=0.45)
    blue = make_material("RuntimeBlue", (0.04, 0.15, 0.55), metallic=0.25)

    offset_box = box(
        "OffsetBox",
        (1.2, 0.5, 0.4),
        (2.0, 0.0, 0.3),
        red,
        rotation=(0.0, 0.0, math.radians(35.0)),
        bevel=0.025,
        form=("secondary", "assembled", "primitive-csg"),
    )
    add_joint(
        "OffsetBoxJoint",
        root,
        offset_box,
        "revolute",
        (0.0, 0.0, 1.0),
        (-25.0, 25.0),
        origin=(2.0, 0.0, 0.3),
    )

    parts = [
        ellipsoid(
            "RuntimeEllipsoid",
            (0.7, 0.5, 0.5),
            (0.8, 0.0, 0.3),
            blue,
            form=("secondary", "assembled", "primitive-csg"),
        ),
        cylinder_between(
            "RuntimeCylinder",
            (-0.2, -0.35, 0.1),
            (-0.2, 0.35, 0.8),
            0.1,
            red,
            form=("secondary", "assembled", "primitive-csg"),
        ),
        loft_rings(
            "RuntimeLoft",
            [
                [(-1.3, -0.25, 0.0), (-1.3, 0.25, 0.0),
                 (-1.3, 0.25, 0.45), (-1.3, -0.25, 0.45)],
                [(-0.9, -0.32, 0.0), (-0.9, 0.32, 0.0),
                 (-0.9, 0.32, 0.58), (-0.9, -0.32, 0.58)],
                [(-0.5, -0.18, 0.0), (-0.5, 0.18, 0.0),
                 (-0.5, 0.18, 0.35), (-0.5, -0.18, 0.35)],
            ],
            blue,
        ),
        sweep_profile(
            "RuntimeSweep",
            [(-1.2, -0.7, 0.7), (-0.8, -0.7, 0.95), (-0.35, -0.7, 0.75)],
            [(-1.0, -0.7), (1.0, -0.7), (1.0, 0.7), (-1.0, 0.7)],
            [0.12, (0.16, 0.12), 0.08],
            red,
        ),
        revolve_profile(
            "RuntimeRevolve",
            [(0.0, 0.0), (0.28, 0.0), (0.34, 0.18),
             (0.22, 0.48), (0.0, 0.58)],
            blue,
        ),
    ]
    parts[-1].location = (0.25, 0.85, 0.0)
    for part in parts:
        reparent_keep_world(part, root)
    return root


if __name__ == "__main__":
    build()
