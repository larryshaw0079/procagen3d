"""ProcAgen3D canonical modeling runtime, vendored into delivered programs.

Authoring programs may import selected helpers from ``procagen3d_runtime``.
The stdlib driver replaces that import with this source before Blender runs,
so the retained ``program.py`` remains deterministic and self-contained.
"""

import math

import bpy
from mathutils import Vector


PROCAGEN3D_RUNTIME_VERSION = "1.0.0"

__all__ = (
    "PROCAGEN3D_RUNTIME_VERSION",
    "activate_only",
    "add_joint",
    "apply_transform",
    "box",
    "cylinder_between",
    "ellipsoid",
    "loft_rings",
    "make_material",
    "mark_form",
    "new_group",
    "reparent_keep_world",
    "revolve_profile",
    "sweep_profile",
)


def activate_only(obj):
    """Make ``obj`` the sole selected and active object for operator safety."""

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    return obj


def apply_transform(obj, *, location=False, rotation=False, scale=False):
    """Apply explicitly selected transform channels without changing defaults.

    Requiring explicit keyword choices at each call site avoids the failure
    where ``transform_apply(scale=True)`` also uses Blender's true defaults
    for location and rotation, baking positioned mesh data around the world
    origin before a later rotation.
    """

    choices = (location, rotation, scale)
    if not all(isinstance(value, bool) for value in choices):
        raise TypeError("location, rotation, and scale must be booleans")
    activate_only(obj)
    bpy.ops.object.transform_apply(
        location=location,
        rotation=rotation,
        scale=scale,
    )
    return obj


def make_material(
    name,
    color,
    roughness=0.6,
    metallic=0.0,
    *,
    alpha=1.0,
    transmission=0.0,
):
    """Create or reuse one Principled material with stable viewport color."""

    rgb = tuple(float(value) for value in color[:3])
    if len(rgb) != 3:
        raise ValueError("color must contain at least three channels")
    alpha = float(alpha)
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is None:
        raise RuntimeError(f"material {name!r} has no Principled BSDF node")
    bsdf.inputs["Base Color"].default_value = (*rgb, alpha)
    bsdf.inputs["Roughness"].default_value = float(roughness)
    bsdf.inputs["Metallic"].default_value = float(metallic)
    if "Alpha" in bsdf.inputs:
        bsdf.inputs["Alpha"].default_value = alpha
    transmission_input = bsdf.inputs.get("Transmission Weight")
    if transmission_input is None:
        transmission_input = bsdf.inputs.get("Transmission")
    if transmission_input is not None:
        transmission_input.default_value = float(transmission)
    mat.diffuse_color = (*rgb, alpha)
    return mat


def new_group(name, *, parent=None, location=(0.0, 0.0, 0.0)):
    """Create a semantic EMPTY assembly node."""

    group = bpy.data.objects.new(name, None)
    group.location = location
    bpy.context.scene.collection.objects.link(group)
    if parent is not None:
        reparent_keep_world(group, parent)
    return group


def reparent_keep_world(obj, new_parent):
    """Change parent without moving the object in world space."""

    bpy.context.view_layer.update()
    matrix_world = obj.matrix_world.copy()
    obj.parent = new_parent
    bpy.context.view_layer.update()
    obj.matrix_world = matrix_world
    return obj


def mark_form(obj, role, topology, method, section_count=None):
    """Attach the semantic form contract consumed by ``procagen3d check``."""

    obj["procagen3d_form_role"] = str(role)
    obj["procagen3d_topology"] = str(topology)
    obj["procagen3d_form_method"] = str(method)
    if section_count is not None:
        obj["procagen3d_section_count"] = int(section_count)
    return obj


def _assign_material(obj, material):
    if material is not None:
        obj.data.materials.append(material)


def _add_bevel(obj, width, segments):
    width = float(width)
    if width <= 0.0:
        return
    modifier = obj.modifiers.new("Edge_Bevel", "BEVEL")
    modifier.width = width
    modifier.segments = max(1, int(segments))


def box(
    name,
    size,
    location,
    material=None,
    *,
    rotation=(0.0, 0.0, 0.0),
    bevel=0.0,
    bevel_segments=2,
    form=None,
):
    """Build a named box while preserving its center as the rotation origin."""

    dimensions = tuple(float(value) for value in size)
    if len(dimensions) != 3 or any(value <= 0.0 for value in dimensions):
        raise ValueError("box size must contain three positive values")
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dimensions
    apply_transform(obj, location=False, rotation=False, scale=True)
    obj.rotation_euler = rotation
    _assign_material(obj, material)
    _add_bevel(obj, bevel, bevel_segments)
    if form is not None:
        mark_form(obj, *form)
    return obj


def ellipsoid(
    name,
    size,
    location,
    material=None,
    *,
    rotation=(0.0, 0.0, 0.0),
    segments=48,
    ring_count=24,
    smooth=True,
    form=None,
):
    """Build a named UV ellipsoid with applied scale and a stable origin."""

    dimensions = tuple(float(value) for value in size)
    if len(dimensions) != 3 or any(value <= 0.0 for value in dimensions):
        raise ValueError("ellipsoid size must contain three positive values")
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=max(8, int(segments)),
        ring_count=max(4, int(ring_count)),
        radius=1.0,
        location=location,
    )
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dimensions
    apply_transform(obj, location=False, rotation=False, scale=True)
    obj.rotation_euler = rotation
    _assign_material(obj, material)
    if smooth:
        for polygon in obj.data.polygons:
            polygon.use_smooth = True
    if form is not None:
        mark_form(obj, *form)
    return obj


def cylinder_between(
    name,
    start,
    end,
    radius,
    material=None,
    *,
    vertices=24,
    bevel=0.0,
    bevel_segments=2,
    form=None,
):
    """Build a cylinder whose local +Z axis runs from ``start`` to ``end``."""

    start = Vector(start)
    end = Vector(end)
    direction = end - start
    depth = direction.length
    if depth <= 1e-9 or float(radius) <= 0.0:
        raise ValueError("cylinder_between needs distinct endpoints and positive radius")
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=max(8, int(vertices)),
        radius=float(radius),
        depth=depth,
        location=(start + end) * 0.5,
    )
    obj = bpy.context.object
    obj.name = name
    obj.rotation_euler = direction.to_track_quat("Z", "Y").to_euler()
    _assign_material(obj, material)
    _add_bevel(obj, bevel, bevel_segments)
    if form is not None:
        mark_form(obj, *form)
    return obj


def loft_rings(
    name,
    rings,
    material=None,
    *,
    cap=True,
    subdivision=0,
    role="primary",
    topology="continuous",
):
    """Create one continuous mesh through equal-size authored section rings."""

    rings = [[Vector(point) for point in ring] for ring in rings]
    if len(rings) < 2 or len(rings[0]) < 3:
        raise ValueError("loft needs at least two rings with three points each")
    ring_size = len(rings[0])
    if any(len(ring) != ring_size for ring in rings):
        raise ValueError("all loft rings must have equal point counts")
    vertices = [tuple(point) for ring in rings for point in ring]
    faces = []
    for row in range(len(rings) - 1):
        first = row * ring_size
        following = (row + 1) * ring_size
        for column in range(ring_size):
            nxt = (column + 1) % ring_size
            faces.append((
                first + column,
                first + nxt,
                following + nxt,
                following + column,
            ))
    if cap:
        faces.append(tuple(reversed(range(ring_size))))
        final = (len(rings) - 1) * ring_size
        faces.append(tuple(final + column for column in range(ring_size)))

    mesh = bpy.data.meshes.new(name + "_Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.validate(verbose=True)
    mesh.update(calc_edges=True)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    _assign_material(obj, material)
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    if subdivision:
        modifier = obj.modifiers.new("Form_Subdivision", "SUBSURF")
        modifier.subdivision_type = "CATMULL_CLARK"
        modifier.levels = int(subdivision)
        modifier.render_levels = int(subdivision)
    mark_form(obj, role, topology, "loft", len(rings))
    obj["procagen3d_ring_points"] = ring_size
    return obj


def sweep_profile(
    name,
    spine,
    profile,
    scales,
    material=None,
    *,
    role="primary",
    topology="continuous",
):
    """Sweep one 2D profile through a parallel-transported varying spine."""

    spine = [Vector(point) for point in spine]
    profile = [tuple(float(value) for value in point) for point in profile]
    if len(spine) < 2 or len(scales) != len(spine):
        raise ValueError("sweep needs matching spine and scale stations")
    if len(profile) < 3 or any(len(point) != 2 for point in profile):
        raise ValueError("sweep profile needs at least three 2D points")

    tangents = []
    for index in range(len(spine)):
        before = spine[max(0, index - 1)]
        after = spine[min(len(spine) - 1, index + 1)]
        delta = after - before
        if delta.length <= 1e-9:
            raise ValueError("sweep spine contains a zero-length station")
        tangents.append(delta.normalized())

    reference = Vector((0.0, 0.0, 1.0))
    if abs(tangents[0].dot(reference)) > 0.95:
        reference = Vector((1.0, 0.0, 0.0))
    normal = (reference - tangents[0] * reference.dot(tangents[0])).normalized()
    previous = tangents[0]
    rings = []
    for center, tangent, scale in zip(spine, tangents, scales):
        normal = previous.rotation_difference(tangent) @ normal
        normal = (normal - tangent * normal.dot(tangent)).normalized()
        binormal = tangent.cross(normal).normalized()
        if isinstance(scale, (int, float)):
            scale_u = scale_v = float(scale)
        else:
            scale_u, scale_v = (float(value) for value in scale)
        rings.append([
            center + normal * (u * scale_u) + binormal * (v * scale_v)
            for u, v in profile
        ])
        previous = tangent
    obj = loft_rings(
        name,
        rings,
        material,
        role=role,
        topology=topology,
    )
    mark_form(obj, role, topology, "sweep", len(rings))
    return obj


def _revolve_point(axis, radial_x, radial_y, axial):
    if axis == "X":
        return (axial, radial_x, radial_y)
    if axis == "Y":
        return (radial_x, axial, radial_y)
    return (radial_x, radial_y, axial)


def revolve_profile(
    name,
    profile,
    material=None,
    *,
    axis="Z",
    segments=64,
    cap=True,
    smooth=True,
    role="primary",
    topology="continuous",
):
    """Revolve ``(radius, axial)`` stations into an editable profile mesh."""

    axis = str(axis).upper()
    if axis not in {"X", "Y", "Z"}:
        raise ValueError("revolve axis must be X, Y, or Z")
    profile = [(float(radius), float(axial)) for radius, axial in profile]
    if len(profile) < 2 or any(radius < 0.0 for radius, _ in profile):
        raise ValueError("revolve profile needs two or more non-negative radii")
    segments = max(8, int(segments))
    vertices = []
    rings = []
    for radius, axial in profile:
        if radius <= 1e-9:
            rings.append([len(vertices)])
            vertices.append(_revolve_point(axis, 0.0, 0.0, axial))
            continue
        ring = []
        for step in range(segments):
            angle = 2.0 * math.pi * step / segments
            ring.append(len(vertices))
            vertices.append(_revolve_point(
                axis,
                radius * math.cos(angle),
                radius * math.sin(angle),
                axial,
            ))
        rings.append(ring)

    faces = []
    for lower, upper in zip(rings, rings[1:]):
        if len(lower) == 1 and len(upper) == 1:
            continue
        if len(lower) == 1:
            for step in range(segments):
                nxt = (step + 1) % segments
                faces.append((lower[0], upper[step], upper[nxt]))
        elif len(upper) == 1:
            for step in range(segments):
                nxt = (step + 1) % segments
                faces.append((lower[step], lower[nxt], upper[0]))
        else:
            for step in range(segments):
                nxt = (step + 1) % segments
                faces.append((lower[step], lower[nxt], upper[nxt], upper[step]))

    if cap and len(rings[0]) > 1:
        center = len(vertices)
        vertices.append(_revolve_point(axis, 0.0, 0.0, profile[0][1]))
        for step in range(segments):
            nxt = (step + 1) % segments
            faces.append((center, rings[0][nxt], rings[0][step]))
    if cap and len(rings[-1]) > 1:
        center = len(vertices)
        vertices.append(_revolve_point(axis, 0.0, 0.0, profile[-1][1]))
        for step in range(segments):
            nxt = (step + 1) % segments
            faces.append((center, rings[-1][step], rings[-1][nxt]))

    mesh = bpy.data.meshes.new(name + "_Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.validate(verbose=True)
    mesh.update(calc_edges=True)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    _assign_material(obj, material)
    if smooth:
        for polygon in mesh.polygons:
            polygon.use_smooth = True
    mark_form(obj, role, topology, "revolve", len(profile))
    obj["procagen3d_ring_points"] = segments
    return obj


def add_joint(name, parent, child, jtype, axis, limits=None, origin=None):
    """Create the canonical parent → joint → moving-child transform chain."""

    joint = bpy.data.objects.new(name, None)
    joint.empty_display_type = "ARROWS"
    joint.empty_display_size = 0.05
    bpy.context.scene.collection.objects.link(joint)
    joint.location = (
        Vector(origin) if origin is not None else child.matrix_world.translation.copy()
    )
    joint["procagen3d_joint_type"] = str(jtype)
    joint["procagen3d_joint_axis"] = [float(value) for value in axis]
    if limits is not None:
        joint["procagen3d_joint_limits"] = [float(value) for value in limits]
    joint["procagen3d_joint_child"] = child.name
    joint["procagen3d_joint_parent"] = parent.name if parent else ""
    if parent is not None:
        reparent_keep_world(joint, parent)
    reparent_keep_world(child, joint)
    return joint
