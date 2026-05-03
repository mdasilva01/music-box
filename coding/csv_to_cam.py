#!/usr/bin/env python3
"""
cam_from_csv_smooth.py

Create a smooth barrel-cam STL from a 2-column CSV trajectory.

This version avoids marching cubes/SDF voxelization.  The cam is built as an
analytic swept mesh, so the edges are smooth instead of blocky/jagged.  The
shaft/hub defaults are matched to 88.05% of the attached reference STL dimensions:

    reference scale factor: 0.8805
    measured reference axis length: 52.6173820496 mm
    locked scaled axis length:      46.3296048946 mm
    small end shaft radius:         ~4.41 mm
    larger hub radius:              ~6.62 mm
    square-hole diagonal:           ~6.47 mm

The shaft X span is locked so the distance from one peg end to the other is
always exactly 88.05% of the reference STL's measured X-axis distance.

The cam track is intentionally lipped/captive.  The groove is modeled as a
round ball path whose top opening is narrower than the ball diameter.  A ball
follower can sit inside the track, while the small overhanging lips help keep it
from lifting straight out.  Control this with --lip-cover / --top-cover.

Usage:
    python cam_from_csv_smooth.py trajectory.csv -o cam.stl --preview cam.png

CSV format:
    x,y
    x,y
    ...
No header.  The row order is interpreted as one full revolution around the cam.
"""

from __future__ import annotations

import argparse
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------

REFERENCE_SCALE = 0.8805

# Measured directly from the attached reference STL bounding box along the shaft/X axis:
#   x_min = -25.7891197205 mm
#   x_max =  26.8282623291 mm
#   length = 52.6173820496 mm
# Scaled to 88.05% for holder fit.
REFERENCE_AXIS_LENGTH = 52.61738204956055
LOCKED_AXIS_LENGTH = REFERENCE_AXIS_LENGTH * REFERENCE_SCALE
LOCKED_AXIS_HALF_LENGTH = LOCKED_AXIS_LENGTH * 0.5

# Keep the same scaled peg/shoulder distances used in the 88.05% reference match.
# These are distances measured along X from the peg end inward.
LOCKED_SMALL_PEG_LENGTH = 4.76651
LOCKED_HUB_BLEND_LENGTH = 1.45282
LOCKED_SMALL_SHAFT_HALF_START = LOCKED_AXIS_HALF_LENGTH - LOCKED_SMALL_PEG_LENGTH
LOCKED_HUB_FULL_HALF_END = LOCKED_SMALL_SHAFT_HALF_START - LOCKED_HUB_BLEND_LENGTH


@dataclass
class CamParams:
    # Reference-style shaft / hub dimensions, in mm.
    # These are 0.8805x the dimensions measured from the reference STL.
    shaft_radius: float = 4.40954
    hub_radius: float = 6.61608
    shaft_half_length: float = LOCKED_AXIS_HALF_LENGTH
    small_shaft_half_start: float = LOCKED_SMALL_SHAFT_HALF_START  # small shaft starts at |x| >= this
    hub_full_half_end: float = LOCKED_HUB_FULL_HALF_END       # full hub radius for |x| <= this
    square_hole_diagonal: float = 6.47168     # diamond diagonal in the Y/Z view

    # Cam-body dimensions, in mm. These control the generated moving cam lobe.
    axial_travel: float = 12.32700          # x-center travel range from CSV x
    track_width: float = 7.04400           # axial width of the swept cam track
    cam_inner_radius: float = 6.47168      # slightly overlaps the hub radius
    radius_min: float = 16.72950           # groove center radius at min CSV y
    radius_max: float = 27.73575           # groove center radius at max CSV y
    groove_radius: float = 2.20125         # rounded groove / roller radius
    top_cover: float = 0.65                # lip amount: top surface is r + top_cover*g_r

    # Mesh smoothness.
    angular_segments: int = 720            # around the barrel
    circle_segments: int = 128             # shaft/hub circular resolution
    groove_arc_segments: int = 24
    top_flat_segments: int = 4
    blend_steps: int = 18                  # rounded radius transitions on shaft


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------

def normalize_to_range(values: np.ndarray, out_min: float, out_max: float) -> np.ndarray:
    """Linearly normalize values into [out_min, out_max]."""
    values = np.asarray(values, dtype=float)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if abs(hi - lo) < 1e-12:
        return np.full_like(values, (out_min + out_max) * 0.5, dtype=float)
    u = (values - lo) / (hi - lo)
    return out_min + u * (out_max - out_min)


def smoothstep(u: np.ndarray | float) -> np.ndarray | float:
    """C1 smooth blend from 0 to 1."""
    return np.asarray(u) * np.asarray(u) * (3.0 - 2.0 * np.asarray(u))


def resample_closed_xy(points: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Resample an ordered closed CSV trajectory to exactly n samples."""
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("CSV must contain at least two numeric columns: x,y")
    if len(points) < 3:
        raise ValueError("Need at least 3 trajectory points")

    src_t = np.linspace(0.0, 1.0, len(points), endpoint=False)
    dst_t = np.linspace(0.0, 1.0, n, endpoint=False)

    # Periodic extension so interpolation wraps cleanly at t = 1.
    src_t_ext = np.r_[src_t, 1.0]
    x_ext = np.r_[points[:, 0], points[0, 0]]
    y_ext = np.r_[points[:, 1], points[0, 1]]

    x = np.interp(dst_t, src_t_ext, x_ext)
    y = np.interp(dst_t, src_t_ext, y_ext)
    return x, y


def tri_normal(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    n = np.cross(v1 - v0, v2 - v0)
    nn = np.linalg.norm(n)
    if nn <= 1e-12:
        return np.zeros(3, dtype=float)
    return n / nn


def write_binary_stl(path: str | Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Write a binary STL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(b"smooth csv barrel cam".ljust(80, b"\0"))
        f.write(struct.pack("<I", len(faces)))
        for face in faces:
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            n = tri_normal(v0, v1, v2)
            f.write(struct.pack("<3f", *n))
            f.write(struct.pack("<3f", *v0))
            f.write(struct.pack("<3f", *v1))
            f.write(struct.pack("<3f", *v2))
            f.write(struct.pack("<H", 0))


def append_mesh(
    all_vertices: list[np.ndarray],
    all_faces: list[np.ndarray],
    vertices: np.ndarray,
    faces: np.ndarray,
) -> None:
    offset = 0 if not all_vertices else sum(len(v) for v in all_vertices)
    all_vertices.append(vertices)
    all_faces.append(faces + offset)


# -----------------------------------------------------------------------------
# Cam body: swept concave cross-section around the x axis
# -----------------------------------------------------------------------------

def build_cam_section(params: CamParams) -> list[tuple[float, str, float]]:
    """
    Return cross-section boundary points in local axial/radial coordinates.

    Each entry is (dx, mode, value):
      - mode == "fixed": radius = value
      - mode == "offset": radius = groove_center_radius + value

    The section is roughly a radial block with a circular bite/groove on top,
    matching what the old sphere subtraction tried to do, but without voxels.
    """
    hw = params.track_width * 0.5
    gr = params.groove_radius
    top = params.top_cover * gr

    if not (0.0 < top < gr):
        raise ValueError("top_cover must satisfy 0 < top_cover < 1")

    x_intersect = math.sqrt(max(gr * gr - top * top, 0.0))
    if x_intersect >= hw:
        raise ValueError(
            "track_width is too small for the groove_radius/top_cover. "
            "Increase --track-width or decrease --groove-radius."
        )

    section: list[tuple[float, str, float]] = []

    # Inner/lower boundary, slightly inside the hub so the cam visually joins it.
    section.append((-hw, "fixed", params.cam_inner_radius))
    section.append(( hw, "fixed", params.cam_inner_radius))

    # Right vertical wall up to the covered top surface.
    section.append(( hw, "offset", top))

    # Flat top from right side to the groove/cutter intersection.
    for u in np.linspace(0.0, 1.0, params.top_flat_segments + 1)[1:]:
        x = hw + u * (x_intersect - hw)
        section.append((float(x), "offset", top))

    # Concave circular groove arc: from right top intersection, through the
    # lower part of the circle, to the left top intersection.
    phi = math.asin(top / gr)
    angles = np.linspace(phi, -(math.pi + phi), params.groove_arc_segments)
    for a in angles[1:]:  # first point is already on the top segment
        dx = gr * math.cos(float(a))
        dr = gr * math.sin(float(a))
        section.append((dx, "offset", dr))

    # Flat top from groove intersection back to the left side.
    for u in np.linspace(0.0, 1.0, params.top_flat_segments + 1)[1:]:
        x = -x_intersect + u * (-hw + x_intersect)
        section.append((float(x), "offset", top))

    # Closing edge from left top to left inner happens automatically when meshed.
    return section


def groove_lip_dimensions(params: CamParams) -> tuple[float, float, float]:
    """
    Return useful captive-track dimensions.

    Returns:
        ball_diameter, slot_opening, lip_overhang_each_side

    The slot opening is the narrowest axial opening at the outer top surface.
    With top_cover > 0, slot_opening is smaller than the ball diameter, which
    creates the lipped/captive track.
    """
    gr = params.groove_radius
    slot_opening = 2.0 * gr * math.sqrt(max(1.0 - params.top_cover * params.top_cover, 0.0))
    ball_diameter = 2.0 * gr
    lip_each_side = 0.5 * (ball_diameter - slot_opening)
    return ball_diameter, slot_opening, lip_each_side


def locked_axis_dimensions(params: CamParams) -> tuple[float, float, float, float]:
    """Return locked axis length and shoulder distances in mm."""
    total_length = 2.0 * params.shaft_half_length
    small_peg_length = params.shaft_half_length - params.small_shaft_half_start
    hub_blend_length = params.small_shaft_half_start - params.hub_full_half_end
    full_hub_length = 2.0 * params.hub_full_half_end
    return total_length, small_peg_length, hub_blend_length, full_hub_length


def build_cam_body_mesh(x_csv: np.ndarray, y_csv: np.ndarray, params: CamParams) -> tuple[np.ndarray, np.ndarray]:
    """Build the smooth swept cam body from normalized CSV x/y."""
    n = params.angular_segments
    x_resampled, y_resampled = resample_closed_xy(np.column_stack([x_csv, y_csv]), n)

    x_center = normalize_to_range(
        x_resampled,
        -0.5 * params.axial_travel,
         0.5 * params.axial_travel,
    )
    r_center = normalize_to_range(y_resampled, params.radius_min, params.radius_max)
    section = build_cam_section(params)
    k = len(section)

    vertices = np.zeros((n * k, 3), dtype=float)

    for i in range(n):
        theta = 2.0 * math.pi * i / n
        c = math.cos(theta)
        s = math.sin(theta)
        for j, (dx, mode, val) in enumerate(section):
            radius = val if mode == "fixed" else r_center[i] + val
            idx = i * k + j
            vertices[idx] = [x_center[i] + dx, radius * c, radius * s]

    faces: list[list[int]] = []
    for i in range(n):
        ni = (i + 1) % n
        for j in range(k):
            nj = (j + 1) % k
            a = i * k + j
            b = ni * k + j
            c = ni * k + nj
            d = i * k + nj
            faces.append([a, b, c])
            faces.append([a, c, d])

    return vertices, np.asarray(faces, dtype=np.int64)


# -----------------------------------------------------------------------------
# Reference-style shaft/hub with a diamond/square hole
# -----------------------------------------------------------------------------

def radius_profile(params: CamParams) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a smooth turned-shaft radius profile.

    The profile is symmetric about x=0 and approximates the reference STL at 88.05% scale:
      small end shaft -> rounded blend -> larger hub through the cam.
    """
    xh = params.shaft_half_length
    xs: list[float] = []
    rs: list[float] = []

    def add(x: float, r: float) -> None:
        if xs and abs(xs[-1] - x) < 1e-9:
            rs[-1] = r
        else:
            xs.append(float(x))
            rs.append(float(r))

    # Left small shaft.
    add(-xh, params.shaft_radius)
    add(-params.small_shaft_half_start, params.shaft_radius)

    # Left smooth blend from small shaft to hub.
    blend_start = -params.small_shaft_half_start
    blend_end = -params.hub_full_half_end
    for u in np.linspace(0.0, 1.0, params.blend_steps + 1)[1:]:
        uu = float(smoothstep(u))
        x = blend_start + u * (blend_end - blend_start)
        r = params.shaft_radius + uu * (params.hub_radius - params.shaft_radius)
        add(x, r)

    # Large hub through the cam.
    add(params.hub_full_half_end, params.hub_radius)

    # Right smooth blend from hub back to small shaft.
    blend_start = params.hub_full_half_end
    blend_end = params.small_shaft_half_start
    for u in np.linspace(0.0, 1.0, params.blend_steps + 1)[1:]:
        uu = float(smoothstep(u))
        x = blend_start + u * (blend_end - blend_start)
        r = params.hub_radius + uu * (params.shaft_radius - params.hub_radius)
        add(x, r)

    # Right small shaft.
    add(xh, params.shaft_radius)

    return np.asarray(xs, dtype=float), np.asarray(rs, dtype=float)


def diamond_radius_at_angle(theta: float, diagonal: float) -> float:
    """
    Radius from the center to a diamond/square boundary.

    The diamond vertices lie on +Y, +Z, -Y, -Z.  Its full diagonal is `diagonal`.
    Equation in the Y/Z plane: |y| + |z| = diagonal / 2.
    """
    d = 0.5 * diagonal
    denom = abs(math.cos(theta)) + abs(math.sin(theta))
    return d / max(denom, 1e-12)


def build_axis_mesh(params: CamParams) -> tuple[np.ndarray, np.ndarray]:
    """Build the shaft/hub mesh with a square/diamond through-hole."""
    x_prof, r_prof = radius_profile(params)
    n_x = len(x_prof)
    n_a = params.circle_segments

    outer = np.zeros((n_x, n_a), dtype=np.int64)
    inner = np.zeros((n_x, n_a), dtype=np.int64)
    vertices: list[list[float]] = []

    for i, (x, r) in enumerate(zip(x_prof, r_prof)):
        for j in range(n_a):
            theta = 2.0 * math.pi * j / n_a
            y = r * math.cos(theta)
            z = r * math.sin(theta)
            outer[i, j] = len(vertices)
            vertices.append([x, y, z])

        for j in range(n_a):
            theta = 2.0 * math.pi * j / n_a
            rho = diamond_radius_at_angle(theta, params.square_hole_diagonal)
            y = rho * math.cos(theta)
            z = rho * math.sin(theta)
            inner[i, j] = len(vertices)
            vertices.append([x, y, z])

    faces: list[list[int]] = []

    # Outer turned surface.
    for i in range(n_x - 1):
        for j in range(n_a):
            jn = (j + 1) % n_a
            a = outer[i, j]
            b = outer[i + 1, j]
            c = outer[i + 1, jn]
            d = outer[i, jn]
            faces.append([a, b, c])
            faces.append([a, c, d])

    # Inner square-hole wall. Reverse orientation so normals face into the hole.
    for i in range(n_x - 1):
        for j in range(n_a):
            jn = (j + 1) % n_a
            a = inner[i, j]
            b = inner[i, jn]
            c = inner[i + 1, jn]
            d = inner[i + 1, j]
            faces.append([a, b, c])
            faces.append([a, c, d])

    # End caps: annular faces between outer circle and square hole.
    for i in [0, n_x - 1]:
        reverse = i == 0
        for j in range(n_a):
            jn = (j + 1) % n_a
            o0 = outer[i, j]
            o1 = outer[i, jn]
            h0 = inner[i, j]
            h1 = inner[i, jn]
            if reverse:
                faces.append([o0, h1, h0])
                faces.append([o0, o1, h1])
            else:
                faces.append([o0, h0, h1])
                faces.append([o0, h1, o1])

    v_arr = np.asarray(vertices, dtype=float)
    # Snap the end rings exactly to the locked scaled reference span.
    v_arr[np.isclose(v_arr[:, 0], -params.shaft_half_length), 0] = -params.shaft_half_length
    v_arr[np.isclose(v_arr[:, 0],  params.shaft_half_length), 0] =  params.shaft_half_length
    return v_arr, np.asarray(faces, dtype=np.int64)


# -----------------------------------------------------------------------------
# Optional preview
# -----------------------------------------------------------------------------

def save_preview(path: str | Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection="3d")

    step = max(1, len(faces) // 14000)
    tris = vertices[faces[::step]]
    poly = Poly3DCollection(tris, alpha=0.55, linewidth=0.0)
    ax.add_collection3d(poly)

    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins))
    half = 0.5 * span
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("X / axial")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Smooth CSV Barrel Cam — {len(faces):,} triangles")
    ax.view_init(elev=18, azim=-55)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a smooth barrel-cam STL from x,y CSV trajectory. Shaft length is locked to 88.05% of the reference STL.")
    parser.add_argument("csv", type=Path, help="Input CSV with two columns: x,y")
    parser.add_argument("-o", "--output", type=Path, default=Path("cam_from_csv_smooth.stl"), help="Output STL path")
    parser.add_argument("--preview", type=Path, default=None, help="Optional PNG preview path")

    # Most useful tuning knobs.
    parser.add_argument("--angular-segments", type=int, default=CamParams.angular_segments)
    parser.add_argument("--axial-travel", type=float, default=CamParams.axial_travel)
    parser.add_argument("--track-width", type=float, default=CamParams.track_width)
    parser.add_argument("--radius-min", type=float, default=CamParams.radius_min)
    parser.add_argument("--radius-max", type=float, default=CamParams.radius_max)
    parser.add_argument("--groove-radius", type=float, default=CamParams.groove_radius)
    parser.add_argument(
        "--top-cover", "--lip-cover",
        dest="top_cover",
        type=float,
        default=CamParams.top_cover,
        help=(
            "Captive lip amount as a fraction of groove radius. "
            "0.0 gives an open half-round groove; larger values narrow the opening. "
            "Default 0.65 makes the opening about 76%% of the ball diameter."
        ),
    )

    # Reference shaft overrides.
    parser.add_argument("--shaft-radius", type=float, default=CamParams.shaft_radius)
    parser.add_argument("--hub-radius", type=float, default=CamParams.hub_radius)
    parser.add_argument("--square-hole-diagonal", type=float, default=CamParams.square_hole_diagonal)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pts = np.loadtxt(args.csv, delimiter=",")
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)

    params = CamParams(
        angular_segments=args.angular_segments,
        axial_travel=args.axial_travel,
        track_width=args.track_width,
        radius_min=args.radius_min,
        radius_max=args.radius_max,
        groove_radius=args.groove_radius,
        top_cover=args.top_cover,
        shaft_radius=args.shaft_radius,
        hub_radius=args.hub_radius,
        shaft_half_length=CamParams.shaft_half_length,
        small_shaft_half_start=CamParams.small_shaft_half_start,
        hub_full_half_end=CamParams.hub_full_half_end,
        square_hole_diagonal=args.square_hole_diagonal,
        cam_inner_radius=max(0.1, args.hub_radius - 0.15),
    )

    cam_v, cam_f = build_cam_body_mesh(pts[:, 0], pts[:, 1], params)
    axis_v, axis_f = build_axis_mesh(params)

    vertices_parts: list[np.ndarray] = []
    faces_parts: list[np.ndarray] = []
    append_mesh(vertices_parts, faces_parts, axis_v, axis_f)
    append_mesh(vertices_parts, faces_parts, cam_v, cam_f)

    vertices = np.vstack(vertices_parts)
    faces = np.vstack(faces_parts)

    write_binary_stl(args.output, vertices, faces)
    print(f"Wrote STL: {args.output}  ({len(faces):,} triangles)")

    total_len, small_peg_len, hub_blend_len, full_hub_len = locked_axis_dimensions(params)
    print(
        "Locked reference axis dimensions: "
        f"end-to-end = {total_len:.9f} mm, "
        f"small peg length = {small_peg_len:.5f} mm each side, "
        f"blend/shoulder length = {hub_blend_len:.5f} mm each side, "
        f"full hub region = {full_hub_len:.5f} mm"
    )

    ball_d, slot_opening, lip_each_side = groove_lip_dimensions(params)
    print(
        "Captive groove/lip dimensions: "
        f"ball diameter = {ball_d:.3f} mm, "
        f"top opening = {slot_opening:.3f} mm, "
        f"lip overhang = {lip_each_side:.3f} mm per side"
    )

    if args.preview is not None:
        save_preview(args.preview, vertices, faces)
        print(f"Wrote preview: {args.preview}")


if __name__ == "__main__":
    main()
