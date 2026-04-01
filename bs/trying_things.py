import math
import os
import sys

import numpy as np
import trimesh

# ============================================================
# SETTINGS
# ============================================================

MAX_POINTS = 240
TARGET_SCALE = 12.0

GROOVE_RADIUS = 0.8
N_PROFILE = 18

BLANK_WALL = 1.5
BLANK_HEIGHT = 8.0
CUTAWAY_MISSING_ANGLE_DEG = 140.0

DEFAULT_OUT = "cam2T_preview.stl"

# ============================================================
# CSV LOADING
# ============================================================

def load_csv_2d(path: str) -> np.ndarray:
    """Load ordered 2D samples from a CSV-like file."""
    pts = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = [p.strip() for p in line.replace(";", ",").split(",")]
            if len(parts) < 2:
                continue

            try:
                x = float(parts[0])
                y = float(parts[1])
                pts.append([x, y])
            except ValueError:
                continue

    if len(pts) < 3:
        raise ValueError("Need at least 3 valid 2D points in the CSV.")

    return np.array(pts, dtype=float)

# ============================================================
# PATH PROCESSING
# ============================================================

def remove_duplicate_points(points: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Remove consecutive duplicate points and a duplicated closing point."""
    points = np.asarray(points, dtype=float)
    out = [points[0]]

    for p in points[1:]:
        if np.linalg.norm(p - out[-1]) > tol:
            out.append(p)

    out = np.array(out, dtype=float)

    if len(out) > 2 and np.linalg.norm(out[0] - out[-1]) < tol:
        out = out[:-1]

    return out

def resample_closed_curve(points: np.ndarray, n_samples: int) -> np.ndarray:
    """Arc-length resampling of a closed 2D polyline."""
    pts = remove_duplicate_points(points)

    if len(pts) < 3:
        return pts

    closed = np.vstack([pts, pts[0]])
    segs = np.diff(closed, axis=0)
    seg_lens = np.linalg.norm(segs, axis=1)
    total_len = np.sum(seg_lens)

    if total_len < 1e-12:
        return pts

    cumulative = np.concatenate([[0.0], np.cumsum(seg_lens)])
    sample_d = np.linspace(0.0, total_len, n_samples, endpoint=False)

    out = []
    seg_idx = 0

    for d in sample_d:
        while seg_idx < len(seg_lens) - 1 and cumulative[seg_idx + 1] < d:
            seg_idx += 1

        d0 = cumulative[seg_idx]
        d1 = cumulative[seg_idx + 1]
        t = 0.0 if abs(d1 - d0) < 1e-12 else (d - d0) / (d1 - d0)
        p = (1 - t) * closed[seg_idx] + t * closed[seg_idx + 1]
        out.append(p)

    return np.array(out, dtype=float)

def normalize_and_scale_planar_trajectory(points: np.ndarray, target_scale: float) -> np.ndarray:
    """Center the planar trajectory and scale it to a target radius."""
    pts = remove_duplicate_points(points)
    pts = pts - pts.mean(axis=0)

    radii = np.linalg.norm(pts, axis=1)
    max_radius = np.max(radii)

    if max_radius < 1e-12:
        return pts

    return pts / max_radius * target_scale

# ============================================================
# 2T PITCH CURVE
# ============================================================

def rz(theta: float) -> np.ndarray:
    """Rotation matrix about z-axis."""
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)

def build_target_trajectory_q(points: np.ndarray) -> np.ndarray:
    """
    2T planar follower trajectory in world coordinates:
        Q(t) = [x(t), y(t), 0]
    """
    pts = np.asarray(points, dtype=float)

    q = np.zeros((len(pts), 3), dtype=float)
    q[:, 0] = pts[:, 0]
    q[:, 1] = pts[:, 1]
    return q

def compute_pitch_curve(q: np.ndarray) -> np.ndarray:
    """
    2T pitch curve:
        C(t) = Rz(-theta(t)) @ Q(t)
    """
    n = len(q)
    c = np.zeros_like(q)

    for i in range(n):
        theta = 2.0 * math.pi * i / n
        c[i] = rz(-theta) @ q[i]

    return c

# ============================================================
# TUBE GEOMETRY
# ============================================================

def compute_frames(centerline: np.ndarray):
    """Build a simple moving frame along a closed 3D curve."""
    c = np.asarray(centerline, dtype=float)
    n = len(c)

    tangents = np.zeros((n, 3), dtype=float)
    normals = np.zeros((n, 3), dtype=float)
    binormals = np.zeros((n, 3), dtype=float)

    for i in range(n):
        prev_pt = c[(i - 1) % n]
        next_pt = c[(i + 1) % n]
        t = next_pt - prev_pt
        t_norm = np.linalg.norm(t)

        if t_norm < 1e-12:
            t = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            t = t / t_norm

        tangents[i] = t

    ref = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(np.dot(ref, tangents[0])) > 0.9:
        ref = np.array([1.0, 0.0, 0.0], dtype=float)

    n0 = ref - np.dot(ref, tangents[0]) * tangents[0]
    n0_norm = np.linalg.norm(n0)
    if n0_norm < 1e-12:
        n0 = np.array([1.0, 0.0, 0.0], dtype=float)
        n0_norm = np.linalg.norm(n0)

    normals[0] = n0 / n0_norm
    binormals[0] = np.cross(tangents[0], normals[0])
    binormals[0] /= max(np.linalg.norm(binormals[0]), 1e-12)

    for i in range(1, n):
        prev_n = normals[i - 1]
        t = tangents[i]

        nvec = prev_n - np.dot(prev_n, t) * t
        n_norm = np.linalg.norm(nvec)

        if n_norm < 1e-12:
            fallback = np.array([0.0, 0.0, 1.0], dtype=float)
            if abs(np.dot(fallback, t)) > 0.9:
                fallback = np.array([1.0, 0.0, 0.0], dtype=float)
            nvec = fallback - np.dot(fallback, t) * t
            n_norm = np.linalg.norm(nvec)

        if n_norm < 1e-12:
            nvec = np.array([1.0, 0.0, 0.0], dtype=float)
            n_norm = np.linalg.norm(nvec)

        normals[i] = nvec / n_norm
        binormals[i] = np.cross(tangents[i], normals[i])
        binormals[i] /= max(np.linalg.norm(binormals[i]), 1e-12)

    return tangents, normals, binormals

def build_tube_mesh(centerline: np.ndarray, tube_radius: float, n_profile: int = 16):
    """Build a round preview tube around the pitch curve."""
    c = np.asarray(centerline, dtype=float)
    n = len(c)

    _, normals, binormals = compute_frames(c)
    angles = np.linspace(0.0, 2.0 * math.pi, n_profile, endpoint=False)

    verts = []
    for i in range(n):
        p = c[i]
        nvec = normals[i]
        bvec = binormals[i]

        for a in angles:
            offset = tube_radius * math.cos(a) * nvec + tube_radius * math.sin(a) * bvec
            verts.append(p + offset)

    verts = np.array(verts, dtype=float)

    faces = []
    for i in range(n):
        i2 = (i + 1) % n
        for j in range(n_profile):
            j2 = (j + 1) % n_profile

            a = i  * n_profile + j
            b = i2 * n_profile + j
            c2 = i2 * n_profile + j2
            d = i  * n_profile + j2

            faces.append([a, b, c2])
            faces.append([a, c2, d])

    return np.array(verts, dtype=float), np.array(faces, dtype=np.int64)

# ============================================================
# MESH HELPERS
# ============================================================

def trimesh_from_verts_faces(verts: np.ndarray, faces: np.ndarray) -> trimesh.Trimesh:
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)

def clean_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Basic cleanup for preview/export."""
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()
    trimesh.repair.fix_normals(mesh)
    return mesh

def build_cutaway_blank(
    centerline: np.ndarray,
    groove_radius: float,
    wall: float,
    height: float,
    sections: int = 128,
    missing_angle_deg: float = 120.0,
) -> trimesh.Trimesh:
    """
    Build a cylindrical blank with a wedge removed so the inner groove is visible.
    """
    r = np.linalg.norm(centerline[:, :2], axis=1)
    outer_r = float(np.max(r) + groove_radius + wall)

    keep_angle = 360.0 - missing_angle_deg
    n = max(8, int(sections * keep_angle / 360.0))

    start = math.radians(missing_angle_deg / 2.0)
    end = math.radians(360.0 - missing_angle_deg / 2.0)
    angles = np.linspace(start, end, n, endpoint=True)

    z0 = -height / 2.0
    z1 =  height / 2.0

    verts = []

    # bottom arc
    for a in angles:
        verts.append([outer_r * math.cos(a), outer_r * math.sin(a), z0])

    # top arc
    for a in angles:
        verts.append([outer_r * math.cos(a), outer_r * math.sin(a), z1])

    bottom_center = len(verts)
    verts.append([0.0, 0.0, z0])

    top_center = len(verts)
    verts.append([0.0, 0.0, z1])

    faces = []

    # curved outer wall
    for i in range(n - 1):
        b0 = i
        b1 = i + 1
        t0 = n + i
        t1 = n + i + 1

        faces.append([b0, b1, t1])
        faces.append([b0, t1, t0])

    # radial wall at start cut
    faces.append([0, n, top_center])
    faces.append([0, top_center, bottom_center])
    faces.append([0, bottom_center, n])

    # radial wall at end cut
    faces.append([n - 1, bottom_center, top_center])
    faces.append([n - 1, top_center, 2 * n - 1])
    faces.append([n - 1, 2 * n - 1, bottom_center])

    # bottom cap
    for i in range(n - 1):
        faces.append([bottom_center, i + 1, i])

    # top cap
    for i in range(n - 1):
        faces.append([top_center, n + i, n + i + 1])

    return trimesh.Trimesh(
        vertices=np.array(verts, dtype=float),
        faces=np.array(faces, dtype=np.int64),
        process=False,
    )

# ============================================================
# MAIN
# ============================================================

def main():
    args = sys.argv[1:]

    if len(args) >= 1:
        csv_path = args[0]
        if not os.path.exists(csv_path):
            raise FileNotFoundError(csv_path)

        print(f"Loading points from: {csv_path}")
        raw_points = load_csv_2d(csv_path)

        if len(args) >= 2:
            out_stl = args[1]
        else:
            base = os.path.splitext(csv_path)[0]
            out_stl = base + "_2T_preview.stl"
    else:
        print("No CSV provided, using default closed planar test path.")
        raw_points = np.array([
            [0.0,  1.0],
            [0.8,  0.3],
            [0.5, -0.8],
            [-0.5, -0.8],
            [-0.8,  0.3],
        ], dtype=float)
        out_stl = DEFAULT_OUT

    pts = normalize_and_scale_planar_trajectory(raw_points, TARGET_SCALE)
    n_use = min(MAX_POINTS, len(pts))
    pts = resample_closed_curve(pts, n_use)
    print(f"Using {len(pts)} sampled points")

    q = build_target_trajectory_q(pts)
    c = compute_pitch_curve(q)

    groove_verts, groove_faces = build_tube_mesh(
        centerline=c,
        tube_radius=GROOVE_RADIUS,
        n_profile=N_PROFILE,
    )
    groove_mesh = clean_mesh(trimesh_from_verts_faces(groove_verts, groove_faces))

    blank_mesh = clean_mesh(
        build_cutaway_blank(
            centerline=c,
            groove_radius=GROOVE_RADIUS,
            wall=BLANK_WALL,
            height=BLANK_HEIGHT,
            missing_angle_deg=CUTAWAY_MISSING_ANGLE_DEG,
        )
    )

    print("blank watertight:", blank_mesh.is_watertight)
    print("blank volume:", blank_mesh.is_volume)
    print("groove watertight:", groove_mesh.is_watertight)
    print("groove volume:", groove_mesh.is_volume)

    preview_mesh = trimesh.util.concatenate([blank_mesh, groove_mesh])
    preview_mesh.export(out_stl)

    print(f"Wrote preview STL: {out_stl}")
    print(f"Combined mesh vertices: {len(preview_mesh.vertices)}, faces: {len(preview_mesh.faces)}")

if __name__ == "__main__":
    main()