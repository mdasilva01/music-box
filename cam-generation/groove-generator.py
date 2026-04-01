import numpy as np
import trimesh
import trimesh.boolean
import sys
import os
import math

global r
r = 25.0

CAM_RADIUS = 25.0        # cam disk radius (mm) — groove must fit inside this
GROOVE_RADIUS = 2.5      # follower pin radius (mm)
N_PROFILE = 24           # groove tube cross-section resolution

# ============================================================
# LOAD CSV
# ============================================================

def load_csv_2d(path):
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
        raise ValueError("Need at least 3 valid points in CSV")

    return pts

# ============================================================
# YOUR MAPPING
# ============================================================

def coord_map(points):
    pts = np.array(points, dtype=float)
    n = len(points)

    if n == 0:
        print("nope. no points.")
        return None

    x = pts[:, 0].copy()
    y = pts[:, 1].copy()

    # center x
    mid_x = 0.5 * (np.max(x) + np.min(x))
    x -= mid_x

    # shift y outward
    y += r - np.min(y)

    theta = np.array([2 * np.pi * i / n for i in range(n)], dtype=float)

    new_y = y * np.cos(theta)
    z = y * np.sin(theta)

    new_pts = np.vstack((x, new_y, z)).T
    return new_pts

# ============================================================
# FACE CAM MAPPING
# Cam rotates around Z. At cam angle theta_i = 2*pi*i/n, the
# follower is at world-frame position Q[i]. In the cam's own
# rotating frame the groove point is R(-theta_i) @ Q[i].
# The result is a flat 3D curve (z=0) — groove on the disk face.
# ============================================================

def two_cam_map(points, cam_radius, groove_radius):
    """
    Two coaxial face cams rotating around Z.

    X cam: follower constrained to X axis.
      World pos at theta_i = (camX[i], 0).
      Cam-frame groove = R(-theta_i) @ (camX[i], 0)
                       = (camX[i]*cos(theta_i), -camX[i]*sin(theta_i), 0)

    Y cam: follower constrained to Y axis.
      World pos at theta_i = (0, camY[i]).
      Cam-frame groove = R(-theta_i) @ (0, camY[i])
                       = (camY[i]*sin(theta_i), camY[i]*cos(theta_i), 0)

    Grooves are limaçon-shaped — NOT copies of the path.
    Together they drive the follower through the full 2D trajectory.
    """
    pts = np.array(points, dtype=float)
    n = len(pts)

    mid = 0.5 * (pts.max(axis=0) + pts.min(axis=0))
    pts -= mid

    max_disp = np.abs(pts).max()
    amp = (cam_radius - groove_radius) * 0.35
    if max_disp > 1e-12:
        pts *= amp / max_disp
    base_r = cam_radius - groove_radius - amp

    cam_x = base_r + pts[:, 0]
    cam_y = base_r + pts[:, 1]
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)

    groove_x = np.column_stack([cam_x * np.cos(theta), -cam_x * np.sin(theta), np.zeros(n)])
    groove_y = np.column_stack([cam_y * np.sin(theta),  cam_y * np.cos(theta), np.zeros(n)])

    return groove_x, groove_y

# ============================================================
# SCALE TO TARGET SIZE
# ============================================================

def scale_to_target_size(points_3d, target_size_mm=100.0):
    pts = np.array(points_3d, dtype=float)

    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    spans = maxs - mins
    max_span = np.max(spans)

    if max_span < 1e-12:
        return pts

    scale = target_size_mm / max_span
    return pts * scale

# ============================================================
# MOVING FRAMES along a closed 3D curve
# ============================================================

def compute_frames(centerline):
    c = np.asarray(centerline, dtype=float)
    n = len(c)

    tangents = np.zeros((n, 3), dtype=float)
    normals  = np.zeros((n, 3), dtype=float)
    binormals = np.zeros((n, 3), dtype=float)

    for i in range(n):
        d = c[(i + 1) % n] - c[(i - 1) % n]
        ln = np.linalg.norm(d)
        tangents[i] = d / ln if ln > 1e-12 else np.array([1.0, 0.0, 0.0])

    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(ref, tangents[0])) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])

    n0 = ref - np.dot(ref, tangents[0]) * tangents[0]
    n0 /= max(np.linalg.norm(n0), 1e-12)
    normals[0] = n0
    binormals[0] = np.cross(tangents[0], normals[0])
    binormals[0] /= max(np.linalg.norm(binormals[0]), 1e-12)

    for i in range(1, n):
        t = tangents[i]
        nv = normals[i - 1] - np.dot(normals[i - 1], t) * t
        nn = np.linalg.norm(nv)
        if nn < 1e-12:
            fb = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(fb, t)) > 0.9:
                fb = np.array([1.0, 0.0, 0.0])
            nv = fb - np.dot(fb, t) * t
            nn = np.linalg.norm(nv)
        normals[i] = nv / max(nn, 1e-12)
        binormals[i] = np.cross(t, normals[i])
        binormals[i] /= max(np.linalg.norm(binormals[i]), 1e-12)

    return tangents, normals, binormals

# ============================================================
# GROOVE TUBE MESH (uniform or per-point varying radius)
# ============================================================

def build_groove_tube(centerline, tube_radius, n_profile=N_PROFILE):
    c = np.asarray(centerline, dtype=float)
    n = len(c)
    _, normals, binormals = compute_frames(c)
    angles = np.linspace(0.0, 2.0 * math.pi, n_profile, endpoint=False)

    verts = []
    for i in range(n):
        for a in angles:
            offset = tube_radius * math.cos(a) * normals[i] + tube_radius * math.sin(a) * binormals[i]
            verts.append(c[i] + offset)
    verts = np.array(verts, dtype=float)

    faces = []
    for i in range(n):
        i2 = (i + 1) % n
        for j in range(n_profile):
            j2 = (j + 1) % n_profile
            a = i  * n_profile + j
            b = i2 * n_profile + j
            cc = i2 * n_profile + j2
            d  = i  * n_profile + j2
            faces.append([a, b, cc])
            faces.append([a, cc, d])

    mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces, dtype=np.int64), process=True)
    trimesh.repair.fix_normals(mesh)
    return mesh

# ============================================================
# CAM BODY
# Built as the groove tube inflated by WALL — wraps tightly around
# the groove path so the groove is open on the outer surface everywhere.
# ============================================================

def build_cam_body(centerline, groove_radius, wall, n_profile=N_PROFILE):
    return build_groove_tube(centerline, groove_radius + wall, n_profile)


# ============================================================
# BOOLEAN DIFFERENCE (manifold3d via trimesh)
# ============================================================

def boolean_subtract(body, cutter):
    return trimesh.boolean.difference([body, cutter], engine="manifold")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    args = sys.argv[1:]

    if len(args) >= 1:
        csv_path = args[0]

        if not os.path.exists(csv_path):
            raise FileNotFoundError(csv_path)

        print(f"Loading CSV: {csv_path}")
        input_pts = load_csv_2d(csv_path)

        base = os.path.splitext(args[1])[0] if len(args) >= 2 else os.path.splitext(csv_path)[0]

    else:
        print("No CSV provided, using default test points.")
        input_pts = [
            [1,1],[1.5,1.3],[1.7,1.4],[1.8,1.6],
            [1.7,1.7],[1.5,1.4],[1.3,1.2],[1.15,1.1],[1,1]
        ]
        base = "cam_output"

    mapped = coord_map(input_pts)
    mapped = scale_to_target_size(mapped, CAM_RADIUS * 2)

    groove = build_groove_tube(mapped, GROOVE_RADIUS)
    groove.export(base + "_groove.stl")
    print(f"Wrote groove: {base}_groove.stl")
