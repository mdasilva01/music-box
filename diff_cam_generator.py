import sys
import os
import struct
import numpy as np

# ============================================================
# PARAMETERS
# ============================================================

CAM_RADIUS      = 50.0   # mm
CAM_THICKNESS   = 8.0    # mm
SHAFT_RADIUS    = 4.0    # mm

PIN_RADIUS      = 2.5    # mm   follower pin radius
GROOVE_CLEAR    = 0.6    # mm   extra side clearance
GROOVE_DEPTH    = 4.0    # mm   depth of trench into disk
CUTTER_EXTRA_Z  = 1.0    # mm   extend cutter slightly above top face for clean boolean

N_SAMPLES       = 360
N_CIRCLE        = 180
N_PROFILE       = 20

OUTER_MARGIN    = 5.0
INNER_MARGIN    = 5.0

TOP_Z           = CAM_THICKNESS
CUTTER_TOP_Z    = CAM_THICKNESS + CUTTER_EXTRA_Z
CUTTER_BOT_Z    = CAM_THICKNESS - GROOVE_DEPTH

GROOVE_R        = PIN_RADIUS + GROOVE_CLEAR


# ============================================================
# BASIC MATH
# ============================================================

def R2(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)

def resample_closed_curve_2d(pts, n):
    pts = np.asarray(pts, dtype=float)
    closed = np.vstack([pts, pts[0]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    t_new = np.linspace(0, total, n, endpoint=False)

    x_new = np.interp(t_new, cum, np.append(pts[:, 0], pts[0, 0]))
    y_new = np.interp(t_new, cum, np.append(pts[:, 1], pts[0, 1]))
    return np.column_stack([x_new, y_new])

def load_csv_2d(path):
    pts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.replace(";", ",").split(",")]
            if len(parts) < 2:
                continue
            try:
                pts.append([float(parts[0]), float(parts[1])])
            except ValueError:
                continue
    if len(pts) < 4:
        raise ValueError(f"Need at least 4 planar points, got {len(pts)}")
    return np.array(pts, dtype=float)

def normalize_and_scale_path(pts):
    """
    Center and scale desired WORLD-frame follower path so it fits safely
    in the disk annulus.
    """
    pts = resample_closed_curve_2d(pts, N_SAMPLES)
    pts = pts - pts.mean(axis=0)

    r = np.linalg.norm(pts, axis=1)
    rmax = max(r.max(), 1e-9)

    r_min = SHAFT_RADIUS + GROOVE_R + INNER_MARGIN
    r_max = CAM_RADIUS   - GROOVE_R - OUTER_MARGIN

    r_center = 0.5 * (r_min + r_max)
    amp_max  = 0.5 * (r_max - r_min)

    scale = amp_max / rmax
    Q = pts * scale

    # shift curve outward so it lives in a safe annulus
    Q[:, 0] += r_center

    rr = np.linalg.norm(Q, axis=1)
    if rr.min() < r_min or rr.max() > r_max:
        raise ValueError("Scaled path violates radial limits.")

    return Q

def compute_cam_frame_centerline(Q):
    """
    Desired follower path in world frame -> groove centerline in rotating cam frame.
    """
    N = len(Q)
    C = np.zeros_like(Q)
    for i in range(N):
        theta = 2.0 * np.pi * i / N
        C[i] = R2(-theta) @ Q[i]
    return C

def compute_2d_frames(C):
    """
    Tangent and left-normal for a closed planar curve.
    """
    n = len(C)
    T = np.zeros((n, 2))
    Nrm = np.zeros((n, 2))

    for i in range(n):
        d = C[(i + 1) % n] - C[(i - 1) % n]
        l = np.linalg.norm(d)
        if l < 1e-12:
            T[i] = np.array([1.0, 0.0])
        else:
            T[i] = d / l
        Nrm[i] = np.array([-T[i, 1], T[i, 0]])

    return T, Nrm


# ============================================================
# STL HELPERS
# ============================================================

def tri_normal(v0, v1, v2):
    e1 = v1 - v0
    e2 = v2 - v0
    n = np.cross(e1, e2)
    l = np.linalg.norm(n)
    return n / l if l > 1e-15 else np.array([0.0, 0.0, 1.0])

def write_stl(path, verts, tris, header=b"mesh"):
    with open(path, "wb") as f:
        f.write(header.ljust(80, b"\x00"))
        f.write(struct.pack("<I", len(tris)))
        for tri in tris:
            v0, v1, v2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
            n = tri_normal(v0, v1, v2)
            f.write(struct.pack("<3f", *n))
            f.write(struct.pack("<3f", *v0))
            f.write(struct.pack("<3f", *v1))
            f.write(struct.pack("<3f", *v2))
            f.write(struct.pack("<H", 0))


# ============================================================
# MESH BUILDERS
# ============================================================

def build_annular_disk(radius, shaft_radius, thickness, n=N_CIRCLE):
    verts = []

    def ring(r, z):
        return [[r*np.cos(2*np.pi*i/n), r*np.sin(2*np.pi*i/n), z] for i in range(n)]

    ob = len(verts); verts.extend(ring(radius, 0.0))
    ot = len(verts); verts.extend(ring(radius, thickness))
    ib = len(verts); verts.extend(ring(shaft_radius, 0.0))
    it = len(verts); verts.extend(ring(shaft_radius, thickness))

    tris = []
    for i in range(n):
        i2 = (i + 1) % n

        # outer wall
        tris += [(ob+i, ob+i2, ot+i2), (ob+i, ot+i2, ot+i)]

        # inner wall
        tris += [(ib+i, it+i, it+i2), (ib+i, it+i2, ib+i2)]

        # bottom face
        tris += [(ib+i, ib+i2, ob+i2), (ib+i, ob+i2, ob+i)]

        # top face
        tris += [(it+i, ot+i, ot+i2), (it+i, ot+i2, it+i2)]

    return np.array(verts, dtype=float), np.array(tris, dtype=np.int32)

def build_tube_cutter(C, tube_r=GROOVE_R, z_top=CUTTER_TOP_Z, z_bot=CUTTER_BOT_Z, n_prof=N_PROFILE):
    """
    Build a closed cutter solid by sweeping a vertical capsule/circle-like profile
    along the groove centerline C.

    Cross-section is circular in the normal-z plane so the follower pin can ride
    smoothly in the trench after subtraction.
    """
    _, Nrm = compute_2d_frames(C)
    n = len(C)

    # circular profile in local (u,z) frame
    ang = np.linspace(0, 2*np.pi, n_prof, endpoint=False)
    # vertically center the circle inside the groove depth, with some top overshoot
    z_mid = 0.5 * (z_top + z_bot)
    prof = np.column_stack([tube_r*np.cos(ang), z_mid + tube_r*np.sin(ang)])

    verts = np.zeros((n * n_prof, 3), dtype=float)

    for i in range(n):
        cx, cy = C[i]
        nx, ny = Nrm[i]

        for j, (u, z) in enumerate(prof):
            px = cx + u * nx
            py = cy + u * ny
            verts[i*n_prof + j] = np.array([px, py, z])

    tris = []

    # side surface
    for i in range(n):
        i2 = (i + 1) % n
        for j in range(n_prof):
            j2 = (j + 1) % n_prof
            a = i  * n_prof + j
            b = i2 * n_prof + j
            c = i2 * n_prof + j2
            d = i  * n_prof + j2
            tris.append((a, b, c))
            tris.append((a, c, d))

    return verts, np.array(tris, dtype=np.int32)


# ============================================================
# MAIN
# ============================================================

def generate_cam_and_cutter(path2d, out_base):
    print("\n====================================================")
    print("Generating planar XY face cam")
    print("====================================================")

    print("[1/4] Scaling desired follower path...")
    Q = normalize_and_scale_path(path2d)
    rq = np.linalg.norm(Q, axis=1)
    print(f"      world-path radius range: {rq.min():.2f} to {rq.max():.2f} mm")

    print("[2/4] Computing groove centerline in cam frame...")
    C = compute_cam_frame_centerline(Q)
    rc = np.linalg.norm(C, axis=1)
    closure = np.linalg.norm(C[0] - C[-1])
    print(f"      cam-frame groove radius range: {rc.min():.2f} to {rc.max():.2f} mm")
    print(f"      closure error: {closure:.6f} mm")

    if rc.min() < SHAFT_RADIUS + GROOVE_R + INNER_MARGIN:
        raise ValueError("Groove too close to shaft.")
    if rc.max() > CAM_RADIUS - GROOVE_R - OUTER_MARGIN:
        raise ValueError("Groove too close to rim.")

    print("[3/4] Building meshes...")
    disk_v, disk_t = build_annular_disk(CAM_RADIUS, SHAFT_RADIUS, CAM_THICKNESS)
    cut_v, cut_t   = build_tube_cutter(C)

    cam_path    = f"{out_base}_disk.stl"
    cutter_path = f"{out_base}_groove_cutter.stl"

    print("[4/4] Writing STL files...")
    write_stl(cam_path, disk_v, disk_t, header=b"face cam disk")
    write_stl(cutter_path, cut_v, cut_t, header=b"face cam groove cutter")

    print(f"      disk:   {cam_path}")
    print(f"      cutter: {cutter_path}")
    print("\nUse a Boolean difference:")
    print("    final_cam = disk - groove_cutter")
    print("====================================================\n")


def demo_path(n=240):
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = 0.9*np.cos(2*t)
    y = 0.7*np.sin(3*t)
    return np.column_stack([x, y])


if __name__ == "__main__":
    args = sys.argv[1:]

    if not args:
        pts = demo_path()
        out_base = "xy_face_cam"
        print("No CSV provided; using built-in demo.")
    else:
        csv_path = args[0]
        if not os.path.exists(csv_path):
            raise FileNotFoundError(csv_path)
        pts = load_csv_2d(csv_path)
        out_base = args[1] if len(args) > 1 else os.path.splitext(csv_path)[0]

    generate_cam_and_cutter(pts, out_base)