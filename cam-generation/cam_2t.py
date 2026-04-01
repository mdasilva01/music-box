"""
2T Flat Face Cam — cylinder-pin approach

For each sampled point on a 2D closed trajectory, place a small cylinder
whose outer tip lies on the boundary. Fill the space between all cylinders
with a base disk around a central shaft hole. The result is a flat disk
cam parallel to the XY plane.

As the cam rotates about Z, the follower ball rides the cylinder surfaces
and is pushed to trace the 2D trajectory.

Outputs:
  cam_2t.stl
"""

import numpy as np
import struct, os

# ── Parameters ────────────────────────────────────────────────────────────────
N_SAMPLES  =  72     # number of sample points around the trajectory
CYL_R      =  2.0    # mm  radius of each sample cylinder
CAM_H      =  8.0    # mm  cam disk height (Z)
SHAFT_R    =  4.0    # mm  central shaft hole radius

N_CYL      =  16     # circumference resolution per cylinder (small — lots of them)
N_SHAFT    = 120     # resolution of shaft hole circle

# ── Demo trajectory (2D closed curve in XY) ──────────────────────────────────

def demo_lemniscate(n=N_SAMPLES):
    """Figure-8 lemniscate in XY."""
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    R = 16.0
    # Lemniscate of Bernoulli: r² = 2R² cos(2θ)  → parametric form
    x = R * np.cos(t) / (1 + np.sin(t)**2)
    y = R * np.cos(t) * np.sin(t) / (1 + np.sin(t)**2)
    return np.column_stack([x, y])

def demo_star(n=N_SAMPLES):
    """5-lobed star."""
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    r = 14.0 + 5.0 * np.cos(5 * t)
    return np.column_stack([r * np.cos(t), r * np.sin(t)])

def demo_ellipse(n=N_SAMPLES):
    """Simple ellipse."""
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([18.0 * np.cos(t), 11.0 * np.sin(t)])

DEMOS = dict(lemniscate=demo_lemniscate, star=demo_star, ellipse=demo_ellipse)

# ── Mesh helpers ──────────────────────────────────────────────────────────────

def merge(*meshes):
    verts_all, tris_all, offset = [], [], 0
    for v, t in meshes:
        verts_all.append(v)
        tris_all.append(t + offset)
        offset += len(v)
    return np.vstack(verts_all), np.vstack(tris_all)


def cylinder_z(cx, cy, z0, z1, r, n=N_CYL):
    """Closed cylinder along Z, centred at (cx, cy)."""
    verts, tris = [], []
    bc = len(verts); verts.append([cx, cy, z0])
    tc = len(verts); verts.append([cx, cy, z1])
    br = len(verts)
    for j in range(n):
        a = 2*np.pi*j/n
        verts.append([cx + r*np.cos(a), cy + r*np.sin(a), z0])
    tr = len(verts)
    for j in range(n):
        a = 2*np.pi*j/n
        verts.append([cx + r*np.cos(a), cy + r*np.sin(a), z1])
    for j in range(n):
        j2 = (j+1) % n
        tris += [[br+j, tr+j2, tr+j], [br+j, br+j2, tr+j2]]  # side
        tris.append([bc, br+j, br+j2])   # bottom cap
        tris.append([tc, tr+j2, tr+j])   # top cap
    return np.array(verts, dtype=float), np.array(tris, dtype=np.int32)


def annular_disk(outer_pts, shaft_r, z0, z1, n_shaft=N_SHAFT):
    """
    Extruded annular disk:
      outer boundary = outer_pts  (Nx2 polygon, follows cam profile)
      inner boundary = circle of shaft_r  (N_SHAFT vertices)

    Builds: top face, bottom face, outer wall, inner wall (shaft hole).
    """
    N = len(outer_pts)

    # Outer ring at z0 and z1
    outer_bot = np.column_stack([outer_pts, np.full(N, z0)])
    outer_top = np.column_stack([outer_pts, np.full(N, z1)])

    # Inner ring (shaft circle) at z0 and z1
    shaft_angles = np.linspace(0, 2*np.pi, n_shaft, endpoint=False)
    inner_bot = np.column_stack([
        shaft_r * np.cos(shaft_angles),
        shaft_r * np.sin(shaft_angles),
        np.full(n_shaft, z0)
    ])
    inner_top = np.column_stack([
        shaft_r * np.cos(shaft_angles),
        shaft_r * np.sin(shaft_angles),
        np.full(n_shaft, z1)
    ])

    verts = np.vstack([outer_bot, outer_top, inner_bot, inner_top])
    # Index offsets
    ob = 0;          ot = N
    ib = 2*N;        it = 2*N + n_shaft

    tris = []

    # ── Top face: triangulate annulus (outer_top → inner_top) ──
    # Walk both rings simultaneously, advancing whichever has smaller angle
    o_angles = np.arctan2(outer_pts[:,1], outer_pts[:,0]) % (2*np.pi)
    i_angles = shaft_angles % (2*np.pi)

    oi, ii = 0, 0
    while oi < N or ii < n_shaft:
        oi2 = (oi+1) % N
        ii2 = (ii+1) % n_shaft
        o_next = o_angles[oi2] if oi+1 < N else o_angles[0] + 2*np.pi
        i_next = i_angles[ii2] if ii+1 < n_shaft else i_angles[0] + 2*np.pi
        if oi >= N:
            # Only inner left
            tris.append([ot+oi%N, it+ii2, it+ii])
            ii += 1
        elif ii >= n_shaft:
            # Only outer left
            tris.append([ot+oi, ot+oi2, it+ii%n_shaft])
            oi += 1
        elif o_next <= i_next:
            tris.append([ot+oi, ot+oi2, it+ii])
            oi += 1
        else:
            tris.append([ot+oi, it+ii2, it+ii])
            ii += 1

    # ── Bottom face (winding flipped for -Z normal) ──
    oi, ii = 0, 0
    while oi < N or ii < n_shaft:
        oi2 = (oi+1) % N
        ii2 = (ii+1) % n_shaft
        o_next = o_angles[oi2] if oi+1 < N else o_angles[0] + 2*np.pi
        i_next = i_angles[ii2] if ii+1 < n_shaft else i_angles[0] + 2*np.pi
        if oi >= N:
            tris.append([ob+oi%N, ib+ii, ib+ii2])
            ii += 1
        elif ii >= n_shaft:
            tris.append([ob+oi, ib+ii%n_shaft, ob+oi2])
            oi += 1
        elif o_next <= i_next:
            tris.append([ob+oi, ib+ii, ob+oi2])
            oi += 1
        else:
            tris.append([ob+oi, ib+ii, ib+ii2])
            ii += 1

    # ── Outer wall ──
    for i in range(N):
        i2 = (i+1) % N
        tris += [[ob+i, ot+i, ot+i2], [ob+i, ot+i2, ob+i2]]

    # ── Inner wall (shaft hole, normal inward = winding flipped) ──
    for i in range(n_shaft):
        i2 = (i+1) % n_shaft
        tris += [[ib+i, it+i2, it+i], [ib+i, ib+i2, it+i2]]

    return verts, np.array(tris, dtype=np.int32)


# ── Build cam ────────────────────────────────────────────────────────────────

def build_cam_2t(Q_2d):
    """
    Q_2d: Nx2 array of trajectory sample points in XY.

    For each point P_i:
      - Place a cylinder of radius CYL_R centred at P_i.
      - The cylinder's outermost surface (tip) lies on the cam boundary
        at radius |P_i| + CYL_R from the origin.

    The base disk fills the space between the shaft hole and the cylinders.
    Its outer boundary sits at each P_i inset inward by CYL_R, so it is
    flush with the inner face of each cylinder (no gap, no overlap with tips).
    """
    N = len(Q_2d)
    meshes = []

    # Small cylinder at every sample point
    for i in range(N):
        cx, cy = Q_2d[i]
        v, t = cylinder_z(cx, cy, 0, CAM_H, CYL_R)
        meshes.append((v, t))

    # Base disk outer boundary: each sample point pulled inward by CYL_R
    # so the base is flush with the inner face of each cylinder
    outer_boundary = []
    for p in Q_2d:
        r = np.linalg.norm(p)
        if r > CYL_R + SHAFT_R + 0.5:
            scale = (r - CYL_R) / r
        else:
            scale = 0.5  # fallback if point is very close to shaft
        outer_boundary.append(p * scale)
    outer_boundary = np.array(outer_boundary)

    base_v, base_t = annular_disk(outer_boundary, SHAFT_R, 0, CAM_H)
    meshes.append((base_v, base_t))

    return merge(*meshes)


# ── STL writer ────────────────────────────────────────────────────────────────

def tri_normal(v0, v1, v2):
    n = np.cross(v1 - v0, v2 - v0)
    l = np.linalg.norm(n)
    return n / l if l > 1e-15 else np.array([0., 0., 1.])


def write_stl(path, verts, tris):
    with open(path, 'wb') as f:
        f.write(b'cam_2t cylinder-pin'.ljust(80, b'\x00'))
        f.write(struct.pack('<I', len(tris)))
        for tri in tris:
            v0, v1, v2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
            f.write(struct.pack('<3f', *tri_normal(v0, v1, v2)))
            f.write(struct.pack('<3f', *v0))
            f.write(struct.pack('<3f', *v1))
            f.write(struct.pack('<3f', *v2))
            f.write(struct.pack('<H', 0))
    print(f"  Wrote {len(tris):,} tris  →  {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    demo_name = sys.argv[1] if len(sys.argv) > 1 else "lemniscate"
    if demo_name not in DEMOS:
        print(f"Unknown demo '{demo_name}'. Options: {', '.join(DEMOS)}")
        sys.exit(1)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, f"cam_2t_{demo_name}.stl")

    Q = DEMOS[demo_name]()
    print(f"\n2T face cam  —  {demo_name}")
    print(f"  {N_SAMPLES} sample points,  cylinder R={CYL_R} mm,  cam H={CAM_H} mm")
    print(f"  Shaft hole R={SHAFT_R} mm")
    print(f"  Trajectory extents: X [{Q[:,0].min():.1f}, {Q[:,0].max():.1f}]  "
          f"Y [{Q[:,1].min():.1f}, {Q[:,1].max():.1f}]\n")

    print("Building cam...")
    v, t = build_cam_2t(Q)
    write_stl(out_path, v, t)
    print("Done.")
