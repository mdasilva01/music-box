"""
3D Cylindrical Cam Generator
Based on: "Spatial-Temporal Motion Control via Composite Cam-follower Mechanisms"
         Cheng et al., SIGGRAPH Asia 2021
         https://dl.acm.org/doi/10.1145/3478513.3480477

MECHANISM TYPE: camMech_1T1R (Table 1 in paper)
  - The cam is a cylinder rotating about its own Z-axis
  - The follower has a CYLINDRICAL JOINT: it can translate in Z
    AND rotate about Z (2 DOF total)
  - The follower arm is fixed-length, so the follower ball stays at
    a fixed radius FOLLOWER_RADIUS from the Z-axis at all times
  - As the cam rotates, the groove profile pushes the follower ball
    to different (angle, height) positions on the cylinder surface
  - This lets the follower trace ANY closed curve on a cylinder

THE KEY MATH (Section 4.1-4.2 of paper):
  The pitch curve C(s) is the path of the follower BALL CENTER
  as seen in the CAM'S LOCAL (ROTATING) FRAME.

  In world space, at cam rotation angle θ_c, the cam frame has
  rotated by θ_c. The follower ball is at world position Q(u(t)).

  To get the pitch curve (ball position in cam frame), we apply
  the INVERSE cam rotation:
      C(s) = R_z(-θ_c(t)) * Q(u(t))

  With uniform timing: θ_c = 2π * (sample_index / N)

  So the groove is simply the target trajectory "unrolled" into
  the cam's rotating frame. As the cam spins, it re-applies the
  rotation, and the follower (constrained to the cylinder surface)
  gets pushed to exactly trace the original target path.

USAGE:
  python cam_generator.py                   # built-in lemniscate demo
  python cam_generator.py sine              # other built-in demos
  python cam_generator.py my_path.csv       # your own trajectory
  python cam_generator.py my_path.csv out.stl

INPUT CSV FORMAT (3 columns, no header required):
  x, y, z   (one point per row)
  The curve will be auto-projected onto a cylinder of radius FOLLOWER_RADIUS.

BUILT-IN DEMOS: lemniscate, sine, star, spiral, trefoil
"""

import sys, os, struct
import numpy as np

# ─── PARAMETERS ────────────────────────────────────────────────────────────────

CAM_RADIUS        = 30.0   # mm  outer radius of the cam cylinder
CAM_HEIGHT        = 50.0   # mm  total height
SHAFT_RADIUS      =  4.0   # mm  central hole radius (motor axle)
FOLLOWER_RADIUS   = 20.0   # mm  radius of follower ball path (< CAM_RADIUS - groove width)
GROOVE_BALL_R     =  2.5   # mm  follower ball radius
GROOVE_WALL       =  1.5   # mm  groove wall thickness on each side

N_THETA   = 360   # angular resolution of pitch curve
N_PROFILE =  16   # cross-section resolution of groove tube
N_CAP     = 180   # resolution of cylinder walls/caps

# ─── DEMO TRAJECTORIES ─────────────────────────────────────────────────────────
# Target Q(u): closed curve on cylinder of radius FOLLOWER_RADIUS.
# Parameterized as angle φ(u) about Z and height z(u),
# giving world coords (FOLLOWER_RADIUS*cos(φ), FOLLOWER_RADIUS*sin(φ), z).
# The cam's rotation uniformly covers φ_cam ∈ [0,2π]; the follower's
# own angular DOF φ_follower can vary independently.

def demo_lemniscate(n=N_THETA):
    """Figure-8: follower swings ±φ_max while rising and falling twice per revolution."""
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    phi = np.radians(35) * np.sin(t)             # angular swing ±35°
    z   = CAM_HEIGHT/2 + (CAM_HEIGHT*0.38)*np.sin(2*t)  # two up-downs per rev
    return np.column_stack([FOLLOWER_RADIUS*np.cos(phi),
                             FOLLOWER_RADIUS*np.sin(phi), z])

def demo_sine(n=N_THETA):
    """Pure axial sine wave — classic barrel cam."""
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    phi = np.zeros(n)
    z   = CAM_HEIGHT/2 + (CAM_HEIGHT*0.4)*np.sin(t)
    return np.column_stack([FOLLOWER_RADIUS*np.cos(phi),
                             FOLLOWER_RADIUS*np.sin(phi), z])

def demo_star(n=N_THETA):
    """5-pointed star: follower traces a star in (φ, z) space."""
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    r_mod  = 1 + 0.35*np.cos(5*t)
    phi    = np.radians(45) * r_mod * np.sin(t)
    z      = CAM_HEIGHT/2 + (CAM_HEIGHT*0.38) * r_mod * np.cos(t)
    return np.column_stack([FOLLOWER_RADIUS*np.cos(phi),
                             FOLLOWER_RADIUS*np.sin(phi), z])

def demo_spiral(n=N_THETA):
    """Compound axial + angular motion."""
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    phi = np.radians(25) * np.sin(t)
    z   = CAM_HEIGHT/2 + (CAM_HEIGHT*0.40)*np.sin(t) + (CAM_HEIGHT*0.12)*np.sin(3*t)
    return np.column_stack([FOLLOWER_RADIUS*np.cos(phi),
                             FOLLOWER_RADIUS*np.sin(phi), z])

def demo_trefoil(n=N_THETA):
    """Three-lobed pattern."""
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    phi = np.radians(40) * np.sin(3*t) / (1 + 0.3*np.cos(t))
    z   = CAM_HEIGHT/2 + (CAM_HEIGHT*0.35)*np.sin(2*t) + (CAM_HEIGHT*0.1)*np.cos(4*t)
    return np.column_stack([FOLLOWER_RADIUS*np.cos(phi),
                             FOLLOWER_RADIUS*np.sin(phi), z])

DEMOS = dict(lemniscate=demo_lemniscate, sine=demo_sine,
             star=demo_star, spiral=demo_spiral, trefoil=demo_trefoil)

# ─── UTILITY ────────────────────────────────────────────────────────────────────

def resample_arclength(pts, n):
    """Resample closed curve to n uniformly-spaced points by arc-length."""
    closed = np.vstack([pts, pts[0]])
    seg    = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumlen = np.concatenate([[0], np.cumsum(seg)])
    t_new  = np.linspace(0, cumlen[-1], n, endpoint=False)
    return np.column_stack([np.interp(t_new, cumlen,
                                      np.append(pts[:,i], pts[0,i])) for i in range(3)])

def Rz(theta):
    """3×3 rotation matrix about Z by theta radians."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=float)

def load_csv(path):
    pts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = [p.strip() for p in line.replace(';',',').split(',')]
            if len(parts) >= 3:
                try: pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except ValueError: continue
    if len(pts) < 4:
        raise ValueError(f"Need ≥4 points in CSV, got {len(pts)}")
    return np.array(pts, dtype=float)

def project_to_cylinder(pts, n=N_THETA):
    """Project arbitrary 3D curve onto cylinder of radius FOLLOWER_RADIUS."""
    resampled = resample_arclength(pts, n)
    xy  = resampled[:, :2]
    r   = np.linalg.norm(xy, axis=1, keepdims=True)
    r   = np.where(r < 1e-9, 1.0, r)
    xy  = xy / r * FOLLOWER_RADIUS
    z   = resampled[:, 2]
    mg  = CAM_HEIGHT * 0.10
    zmn, zmx = z.min(), z.max()
    z   = mg + (z - zmn)/(zmx - zmn + 1e-12) * (CAM_HEIGHT - 2*mg) if zmx > zmn else np.full(n, CAM_HEIGHT/2)
    return np.column_stack([xy, z])

# ─── CORE: PITCH CURVE COMPUTATION ─────────────────────────────────────────────

def compute_pitch_curve(Q):
    """
    Compute pitch curve C(s) in cam's local frame.

    Paper Section 4.2, Eq. 2:  p_p(t) = T_c(t) * C(s)
    => C(s) = T_c(t)^{-1} * p_p(t) = R_z(-θ_c) * Q(u)

    With uniform timing θ_c = 2π * i/N.
    The pitch curve is the target trajectory "wound" into the cam frame.
    """
    N = len(Q)
    C = np.zeros((N, 3))
    for i in range(N):
        theta_c = 2 * np.pi * i / N
        C[i] = Rz(-theta_c) @ Q[i]
    return C

# ─── FRENET FRAMES ──────────────────────────────────────────────────────────────

def compute_frames(C):
    """
    Parallel-transport Frenet frames along closed curve C.
    Returns T (tangent), N (normal), B (binormal), each (n,3).
    """
    n = len(C)
    T = np.zeros((n, 3)); N_f = np.zeros((n, 3)); B_f = np.zeros((n, 3))

    for i in range(n):
        t = C[(i+1)%n] - C[(i-1)%n]
        l = np.linalg.norm(t)
        T[i] = t/l if l > 1e-12 else np.array([1.,0.,0.])

    # Seed normal perpendicular to T[0]
    ref = np.array([0.,0.,1.]) if abs(T[0,2]) < 0.9 else np.array([1.,0.,0.])
    n0  = ref - np.dot(ref, T[0]) * T[0]
    N_f[0] = n0 / np.linalg.norm(n0)
    B_f[0] = np.cross(T[0], N_f[0])

    for i in range(1, n):
        # Rodrigues rotation to transport N_{i-1} to be perpendicular to T[i]
        v  = np.cross(T[i-1], T[i])
        vl = np.linalg.norm(v)
        if vl < 1e-10:
            N_f[i] = N_f[i-1]; B_f[i] = B_f[i-1]
        else:
            v  /= vl
            phi = np.arcsin(np.clip(vl, -1., 1.))
            ni = N_f[i-1]
            N_f[i] = ni*np.cos(phi) + np.cross(v,ni)*np.sin(phi) + v*np.dot(v,ni)*(1-np.cos(phi))
            nl = np.linalg.norm(N_f[i])
            N_f[i] /= nl if nl > 1e-12 else 1.
            B_f[i]  = np.cross(T[i], N_f[i])

    return T, N_f, B_f

# ─── GROOVE TUBE MESH ───────────────────────────────────────────────────────────

def build_groove_tube(C, T, N_f, B_f):
    """
    Build a swept tube along the pitch curve.

    Cross-section: U-channel open toward the cam's outer surface.
    - Width  = 2*(GROOVE_BALL_R + GROOVE_WALL)  (follower ball + walls)
    - Depth  = GROOVE_BALL_R + GROOVE_WALL       (deep enough to capture ball)
    - The opening faces RADIALLY OUTWARD (toward cam surface),
      letting the follower arm exit and connect to the mechanism.

    The groove tube surface, combined with the cam cylinder, defines
    the groove channel that guides the follower ball.
    """
    n    = len(C)
    hw   = GROOVE_BALL_R + GROOVE_WALL          # half-width
    dep  = GROOVE_BALL_R + GROOVE_WALL          # depth

    # Build U-channel cross-section profile (in local 2D frame)
    # u-axis = lateral (along cam tangential direction on surface)
    # v-axis = radially inward (into the cam body)
    # Profile goes: left-wall-top → left-wall-bottom → bottom-arc → right-wall-bottom → right-wall-top
    np_bot  = max(N_PROFILE, 8)
    np_side = 4
    profile_uv = []

    # Left wall (u = -hw, v from 0 to dep)
    for k in range(np_side):
        profile_uv.append((-hw, k * dep / (np_side - 1)))

    # Bottom arc (semicircle at bottom of groove)
    for k in range(np_bot):
        angle = np.pi + np.pi * k / (np_bot - 1)   # 180° → 360° (bottom half)
        profile_uv.append((hw * np.cos(angle), dep - GROOVE_BALL_R * np.sin(angle - np.pi)))

    # Right wall (u = +hw, v from dep to 0)
    for k in range(np_side):
        profile_uv.append((hw, dep - k * dep / (np_side - 1)))

    profile_uv = np.array(profile_uv)
    n_prof = len(profile_uv)

    verts = np.zeros((n * n_prof, 3))

    for i in range(n):
        c  = C[i]
        t  = T[i]

        # Radially outward direction in cam frame at this pitch curve point
        cxy   = c[:2]
        cr    = np.linalg.norm(cxy)
        r_out = np.array([cxy[0]/cr, cxy[1]/cr, 0.]) if cr > 1e-9 else np.array([1.,0.,0.])

        # u-axis: tangential along the groove (perpendicular to both T and r_out)
        u_axis = np.cross(t, r_out)
        ul = np.linalg.norm(u_axis)
        if ul < 1e-9:
            u_axis = np.cross(t, np.array([0.,0.,1.]))
            ul = np.linalg.norm(u_axis)
        u_axis /= ul

        # v-axis: radially inward (into cam body, for groove depth)
        v_axis = -r_out

        for j, (pu, pv) in enumerate(profile_uv):
            verts[i * n_prof + j] = c + pu * u_axis + pv * v_axis

    # Triangle strip connecting rings (closed loop)
    tris = []
    for i in range(n):
        i2 = (i + 1) % n
        for j in range(n_prof - 1):
            a = i  * n_prof + j
            b = i2 * n_prof + j
            c2 = i2 * n_prof + j + 1
            d  = i  * n_prof + j + 1
            # Winding: normals point INTO the groove (toward follower ball)
            tris.append((a, c2, b))
            tris.append((a, d,  c2))

    return verts, np.array(tris, dtype=np.int32)

# ─── CYLINDER BODY ──────────────────────────────────────────────────────────────

def build_hollow_cylinder(outer_r, inner_r, height, n=N_CAP):
    """Closed hollow cylinder (annular tube): outer wall, inner wall, top cap, bottom cap."""
    verts = []

    def ring(r, z, count=n):
        return [[r*np.cos(2*np.pi*i/count), r*np.sin(2*np.pi*i/count), z]
                for i in range(count)]

    ob = len(verts); verts.extend(ring(outer_r, 0))
    ot = len(verts); verts.extend(ring(outer_r, height))
    ib = len(verts); verts.extend(ring(inner_r, 0))
    it = len(verts); verts.extend(ring(inner_r, height))

    tris = []
    for i in range(n):
        i2 = (i+1) % n
        # Outer wall (normals outward)
        tris += [(ob+i, ob+i2, ot+i2), (ob+i, ot+i2, ot+i)]
        # Inner wall (normals inward → flipped winding)
        tris += [(ib+i, it+i,  it+i2), (ib+i, it+i2, ib+i2)]
        # Bottom annular cap (normal −z)
        tris += [(ib+i, ib+i2, ob+i2), (ib+i, ob+i2, ob+i)]
        # Top annular cap (normal +z)
        tris += [(it+i, ot+i,  ot+i2), (it+i, ot+i2, it+i2)]

    return np.array(verts, dtype=float), np.array(tris, dtype=np.int32)

# ─── STL WRITER ─────────────────────────────────────────────────────────────────

def tri_normal(v0, v1, v2):
    e1 = v1-v0; e2 = v2-v0
    n  = np.cross(e1, e2)
    l  = np.linalg.norm(n)
    return n/l if l > 1e-15 else np.array([0.,0.,1.])

def write_stl(path, meshes):
    """meshes: list of (verts Nx3, tris Mx3)"""
    all_tris = []
    for verts, tris in meshes:
        for tri in tris:
            all_tris.append((verts[tri[0]], verts[tri[1]], verts[tri[2]]))
    with open(path, 'wb') as f:
        f.write(b'3D Cam - Cheng 2021 camMech_1T1R'.ljust(80, b'\x00'))
        f.write(struct.pack('<I', len(all_tris)))
        for v0, v1, v2 in all_tris:
            f.write(struct.pack('<3f', *tri_normal(v0,v1,v2)))
            f.write(struct.pack('<3f', *v0))
            f.write(struct.pack('<3f', *v1))
            f.write(struct.pack('<3f', *v2))
            f.write(struct.pack('<H', 0))
    return len(all_tris)

# ─── MAIN ───────────────────────────────────────────────────────────────────────

def generate_cam(Q_input, output_path):
    print(f"\n{'='*60}")
    print(f"  3D Cam Generator  (camMech_1T1R, Cheng et al. 2021)")
    print(f"{'='*60}")
    print(f"  Cam R={CAM_RADIUS}mm  H={CAM_HEIGHT}mm  shaft={SHAFT_RADIUS}mm")
    print(f"  Follower R={FOLLOWER_RADIUS}mm  Ball R={GROOVE_BALL_R}mm  Wall={GROOVE_WALL}mm")
    print()

    # 1. Resample target trajectory uniformly by arc-length
    print("  [1/5]  Resampling target trajectory...")
    Q = resample_arclength(Q_input, N_THETA)
    z_travel = Q[:,2].max() - Q[:,2].min()
    phi = np.arctan2(Q[:,1], Q[:,0])
    phi_travel = np.degrees(phi.max() - phi.min())
    print(f"         Z travel: {z_travel:.1f} mm   φ travel: {phi_travel:.1f}°")

    # 2. Compute pitch curve C(s) = R_z(-θ_c) * Q(u)
    print("  [2/5]  Computing pitch curve in cam frame...")
    print("         (Paper Eq.2: C(s) = R_z(-θ_c) * Q(u))")
    C = compute_pitch_curve(Q)
    r_C = np.linalg.norm(C[:,:2], axis=1)
    closure = np.linalg.norm(C[0] - C[-1])
    print(f"         Pitch curve radii: {r_C.min():.2f} – {r_C.max():.2f} mm")
    print(f"         Z range: {C[:,2].min():.1f} – {C[:,2].max():.1f} mm")
    print(f"         Closure error: {closure:.4f} mm  {'✓' if closure < 0.5 else '⚠ check params'}")

    if r_C.min() < SHAFT_RADIUS + 3:
        print(f"  ⚠  WARNING: Pitch curve too close to shaft!")
    if r_C.max() > CAM_RADIUS - GROOVE_BALL_R - GROOVE_WALL - 1:
        print(f"  ⚠  WARNING: Pitch curve may exceed cam radius!")

    # 3. Frenet frames
    print("  [3/5]  Computing Frenet frames...")
    T_f, N_f, B_f = compute_frames(C)

    # 4. Build meshes
    print("  [4/5]  Building geometry...")
    groove_verts, groove_tris = build_groove_tube(C, T_f, N_f, B_f)
    cyl_verts,    cyl_tris    = build_hollow_cylinder(CAM_RADIUS, SHAFT_RADIUS, CAM_HEIGHT)
    print(f"         Cylinder: {len(cyl_tris):,} tris")
    print(f"         Groove:   {len(groove_tris):,} tris")

    # 5. Write STL
    print(f"  [5/5]  Writing {output_path} ...")
    total = write_stl(output_path, [(cyl_verts, cyl_tris), (groove_verts, groove_tris)])
    print(f"         Total triangles: {total:,}")
    print(f"\n  ✓  Done: {output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    args = sys.argv[1:]
    csv_file = out_file = None
    demo_name = "lemniscate"

    for a in args:
        al = a.lower()
        if al.endswith('.csv') or al.endswith('.txt'): csv_file = a
        elif al.endswith('.stl'):                       out_file = a
        elif al in DEMOS:                              demo_name = al

    if csv_file:
        if not os.path.exists(csv_file):
            print(f"ERROR: '{csv_file}' not found"); sys.exit(1)
        print(f"Loading: {csv_file}")
        Q = project_to_cylinder(load_csv(csv_file))
    else:
        print(f"Demo: '{demo_name}'  (options: {', '.join(DEMOS)})")
        print(f"Usage: python cam_generator.py [curve.csv] [out.stl] [demo_name]\n")
        Q = DEMOS[demo_name]()

    if not out_file:
        base = os.path.splitext(csv_file)[0] if csv_file else demo_name
        out_file = f"cam_{base}.stl"

    generate_cam(Q, out_file)
