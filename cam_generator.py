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

FOLLOWER GEOMETRY (camMech_1T1R):
  The follower is an L-shaped assembly consisting of:
    1. Shaft sleeve  — hollow cylinder around the central shaft,
                       slides and rotates freely (cylindrical joint).
                       Inner radius = SHAFT_RADIUS + FOLLOWER_CLEARANCE
                       Outer radius = SHAFT_RADIUS + FOLLOWER_SLEEVE_WALL
                       Height       = CAM_HEIGHT
    2. Horizontal arm — solid cylinder extending radially at Z=CAM_HEIGHT/2,
                        from sleeve outer surface to FOLLOWER_RADIUS.
                        Radius = GROOVE_BALL_R + GROOVE_WALL (same as groove width)
    3. Hemisphere tip — half-sphere at the arm end, flat face flush with arm tip,
                        dome pointing radially inward into the groove.
                        Radius = GROOVE_BALL_R
  All dimensions auto-derive from the cam parameters — no extra constants needed
  except FOLLOWER_CLEARANCE (fit gap) and FOLLOWER_SLEEVE_WALL (wall thickness).

USAGE:
  python cam_generator.py                   # built-in lemniscate demo
  python cam_generator.py sine              # other built-in demos
  python cam_generator.py my_path.csv       # your own trajectory
  python cam_generator.py my_path.csv out.stl

  # Follower STL is always written alongside the cam STL:
  #   <base>_cam.stl  and  <base>_follower.stl

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

# Follower-specific parameters (derived from cam params where possible)
FOLLOWER_CLEARANCE  = 0.2   # mm  radial clearance between shaft and sleeve bore
FOLLOWER_SLEEVE_WALL = 3.0  # mm  sleeve wall thickness (bore to outer surface)

N_THETA   = 360   # angular resolution of pitch curve
N_PROFILE =  16   # cross-section resolution of groove tube
N_CAP     = 180   # resolution of cylinder walls/caps
N_SPHERE  =  32   # latitude/longitude resolution of hemisphere tip

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

# ─── FOLLOWER GEOMETRY ──────────────────────────────────────────────────────────
# The follower is an L-shaped assembly:
#
#   Z
#   ▲
#   │  ┌──────────────────────────────────┐  ← top cap
#   │  │         shaft sleeve             │
#   │  │  (hollow cylinder, slides on Z   │
#   │  │   and rotates about Z freely)    │
#   │  │                                  │
#   │  │          ┌═══════════════╗       │  ← horizontal arm at Z=CAM_HEIGHT/2
#   │  │          │    arm        ║●)     │    hemisphere tip at FOLLOWER_RADIUS
#   │  │          └═══════════════╝       │
#   │  │                                  │
#   │  └──────────────────────────────────┘  ← bottom cap
#   └──────────────────────────────────────► X
#      ├──┤                        ├──────┤
#   shaft bore              arm length
#
# Dimensions (all auto-derived from cam parameters):
#   Sleeve bore inner radius = SHAFT_RADIUS + FOLLOWER_CLEARANCE
#   Sleeve outer radius      = SHAFT_RADIUS + FOLLOWER_CLEARANCE + FOLLOWER_SLEEVE_WALL
#   Sleeve height            = CAM_HEIGHT
#   Arm inner radius (rod)   = GROOVE_BALL_R + GROOVE_WALL   (matches groove half-width)
#   Arm start                = sleeve outer radius
#   Arm end                  = FOLLOWER_RADIUS               (ball center at pitch radius)
#   Arm Z center             = CAM_HEIGHT / 2
#   Hemisphere radius        = GROOVE_BALL_R
#   Hemisphere flat face     = flush with arm tip (at FOLLOWER_RADIUS)
#   Hemisphere dome          = points radially inward (−X direction)

def build_solid_cylinder(outer_r, inner_r, z_bot, z_top, n=N_CAP):
    """
    Annular cylinder (tube) with closed end caps.
    Outer normal outward, inner normal inward (manifold solid).
    """
    verts = []

    def ring(r, z):
        return [[r*np.cos(2*np.pi*i/n), r*np.sin(2*np.pi*i/n), z] for i in range(n)]

    ob = len(verts); verts.extend(ring(outer_r, z_bot))
    ot = len(verts); verts.extend(ring(outer_r, z_top))
    ib = len(verts); verts.extend(ring(inner_r, z_bot))
    it = len(verts); verts.extend(ring(inner_r, z_top))

    tris = []
    for i in range(n):
        i2 = (i+1) % n
        tris += [(ob+i, ob+i2, ot+i2), (ob+i, ot+i2, ot+i)]   # outer wall
        tris += [(ib+i, it+i,  it+i2), (ib+i, it+i2, ib+i2)]  # inner wall (flipped)
        tris += [(ib+i, ib+i2, ob+i2), (ib+i, ob+i2, ob+i)]   # bottom cap
        tris += [(it+i, ot+i,  ot+i2), (it+i, ot+i2, it+i2)]  # top cap

    return np.array(verts, dtype=float), np.array(tris, dtype=np.int32)


def build_follower_arm(sleeve_outer_r, arm_end_r, arm_r, arm_z, n=N_CAP):
    """
    Solid horizontal arm (filled cylinder) aligned with +X axis,
    spanning from sleeve_outer_r to arm_end_r at height arm_z.

    Because the arm is a radial rod (not an annulus), inner_r = 0 would give
    a solid disk cap, but we model it as a thin-walled tube with a solid end
    cap at each end for cleanliness. We use inner_r=0 (solid rod).

    The arm is a capped solid cylinder lying along X, so we build it as
    a stack of rings swept from x=sleeve_outer_r to x=arm_end_r along +X,
    each ring in the Y-Z plane.
    """
    x0 = sleeve_outer_r
    x1 = arm_end_r   # stop before hemisphere center so tip is flush

    verts = []
    tris  = []

    # Ring at x0 (back, facing −X)
    base = len(verts)
    for i in range(n):
        angle = 2*np.pi*i/n
        verts.append([x0, arm_r*np.cos(angle), arm_z + arm_r*np.sin(angle)])

    # Ring at x1 (front, facing +X)
    front = len(verts)
    for i in range(n):
        angle = 2*np.pi*i/n
        verts.append([x1, arm_r*np.cos(angle), arm_z + arm_r*np.sin(angle)])

    # Side wall
    for i in range(n):
        i2 = (i+1) % n
        a, b, c2, d = base+i, front+i, front+i2, base+i2
        tris += [(a, b, c2), (a, c2, d)]

    # Back cap (normal −X): fan from center at x0
    ctr_back = len(verts)
    verts.append([x0, 0.0, arm_z])
    for i in range(n):
        i2 = (i+1) % n
        # Winding for −X normal: clockwise when viewed from −X
        tris.append((ctr_back, base+i2, base+i))

    # Front cap (normal +X): fan from center at x1
    # NOTE: The front cap is NOT added here because the hemisphere base
    # will be welded flush against it.  We leave the front open so the
    # hemisphere base disk seals it.

    return np.array(verts, dtype=float), np.array(tris, dtype=np.int32)


def build_hemisphere_tip(center_x, arm_z, n_lat=N_SPHERE, n_lon=N_SPHERE):
    """
    Hemisphere with flat face at x=center_x, dome pointing in the −X direction.

    The flat face disk seals the open end of the follower arm.
    The dome sits inside the cam groove, contacting the groove walls.

      flat face (x = center_x)     dome apex (x = center_x − GROOVE_BALL_R)
           |←────── GROOVE_BALL_R ──────→|
           ●━━━━━━━━━━━━━━━━━━━━━━━━━━━━●
           │         hemisphere          │
           ●━━━━━━━━━━━━━━━━━━━━━━━━━━━━●

    Latitude 0 = equator (the flat rim), latitude π/2 = dome apex.
    """
    R  = GROOVE_BALL_R
    cx = center_x   # x-coordinate of the flat face / ball center
    cy = 0.0        # arm lies along +X axis at y=0
    cz = arm_z      # arm Z-center

    verts = []
    tris  = []

    # Build latitude rings from equator (lat=0) to near-apex
    # lat goes 0 → π/2 (equator to pole)
    n_lat_rings = max(n_lat // 2, 4)
    lats = np.linspace(0, np.pi/2, n_lat_rings + 1)

    rings = []
    for lat in lats:
        ring_r = R * np.cos(lat)   # radius of this latitude circle
        ring_x = cx - R * np.sin(lat)  # x recedes into −X as lat increases
        ring = []
        for j in range(n_lon):
            angle = 2*np.pi*j/n_lon
            ring.append([ring_x,
                         cy + ring_r*np.cos(angle),
                         cz + ring_r*np.sin(angle)])
        rings.append(ring)

    # Flatten rings into vertex list
    ring_starts = []
    for ring in rings:
        ring_starts.append(len(verts))
        verts.extend(ring)

    # Apex vertex
    apex_idx = len(verts)
    verts.append([cx - R, cy, cz])

    # Side quads between consecutive latitude rings
    for ri in range(len(rings) - 1):
        r0 = ring_starts[ri]
        r1 = ring_starts[ri + 1]
        for j in range(n_lon):
            j2 = (j+1) % n_lon
            a, b  = r0+j,  r0+j2
            c2, d = r1+j2, r1+j
            tris += [(a, b, c2), (a, c2, d)]

    # Apex fan from last ring
    last_r = ring_starts[-1]
    for j in range(n_lon):
        j2 = (j+1) % n_lon
        tris.append((last_r+j, apex_idx, last_r+j2))

    # Flat base disk (normal +X, seals the arm opening)
    # Fan from center of flat face outward
    flat_center_idx = len(verts)
    verts.append([cx, cy, cz])
    equator_start = ring_starts[0]
    for j in range(n_lon):
        j2 = (j+1) % n_lon
        # +X normal: counter-clockwise when viewed from +X
        tris.append((flat_center_idx, equator_start+j2, equator_start+j))

    return np.array(verts, dtype=float), np.array(tris, dtype=np.int32)


def build_follower():
    """
    Assemble the complete L-shaped follower as a single combined mesh.

    Parts:
      1. Shaft sleeve  — hollow cylinder, bore = SHAFT_RADIUS + FOLLOWER_CLEARANCE,
                         outer = bore + FOLLOWER_SLEEVE_WALL, height = CAM_HEIGHT
      2. Horizontal arm — solid rod along +X, from sleeve outer to FOLLOWER_RADIUS,
                          radius = GROOVE_BALL_R + GROOVE_WALL
      3. Hemisphere tip — GROOVE_BALL_R dome at x=FOLLOWER_RADIUS, into the groove

    The follower rests at its HOME position: arm pointing along +X at Z=CAM_HEIGHT/2.
    The shaft bore is centered on the Z-axis, matching the cam's rotation axis.
    """
    bore_r   = SHAFT_RADIUS + FOLLOWER_CLEARANCE
    sleeve_r = bore_r + FOLLOWER_SLEEVE_WALL
    arm_r    = GROOVE_BALL_R + GROOVE_WALL      # arm cross-section radius
    arm_z    = CAM_HEIGHT / 2.0                 # arm exits sleeve at mid-height
    arm_end  = FOLLOWER_RADIUS                  # arm tip x-coordinate (ball center)

    print(f"         Sleeve: bore={bore_r:.1f}mm  outer={sleeve_r:.1f}mm  H={CAM_HEIGHT:.1f}mm")
    print(f"         Arm:    R={arm_r:.1f}mm  from x={sleeve_r:.1f} to x={arm_end:.1f}mm  at Z={arm_z:.1f}mm")
    print(f"         Tip:    hemisphere R={GROOVE_BALL_R:.1f}mm  flat at x={arm_end:.1f}mm  dome→−X")

    # 1. Shaft sleeve
    sv, st = build_solid_cylinder(sleeve_r, bore_r, 0, CAM_HEIGHT)

    # 2. Horizontal arm (leaves sleeve outer wall, ends at ball center)
    av, at = build_follower_arm(sleeve_r, arm_end, arm_r, arm_z)

    # 3. Hemisphere tip
    hv, ht = build_hemisphere_tip(arm_end, arm_z)

    # Combine all meshes: offset triangle indices by cumulative vertex counts
    offset_a = len(sv)
    offset_h = offset_a + len(av)

    at_off = at + offset_a
    ht_off = ht + offset_h

    all_verts = np.vstack([sv, av, hv])
    all_tris  = np.vstack([st, at_off, ht_off])

    print(f"         Follower tris: sleeve={len(st):,}  arm={len(at):,}  tip={len(ht):,}  total={len(all_tris):,}")

    return all_verts, all_tris

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

def generate_cam(Q_input, cam_path, follower_path):
    print(f"\n{'='*60}")
    print(f"  3D Cam Generator  (camMech_1T1R, Cheng et al. 2021)")
    print(f"{'='*60}")
    print(f"  Cam R={CAM_RADIUS}mm  H={CAM_HEIGHT}mm  shaft={SHAFT_RADIUS}mm")
    print(f"  Follower R={FOLLOWER_RADIUS}mm  Ball R={GROOVE_BALL_R}mm  Wall={GROOVE_WALL}mm")
    print()

    # 1. Resample target trajectory uniformly by arc-length
    print("  [1/6]  Resampling target trajectory...")
    Q = resample_arclength(Q_input, N_THETA)
    z_travel = Q[:,2].max() - Q[:,2].min()
    phi = np.arctan2(Q[:,1], Q[:,0])
    phi_travel = np.degrees(phi.max() - phi.min())
    print(f"         Z travel: {z_travel:.1f} mm   φ travel: {phi_travel:.1f}°")

    # 2. Compute pitch curve C(s) = R_z(-θ_c) * Q(u)
    print("  [2/6]  Computing pitch curve in cam frame...")
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
    print("  [3/6]  Computing Frenet frames...")
    T_f, N_f, B_f = compute_frames(C)

    # 4. Build cam meshes
    print("  [4/6]  Building cam geometry...")
    groove_verts, groove_tris = build_groove_tube(C, T_f, N_f, B_f)
    cyl_verts,    cyl_tris    = build_hollow_cylinder(CAM_RADIUS, SHAFT_RADIUS, CAM_HEIGHT)
    print(f"         Cylinder: {len(cyl_tris):,} tris")
    print(f"         Groove:   {len(groove_tris):,} tris")

    # 5. Build follower mesh
    print("  [5/6]  Building follower geometry...")
    fol_verts, fol_tris = build_follower()

    # 6. Write STLs
    print(f"  [6/6]  Writing STL files...")
    cam_total = write_stl(cam_path,      [(cyl_verts, cyl_tris), (groove_verts, groove_tris)])
    fol_total = write_stl(follower_path, [(fol_verts, fol_tris)])
    print(f"         Cam:      {cam_total:,} tris  →  {cam_path}")
    print(f"         Follower: {fol_total:,} tris  →  {follower_path}")
    print(f"\n  ✓  Done.")
    print(f"{'='*60}\n")
    print(f"  ASSEMBLY NOTES:")
    print(f"    • Follower home position: arm along +X, tip at x={FOLLOWER_RADIUS:.1f}mm, z={CAM_HEIGHT/2:.1f}mm")
    print(f"    • Sleeve bore inner R = {SHAFT_RADIUS + FOLLOWER_CLEARANCE:.2f}mm  (shaft R + {FOLLOWER_CLEARANCE}mm clearance)")
    print(f"    • Arm tip hemisphere sits in cam groove at R={FOLLOWER_RADIUS:.1f}mm")
    print(f"    • Both parts share the same origin (Z-axis = cam/shaft axis)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    args = sys.argv[1:]
    csv_file = out_cam = None
    demo_name = "lemniscate"

    for a in args:
        al = a.lower()
        if al.endswith('.csv') or al.endswith('.txt'): csv_file = a
        elif al.endswith('.stl'):
            if out_cam is None: out_cam = a
        elif al in DEMOS: demo_name = al

    if csv_file:
        if not os.path.exists(csv_file):
            print(f"ERROR: '{csv_file}' not found"); sys.exit(1)
        print(f"Loading: {csv_file}")
        Q = project_to_cylinder(load_csv(csv_file))
        base = os.path.splitext(csv_file)[0]
    else:
        print(f"Demo: '{demo_name}'  (options: {', '.join(DEMOS)})")
        print(f"Usage: python cam_generator.py [curve.csv] [out_cam.stl] [demo_name]\n")
        Q = DEMOS[demo_name]()
        base = demo_name

    # Derive output paths: <base>_cam.stl and <base>_follower.stl
    if out_cam is None:
        out_cam = f"{base}_cam.stl"
    # Follower always mirrors cam filename with _follower suffix
    cam_stem      = out_cam[:-4] if out_cam.endswith('.stl') else out_cam
    # Strip trailing _cam if present, then add _follower
    if cam_stem.endswith('_cam'):
        fol_stem = cam_stem[:-4]
    else:
        fol_stem = cam_stem
    out_follower = f"{fol_stem}_follower.stl"

    generate_cam(Q, out_cam, out_follower)