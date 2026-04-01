"""
camMech_2T Follower Mechanism

Holder: rectangular prism with a rectangular channel cut through it.
        Split into top and bottom halves for assembly around the stick.
        Channel is WIDE in X (left/right travel), TIGHT in Z (no up/down).

Stick:  long rod, round outside the holder, SQUARE inside the holder
        so it cannot rotate. Sphere follower at one end. Small upward
        tip at the other end.

Outputs:
  follower_stick.stl         – the moving part
  follower_holder_bottom.stl – lower half of the housing
  follower_holder_top.stl    – upper half of the housing
"""

import numpy as np
import struct, os

# ── Parameters ────────────────────────────────────────────────────────────────
STICK_SQ    =  5.0   # mm  square cross-section side (inside holder)
STICK_R     =  2.5   # mm  round shaft radius (outside holder)
STICK_L     = 90.0   # mm  total stick length (along Y)
BALL_R      =  3.5   # mm  follower sphere radius
TIP_R       =  1.5   # mm  upward tip cylinder radius
TIP_H       =  5.0   # mm  upward tip cylinder height
TIP_CONE_H  =  2.0   # mm  cone point on top of tip

HOLDER_X    = 36.0   # mm  holder width  (X) — sets left/right travel room
HOLDER_Y    = 22.0   # mm  holder depth  (Y) — stick slides through this
HOLDER_Z    = 14.0   # mm  holder height (Z)
WALL_X      =  5.0   # mm  wall thickness left/right of channel
WALL_Z      =  4.0   # mm  wall thickness above/below channel
SLOT_CL     =  0.3   # mm  clearance between stick square and channel

# Derived channel dimensions
CHANNEL_X   = HOLDER_X - 2 * WALL_X        # wide: allows left/right travel
CHANNEL_Z   = STICK_SQ + 2 * SLOT_CL       # tight: prevents up/down + rotation

# Square section of stick: long enough to stay engaged as stick travels
SQ_SECTION_L = HOLDER_Y + CHANNEL_X        # covers full travel range inside

N_LON = 32
N_LAT = 20


# ── Mesh helpers ──────────────────────────────────────────────────────────────

def merge(*meshes):
    verts_all, tris_all, offset = [], [], 0
    for v, t in meshes:
        verts_all.append(v)
        tris_all.append(t + offset)
        offset += len(v)
    return np.vstack(verts_all), np.vstack(tris_all)


def box_mesh(x0, x1, y0, y1, z0, z1):
    """Closed solid box, normals outward."""
    v = np.array([
        [x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
        [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1],
    ], dtype=float)
    t = np.array([
        [0,2,1],[0,3,2],  # bottom  -Z
        [4,5,6],[4,6,7],  # top     +Z
        [0,1,5],[0,5,4],  # front   -Y
        [3,7,6],[3,6,2],  # back    +Y
        [0,4,7],[0,7,3],  # left    -X
        [1,2,6],[1,6,5],  # right   +X
    ], dtype=np.int32)
    return v, t


def cylinder_mesh(cx, cy, z0, z1, r):
    """Closed solid cylinder along Z axis."""
    verts, tris = [], []
    bc = len(verts); verts.append([cx, cy, z0])
    tc = len(verts); verts.append([cx, cy, z1])
    br = len(verts)
    for j in range(N_LON):
        a = 2*np.pi*j/N_LON
        verts.append([cx + r*np.cos(a), cy + r*np.sin(a), z0])
    tr = len(verts)
    for j in range(N_LON):
        a = 2*np.pi*j/N_LON
        verts.append([cx + r*np.cos(a), cy + r*np.sin(a), z1])
    for j in range(N_LON):
        j2 = (j+1) % N_LON
        tris += [[br+j, tr+j2, tr+j], [br+j, br+j2, tr+j2]]
        tris.append([bc, br+j, br+j2])
        tris.append([tc, tr+j2, tr+j])
    return np.array(verts, dtype=float), np.array(tris, dtype=np.int32)


def sphere_mesh(cx, cy, cz, r):
    """Full UV sphere."""
    verts, tris = [], []
    south = 0; verts.append([cx, cy, cz - r])
    lats = np.linspace(-np.pi/2, np.pi/2, N_LAT + 2)[1:-1]
    rings = []
    for lat in lats:
        rl = r * np.cos(lat); z = cz + r * np.sin(lat)
        rb = len(verts); rings.append(rb)
        for j in range(N_LON):
            a = 2*np.pi*j/N_LON
            verts.append([cx + rl*np.cos(a), cy + rl*np.sin(a), z])
    north = len(verts); verts.append([cx, cy, cz + r])
    for j in range(N_LON):
        tris.append([south, rings[0]+(j+1)%N_LON, rings[0]+j])
    for i in range(len(rings)-1):
        ra, rb = rings[i], rings[i+1]
        for j in range(N_LON):
            j2 = (j+1)%N_LON
            tris += [[ra+j, ra+j2, rb+j2], [ra+j, rb+j2, rb+j]]
    for j in range(N_LON):
        tris.append([rings[-1]+j, rings[-1]+(j+1)%N_LON, north])
    return np.array(verts, dtype=float), np.array(tris, dtype=np.int32)


def cone_mesh(cx, cy, z_base, z_apex, r_base):
    """Closed cone along Z."""
    verts, tris = [], []
    bci = len(verts); verts.append([cx, cy, z_base])
    api = len(verts); verts.append([cx, cy, z_apex])
    bri = len(verts)
    for j in range(N_LON):
        a = 2*np.pi*j/N_LON
        verts.append([cx + r_base*np.cos(a), cy + r_base*np.sin(a), z_base])
    for j in range(N_LON):
        j2 = (j+1) % N_LON
        tris.append([bci, bri+j2, bri+j])
        tris.append([bri+j, bri+j2, api])
    return np.array(verts, dtype=float), np.array(tris, dtype=np.int32)


# ── Stick ─────────────────────────────────────────────────────────────────────

def build_stick():
    """
    Stick runs along Y axis, centred at origin.

    Layout (Y axis):
      Ball end:   Y = -STICK_L/2  (sphere, contacts cam groove)
      Round shaft: Y = -STICK_L/2 to -SQ_SECTION_L/2  (round, below holder)
      Square section: Y = -SQ_SECTION_L/2 to +SQ_SECTION_L/2  (inside holder)
      Round shaft: Y = +SQ_SECTION_L/2 to +STICK_L/2  (round, above holder)
      Upward tip: on top face of stick at Y = +STICK_L/2 end

    The square section fits the holder channel with SLOT_CL clearance in Z.
    Rotation is blocked because the channel height = STICK_SQ + 2*SLOT_CL only.
    """
    hs = STICK_SQ / 2

    # Square middle section
    sq_v, sq_t = box_mesh(-hs, hs, -SQ_SECTION_L/2, SQ_SECTION_L/2, -hs, hs)

    # Round shaft — ball end (Y negative)
    rnd_bot_v, rnd_bot_t = cylinder_mesh(
        0, 0,   # cx, cy — but cylinder is along Z; need along Y instead
        0, 0, STICK_R   # placeholder — build as box rotated?
    )
    # cylinder_mesh builds along Z, so we build round sections as
    # cylinders along Y by treating Y as the axis manually:
    def cyl_along_y(y0, y1, r):
        verts, tris = [], []
        bc = len(verts); verts.append([0, y0, 0])
        tc = len(verts); verts.append([0, y1, 0])
        br = len(verts)
        for j in range(N_LON):
            a = 2*np.pi*j/N_LON
            verts.append([r*np.cos(a), y0, r*np.sin(a)])
        tr = len(verts)
        for j in range(N_LON):
            a = 2*np.pi*j/N_LON
            verts.append([r*np.cos(a), y1, r*np.sin(a)])
        for j in range(N_LON):
            j2 = (j+1) % N_LON
            tris += [[br+j, tr+j, tr+j2], [br+j, tr+j2, br+j2]]   # side
            tris.append([bc, br+j2, br+j])   # back cap  (−Y normal)
            tris.append([tc, tr+j, tr+j2])   # front cap (+Y normal)
        return np.array(verts, dtype=float), np.array(tris, dtype=np.int32)

    # Round section: ball end
    bot_shaft_v, bot_shaft_t = cyl_along_y(-STICK_L/2, -SQ_SECTION_L/2, STICK_R)

    # Round section: tip end
    top_shaft_v, top_shaft_t = cyl_along_y(SQ_SECTION_L/2, STICK_L/2, STICK_R)

    # Sphere ball at Y = -STICK_L/2 (tangent to end of shaft)
    ball_v, ball_t = sphere_mesh(0, -STICK_L/2 - BALL_R, 0, BALL_R)

    # Upward tip at the far end (Y = +STICK_L/2), pointing in +Z from top of shaft
    tip_base_z = STICK_R          # top of round shaft
    tip_y      = STICK_L/2        # at the very tip end of stick
    tip_v,  tip_t  = cylinder_mesh(0, tip_y, tip_base_z, tip_base_z + TIP_H, TIP_R)
    cone_v, cone_t = cone_mesh(0, tip_y, tip_base_z + TIP_H,
                                tip_base_z + TIP_H + TIP_CONE_H, TIP_R)

    return merge(
        (sq_v,        sq_t),
        (bot_shaft_v, bot_shaft_t),
        (top_shaft_v, top_shaft_t),
        (ball_v,      ball_t),
        (tip_v,       tip_t),
        (cone_v,      cone_t),
    )


# ── Holder halves ─────────────────────────────────────────────────────────────

def holder_half(is_bottom):
    """
    Build one half of the holder.

    The holder is a rectangular box with a rectangular channel cut through
    it in the Y direction. Channel is centred in X and Z.

    Channel:
      X: -CHANNEL_X/2 to +CHANNEL_X/2  (wide → left/right travel)
      Z: -CHANNEL_Z/2 to +CHANNEL_Z/2  (tight → no vertical play)
      Y: full depth (stick slides through)

    Split at Z=0 (mid-channel):
      bottom half: z from -HOLDER_Z/2 to 0
      top half:    z from 0 to +HOLDER_Z/2

    Each half is built as three boxes:
      1. Full slab (solid, away from channel)
      2. Left wall (beside channel)
      3. Right wall (beside channel)
    """
    hx = HOLDER_X / 2
    hy = HOLDER_Y / 2
    hz = HOLDER_Z / 2
    cx = CHANNEL_X / 2
    cz = CHANNEL_Z / 2

    if is_bottom:
        z0, z1 = -hz, 0.0
        slab_z0, slab_z1 = -hz, -cz        # solid below channel
        wall_z0, wall_z1 = -cz, 0.0        # walls beside lower half of channel
    else:
        z0, z1 = 0.0, hz
        slab_z0, slab_z1 = cz, hz          # solid above channel
        wall_z0, wall_z1 = 0.0, cz         # walls beside upper half of channel

    meshes = []
    # Full slab (no channel here)
    meshes.append(box_mesh(-hx, hx, -hy, hy, slab_z0, slab_z1))
    # Left wall
    meshes.append(box_mesh(-hx, -cx, -hy, hy, wall_z0, wall_z1))
    # Right wall
    meshes.append(box_mesh( cx,  hx, -hy, hy, wall_z0, wall_z1))

    return merge(*meshes)


# ── STL writer ────────────────────────────────────────────────────────────────

def tri_normal(v0, v1, v2):
    n = np.cross(v1 - v0, v2 - v0)
    l = np.linalg.norm(n)
    return n / l if l > 1e-15 else np.array([0., 0., 1.])


def write_stl(path, verts, tris):
    with open(path, 'wb') as f:
        f.write(b'camMech_2T follower'.ljust(80, b'\x00'))
        f.write(struct.pack('<I', len(tris)))
        for tri in tris:
            v0, v1, v2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
            f.write(struct.pack('<3f', *tri_normal(v0, v1, v2)))
            f.write(struct.pack('<3f', *v0))
            f.write(struct.pack('<3f', *v1))
            f.write(struct.pack('<3f', *v2))
            f.write(struct.pack('<H', 0))
    print(f"  Wrote {len(tris):,} tris  →  {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"\ncamMech_2T follower mechanism")
    print(f"  Stick:   round R={STICK_R}mm outside, square {STICK_SQ}×{STICK_SQ}mm inside holder")
    print(f"  Ball:    R={BALL_R} mm at one end")
    print(f"  Tip:     R={TIP_R} mm, H={TIP_H+TIP_CONE_H:.1f} mm, at other end pointing up")
    print(f"  Holder:  {HOLDER_X}×{HOLDER_Y}×{HOLDER_Z} mm outer")
    print(f"  Channel: {CHANNEL_X:.1f} mm wide (X travel)  ×  {CHANNEL_Z:.1f} mm tall (tight fit)\n")

    print("Building stick...")
    sv, st = build_stick()
    write_stl(os.path.join(out_dir, "follower_stick.stl"), sv, st)

    print("Building holder bottom half...")
    bv, bt = holder_half(is_bottom=True)
    write_stl(os.path.join(out_dir, "follower_holder_bottom.stl"), bv, bt)

    print("Building holder top half...")
    tv, tt = holder_half(is_bottom=False)
    write_stl(os.path.join(out_dir, "follower_holder_top.stl"), tv, tt)

    print(f"""
Assembly:
  1. Lay stick into holder bottom half (square section sits in the trough)
  2. Press holder top half down — channel encloses the square section
  3. Screw/pin the two halves together through the walls
  4. Ball end (Y−) hangs into cam groove; tip end (Y+) points upward
  5. Stick slides ±{CHANNEL_X/2 - STICK_SQ/2:.1f} mm left/right, freely forward/back
  6. Zero vertical play — rotation blocked by square-in-rectangle channel
""")
