"""
cam_groove_v3.py
================
Per spec:

Trajectory: x(t), y(t), t in [0,1], N=1000 samples.
  r(t) = R + y(t) - min(y)

Pre-rotation coordinates:
  x_t = x(t)   (axial)
  y_t = 0
  z_t = r(t)

Sphere at time t (pre-rotation):
  centre (x_t, 0, r_t), radius r_g
  --> rotate by theta=2*pi*t around x-axis

Prism at time t (pre-rotation), vertices:
  x in [x_t - l/2, x_t + l/2]   (axial,      l = 2*r_g + P)
  y in [-r_g,       r_g      ]   (tangential, w = 2*r_g)
  z in [0,           h_t     ]   (radial,     h_t = r(t) + 0.8*r_g)
  --> rotate by theta=2*pi*t around x-axis

Final solid = union(rotated prisms) - union(rotated spheres)
"""

import numpy as np
from pathlib import Path
import struct, os
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

inp = input()

# ── Parameters ────────────────────────────────────────────────────────────────
R        = 10.0 # mm
r_g      = 2.5 # mm
P        = 2 # mm
HALF_H   = 15.0 # mm
GRID_RES = 100 # size of 3d grid
CSV_PATH = f"~/Downloads/{inp}.csv"   # List of x,y coordinates
SCALE = 0.2     # Scaling of x,y points (0,1)
COVERAGE = 0.65 # value to determine how much the groove is covered (0,1)

# ── Read arbitrary closed curve from CSV ─────────────────────────────────────
pts = np.loadtxt(Path(CSV_PATH).expanduser(), delimiter=",")
x_t, y_t = pts[:, 0], pts[:, 1]
N = len(x_t)                     
t = np.linspace(0, 1, N, endpoint=False)

# Shift/scale x/y as needed to fit your cam height budget
avg = SCALE * (np.mean(x_t + y_t)) / 2
x_t = (x_t - np.mean(x_t)) / avg
y_t = (y_t - np.min(y_t)) / avg  
print(x_t)
print(y_t)
exit

# Make radius always valid
r_t = R + y_t          # always >= R

theta = 2 * np.pi * t

# Prism dims per spec
l   = 2 * r_g + P                # axial width      (x direction)
w   = 2 * r_g                    # tangential width  (y direction)
h_t = r_t + COVERAGE * r_g            # radial height     (z direction, per-sample)

# ── Grid ──────────────────────────────────────────────────────────────────────
lim = r_t.max() + r_g + 2.0
xs  = np.linspace(-HALF_H, HALF_H, GRID_RES)
ys  = np.linspace(-lim,    lim,    GRID_RES)
zs  = np.linspace(-lim,    lim,    GRID_RES)
Xg, Yg, Zg = np.meshgrid(xs, ys, zs, indexing='ij')   # (G,G,G)

# ── SDF accumulation ─────────────────────────────────────────────────────────
# We build the union SDF for prisms and spheres separately.
# Union SDF = max over all primitives (smooth or hard).
# Positive = inside, negative = outside.

prism_field  = np.full(Xg.shape, -np.inf, dtype=np.float32)
sphere_field = np.full(Xg.shape, -np.inf, dtype=np.float32)

print(f"Computing SDFs over {GRID_RES}³ grid with N={N} samples…")

for i in range(N):
    if i % 100 == 0:
        print(f"  {i}/{N}…")

    th = theta[i]
    xi = x_t[i]
    ri = r_t[i]
    hi = h_t[i]
    c, s = np.cos(th), np.sin(th)

    # ── Sphere (rotated) ──────────────────────────────────────────────────────
    # Pre-rotation centre: (xi, 0, ri)
    # After rotating by th around x-axis:
    #   x' = xi
    #   y' = 0*c - ri*s = -ri*s
    #   z' = 0*s + ri*c =  ri*c
    scx = xi
    scy = -ri * s
    scz =  ri * c

    dist2 = (Xg - scx)**2 + (Yg - scy)**2 + (Zg - scz)**2
    # SDF positive inside: r_g - dist  (but we store r_g² - dist² for speed, same sign)
    s_sdf = r_g - np.sqrt(dist2)
    sphere_field = np.maximum(sphere_field, s_sdf)

    # ── Elliptical cylinder (rotated), height along local z ───────────────────
    dx     = Xg - xi
    y_loc  =  c * Yg + s * Zg
    z_loc  = -s * Yg + c * Zg

    # Ellipse in local x-y plane
    # MAYBE CHANGE BOTH TO l / 2
    a = l / 2        # semi-axis along x
    b = w / 2        # semi-axis along local y

    ell_val = (dx / a)**2 + (y_loc / b)**2
    inside_ell = 1.0 - ell_val              # positive inside ellipse

    # Original hard cap
    inside_z = np.minimum(z_loc, hi - z_loc)

    # Soften only the top edge: where z_loc is within `blend` of hi, 
    # replace the linear ramp with a circular arc
    blend = 1.5   # in world units, tune to taste
    top_dist = hi - z_loc                          # 0 at top, positive below
    arc = np.sqrt(np.maximum(blend**2 - (top_dist - blend)**2, 0)) - blend
    inside_z = np.where(top_dist < blend, arc + top_dist, inside_z)

    p_sdf = np.minimum(inside_ell, inside_z)
    prism_field = np.maximum(prism_field, p_sdf)
    
    # ── Cylinder along x-axis (no rotation needed) ──
    # Equation: y^2 + z^2 <= R^2
    cyl_sdf = R / 2 - np.sqrt(Yg**2 + Zg**2)

    prism_field = np.maximum(prism_field, p_sdf)
    # Union with prisms
    prism_field = np.maximum(prism_field, cyl_sdf)


# ── Final solid SDF ───────────────────────────────────────────────────────────
# inside prisms AND outside spheres
solid_sdf = np.minimum(prism_field, -sphere_field)

print("Running marching cubes…")
verts, faces, _, _ = measure.marching_cubes(solid_sdf, level=0.0)

# Rescale from grid indices to world coords
verts[:, 0] = verts[:, 0] / (GRID_RES - 1) * (2 * HALF_H) - HALF_H
verts[:, 1] = verts[:, 1] / (GRID_RES - 1) * (2 * lim)    - lim
verts[:, 2] = verts[:, 2] / (GRID_RES - 1) * (2 * lim)    - lim

# ── STL export ────────────────────────────────────────────────────────────────
def write_stl(path, verts, faces):
    with open(path, 'wb') as f:
        f.write(b'\x00' * 80)
        f.write(struct.pack('<I', len(faces)))
        for tri in faces:
            v0, v1, v2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
            n = np.cross(v1-v0, v2-v0)
            nn = np.linalg.norm(n)
            n = n/nn if nn > 1e-12 else n
            f.write(struct.pack('<3f', *n))
            f.write(struct.pack('<3f', *v0))
            f.write(struct.pack('<3f', *v1))
            f.write(struct.pack('<3f', *v2))
            f.write(struct.pack('<H', 0))

# os.makedirs('/mnt/user-data/outputs', exist_ok=True)
stl_path = 'coding/outputs/cam_groove_v3.stl'
write_stl(stl_path, verts, faces)
print(f"STL written → {stl_path}  ({len(faces):,} triangles)")

# ── Preview ───────────────────────────────────────────────────────────────────
gcx = x_t
gcy = -r_t * np.sin(theta)
gcz =  r_t * np.cos(theta)

fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(121, projection='3d')
sc = ax1.scatter(gcx, gcy, gcz, c=t, cmap='plasma', s=4)
plt.colorbar(sc, ax=ax1, label='t', shrink=0.6)
ax1.set_title('Groove centreline (sphere centres)')
ax1.set_xlabel('X axial'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')

ax2 = fig.add_subplot(122, projection='3d')
step = max(1, len(faces) // 8000)
tri  = verts[faces[::step]]
poly = Poly3DCollection(tri, alpha=0.3, linewidth=0,
                        facecolor='steelblue', edgecolor='none')
ax2.add_collection3d(poly)
ax2.set_xlim(-HALF_H, HALF_H)
ax2.set_ylim(-lim, lim)
ax2.set_zlim(-lim, lim)
ax2.set_title(f'Cam mesh ({len(faces):,} triangles)')
ax2.set_xlabel('X axial'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')

plt.tight_layout()
png_path = 'coding/outputs/cam_groove_v3_preview.png'
plt.savefig(png_path, dpi=150, bbox_inches='tight')
print(f"Preview → {png_path}")
