import math
import csv
import subprocess
import os
from solid2 import *

# --- 1. CONFIGURATION ---
# Updated to your specific file location
OPENSCAD_PATH = r"C:\Program Files (x86)\OpenSCAD\openscad.exe"

CAM_RADIUS = 20
CAM_HEIGHT = 60
GROOVE_WIDTH = 4.4   # Track width (4.4mm for a 4.0mm pin)
GROOVE_DEPTH = 3     # How deep the cut goes
AXLE_RADIUS = 2.5    # 5mm motor shaft hole
RESOLUTION = 120     # Number of segments in the curve

# --- 2. CORE LOGIC ---

def get_3d_point(pt, min_x, span_x, min_y, span_y):
    """Maps 2D (x,y) to Cylindrical (Angle, Z)"""
    angle = ((pt[0] - min_x) / (span_x if span_x != 0 else 1)) * 360
    # Add vertical padding so the groove doesn't hit the top/bottom edges
    padding = GROOVE_WIDTH * 1.5
    usable_h = CAM_HEIGHT - (2 * padding)
    z_pos = (((pt[1] - min_y) / (span_y if span_y != 0 else 1)) - 0.5) * usable_h
    return angle, z_pos

def create_cam_geometry(points):
    """Builds the Cam with Axle and Set Screw Holes"""
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    span_x, span_y = max_x - min_x, max_y - min_y

    # Main Body
    base = cylinder(r=CAM_RADIUS, h=CAM_HEIGHT, center=True, _fn=100)
    
    # Holes
    axle = cylinder(r=AXLE_RADIUS, h=CAM_HEIGHT + 2, center=True, _fn=32)
    set_screw = rotate([0, 90, 0])(cylinder(d=3.2, h=CAM_RADIUS * 2, _fn=20))
    
    cam_solid = base - axle - set_screw

    # Groove Path
    segments = []
    for i in range(len(points) - 1):
        ang1, z1 = get_3d_point(points[i], min_x, span_x, min_y, span_y)
        ang2, z2 = get_3d_point(points[i+1], min_x, span_x, min_y, span_y)
        
        # We use spheres as the 'cutting bit'
        p1 = rotate([0, 0, ang1])(translate([CAM_RADIUS - GROOVE_DEPTH/2, 0, z1])(sphere(d=GROOVE_WIDTH, _fn=16)))
        p2 = rotate([0, 0, ang2])(translate([CAM_RADIUS - GROOVE_DEPTH/2, 0, z2])(sphere(d=GROOVE_WIDTH, _fn=16)))
        segments.append(hull()(p1, p2))
    
    return cam_solid - union()(*segments)

def create_follower_pin():
    """Builds a pin that fits the generated groove"""
    pin = cylinder(d=4.0, h=GROOVE_DEPTH + 10, _fn=32)
    base = cube([12, 12, 3], center=True).translate([0, 0, -1.5])
    return pin + base

def export_stl(model, filename):
    """Saves SCAD and triggers OpenSCAD for STL export"""
    scad_name = filename.replace(".stl", ".scad")
    model.save_as_scad(scad_name)
    
    if not os.path.exists(OPENSCAD_PATH):
        print(f"Error: Could not find OpenSCAD at {OPENSCAD_PATH}")
        return

    print(f"Rendering {filename}... This takes about 60-90 seconds.")
    try:
        # Command line: openscad -o output.stl input.scad
        subprocess.run([OPENSCAD_PATH, "-o", filename, scad_name], check=True)
        print(f"Success! {filename} is ready.")
    except Exception as e:
        print(f"Failed to render {filename}: {e}")

# --- 3. EXECUTION ---

if __name__ == "__main__":
    # Generate CSV of a Figure-Eight for your records
    with open('figure_eight.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        curve_pts = []
        for i in range(RESOLUTION + 1):
            t = i / RESOLUTION
            x = t
            y = math.sin(2 * math.pi * t) * math.cos(2 * math.pi * t)
            writer.writerow([x, y])
            curve_pts.append((x, y))

    # Render STL for Cam
    cam = create_cam_geometry(curve_pts)
    export_stl(cam, "barrel_cam.stl")

    # Render STL for Follower
    follower = create_follower_pin()
    export_stl(follower, "follower_pin.stl")