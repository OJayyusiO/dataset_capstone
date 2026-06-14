"""
Analytics Setup Tool — for REAL-WORLD video sources

Lets the user define calibration (and later, zones/lines) by clicking on a frame.
Saves config to analytics_config.yaml next to the source.

NOTE: CARLA recordings auto-generate analytics_config.yaml when record_test.py
finishes — you do NOT need to run this script for them.

Use this script for:
- Real footage from CCTV cameras, dashcams, etc.
- Webcam / RTSP streams
- Any video where you need to manually calibrate using known reference points

Sources supported:
- MP4 / video file
- Frame sequence directory (with frames/ subfolder)
- Webcam: pass an integer like 0
- RTSP / HTTP stream: pass a URL

Usage:
    python setup_analytics.py path/to/video.mp4
    python setup_analytics.py 0                             # webcam
    python setup_analytics.py rtsp://camera.local/stream
"""

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml


# --------------------------------------------------------------------------- #
# Calibration: AUTO mode (CARLA recording)
# --------------------------------------------------------------------------- #

def _build_intrinsic_matrix(w, h, fov_deg):
    """Camera intrinsic matrix K from image dims and horizontal FOV."""
    focal = w / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    K = np.array([
        [focal, 0.0,   w / 2.0],
        [0.0,   focal, h / 2.0],
        [0.0,   0.0,   1.0],
    ])
    return K


def _camera_to_world_rotation(pitch_deg, yaw_deg, roll_deg):
    """Build a rotation matrix that converts CAMERA-frame vectors to WORLD frame.

    Uses CARLA's API directly when available (always correct).
    Falls back to manual math otherwise.

    CARLA convention:
      - yaw rotates around Z (up); positive = clockwise from above
      - pitch rotates around the rotated Y-axis (right); positive = nose up
      - roll rotates around the rotated X-axis (forward)
    """
    try:
        import carla
        # Build a CARLA transform and use its forward/right/up vectors as columns
        tf = carla.Transform(
            carla.Location(0, 0, 0),
            carla.Rotation(pitch=pitch_deg, yaw=yaw_deg, roll=roll_deg),
        )
        fwd = tf.get_forward_vector()
        right = tf.get_right_vector()
        up = tf.get_up_vector()
        # Each vector is the world direction of the corresponding camera-local axis
        R = np.array([
            [fwd.x,   right.x, up.x],
            [fwd.y,   right.y, up.y],
            [fwd.z,   right.z, up.z],
        ])
        return R
    except ImportError:
        pass

    # Fallback: manual math (CARLA convention)
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)
    r = math.radians(roll_deg)

    cy, sy = math.cos(y), math.sin(y)
    cp, sp = math.cos(p), math.sin(p)
    cr, sr = math.cos(r), math.sin(r)

    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    # Positive pitch in CARLA = nose up, so rotation is in the opposite direction
    # from the standard right-hand convention
    Ry = np.array([[cp, 0, -sp], [0, 1, 0], [sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def _pixel_to_ground(u, v, K, cam_pos, cam_rot_matrix):
    """Project a single pixel (u, v) onto the world ground plane (z = 0).

    Returns (x_world, y_world) in meters, or None if the ray doesn't hit the ground.
    """
    K_inv = np.linalg.inv(K)
    # Pixel -> camera-frame ray (in standard pinhole convention: x right, y down, z forward)
    pixel_ray_cam = K_inv @ np.array([u, v, 1.0])

    # Convert from standard pinhole (x_right, y_down, z_forward) to UE4 camera frame
    # UE4 camera local axes: X forward, Y right, Z up
    ray_ue4 = np.array([pixel_ray_cam[2], pixel_ray_cam[0], -pixel_ray_cam[1]])

    # Rotate ray to world frame
    ray_world = cam_rot_matrix @ ray_ue4

    # Find intersection with ground plane (z = 0): cam_pos.z + t * ray_world[2] = 0
    if abs(ray_world[2]) < 1e-9:
        return None
    t = -cam_pos[2] / ray_world[2]
    if t <= 0:
        return None  # ray points away from ground
    x = cam_pos[0] + t * ray_world[0]
    y = cam_pos[1] + t * ray_world[1]
    return (x, y)


def auto_calibrate_from_carla(meta_path, camera_index=0):
    """Compute homography from CARLA recording metadata using known camera params.

    Picks 4 image points (the corners of a centered rectangle in the image),
    projects each to the world ground plane via pinhole + camera transform,
    then computes the homography image -> world from those 4 correspondences.
    """
    with open(meta_path) as f:
        meta = yaml.safe_load(f)

    cameras = meta.get('cameras')
    if not cameras:
        print("  No 'cameras' field in recording_meta.yaml — re-record with the latest record_test.py to enable auto-calibration.")
        return None

    if camera_index >= len(cameras):
        print(f"  Requested camera_index={camera_index} but only {len(cameras)} camera(s) in metadata.")
        return None

    cam = cameras[camera_index]
    image_w = cam['image_width']
    image_h = cam['image_height']
    fov = cam['fov']
    cam_pos = (cam['location']['x'], cam['location']['y'], cam['location']['z'])
    rotation = cam['rotation']
    R = _camera_to_world_rotation(rotation['pitch'], rotation['yaw'], rotation['roll'])
    K = _build_intrinsic_matrix(image_w, image_h, fov)

    # Pick 4 image points spread across the lower portion of the frame
    # (where the ground is visible in a typical CCTV-style angled camera)
    margin_x = int(image_w * 0.1)
    upper_y = int(image_h * 0.55)   # avoid sky
    lower_y = int(image_h * 0.95)
    image_pts = [
        (margin_x,             upper_y),
        (image_w - margin_x,   upper_y),
        (image_w - margin_x,   lower_y),
        (margin_x,             lower_y),
    ]

    world_pts = []
    for (u, v) in image_pts:
        wp = _pixel_to_ground(u, v, K, cam_pos, R)
        if wp is None:
            print(f"  Pixel ({u},{v}) didn't hit the ground plane. Camera angle may not see ground.")
            return None
        world_pts.append(wp)

    img_arr = np.array(image_pts, dtype=np.float32)
    world_arr = np.array(world_pts, dtype=np.float32)

    H, _ = cv2.findHomography(img_arr, world_arr)
    if H is None:
        print("  Homography computation failed.")
        return None

    # Sanity check: meters per pixel near image center
    center_world = _pixel_to_ground(image_w // 2, image_h // 2 + image_h // 4, K, cam_pos, R)
    center_world_right = _pixel_to_ground(image_w // 2 + 1, image_h // 2 + image_h // 4, K, cam_pos, R)
    if center_world and center_world_right:
        approx_mpp = math.dist(center_world, center_world_right)
    else:
        approx_mpp = None

    return {
        'mode': 'auto_carla',
        'camera_index': int(camera_index),
        'image_width': int(image_w),
        'image_height': int(image_h),
        'fov': float(fov),
        'camera_position': [float(v) for v in cam_pos],
        'camera_rotation': {k: float(v) for k, v in rotation.items()},
        'image_points': [[int(u), int(v)] for (u, v) in image_pts],
        'world_points': [[float(x), float(y)] for (x, y) in world_pts],
        'homography_matrix': [[float(v) for v in row] for row in H.tolist()],
        'approx_meters_per_pixel_at_mid_ground': round(float(approx_mpp), 5) if approx_mpp else None,
    }


# --------------------------------------------------------------------------- #
# Calibration: MANUAL mode (4-point homography)
# --------------------------------------------------------------------------- #

class PointPicker:
    """Lets user click N points on an OpenCV image."""

    def __init__(self, image, num_points, window_name='Click points'):
        self.image = image.copy()
        self.display = image.copy()
        self.num_points = num_points
        self.window_name = window_name
        self.points = []

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < self.num_points:
            self.points.append((x, y))
            # Draw click marker
            cv2.circle(self.display, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(self.display, str(len(self.points)), (x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # If we have multiple points, draw lines between them
            if len(self.points) > 1:
                cv2.line(self.display, self.points[-2], self.points[-1], (0, 255, 0), 2)
            # Close the loop visually if we have all 4 points
            if len(self.points) == self.num_points and self.num_points >= 3:
                cv2.line(self.display, self.points[-1], self.points[0], (0, 255, 0), 2)
                # Big overlay so user knows what to do
                msg = "Press ENTER or SPACE to confirm  |  R to reset  |  Q to cancel"
                (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(self.display, (10, 10), (10 + tw + 20, 10 + th + 20), (0, 0, 0), -1)
                cv2.putText(self.display, msg, (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow(self.window_name, self.display)

    def pick(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(self.window_name, self.display)
        cv2.setMouseCallback(self.window_name, self._on_mouse)

        print(f"Click {self.num_points} points in the image.")
        print("Press 'r' to reset, 'q' to cancel, ENTER or SPACE to confirm.")
        print("(Make sure the image window is focused when pressing keys.)")

        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == 0xFF:
                continue  # no key pressed
            if key == ord('q'):
                cv2.destroyWindow(self.window_name)
                return None
            if key == ord('r'):
                self.display = self.image.copy()
                self.points = []
                cv2.imshow(self.window_name, self.display)
            # Accept ENTER (13 on Win, 10 on Linux/Mac) or SPACE (32) as confirm
            if key in (13, 10, 32) and len(self.points) == self.num_points:
                cv2.destroyWindow(self.window_name)
                return self.points
            # Also auto-confirm right after the last click (no key press needed)
            if len(self.points) == self.num_points:
                # Show a "press ENTER to confirm" message after all points are clicked
                pass


class PolygonPicker:
    """Lets user click an arbitrary number of points to form a polygon.

    Press 'd' to finish the current polygon, 'r' to reset, 'q' to cancel.
    """

    COLOR_LINE = (0, 255, 255)  # yellow
    COLOR_POINT = (0, 255, 0)   # green

    def __init__(self, image, window_name='Click polygon corners'):
        self.image = image.copy()
        self.display = image.copy()
        self.window_name = window_name
        self.points = []

    def _redraw(self):
        self.display = self.image.copy()
        for i, p in enumerate(self.points):
            cv2.circle(self.display, p, 6, self.COLOR_POINT, -1)
            cv2.putText(self.display, str(i + 1), (p[0] + 10, p[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_POINT, 2)
            if i > 0:
                cv2.line(self.display, self.points[i - 1], p, self.COLOR_LINE, 2)
        # Close the polygon visually once we have 3+ points
        if len(self.points) >= 3:
            cv2.line(self.display, self.points[-1], self.points[0], self.COLOR_LINE, 2)
        # Always show controls overlay
        msg = "Click corners  |  D when done  |  R reset  |  Q cancel"
        (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(self.display, (10, 10), (10 + tw + 20, 10 + th + 20), (0, 0, 0), -1)
        cv2.putText(self.display, msg, (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_LINE, 2)
        cv2.imshow(self.window_name, self.display)

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            self._redraw()

    def pick(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self._redraw()
        cv2.setMouseCallback(self.window_name, self._on_mouse)

        print(f"  Click corners of the polygon. Press 'd' when done.")
        print(f"  ('r' to reset, 'q' to cancel.)")

        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == 0xFF:
                continue
            if key == ord('q'):
                cv2.destroyWindow(self.window_name)
                return None
            if key == ord('r'):
                self.points = []
                self._redraw()
            if key == ord('d'):
                if len(self.points) >= 3:
                    cv2.destroyWindow(self.window_name)
                    return self.points
                else:
                    print("  Need at least 3 points to form a polygon.")


def define_lanes(frame, existing_lanes=None, skip_first_prompt=False):
    """Interactively define lane polygons.

    Args:
        existing_lanes: lanes already defined (will be kept and added to)
        skip_first_prompt: if True, jump straight to defining the first lane
            without asking "Add a lane?" (useful after --redo-lanes)

    Returns list of {'id': str, 'polygon': [[x, y], ...]} dicts.
    """
    lanes = list(existing_lanes or [])

    if lanes:
        print(f"\n{len(lanes)} lane(s) already defined:")
        for ln in lanes:
            print(f"  - {ln['id']}: {len(ln['polygon'])} points")

    print("\nDefine lanes by clicking polygon corners around each lane.")
    print("Use 3+ points per lane (typically 4 for a rectangular lane segment).")

    first_iter = True
    while True:
        # Skip the prompt on the first iteration if requested
        if first_iter and skip_first_prompt:
            choice = 'y'
        else:
            prompt = "\nAdd a new lane? [y/N]: " if lanes else "\nAdd a lane? [Y/n]: "
            choice = input(prompt).strip().lower()
            if choice == 'q':
                break
            default_yes = not lanes
            if choice == '':
                choice = 'y' if default_yes else 'n'
            if choice != 'y':
                break
        first_iter = False

        lane_id = input("  Lane ID (e.g. lane_north_1): ").strip()
        if not lane_id:
            print("  Skipped (no ID provided)")
            continue

        # Show frame with existing lanes overlaid
        preview = frame.copy()
        for ln in lanes:
            pts = np.array(ln['polygon'], dtype=np.int32)
            cv2.polylines(preview, [pts], isClosed=True, color=(100, 100, 255), thickness=2)
            centroid = pts.mean(axis=0).astype(int)
            cv2.putText(preview, ln['id'], tuple(centroid),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)

        picker = PolygonPicker(preview, window_name=f"Define: {lane_id}")
        polygon = picker.pick()
        if polygon is None:
            print(f"  Cancelled lane {lane_id}")
            continue

        lanes.append({
            'id': lane_id,
            'polygon': [[int(x), int(y)] for (x, y) in polygon],
        })
        print(f"  Added {lane_id} with {len(polygon)} corners.")

    return lanes


def define_entry_zones(frame, lanes=None, existing_zones=None, skip_first_prompt=False):
    """Interactively define highway entry zones as polygons.

    Each unique vehicle that enters a zone is counted once, recorded with the
    light state at the moment of entry (for ramp-metering / entry-flow analytics).

    Returns list of {'id': str, 'polygon': [[x, y], ...]} dicts.
    """
    zones = list(existing_zones or [])
    lanes = lanes or []

    if zones:
        print(f"\n{len(zones)} entry zone(s) already defined:")
        for z in zones:
            print(f"  - {z['id']}")

    print("\nDefine entry zones by clicking polygon corners around each highway entry / ramp.")
    print("Vehicles entering the zone are counted, grouped by the light state at entry.")

    first_iter = True
    while True:
        if first_iter and skip_first_prompt:
            choice = 'y'
        else:
            prompt = "\nAdd another entry zone? [y/N]: " if zones else "\nAdd an entry zone? [y/N]: "
            choice = input(prompt).strip().lower()
            if choice == '':
                choice = 'n'
            if choice != 'y':
                break
        first_iter = False

        zone_id = input("  Entry zone ID (e.g. ramp_east): ").strip()
        if not zone_id:
            print("  Skipped (no ID provided)")
            continue

        preview = frame.copy()
        for ln in lanes:
            pts = np.array(ln['polygon'], dtype=np.int32)
            cv2.polylines(preview, [pts], isClosed=True, color=(100, 100, 255), thickness=1)
        for z in zones:
            pts = np.array(z['polygon'], dtype=np.int32)
            cv2.polylines(preview, [pts], isClosed=True, color=(0, 200, 255), thickness=2)

        picker = PolygonPicker(preview, window_name=f"Define entry zone: {zone_id}")
        polygon = picker.pick()
        if polygon is None:
            print(f"  Cancelled zone {zone_id}")
            continue

        zones.append({
            'id': zone_id,
            'polygon': [[int(x), int(y)] for (x, y) in polygon],
        })
        print(f"  Added {zone_id} with {len(polygon)} corners.")

    return zones


def define_forbidden_zones(frame, lanes=None, existing_zones=None, skip_first_prompt=False):
    """Interactively define forbidden (no-go) zones as polygons — e.g. the painted
    chevron / gore area at a highway ramp. Any vehicle that drives into the zone is
    flagged as a violation (no traffic-light state needed).

    Returns list of {'id': str, 'polygon': [[x, y], ...]} dicts.
    """
    zones = list(existing_zones or [])
    lanes = lanes or []

    if zones:
        print(f"\n{len(zones)} forbidden zone(s) already defined:")
        for z in zones:
            print(f"  - {z['id']}")

    print("\nDefine forbidden zones by clicking polygon corners around each no-go area")
    print("(e.g. the chevron gore at a highway ramp). Any vehicle entering is flagged.")

    first_iter = True
    while True:
        if first_iter and skip_first_prompt:
            choice = 'y'
        else:
            prompt = "\nAdd another forbidden zone? [y/N]: " if zones else "\nAdd a forbidden zone? [y/N]: "
            choice = input(prompt).strip().lower()
            if choice == '':
                choice = 'n'
            if choice != 'y':
                break
        first_iter = False

        zone_id = input("  Forbidden zone ID (e.g. ramp_chevron): ").strip()
        if not zone_id:
            print("  Skipped (no ID provided)")
            continue

        preview = frame.copy()
        for ln in lanes:
            pts = np.array(ln['polygon'], dtype=np.int32)
            cv2.polylines(preview, [pts], isClosed=True, color=(100, 100, 255), thickness=1)
        for z in zones:
            pts = np.array(z['polygon'], dtype=np.int32)
            cv2.polylines(preview, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

        picker = PolygonPicker(preview, window_name=f"Define forbidden zone: {zone_id}")
        polygon = picker.pick()
        if polygon is None:
            print(f"  Cancelled zone {zone_id}")
            continue

        zones.append({
            'id': zone_id,
            'polygon': [[int(x), int(y)] for (x, y) in polygon],
        })
        print(f"  Added {zone_id} with {len(polygon)} corners.")

    return zones


def define_forbidden_lines(frame, lanes=None, existing_lines=None, skip_first_prompt=False):
    """Interactively define forbidden lines (e.g. stop lines) as 2-point segments.

    A vehicle crossing one of these lines while the light is red is a violation.

    Returns list of {'id': str, 'points': [[x1, y1], [x2, y2]]} dicts.
    """
    lines = list(existing_lines or [])
    lanes = lanes or []

    if lines:
        print(f"\n{len(lines)} forbidden line(s) already defined:")
        for ln in lines:
            print(f"  - {ln['id']}")

    print("\nDefine forbidden lines by clicking 2 points across the road (e.g. a stop line).")
    print("A vehicle crossing the line while the light is RED counts as a violation.")

    first_iter = True
    while True:
        if first_iter and skip_first_prompt:
            choice = 'y'
        else:
            prompt = "\nAdd another forbidden line? [y/N]: " if lines else "\nAdd a forbidden line? [y/N]: "
            choice = input(prompt).strip().lower()
            default_yes = False
            if choice == '':
                choice = 'y' if default_yes else 'n'
            if choice != 'y':
                break
        first_iter = False

        line_id = input("  Line ID (e.g. stop_line_north): ").strip()
        if not line_id:
            print("  Skipped (no ID provided)")
            continue

        # Show frame with lanes and existing lines overlaid for context
        preview = frame.copy()
        for ln in lanes:
            pts = np.array(ln['polygon'], dtype=np.int32)
            cv2.polylines(preview, [pts], isClosed=True, color=(100, 100, 255), thickness=1)
        for fl in lines:
            p1, p2 = fl['points']
            cv2.line(preview, tuple(p1), tuple(p2), (0, 0, 255), 2)

        picker = PointPicker(preview, num_points=2, window_name=f"Define line: {line_id}")
        pts = picker.pick()
        if pts is None:
            print(f"  Cancelled line {line_id}")
            continue

        lines.append({
            'id': line_id,
            'points': [[int(pts[0][0]), int(pts[0][1])], [int(pts[1][0]), int(pts[1][1])]],
        })
        print(f"  Added {line_id}.")

    return lines


def manual_calibrate(frame):
    """Run interactive 4-point homography calibration.

    Returns dict with homography matrix and meters_per_pixel.
    """
    print("\n" + "=" * 60)
    print("MANUAL CALIBRATION (4-point homography)")
    print("=" * 60)
    print("\nClick 4 points that form a rectangle on the GROUND PLANE.")
    print("Order: top-left, top-right, bottom-right, bottom-left.")
    print("\nGood reference rectangles:")
    print("  - Lane markings (lane width is usually 3.5m)")
    print("  - Pedestrian crossing stripes")
    print("  - A parking spot (usually 2.4m × 4.8m)")
    print("  - Any flat rectangular object you can measure")
    print()

    picker = PointPicker(frame, num_points=4, window_name='Calibration: click 4 corners')
    img_pts = picker.pick()

    if img_pts is None:
        print("Calibration cancelled.")
        return None

    print(f"\nCaptured 4 image points:")
    for i, (x, y) in enumerate(img_pts):
        print(f"  Point {i+1}: ({x}, {y})")

    print("\nNow enter the real-world dimensions of that rectangle:")
    while True:
        try:
            width_m = float(input("  Width  (meters, top edge): ").strip())
            length_m = float(input("  Length (meters, side edge): ").strip())
            break
        except ValueError:
            print("  Please enter valid numbers.")

    # Build the corresponding world-coordinate points (origin at top-left of rectangle)
    world_pts = np.array([
        [0, 0],
        [width_m, 0],
        [width_m, length_m],
        [0, length_m],
    ], dtype=np.float32)

    img_pts_arr = np.array(img_pts, dtype=np.float32)

    # Compute homography from image -> world
    H, _ = cv2.findHomography(img_pts_arr, world_pts)

    if H is None:
        print("ERROR: failed to compute homography. Try again with cleaner points.")
        return None

    # Compute approximate meters_per_pixel as a sanity check (not used for actual conversion)
    avg_image_edge_px = (
        np.linalg.norm(img_pts_arr[1] - img_pts_arr[0]) +
        np.linalg.norm(img_pts_arr[2] - img_pts_arr[3])
    ) / 2
    avg_world_edge_m = width_m
    approx_meters_per_pixel = float(avg_world_edge_m / avg_image_edge_px)

    print(f"\n✓ Calibration complete.")
    print(f"  Homography matrix computed.")
    print(f"  Approx meters/pixel near the rectangle: {approx_meters_per_pixel:.4f}")
    print(f"  (Use the homography for accurate world coords; this number is just a hint.)")

    return {
        'mode': 'manual_homography',
        'image_points': [[int(x), int(y)] for (x, y) in img_pts],
        'world_points': [[float(x), float(y)] for (x, y) in world_pts.tolist()],
        'rectangle_width_m': float(width_m),
        'rectangle_length_m': float(length_m),
        'homography_matrix': [[float(v) for v in row] for row in H.tolist()],
        'approx_meters_per_pixel': round(approx_meters_per_pixel, 5),
    }


# --------------------------------------------------------------------------- #
# Existing config handling
# --------------------------------------------------------------------------- #

def load_existing_config(config_path):
    if config_path.exists():
        try:
            with open(config_path) as f:
                return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            print(f"  Could not load existing config (file may be corrupt): {e}")
            return {}
    return {}


# --------------------------------------------------------------------------- #
# Source resolution: handles MP4 files, frame directories, webcam, streams
# --------------------------------------------------------------------------- #

def resolve_source(source_arg, frame_idx):
    """Returns (frame_image, source_dir, source_label) for any input type.

    source_dir is where analytics_config.yaml will be saved.
    source_label is a short string identifying the source for the YAML.
    """
    # Webcam (integer)
    if source_arg.isdigit():
        cam_idx = int(source_arg)
        cap = cv2.VideoCapture(cam_idx)
        if not cap.isOpened():
            print(f"Could not open webcam {cam_idx}")
            sys.exit(1)
        # Discard a few frames to let auto-exposure settle
        for _ in range(5):
            cap.read()
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"Could not read from webcam {cam_idx}")
            sys.exit(1)
        return frame, Path.cwd(), f"webcam_{cam_idx}"

    # RTSP / HTTP stream
    if source_arg.startswith(('rtsp://', 'http://', 'https://')):
        cap = cv2.VideoCapture(source_arg)
        if not cap.isOpened():
            print(f"Could not open stream: {source_arg}")
            sys.exit(1)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"Could not read from stream")
            sys.exit(1)
        return frame, Path.cwd(), source_arg.replace('://', '_').replace('/', '_')

    # File or directory
    source_path = Path(source_arg)
    if not source_path.exists():
        print(f"Source not found: {source_path}")
        sys.exit(1)

    # Scenario YAML: connect to CARLA, spawn the camera, grab one frame
    if source_path.suffix in ('.yaml', '.yml'):
        with open(source_path) as f:
            scenario = yaml.safe_load(f)
        # Only treat it as a scenario if it has camera/cameras config
        if 'camera' in scenario or 'cameras' in scenario:
            frame = _capture_frame_from_carla_scenario(scenario)
            # Save analytics config in a dedicated folder, named to match the scenario
            analytics_dir = Path(__file__).resolve().parents[2] / 'analytics_configs'
            analytics_dir.mkdir(parents=True, exist_ok=True)
            return frame, analytics_dir, source_path.stem

    # Directory: look for a frames/ subfolder or PNG files directly
    if source_path.is_dir():
        if (source_path / 'frames').is_dir():
            frames_dir = source_path / 'frames'
        else:
            frames_dir = source_path
        png_files = sorted(frames_dir.glob('*.png'))
        if not png_files:
            print(f"No PNG frames found in {frames_dir}")
            sys.exit(1)
        chosen = png_files[min(frame_idx, len(png_files) - 1)]
        frame = cv2.imread(str(chosen))
        if frame is None:
            print(f"Could not read frame: {chosen}")
            sys.exit(1)
        return frame, source_path, source_path.name

    # MP4 / video file
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        print(f"Could not open video: {source_path}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"Could not read frame {frame_idx} from video.")
        sys.exit(1)
    return frame, source_path.parent, source_path.name


def _capture_frame_from_carla_scenario(scenario):
    """Connect to CARLA, spawn camera from scenario, grab one frame, clean up."""
    try:
        import carla
        import queue
        import time as _time
    except ImportError:
        print("CARLA not installed. Cannot capture from scenario YAML.")
        sys.exit(1)

    # Add scripts/utils to path so we can use carla_helpers
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from capstone_sim.scripts.utils.carla_helpers import spawn_camera, apply_weather

    if 'cameras' in scenario:
        cam_cfg = scenario['cameras'][0]
    else:
        cam_cfg = scenario['camera']

    image_w = cam_cfg.get('image_width', 1280)
    image_h = cam_cfg.get('image_height', 720)

    print("Connecting to CARLA...")
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    target_map = scenario.get('map')
    if target_map:
        current_map = client.get_world().get_map().name.split('/')[-1]
        if current_map != target_map:
            print(f"Loading map {target_map}...")
            client.load_world(target_map)
            _time.sleep(5)

    world = client.get_world()
    world.unload_map_layer(carla.MapLayer.ParkedVehicles)
    bp_lib = world.get_blueprint_library()

    # Set synchronous so we get exactly one frame
    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    apply_weather(world, scenario.get('weather', {}))
    camera = spawn_camera(world, bp_lib, cam_cfg)
    image_queue = queue.Queue()
    camera.listen(image_queue.put)

    try:
        # Tick a few times to get a clean frame
        for _ in range(5):
            world.tick()
        carla_img = image_queue.get(timeout=5.0)
        img_data = np.array(carla_img.raw_data)
        frame = img_data.reshape((image_h, image_w, 4))[:, :, :3].copy()
        print(f"Captured frame from CARLA ({image_w}x{image_h})")
        return frame
    finally:
        camera.stop()
        camera.destroy()
        world.apply_settings(original)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def _ask_yes_no(prompt, default_yes=True):
    suffix = "[Y/n]" if default_yes else "[y/N]"
    while True:
        choice = input(f"{prompt} {suffix}: ").strip().lower()
        if choice == '':
            return default_yes
        if choice in ('y', 'yes'):
            return True
        if choice in ('n', 'no'):
            return False


def main():
    parser = argparse.ArgumentParser(description='Setup analytics config (calibration + lanes) for any source')
    parser.add_argument('source', type=str,
                        help='Video file, frame directory, webcam index (e.g. 0), or stream URL')
    parser.add_argument('--output', type=str, default=None,
                        help='Output YAML name (default: analytics_config.yaml next to source)')
    parser.add_argument('--frame', type=int, default=0,
                        help='Which frame to use for setup (default: 0)')
    parser.add_argument('--manual', action='store_true',
                        help='Force manual calibration even if CARLA metadata exists')
    parser.add_argument('--recalibrate', action='store_true',
                        help='Redo calibration even if already present in config')
    parser.add_argument('--redo-lanes', action='store_true',
                        help='Wipe existing lanes and start fresh')
    parser.add_argument('--redo-lines', action='store_true',
                        help='Wipe existing forbidden lines and start fresh')
    parser.add_argument('--redo-entry-zones', action='store_true',
                        help='Wipe existing highway entry zones and start fresh')
    parser.add_argument('--redo-forbidden-zones', action='store_true',
                        help='Wipe existing forbidden (no-go) zones and start fresh')
    args = parser.parse_args()

    frame, source_dir, source_label = resolve_source(args.source, args.frame)

    # Default filename: <scenario_name>.yaml if source was a scenario YAML
    # (source_label is the scenario stem in that case, set by resolve_source);
    # otherwise "analytics_config.yaml" next to the source.
    if args.output:
        output_name = args.output
    elif args.source.endswith(('.yaml', '.yml')):
        output_name = f"{source_label}.yaml"
    else:
        output_name = 'analytics_config.yaml'
    config_path = source_dir / output_name

    # Try to load existing config — but don't fail if it's corrupt
    config = load_existing_config(config_path)
    if config:
        print(f"Found existing config: {config_path}")

    # --- Step 1: Calibration ---
    print("\n" + "=" * 60)
    print("Step 1: Calibration")
    print("=" * 60)

    existing_cal = config.get('calibration')
    do_calibrate = args.recalibrate or existing_cal is None

    if existing_cal and not args.recalibrate:
        print(f"  Calibration already present (mode={existing_cal.get('mode', 'unknown')})")
        do_calibrate = _ask_yes_no("  Redo calibration?", default_yes=False)

    if do_calibrate:
        calibration = None

        # Case 1: source is a scenario YAML — use the camera params from it directly
        source_path = Path(args.source) if not args.source.isdigit() and not args.source.startswith(('rtsp://', 'http://', 'https://')) else None
        if source_path and source_path.is_file() and source_path.suffix in ('.yaml', '.yml') and not args.manual:
            with open(source_path) as f:
                scenario = yaml.safe_load(f)
            if 'camera' in scenario or 'cameras' in scenario:
                print(f"  Source is a scenario YAML, auto-calibrating from its camera params...")
                # Build temp meta and use existing auto-calibrate
                import tempfile
                cams = scenario.get('cameras') or [scenario.get('camera')]
                meta_dict = {'cameras': [{
                    'index': i,
                    'location': c['location'],
                    'rotation': c['rotation'],
                    'image_width': c.get('image_width', 1280),
                    'image_height': c.get('image_height', 720),
                    'fov': c.get('fov', 70),
                } for i, c in enumerate(cams)]}
                with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tf:
                    yaml.dump(meta_dict, tf)
                    tmp_path = Path(tf.name)
                try:
                    calibration = auto_calibrate_from_carla(tmp_path)
                finally:
                    tmp_path.unlink()

        # Case 2: source has a recording_meta.yaml next to it
        if calibration is None:
            meta_path = source_dir / 'recording_meta.yaml'
            if meta_path.exists() and not args.manual:
                print(f"  Found CARLA recording metadata: {meta_path.name}")
                print("  Attempting auto-calibration...")
                calibration = auto_calibrate_from_carla(meta_path)

        if calibration:
            print(f"  Auto-calibration succeeded (mode={calibration['mode']})")
        else:
            print("  Falling back to manual calibration.")
            calibration = manual_calibrate(frame)
            if calibration is None:
                print("Setup aborted.")
                sys.exit(0)

        config['calibration'] = calibration

    # --- Step 2: Lanes ---
    print("\n" + "=" * 60)
    print("Step 2: Lane Definition")
    print("=" * 60)

    existing_lanes = config.get('lanes', [])

    if args.redo_lanes:
        print(f"  --redo-lanes flag set: wiping {len(existing_lanes)} existing lane(s).")
        existing_lanes = []
        lanes = define_lanes(frame, existing_lanes=[], skip_first_prompt=True)
        config['lanes'] = lanes
    elif existing_lanes:
        print(f"  {len(existing_lanes)} lane(s) already defined:")
        for ln in existing_lanes:
            print(f"    - {ln['id']}")
        print("\nOptions:")
        print("  [a] Add more lanes (keep existing)")
        print("  [r] Replace all — wipe existing and start fresh")
        print("  [s] Skip — don't touch lanes")
        while True:
            choice = input("Choice [a/r/s]: ").strip().lower()
            if choice in ('a', 'r', 's'):
                break
            print("Invalid choice.")
        if choice == 'r':
            existing_lanes = []
            lanes = define_lanes(frame, existing_lanes=[], skip_first_prompt=True)
            config['lanes'] = lanes
        elif choice == 'a':
            lanes = define_lanes(frame, existing_lanes=existing_lanes)
            config['lanes'] = lanes
        # 's' = skip, do nothing
    else:
        # No existing lanes — just ask if they want to define some
        if _ask_yes_no("Add lanes?", default_yes=True):
            lanes = define_lanes(frame, existing_lanes=[])
            config['lanes'] = lanes

    # --- Step 3: Forbidden Lines (for red-light violation detection) ---
    print("\n" + "=" * 60)
    print("Step 3: Forbidden Lines (stop lines)")
    print("=" * 60)

    current_lanes = config.get('lanes', [])
    existing_lines = config.get('forbidden_lines', [])

    if args.redo_lines:
        print(f"  --redo-lines flag set: wiping {len(existing_lines)} existing line(s).")
        lines = define_forbidden_lines(frame, lanes=current_lanes, existing_lines=[], skip_first_prompt=True)
        config['forbidden_lines'] = lines
    elif existing_lines:
        print(f"  {len(existing_lines)} forbidden line(s) already defined:")
        for ln in existing_lines:
            print(f"    - {ln['id']}")
        print("\nOptions:")
        print("  [a] Add more lines (keep existing)")
        print("  [r] Replace all — wipe existing and start fresh")
        print("  [s] Skip — don't touch lines")
        while True:
            choice = input("Choice [a/r/s]: ").strip().lower()
            if choice in ('a', 'r', 's'):
                break
            print("Invalid choice.")
        if choice == 'r':
            lines = define_forbidden_lines(frame, lanes=current_lanes, existing_lines=[], skip_first_prompt=True)
            config['forbidden_lines'] = lines
        elif choice == 'a':
            lines = define_forbidden_lines(frame, lanes=current_lanes, existing_lines=existing_lines)
            config['forbidden_lines'] = lines
        # 's' = skip
    else:
        if _ask_yes_no("Add forbidden lines (for red-light violations)?", default_yes=False):
            lines = define_forbidden_lines(frame, lanes=current_lanes, existing_lines=[])
            config['forbidden_lines'] = lines

    # --- Step 4: Highway Entry Zones (for entry counting by light state) ---
    print("\n" + "=" * 60)
    print("Step 4: Highway Entry Zones")
    print("=" * 60)

    existing_zones = config.get('entry_zones', [])

    if args.redo_entry_zones:
        print(f"  --redo-entry-zones flag set: wiping {len(existing_zones)} existing zone(s).")
        zones = define_entry_zones(frame, lanes=current_lanes, existing_zones=[], skip_first_prompt=True)
        config['entry_zones'] = zones
    elif existing_zones:
        print(f"  {len(existing_zones)} entry zone(s) already defined:")
        for z in existing_zones:
            print(f"    - {z['id']}")
        print("\nOptions:")
        print("  [a] Add more zones (keep existing)")
        print("  [r] Replace all — wipe existing and start fresh")
        print("  [s] Skip — don't touch zones")
        while True:
            choice = input("Choice [a/r/s]: ").strip().lower()
            if choice in ('a', 'r', 's'):
                break
            print("Invalid choice.")
        if choice == 'r':
            zones = define_entry_zones(frame, lanes=current_lanes, existing_zones=[], skip_first_prompt=True)
            config['entry_zones'] = zones
        elif choice == 'a':
            zones = define_entry_zones(frame, lanes=current_lanes, existing_zones=existing_zones)
            config['entry_zones'] = zones
        # 's' = skip
    else:
        if _ask_yes_no("Add highway entry zones (for entry counting)?", default_yes=False):
            zones = define_entry_zones(frame, lanes=current_lanes, existing_zones=[])
            config['entry_zones'] = zones

    # --- Step 5: Forbidden Zones (no-go polygons, e.g. ramp chevron) ---
    print("\n" + "=" * 60)
    print("Step 5: Forbidden Zones (no-go polygons)")
    print("=" * 60)

    existing_fzones = config.get('forbidden_zones', [])

    if args.redo_forbidden_zones:
        print(f"  --redo-forbidden-zones flag set: wiping {len(existing_fzones)} existing zone(s).")
        fzones = define_forbidden_zones(frame, lanes=current_lanes, existing_zones=[], skip_first_prompt=True)
        config['forbidden_zones'] = fzones
    elif existing_fzones:
        print(f"  {len(existing_fzones)} forbidden zone(s) already defined:")
        for z in existing_fzones:
            print(f"    - {z['id']}")
        print("\nOptions:")
        print("  [a] Add more zones (keep existing)")
        print("  [r] Replace all — wipe existing and start fresh")
        print("  [s] Skip — don't touch forbidden zones")
        while True:
            choice = input("Choice [a/r/s]: ").strip().lower()
            if choice in ('a', 'r', 's'):
                break
            print("Invalid choice.")
        if choice == 'r':
            fzones = define_forbidden_zones(frame, lanes=current_lanes, existing_zones=[], skip_first_prompt=True)
            config['forbidden_zones'] = fzones
        elif choice == 'a':
            fzones = define_forbidden_zones(frame, lanes=current_lanes, existing_zones=existing_fzones)
            config['forbidden_zones'] = fzones
        # 's' = skip
    else:
        if _ask_yes_no("Add forbidden zones (no-go areas like a ramp chevron)?", default_yes=False):
            fzones = define_forbidden_zones(frame, lanes=current_lanes, existing_zones=[])
            config['forbidden_zones'] = fzones

    # --- Save ---
    config['source'] = source_label

    # Add default queue thresholds if not already present (user can edit later)
    if 'queue' not in config:
        config['queue'] = {
            'speed_threshold_kmh': 7.2,         # below this = "slow"
            'min_stationary_seconds': 2.0,      # must be slow for N seconds to count as queued
        }

    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\n✓ Saved: {config_path.resolve()}")
    print(f"\nConfig summary:")
    print(f"  Calibration: {config.get('calibration', {}).get('mode', 'NONE')}")
    print(f"  Lanes: {len(config.get('lanes', []))}")
    print(f"  Forbidden lines: {len(config.get('forbidden_lines', []))}")
    print(f"  Entry zones: {len(config.get('entry_zones', []))}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(0)
