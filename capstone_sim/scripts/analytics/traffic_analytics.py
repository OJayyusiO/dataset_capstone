"""
Traffic Analytics — Phase A: Speed per car

Runs YOLO + ByteTrack on a video, computes real-world speed for each tracked
vehicle using the homography from analytics_config.yaml, and outputs:
- Annotated video with speed overlays
- CSV per-track-per-frame data

Usage:
    python traffic_analytics.py <source> <model.pt>
    python traffic_analytics.py recordings/town6/ best.pt --output results/

Source can be: MP4 file, frame directory, webcam index (e.g. 0), RTSP URL.
The script auto-finds analytics_config.yaml next to the source.
"""

import argparse
import csv
import sys
import time
from collections import deque, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from capstone_sim.scripts.utils.constants import CLASS_NAMES, CLASS_COLORS
from capstone_sim.scripts.utils.light_state import LightStateProvider, draw_light_indicator

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics not installed. Run: pip install ultralytics")
    sys.exit(1)


# --------------------------------------------------------------------------- #
# Config & source loading
# --------------------------------------------------------------------------- #

def load_analytics_config(source_dir):
    """Load analytics_config.yaml from the source directory."""
    config_path = source_dir / 'analytics_config.yaml'
    if not config_path.exists():
        print(f"Error: analytics_config.yaml not found at {config_path}")
        print(f"Run setup_analytics.py first or ensure the recording has it.")
        sys.exit(1)
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_source(source_arg):
    """Returns (cv2.VideoCapture-like iterator yielding frames, fps, w, h, source_dir).

    For frame directories, builds a generator that yields PNG frames in order.
    """
    if source_arg.isdigit():
        cap = cv2.VideoCapture(int(source_arg))
        return _cap_iterator(cap), cap.get(cv2.CAP_PROP_FPS) or 30, \
               int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), \
               int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), Path.cwd(), None

    if source_arg.startswith(('rtsp://', 'http://', 'https://')):
        cap = cv2.VideoCapture(source_arg)
        return _cap_iterator(cap), cap.get(cv2.CAP_PROP_FPS) or 30, \
               int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), \
               int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), Path.cwd(), None

    source_path = Path(source_arg)
    if not source_path.exists():
        print(f"Source not found: {source_path}")
        sys.exit(1)

    if source_path.is_dir():
        frames_dir = source_path / 'frames' if (source_path / 'frames').is_dir() else source_path
        png_files = sorted(frames_dir.glob('*.png'))
        if not png_files:
            print(f"No PNG frames in {frames_dir}")
            sys.exit(1)
        first = cv2.imread(str(png_files[0]))
        h, w = first.shape[:2]
        # Read FPS from recording_meta.yaml if present
        fps = 20.0
        meta_path = source_path / 'recording_meta.yaml'
        if meta_path.exists():
            with open(meta_path) as f:
                meta = yaml.safe_load(f)
            fps = meta.get('fps', 20.0)
        return _png_iterator(png_files), fps, w, h, source_path, len(png_files)

    # Single video file
    cap = cv2.VideoCapture(str(source_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return _cap_iterator(cap), fps, w, h, source_path.parent, total


def _cap_iterator(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            break
        yield frame


def _png_iterator(paths):
    for p in paths:
        img = cv2.imread(str(p))
        if img is not None:
            yield img


# --------------------------------------------------------------------------- #
# Homography helpers
# --------------------------------------------------------------------------- #

def pixel_to_world(H, px, py):
    """Apply homography H to a pixel (px, py) -> world (X, Y) on the ground plane."""
    src = np.array([px, py, 1.0])
    dst = H @ src
    if abs(dst[2]) < 1e-9:
        return None
    return (float(dst[0] / dst[2]), float(dst[1] / dst[2]))


# Where the vehicle's ground contact point is, expressed as fractions of the bbox.
# Horizontal: 50% (centered). Vertical: 85% from top (just above the very bottom,
# to avoid protruding bumpers, light bars, and shadow effects).
VEHICLE_REF_X_FRAC = 0.50
VEHICLE_REF_Y_FRAC = 0.85


def vehicle_ground_point(bbox_xyxy):
    """Return the pixel point used as the vehicle's ground contact reference.

    Using a point slightly above the very bottom of the bbox avoids bumper/grill
    artifacts and shadows. This gives a more stable lane assignment than the raw
    bbox bottom-center.
    """
    x1, y1, x2, y2 = bbox_xyxy
    bw = x2 - x1
    bh = y2 - y1
    ref_x = x1 + VEHICLE_REF_X_FRAC * bw
    ref_y = y1 + VEHICLE_REF_Y_FRAC * bh
    return float(ref_x), float(ref_y)


# --------------------------------------------------------------------------- #
# Speed tracking
# --------------------------------------------------------------------------- #

class SpeedTracker:
    """Maintains world-position history per track_id and computes smoothed speed."""

    SMOOTHING_WINDOW = 5  # number of past speeds to average

    def __init__(self, fps, homography):
        self.dt = 1.0 / fps
        self.H = homography
        self.world_history = {}    # track_id -> deque of (frame_idx, world_xy)
        self.speed_history = {}    # track_id -> deque of recent speeds (m/s)

    def update(self, frame_idx, track_id, bbox_xyxy):
        """Given a tracked bbox, return current smoothed speed in m/s."""
        ref_x, ref_y = vehicle_ground_point(bbox_xyxy)
        world_xy = pixel_to_world(self.H, ref_x, ref_y)
        if world_xy is None:
            return 0.0

        history = self.world_history.setdefault(track_id, deque(maxlen=10))
        history.append((frame_idx, world_xy))

        if len(history) < 2:
            return 0.0

        # Speed from previous frame to current
        prev_frame, prev_xy = history[-2]
        frames_between = max(1, frame_idx - prev_frame)
        dt = frames_between * self.dt
        dx = world_xy[0] - prev_xy[0]
        dy = world_xy[1] - prev_xy[1]
        raw_speed = (dx**2 + dy**2) ** 0.5 / dt

        # Smooth speed
        speeds = self.speed_history.setdefault(track_id, deque(maxlen=self.SMOOTHING_WINDOW))
        speeds.append(raw_speed)
        smoothed = sum(speeds) / len(speeds)
        return smoothed


# --------------------------------------------------------------------------- #
# Lane / queue logic
# --------------------------------------------------------------------------- #

# Defaults (can be overridden in analytics_config.yaml under `queue:`)
DEFAULT_QUEUE_SPEED_KMH = 7.2          # ~2 m/s
DEFAULT_QUEUE_MIN_STATIONARY_SECONDS = 2.0


def point_in_polygon(point, polygon):
    """polygon is a list of [x, y] image-pixel coordinates."""
    poly_arr = np.array(polygon, dtype=np.int32)
    result = cv2.pointPolygonTest(poly_arr, (float(point[0]), float(point[1])), False)
    return result >= 0


class QueueTracker:
    """Tracks how long each vehicle has been stationary.

    A vehicle is "queued" only after it has been below the speed threshold
    for at least `min_stationary_seconds` of continuous slow movement.

    This avoids counting vehicles that are briefly slowing down (e.g., a car
    momentarily braking) and gives a more reliable queue count.
    """

    def __init__(self, speed_threshold_kmh, min_stationary_seconds, fps):
        self.speed_threshold_mps = speed_threshold_kmh / 3.6
        self.min_stationary_frames = max(1, int(round(min_stationary_seconds * fps)))
        self.stationary_count = {}  # track_id -> consecutive slow frames

    def annotate(self, detections):
        """Sets det['is_queued'] = True/False on each detection in place."""
        for det in detections:
            tid = det['track_id']
            if det['speed_mps'] < self.speed_threshold_mps:
                self.stationary_count[tid] = self.stationary_count.get(tid, 0) + 1
            else:
                self.stationary_count[tid] = 0
            det['is_queued'] = self.stationary_count[tid] >= self.min_stationary_frames


def compute_queue_counts(lanes, detections):
    """Count queued vehicles per lane.

    detections: list of dicts with at least {'point', 'is_queued'}.
    Caller should run QueueTracker.annotate(detections) first to set 'is_queued'.

    Returns dict {lane_id: count}.
    """
    counts = {lane['id']: 0 for lane in lanes}
    for det in detections:
        if not det.get('is_queued', False):
            continue
        for lane in lanes:
            if point_in_polygon(det['point'], lane['polygon']):
                counts[lane['id']] += 1
                break  # each vehicle counts for at most one lane
    return counts


def draw_lanes_overlay(frame, lanes, queue_counts):
    """Draw lane polygons and queue counts as overlay."""
    if not lanes:
        return
    overlay = frame.copy()
    for lane in lanes:
        poly = np.array(lane['polygon'], dtype=np.int32)
        cv2.fillPoly(overlay, [poly], (60, 60, 200))
        cv2.polylines(frame, [poly], isClosed=True, color=(100, 100, 255), thickness=2)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

    # Lane labels with queue counts
    for lane in lanes:
        poly = np.array(lane['polygon'], dtype=np.int32)
        centroid = poly.mean(axis=0).astype(int)
        count = queue_counts.get(lane['id'], 0)
        label = f"{lane['id']}: {count} queued"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        bg_color = (0, 0, 200) if count > 0 else (40, 40, 40)
        cv2.rectangle(frame, (centroid[0] - 5, centroid[1] - th - 6),
                      (centroid[0] + tw + 5, centroid[1] + 6), bg_color, -1)
        cv2.putText(frame, label, (centroid[0], centroid[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


# --------------------------------------------------------------------------- #
# Forbidden lines / red-light violation logic
# --------------------------------------------------------------------------- #

def _segment_side(p, a, b):
    """Signed area (cross product) telling which side of line A->B point P is on."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def _projection_param(p, a, b):
    """Parameter t of the perpendicular projection of P onto segment A->B.

    t in [0, 1] means the foot of the perpendicular is within the segment.
    """
    abx, aby = b[0] - a[0], b[1] - a[1]
    denom = abx * abx + aby * aby
    if denom == 0:
        return -1.0
    return ((p[0] - a[0]) * abx + (p[1] - a[1]) * aby) / denom


class ViolationDetector:
    """Detects vehicles crossing a forbidden line while the light is red.

    A violation fires when a tracked vehicle's ground reference point flips
    from one side of a line segment to the other (and the crossing is within
    the segment span), while the current light state is 'red'.
    """

    def __init__(self, lines):
        self.lines = lines or []
        self.prev_side = {}        # (track_id, line_id) -> last side sign (-1/0/1)
        self.flagged = set()       # (track_id, line_id) already counted, avoid double-count

    def check(self, detections, light_state, frame_idx):
        """Return a list of new violations this frame: {track_id, line_id, frame}."""
        new_violations = []
        is_red = (light_state == 'red')
        for det in detections:
            tid = det['track_id']
            p = det['point']
            for line in self.lines:
                a, b = line['points'][0], line['points'][1]
                raw = _segment_side(p, a, b)
                sign = 1 if raw > 0 else (-1 if raw < 0 else 0)
                key = (tid, line['id'])
                prev = self.prev_side.get(key)
                self.prev_side[key] = sign

                if prev is None or sign == 0 or prev == 0 or sign == prev:
                    continue  # no clean side flip
                # Crossed the infinite line; require it to be within the segment span
                t = _projection_param(p, a, b)
                if not (-0.1 <= t <= 1.1):
                    continue
                if is_red and key not in self.flagged:
                    self.flagged.add(key)
                    new_violations.append({
                        'track_id': tid,
                        'line_id': line['id'],
                        'frame': frame_idx,
                    })
        return new_violations


def draw_forbidden_lines(frame, lines, active_violation=False):
    """Draw forbidden lines. Red normally; brighter/thicker when a violation just fired."""
    if not lines:
        return
    for line in lines:
        a = tuple(line['points'][0])
        b = tuple(line['points'][1])
        thickness = 4 if active_violation else 2
        cv2.line(frame, a, b, (0, 0, 255), thickness)
        mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
        cv2.putText(frame, line['id'], (mid[0], mid[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


# --------------------------------------------------------------------------- #
# Highway entry zones / entry counting
# --------------------------------------------------------------------------- #

class EntryCounter:
    """Counts unique vehicles entering each zone, grouped by light state at entry.

    A vehicle is counted once per zone, on the frame it first transitions from
    outside the zone to inside it. The light state at that moment is recorded —
    e.g. an entry on red can indicate a ramp-metering violation.
    """

    def __init__(self, zones):
        self.zones = zones or []
        self.inside = {}          # (track_id, zone_id) -> currently inside?
        self.counted = set()      # (track_id, zone_id) already counted
        self.counts = {z['id']: {'total': 0, 'by_light': defaultdict(int)} for z in self.zones}

    def check(self, detections, light_state, frame_idx):
        """Return list of new entries this frame: {track_id, zone_id, frame, light_state}."""
        new_entries = []
        for det in detections:
            tid = det['track_id']
            p = det['point']
            for zone in self.zones:
                key = (tid, zone['id'])
                now_inside = point_in_polygon(p, zone['polygon'])
                was_inside = self.inside.get(key, False)
                self.inside[key] = now_inside
                if now_inside and not was_inside and key not in self.counted:
                    self.counted.add(key)
                    self.counts[zone['id']]['total'] += 1
                    self.counts[zone['id']]['by_light'][light_state] += 1
                    new_entries.append({
                        'track_id': tid, 'zone_id': zone['id'],
                        'frame': frame_idx, 'light_state': light_state,
                    })
        return new_entries

    def summary(self):
        """Return a JSON-serializable summary of counts per zone."""
        return {
            zid: {'total': c['total'], 'by_light': dict(c['by_light'])}
            for zid, c in self.counts.items()
        }


def draw_entry_zones(frame, zones, entry_counter):
    """Draw entry zone polygons with running entry counts (total + by light)."""
    if not zones:
        return
    overlay = frame.copy()
    for zone in zones:
        poly = np.array(zone['polygon'], dtype=np.int32)
        cv2.fillPoly(overlay, [poly], (0, 140, 200))
        cv2.polylines(frame, [poly], isClosed=True, color=(0, 200, 255), thickness=2)
    cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)

    for zone in zones:
        poly = np.array(zone['polygon'], dtype=np.int32)
        centroid = poly.mean(axis=0).astype(int)
        c = entry_counter.counts.get(zone['id'], {})
        total = c.get('total', 0)
        by = c.get('by_light', {})
        label = f"{zone['id']}: {total} in (G:{by.get('green', 0)} R:{by.get('red', 0)})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (centroid[0] - 5, centroid[1] - th - 6),
                      (centroid[0] + tw + 5, centroid[1] + 6), (0, 120, 180), -1)
        cv2.putText(frame, label, (centroid[0], centroid[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


# --------------------------------------------------------------------------- #
# Forbidden zones (no-go polygons, e.g. the chevron gore at a highway ramp)
# --------------------------------------------------------------------------- #

class ForbiddenZoneDetector:
    """Flags a vehicle that drives into a forbidden zone polygon (e.g., the
    painted chevron / gore area at a highway ramp). Unlike red-light violations
    this needs no traffic-light state — the zone is always off-limits. Each
    (track_id, zone_id) is counted once."""

    def __init__(self, zones):
        self.zones = zones or []
        self.flagged = set()  # (track_id, zone_id) already counted

    def check(self, detections, frame_idx):
        """Return new forbidden-zone entries this frame: [{track_id, zone_id, frame}]."""
        new = []
        for det in detections:
            tid = det['track_id']
            p = det['point']
            for zone in self.zones:
                key = (tid, zone['id'])
                if key in self.flagged:
                    continue
                if point_in_polygon(p, zone['polygon']):
                    self.flagged.add(key)
                    new.append({'track_id': tid, 'zone_id': zone['id'], 'frame': frame_idx})
        return new


def draw_forbidden_zones(frame, zones, active=False):
    """Draw forbidden-zone polygons in red; brighter/thicker when one just fired."""
    if not zones:
        return
    overlay = frame.copy()
    for zone in zones:
        poly = np.array(zone['polygon'], dtype=np.int32)
        cv2.fillPoly(overlay, [poly], (0, 0, 220))
    cv2.addWeighted(overlay, 0.40 if active else 0.20, frame, 0.60 if active else 0.80, 0, frame)
    for zone in zones:
        poly = np.array(zone['polygon'], dtype=np.int32)
        cv2.polylines(frame, [poly], isClosed=True, color=(0, 0, 255),
                      thickness=4 if active else 2)
        centroid = poly.mean(axis=0).astype(int)
        label = f"NO-GO {zone['id']}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (centroid[0] - 5, centroid[1] - th - 6),
                      (centroid[0] + tw + 5, centroid[1] + 6), (0, 0, 200), -1)
        cv2.putText(frame, label, (centroid[0], centroid[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


# --------------------------------------------------------------------------- #
# Collision detection (experimental / opt-in)
# --------------------------------------------------------------------------- #

# Defaults (overridable in analytics_config.yaml under `collision:`)
DEFAULT_COLLISION_IOU = 0.10            # bbox overlap fraction to consider "touching"
DEFAULT_COLLISION_SPEED_DROP_KMH = 15.0  # sudden deceleration that suggests impact
DEFAULT_COLLISION_WORLD_DIST_M = 6.0    # ground-plane proximity (cuts perspective FPs)
DEFAULT_COLLISION_WINDOW_SECONDS = 0.6  # window over which the speed drop is measured


def _bbox_iou(a, b):
    """IoU of two [x1, y1, x2, y2] boxes."""
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class CollisionDetector:
    """Heuristic collision detector from tracking output.

    Flags a pair of vehicles as a likely collision when ALL hold:
      1. their bounding boxes overlap (IoU >= iou_threshold) — they visually touch
      2. their ground-plane positions are within world_dist_m — rules out
         perspective overlaps of vehicles that are actually far apart in 3D
      3. at least one of them shows a sudden speed drop >= speed_drop within the
         recent window — a real impact causes abrupt deceleration

    Each unordered pair is flagged once. This is a heuristic, not ground truth;
    thresholds are tunable and the feature is opt-in.
    """

    def __init__(self, iou_threshold, speed_drop_kmh, world_dist_m, window_seconds, fps):
        self.iou_threshold = iou_threshold
        self.speed_drop_mps = speed_drop_kmh / 3.6
        self.world_dist_m = world_dist_m
        self.window = max(1, int(round(window_seconds * fps)))
        self.speed_hist = {}      # track_id -> deque of recent speeds (m/s)
        self.flagged = set()      # frozenset({a, b}) already flagged

    def _sudden_drop(self, track_id, speed_mps):
        hist = self.speed_hist.setdefault(track_id, deque(maxlen=self.window))
        recent_max = max(hist) if hist else speed_mps
        hist.append(speed_mps)
        return (recent_max - speed_mps) >= self.speed_drop_mps

    def check(self, detections, frame_idx):
        """detections: list of {track_id, bbox, world, speed_mps}. Returns new collisions."""
        # Update per-track speed-drop flags first
        dropped = {}
        for d in detections:
            dropped[d['track_id']] = self._sudden_drop(d['track_id'], d['speed_mps'])

        new_collisions = []
        n = len(detections)
        for i in range(n):
            for j in range(i + 1, n):
                da, db = detections[i], detections[j]
                pair = frozenset((da['track_id'], db['track_id']))
                if pair in self.flagged:
                    continue
                if _bbox_iou(da['bbox'], db['bbox']) < self.iou_threshold:
                    continue
                wa, wb = da['world'], db['world']
                if wa is None or wb is None:
                    continue
                dist = ((wa[0] - wb[0]) ** 2 + (wa[1] - wb[1]) ** 2) ** 0.5
                if dist > self.world_dist_m:
                    continue
                if not (dropped.get(da['track_id']) or dropped.get(db['track_id'])):
                    continue
                self.flagged.add(pair)
                new_collisions.append({
                    'frame': frame_idx,
                    'track_a': da['track_id'],
                    'track_b': db['track_id'],
                    'world_dist_m': round(dist, 2),
                })
        return new_collisions


def draw_collisions(frame, detections, collision_pairs):
    """Highlight vehicles involved in a collision this frame (red boxes + marker)."""
    if not collision_pairs:
        return
    involved = set()
    for c in collision_pairs:
        involved.add(c['track_a'])
        involved.add(c['track_b'])
    for d in detections:
        if d['track_id'] in involved:
            x1, y1, x2, y2 = [int(v) for v in d['bbox']]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.putText(frame, "COLLISION", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #

def speed_color(speed_kmh):
    """Color-code by speed (BGR)."""
    if speed_kmh < 5:
        return (180, 180, 180)   # gray (stopped/very slow)
    if speed_kmh < 30:
        return (0, 200, 0)       # green (slow)
    if speed_kmh < 60:
        return (0, 200, 200)     # yellow (normal)
    return (0, 0, 255)           # red (fast)


def class_color(class_id):
    """Box color (BGR) for a class id: our palette if known, else a stable
    generated color so an arbitrary model's classes still get distinct colors."""
    if class_id in CLASS_COLORS:
        return CLASS_COLORS[class_id]
    hue = int((class_id * 47) % 180)
    bgr = cv2.cvtColor(np.uint8([[[hue, 200, 230]]]), cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


def draw_detection(frame, bbox, class_id, track_id, conf, speed_kmh, class_names=None):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    box_color = class_color(class_id)
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    names = class_names if class_names is not None else CLASS_NAMES
    name = names.get(class_id, str(class_id))
    label = f"{name} #{track_id} {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), box_color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Speed label below the box, color-coded
    spd_label = f"{speed_kmh:.0f} km/h"
    spd_color = speed_color(speed_kmh)
    (tw, th), _ = cv2.getTextSize(spd_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    spd_y = y2 + th + 8
    cv2.rectangle(frame, (x1, y2 + 2), (x1 + tw + 6, spd_y + 4), (0, 0, 0), -1)
    cv2.putText(frame, spd_label, (x1 + 3, spd_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, spd_color, 2)


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #

def run(source_arg, model_path, output_dir, conf, iou, show, detect_collisions=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_iter, fps, width, height, source_dir, total_frames = resolve_source(source_arg)
    config = load_analytics_config(source_dir)
    H = np.array(config['calibration']['homography_matrix'])

    print("=" * 60)
    lanes = config.get('lanes', [])

    # Queue settings (from analytics_config.yaml `queue:` section)
    queue_cfg = config.get('queue', {})
    queue_speed_kmh = queue_cfg.get('speed_threshold_kmh', DEFAULT_QUEUE_SPEED_KMH)
    queue_min_seconds = queue_cfg.get('min_stationary_seconds', DEFAULT_QUEUE_MIN_STATIONARY_SECONDS)
    queue_tracker = QueueTracker(queue_speed_kmh, queue_min_seconds, fps)

    # Traffic light state: prefer recorded light_states.csv, else a manual schedule
    # in analytics_config.yaml (light_schedule: [{frame, state}, ...])
    light_csv = source_dir / 'light_states.csv'
    if light_csv.exists():
        light_provider = LightStateProvider.from_csv(light_csv)
    else:
        light_provider = LightStateProvider.from_schedule(config.get('light_schedule'))

    # Forbidden lines + red-light violation detection
    forbidden_lines = config.get('forbidden_lines', [])
    violation_detector = ViolationDetector(forbidden_lines)

    # Highway entry zones + entry counting
    entry_zones = config.get('entry_zones', [])
    entry_counter = EntryCounter(entry_zones)

    # Forbidden zones (no-go polygons, e.g. chevron gore at a ramp) — violation on entry
    forbidden_zones = config.get('forbidden_zones', [])
    fzone_detector = ForbiddenZoneDetector(forbidden_zones)

    # Collision detection (opt-in via --collisions; thresholds from `collision:` config)
    collision_detector = None
    if detect_collisions:
        cc = config.get('collision', {})
        collision_detector = CollisionDetector(
            iou_threshold=cc.get('iou_threshold', DEFAULT_COLLISION_IOU),
            speed_drop_kmh=cc.get('speed_drop_kmh', DEFAULT_COLLISION_SPEED_DROP_KMH),
            world_dist_m=cc.get('world_distance_m', DEFAULT_COLLISION_WORLD_DIST_M),
            window_seconds=cc.get('window_seconds', DEFAULT_COLLISION_WINDOW_SECONDS),
            fps=fps,
        )

    print("Traffic Analytics — Speed + Queue per lane")
    print("=" * 60)
    print(f"Source:      {source_arg}")
    print(f"Model:       {model_path}")
    print(f"Resolution:  {width}x{height} @ {fps:.1f} FPS")
    print(f"Total frames: {total_frames if total_frames else 'live stream'}")
    print(f"Calibration: {config['calibration'].get('mode', '?')}")
    print(f"Lanes:       {len(lanes)}")
    print(f"Queue:       slower than {queue_speed_kmh:.1f} km/h for {queue_min_seconds:.1f}+ sec")
    print(f"Light state: {'available (' + light_provider.mode + ')' if light_provider.available else 'none'}")
    print(f"Forbidden lines: {len(forbidden_lines)}")
    print(f"Entry zones: {len(entry_zones)}")
    print(f"Forbidden zones: {len(forbidden_zones)}")
    print(f"Collision detection: {'ON' if collision_detector else 'off'}")
    print(f"Output:      {output_dir.resolve()}")
    print("=" * 60)

    model = YOLO(model_path)
    # Use the model's OWN class names so any trained model labels correctly,
    # not just our 7-class model.
    class_names = model.names
    speed_tracker = SpeedTracker(fps=fps, homography=H)

    # Output video
    output_video = output_dir / 'analytics.mp4'
    writer = cv2.VideoWriter(str(output_video),
                             cv2.VideoWriter_fourcc(*'mp4v'),
                             fps, (width, height))

    # CSV log per track
    csv_path = output_dir / 'per_track.csv'
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['frame', 'track_id', 'class', 'world_x', 'world_y', 'speed_mps', 'speed_kmh'])

    # CSV log per lane (queue counts per frame)
    queue_csv_file = None
    queue_csv_writer = None
    if lanes:
        queue_csv_path = output_dir / 'per_lane_queue.csv'
        queue_csv_file = open(queue_csv_path, 'w', newline='')
        queue_csv_writer = csv.writer(queue_csv_file)
        queue_csv_writer.writerow(['frame'] + [lane['id'] for lane in lanes])

    # CSV log of violations
    violation_csv_file = None
    violation_csv_writer = None
    if forbidden_lines:
        violation_csv_path = output_dir / 'violations.csv'
        violation_csv_file = open(violation_csv_path, 'w', newline='')
        violation_csv_writer = csv.writer(violation_csv_file)
        violation_csv_writer.writerow(['frame', 'track_id', 'line_id', 'light_state'])

    # CSV log of entry-zone entries
    entry_csv_file = None
    entry_csv_writer = None
    if entry_zones:
        entry_csv_path = output_dir / 'entries.csv'
        entry_csv_file = open(entry_csv_path, 'w', newline='')
        entry_csv_writer = csv.writer(entry_csv_file)
        entry_csv_writer.writerow(['frame', 'track_id', 'zone_id', 'light_state'])

    # CSV log of forbidden-zone violations
    fzone_csv_file = None
    fzone_csv_writer = None
    if forbidden_zones:
        fzone_csv_path = output_dir / 'forbidden_zones.csv'
        fzone_csv_file = open(fzone_csv_path, 'w', newline='')
        fzone_csv_writer = csv.writer(fzone_csv_file)
        fzone_csv_writer.writerow(['frame', 'track_id', 'zone_id'])

    # CSV log of collisions
    collision_csv_file = None
    collision_csv_writer = None
    if collision_detector:
        collision_csv_path = output_dir / 'collisions.csv'
        collision_csv_file = open(collision_csv_path, 'w', newline='')
        collision_csv_writer = csv.writer(collision_csv_file)
        collision_csv_writer.writerow(['frame', 'track_a', 'track_b', 'world_dist_m'])

    frame_idx = 0
    start_time = time.time()
    track_ids_seen = set()
    total_violations = 0
    total_entries = 0
    total_fzone = 0
    total_collisions = 0

    try:
        for frame in frame_iter:
            results = model.track(
                source=frame, conf=conf, iou=iou,
                persist=True, tracker='bytetrack.yaml',
                verbose=False,
            )

            # Collect detections in this frame for queue computation
            frame_detections = []

            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                for j in range(len(boxes)):
                    xyxy = boxes.xyxy[j].cpu().numpy()
                    cls = int(boxes.cls[j].cpu().item())
                    conf_score = float(boxes.conf[j].cpu().item())
                    track_id = int(boxes.id[j].cpu().item()) if boxes.id is not None else -1
                    if track_id == -1:
                        continue
                    track_ids_seen.add(track_id)

                    speed_mps = speed_tracker.update(frame_idx, track_id, xyxy)
                    speed_kmh = speed_mps * 3.6

                    ref_x, ref_y = vehicle_ground_point(xyxy)
                    world = pixel_to_world(H, ref_x, ref_y) or (0, 0)
                    frame_detections.append({
                        'point': (ref_x, ref_y),
                        'speed_mps': speed_mps,
                        'track_id': track_id,
                        'bbox': [float(v) for v in xyxy],
                        'world': world,
                    })

                    draw_detection(frame, xyxy, cls, track_id, conf_score, speed_kmh, class_names)

                    csv_writer.writerow([
                        frame_idx, track_id, class_names.get(cls, cls),
                        round(world[0], 3), round(world[1], 3),
                        round(speed_mps, 3), round(speed_kmh, 1),
                    ])

            # Annotate detections with is_queued (requires history across frames)
            queue_tracker.annotate(frame_detections)

            # Queue counts per lane
            queue_counts = compute_queue_counts(lanes, frame_detections)
            if lanes:
                draw_lanes_overlay(frame, lanes, queue_counts)
                if queue_csv_writer:
                    queue_csv_writer.writerow([frame_idx] + [queue_counts.get(l['id'], 0) for l in lanes])

            # Current light state for this frame
            light_state = light_provider.state_at(frame_idx) if light_provider.available else 'unknown'

            # Red-light violations
            violations_now = []
            if forbidden_lines:
                violations_now = violation_detector.check(frame_detections, light_state, frame_idx)
                for v in violations_now:
                    total_violations += 1
                    if violation_csv_writer:
                        violation_csv_writer.writerow([v['frame'], v['track_id'], v['line_id'], light_state])
                    print(f"  VIOLATION: track #{v['track_id']} crossed {v['line_id']} on red (frame {frame_idx})")
                draw_forbidden_lines(frame, forbidden_lines, active_violation=bool(violations_now))

            # Highway entry counting
            if entry_zones:
                entries_now = entry_counter.check(frame_detections, light_state, frame_idx)
                for e in entries_now:
                    total_entries += 1
                    if entry_csv_writer:
                        entry_csv_writer.writerow([e['frame'], e['track_id'], e['zone_id'], e['light_state']])
                draw_entry_zones(frame, entry_zones, entry_counter)

            # Forbidden-zone violations (no-go polygons; no light state needed)
            fzone_now = []
            if forbidden_zones:
                fzone_now = fzone_detector.check(frame_detections, frame_idx)
                for fz in fzone_now:
                    total_fzone += 1
                    if fzone_csv_writer:
                        fzone_csv_writer.writerow([fz['frame'], fz['track_id'], fz['zone_id']])
                    print(f"  FORBIDDEN ZONE: track #{fz['track_id']} entered {fz['zone_id']} (frame {frame_idx})")
                draw_forbidden_zones(frame, forbidden_zones, active=bool(fzone_now))

            # Collision detection (experimental)
            collisions_now = []
            if collision_detector:
                collisions_now = collision_detector.check(frame_detections, frame_idx)
                for c in collisions_now:
                    total_collisions += 1
                    if collision_csv_writer:
                        collision_csv_writer.writerow([c['frame'], c['track_a'], c['track_b'], c['world_dist_m']])
                    print(f"  COLLISION: tracks #{c['track_a']} & #{c['track_b']} "
                          f"({c['world_dist_m']}m apart, frame {frame_idx})")
                draw_collisions(frame, frame_detections, collisions_now)

            # Traffic light indicator
            if light_provider.available:
                draw_light_indicator(frame, light_state)

            # Violation banner (red-light line crossing or forbidden-zone entry)
            vtxt = None
            if violations_now:
                vtxt = f"RED-LIGHT VIOLATION x{len(violations_now)}"
            elif fzone_now:
                vtxt = f"FORBIDDEN ZONE x{len(fzone_now)}"
            if vtxt:
                (tw, th), _ = cv2.getTextSize(vtxt, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                cv2.rectangle(frame, (width // 2 - tw // 2 - 12, 50),
                              (width // 2 + tw // 2 + 12, 50 + th + 16), (0, 0, 255), -1)
                cv2.putText(frame, vtxt, (width // 2 - tw // 2, 50 + th + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            # HUD overlay
            elapsed = time.time() - start_time
            inference_fps = (frame_idx + 1) / max(elapsed, 0.001)
            hud = f"Frame {frame_idx}  |  {inference_fps:.1f} FPS  |  Tracks: {len(track_ids_seen)}  |  Violations: {total_violations}"
            if entry_zones:
                hud += f"  |  Entries: {total_entries}"
            if forbidden_zones:
                hud += f"  |  No-go: {total_fzone}"
            if collision_detector:
                hud += f"  |  Collisions: {total_collisions}"
            cv2.putText(frame, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)

            writer.write(frame)
            if show:
                cv2.imshow('Analytics', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_idx += 1
            if frame_idx % 100 == 0:
                progress = f"{frame_idx}/{total_frames}" if total_frames else str(frame_idx)
                print(f"  Frame {progress}  |  {inference_fps:.1f} FPS  |  Tracks: {len(track_ids_seen)}")

    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        writer.release()
        csv_file.close()
        if queue_csv_file:
            queue_csv_file.close()
        if violation_csv_file:
            violation_csv_file.close()
        if entry_csv_file:
            entry_csv_file.close()
        if fzone_csv_file:
            fzone_csv_file.close()
        if collision_csv_file:
            collision_csv_file.close()
        if show:
            cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    print(f"\nDone — processed {frame_idx} frames in {elapsed:.1f}s ({frame_idx / max(elapsed, 0.001):.1f} FPS)")
    print(f"  Unique tracks: {len(track_ids_seen)}")
    if forbidden_lines:
        print(f"  Red-light violations: {total_violations}")
    if entry_zones:
        print(f"  Entry counts: {entry_counter.summary()}")
    if forbidden_zones:
        print(f"  Forbidden-zone violations: {total_fzone}")
    if collision_detector:
        print(f"  Collisions detected: {total_collisions}")
    print(f"  Video:        {output_video.resolve()}")
    print(f"  Per-track CSV: {csv_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description='Traffic analytics: speed per car')
    parser.add_argument('source', type=str, help='Video file, frame directory, webcam, or stream URL')
    parser.add_argument('model', type=str, help='Path to .pt YOLO model')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: <source_dir>/analytics_output)')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--iou', type=float, default=0.5)
    parser.add_argument('--show', action='store_true', help='Show live preview window')
    parser.add_argument('--collisions', action='store_true',
                        help='Enable experimental collision detection (bbox overlap + world proximity + speed drop)')
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"Model not found: {args.model}")
        sys.exit(1)

    if args.output is None:
        if args.source.isdigit() or args.source.startswith(('rtsp://', 'http://')):
            output_dir = 'analytics_output'
        else:
            src = Path(args.source)
            base = src if src.is_dir() else src.parent
            output_dir = base / 'analytics_output'
    else:
        output_dir = args.output

    run(args.source, args.model, output_dir, args.conf, args.iou, args.show, args.collisions)


if __name__ == '__main__':
    main()
