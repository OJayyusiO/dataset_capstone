"""
Queue length estimation.

Uses frozen ROI polygons (QUEUE_ROIS below) and the frozen mpp(y)
calibration from track_and_speed.py. The reference screenshots in
measurements/ are documentation only — they're not read at runtime.

To re-derive QUEUE_ROIS from updated screenshots, call
extract_roi_from_image() (kept as a utility further down) and copy
the printed vertices back into QUEUE_ROIS.

Pipeline:
  1. Run Ultralytics tracking with the same calibration as track_and_speed.py.
  2. For each tracked vehicle each frame:
        in_roi  = bottom-center is inside ANY of the queue polygons
        queued  = in_roi AND speed_kmh <= MAX_QUEUE_KMH (default 4)
  3. Writes:
        runs/queue/annotated_queue.mp4   ROI overlay + per-car boxes + HUD
        runs/queue/queue_tracks.csv      per-frame per-track row
        runs/queue/queue_log.csv         per-frame per-ROI summary

Usage:
    python queue_length.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from detect import (
    ALLOWED_CLASS_NAMES,
    CONF_THRESHOLD,
    DEFAULT_VIDEO,
    HERE,
    IOU_THRESHOLD,
    MODEL_NAME,
    PER_CLASS_CONF,
    load_model,
)
from track_and_speed import (
    DETECTION_GRACE_FRAMES,
    SPEED_SMOOTHING_FRAMES,
    SPEED_WINDOW_FRAMES,
    STOPPED_ENTER_KMH,
    STOPPED_EXIT_KMH,
    mpp_at_y,
)

# A tracked vehicle whose bottom-center is inside any ROI AND whose current
# rolling-window speed is at or below this value counts as part of the queue.
MAX_QUEUE_KMH = 4.0

# ---------------------------------------------------------------------------
# Queue ROIs — frozen from measurements/queue1.png and measurements/queue2.png.
# Order matches roi_idx in the outputs. Each is an Nx2 array of (x, y) in
# video pixel coordinates.
#
# To regenerate from fresh annotated screenshots, call extract_roi_from_image()
# below and copy the printed vertices here.
# ---------------------------------------------------------------------------
QUEUE_ROIS: list[np.ndarray] = [
    np.array([[822, 117], [543, 243], [201, 250], [611, 110]], dtype=np.int32),
    np.array([[1469, 450], [1257, 470], [1090, 317], [1463, 351]], dtype=np.int32),
]

# Calibration utility constants below are NOT used at runtime — only by
# extract_roi_from_image() if the user wants to regenerate QUEUE_ROIS.
QUEUE_REFERENCE_IMAGES = [
    HERE / "measurements" / "queue1.png",
    HERE / "measurements" / "queue2.png",
]

# HSV range for the yellow boundary outline.
YELLOW_HSV_LO = (20, 120, 120)
YELLOW_HSV_HI = (35, 255, 255)


def extract_roi_from_image(image_path: Path, video_width: int) -> np.ndarray:
    """Detect the yellow polygon outline drawn on the screenshot and return
    its vertices in video pixel space, as an Nx2 int32 array.
    """
    if not image_path.exists():
        sys.exit(f"[queue] queue ROI image not found: {image_path}")
    img = cv2.imread(str(image_path))
    if img is None:
        sys.exit(f"[queue] could not read queue ROI image: {image_path}")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, YELLOW_HSV_LO, YELLOW_HSV_HI)

    # Largest yellow blob = the drawn polygon outline; smaller yellow
    # speckles (logo, signage) are dropped.
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        sys.exit(f"[queue] no yellow pixels in {image_path.name} — can't extract ROI")
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    blob = (labels == largest).astype(np.uint8) * 255

    # Close small stroke gaps so the outline is one ring.
    blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    ys, xs = np.where(blob > 0)
    if len(xs) < 50:
        sys.exit(f"[queue] yellow blob in {image_path.name} is only {len(xs)} px")

    pts = np.column_stack([xs, ys]).astype(np.int32)
    hull = cv2.convexHull(pts)
    peri = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.02 * peri, True).reshape(-1, 2)

    image_width = img.shape[1]
    scale = video_width / image_width
    scaled = (approx.astype(np.float64) * scale).astype(np.int32)
    print(
        f"[queue] {image_path.name}: {len(scaled)}-vertex ROI "
        f"({img.shape[1]}px → {video_width}px, x{scale:.4f})"
    )
    return scaled


def queue_length_meters(queued_ys: list[float], mpp_fn) -> float:
    """Rough physical length of the queue: y-range of queued cars × mpp at
    the midpoint y. Crude but uses the existing perspective calibration.
    """
    if len(queued_ys) < 2:
        return 0.0
    y_min, y_max = min(queued_ys), max(queued_ys)
    mid = (y_min + y_max) / 2
    return abs(y_max - y_min) * mpp_fn(mid)


def run(
    video_path: Path,
    model_name: str,
    out_dir: Path,
    max_queue_kmh: float,
) -> None:
    if not video_path.exists():
        sys.exit(f"[queue] video not found: {video_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"[queue] could not open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    rois = QUEUE_ROIS
    print(f"[queue] using {len(rois)} frozen ROI(s) from QUEUE_ROIS constant")

    model, used_model = load_model(model_name)

    name_to_id = {n.lower(): i for i, n in model.names.items()}
    allowed_ids = [name_to_id[n.lower()] for n in ALLOWED_CLASS_NAMES if n.lower() in name_to_id]
    if not allowed_ids:
        sys.exit("[queue] none of the requested classes exist in this model")
    print(f"[queue] keeping classes: {[model.names[i] for i in allowed_ids]}")

    inference_conf = min(CONF_THRESHOLD, *PER_CLASS_CONF.values())

    video_out_path = out_dir / "annotated_queue.mp4"
    writer = cv2.VideoWriter(
        str(video_out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    per_track_path = out_dir / "queue_tracks.csv"
    per_frame_path = out_dir / "queue_log.csv"
    tf = per_track_path.open("w", newline="")
    ff = per_frame_path.open("w", newline="")
    t_writer = csv.writer(tf)
    f_writer = csv.writer(ff)
    t_writer.writerow(
        [
            "frame", "track_id", "class_name", "x", "y", "speed_kmh",
            "roi_idx", "in_queue",
        ]
    )
    # Per-ROI count + length only — ROIs are independent traffic lights,
    # so no aggregated total column.
    per_roi_count_cols = [f"roi{i}_count" for i in range(len(rois))]
    per_roi_len_cols = [f"roi{i}_length_m" for i in range(len(rois))]
    f_writer.writerow(
        ["frame", "time_s"] + per_roi_count_cols + per_roi_len_cols
    )

    history: dict[int, deque] = {}
    speed_hist: dict[int, deque] = {}
    last_state: dict[int, dict] = {}

    results_iter = model.track(
        source=str(video_path),
        conf=inference_conf,
        iou=IOU_THRESHOLD,
        classes=allowed_ids,
        persist=True,
        stream=True,
        verbose=False,
        tracker="bytetrack.yaml",
    )

    def membership(gx: float, gy: float) -> int:
        for i, roi in enumerate(rois):
            if cv2.pointPolygonTest(roi, (gx, gy), False) >= 0:
                return i
        return -1

    def draw_box(frame, tid, cname, bbox, speed_kmh, roi_idx, in_queue, ghost):
        x1, y1, x2, y2 = bbox
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        if in_queue:
            color = (0, 0, 255)
        elif roi_idx >= 0:
            color = (0, 165, 255)
        else:
            color = (200, 200, 200)
        thickness = 1 if ghost else 2
        cv2.rectangle(frame, p1, p2, color, thickness)
        label = f"#{int(tid)} {cname} {speed_kmh:.0f}km/h"
        if in_queue:
            label += f" [Q{roi_idx}]"
        cv2.putText(
            frame, label, (p1[0], max(12, p1[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
        )

    frame_idx = 0
    try:
        for r in results_iter:
            frame = r.orig_img.copy()
            names = r.names

            # Translucent ROI fill + outline for every queue polygon
            overlay = frame.copy()
            for roi in rois:
                cv2.fillPoly(overlay, [roi], (0, 200, 200))
            frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)
            for i, roi in enumerate(rois):
                cv2.polylines(frame, [roi], True, (0, 255, 255), 2)
                cx, cy = roi.mean(axis=0).astype(int)
                cv2.putText(
                    frame, f"ROI{i}", (int(cx), int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA,
                )

            roi_counts = [0] * len(rois)
            roi_ys: list[list[float]] = [[] for _ in rois]
            detected_this_frame: set[int] = set()

            if r.boxes is not None and r.boxes.id is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy()
                cls_ids = r.boxes.cls.cpu().numpy().astype(int)
                track_ids = r.boxes.id.cpu().numpy().astype(int)
                confs = r.boxes.conf.cpu().numpy()

                for (x1, y1, x2, y2), cid, tid, cf_val in zip(xyxy, cls_ids, track_ids, confs):
                    cname = names.get(int(cid), str(int(cid)))
                    if float(cf_val) < PER_CLASS_CONF.get(cname, CONF_THRESHOLD):
                        continue

                    gx = float((x1 + x2) / 2)
                    gy = float(y2)

                    hist = history.setdefault(int(tid), deque(maxlen=SPEED_WINDOW_FRAMES + 1))
                    hist.append((frame_idx, gx, gy))

                    raw_speed_kmh = 0.0
                    if len(hist) >= 2:
                        f0, x0, y0 = hist[0]
                        df = frame_idx - f0
                        if df > 0:
                            dpx = ((gx - x0) ** 2 + (gy - y0) ** 2) ** 0.5
                            raw_speed_kmh = (dpx * mpp_at_y(gy)) / (df / fps) * 3.6

                    sh = speed_hist.setdefault(int(tid), deque(maxlen=SPEED_SMOOTHING_FRAMES))
                    sh.append(raw_speed_kmh)
                    smoothed = float(np.median(sh))

                    prev = last_state.get(int(tid), {})
                    was_stopped = prev.get("stopped", False)
                    if was_stopped:
                        is_stopped = smoothed < STOPPED_EXIT_KMH
                    else:
                        is_stopped = smoothed < STOPPED_ENTER_KMH
                    speed_kmh = 0.0 if is_stopped else smoothed

                    roi_idx = membership(gx, gy)
                    in_roi = roi_idx >= 0
                    in_queue = in_roi and speed_kmh <= max_queue_kmh

                    if in_queue:
                        roi_counts[roi_idx] += 1
                        roi_ys[roi_idx].append(gy)

                    last_state[int(tid)] = {
                        "bbox": (float(x1), float(y1), float(x2), float(y2)),
                        "class": cname,
                        "speed": speed_kmh,
                        "stopped": is_stopped,
                        "last_frame": frame_idx,
                        "gx": gx,
                        "gy": gy,
                    }
                    detected_this_frame.add(int(tid))

                    draw_box(frame, tid, cname, (x1, y1, x2, y2), speed_kmh, roi_idx, in_queue, ghost=False)
                    t_writer.writerow(
                        [
                            frame_idx, int(tid), cname,
                            f"{gx:.1f}", f"{gy:.1f}", f"{speed_kmh:.1f}",
                            roi_idx, int(in_queue),
                        ]
                    )

            # Carry-forward: count + draw tracks YOLO temporarily missed.
            for tid, st in list(last_state.items()):
                if tid in detected_this_frame:
                    continue
                if frame_idx - st["last_frame"] > DETECTION_GRACE_FRAMES:
                    del last_state[tid]
                    continue
                gx, gy = st["gx"], st["gy"]
                roi_idx = membership(gx, gy)
                in_roi = roi_idx >= 0
                in_queue = in_roi and st["speed"] <= max_queue_kmh
                if in_queue:
                    roi_counts[roi_idx] += 1
                    roi_ys[roi_idx].append(gy)
                draw_box(frame, tid, st["class"], st["bbox"], st["speed"], roi_idx, in_queue, ghost=True)
                t_writer.writerow(
                    [
                        frame_idx, int(tid), st["class"],
                        f"{gx:.1f}", f"{gy:.1f}", f"{st['speed']:.1f}",
                        roi_idx, int(in_queue),
                    ]
                )

            roi_lengths = [queue_length_meters(ys, mpp_at_y) for ys in roi_ys]
            t_s = frame_idx / fps
            f_writer.writerow(
                [frame_idx, f"{t_s:.2f}"]
                + roi_counts
                + [f"{x:.1f}" for x in roi_lengths]
            )

            # Per-ROI HUD lines — each traffic light reported separately.
            cv2.putText(frame, f"t={t_s:.1f}s", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 5, cv2.LINE_AA)
            cv2.putText(frame, f"t={t_s:.1f}s", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            for i, (c, l) in enumerate(zip(roi_counts, roi_lengths)):
                line = f"ROI{i}: queue={c}  length={l:.1f} m"
                yy = 80 + i * 36
                cv2.putText(frame, line, (20, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 6, cv2.LINE_AA)
                cv2.putText(frame, line, (20, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

            writer.write(frame)
            frame_idx += 1
    finally:
        writer.release()
        tf.close()
        ff.close()

    print(f"[queue] done. model used: {used_model}")
    print(f"[queue] frames processed: {frame_idx}")
    print(f"[queue] annotated video : {video_out_path}")
    print(f"[queue] per-track CSV   : {per_track_path}")
    print(f"[queue] per-frame log   : {per_frame_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Queue length detection (ROI + speed rule)")
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--video", default=str(DEFAULT_VIDEO))
    p.add_argument("--out", default=str(HERE / "runs" / "queue"))
    p.add_argument(
        "--max-queue-kmh", type=float, default=MAX_QUEUE_KMH,
        help="Speed at or below which a car in any ROI is counted as queued",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(Path(args.video), args.model, Path(args.out), args.max_queue_kmh)
