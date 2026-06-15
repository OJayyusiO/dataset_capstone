"""
Unified MVP: detect + track + speed + queue length.

One pass over the video producing:
  - per-class detection with class-distinct box colors and class names
    rendered on top of every bounding box
  - ByteTrack persistent IDs across frames + carry-forward through brief
    YOLO misses (no flickering boxes)
  - per-track speed in km/h, perspective-aware via measurements/redline.png
    calibration, with hysteresis so stationary cars stay at 0
  - queue counts + queue length in meters per ROI, ROIs auto-extracted
    from measurements/queue*.png

Outputs (runs/full/):
  annotated_full.mp4
  detections.csv          per-frame per-track row (incl. roi_idx, in_queue)
  queue_log.csv           per-frame queue summary
  rois.json               polygons used, in video pixel space

Usage:
    python full_mvp.py
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
from queue_length import (
    MAX_QUEUE_KMH,
    QUEUE_ROIS,
    queue_length_meters,
)

# Per-class BGR colors, chosen for visual distinctness.
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "car":          ( 80, 220,  60),   # green
    "truck":        (255, 150,   0),   # cyan-blue
    "bus":          ( 30,  90, 250),   # red-orange
    "motorcycle":   (  0, 220, 240),   # yellow
    "fire truck":   ( 60,  60, 255),   # red
    "ambulance":    (255,  90, 255),   # magenta
}
DEFAULT_COLOR = (200, 200, 200)


def color_for(cname: str) -> tuple[int, int, int]:
    return CLASS_COLORS.get(cname, DEFAULT_COLOR)


def draw_detection(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
    cname: str,
    tid: int,
    speed_kmh: float,
    in_queue: bool,
    roi_idx: int,
    ghost: bool,
) -> None:
    x1, y1, x2, y2 = bbox
    p1 = (int(x1), int(y1))
    p2 = (int(x2), int(y2))
    color = color_for(cname)
    thickness = 1 if ghost else (3 if in_queue else 2)
    cv2.rectangle(frame, p1, p2, color, thickness)

    label = f"{cname} #{int(tid)} {speed_kmh:.0f}km/h"
    if in_queue:
        label += f" [Q{roi_idx}]"

    # Filled black tag behind the text so the label stays readable on any
    # background. Sits just above the top edge of the bbox.
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    tx = p1[0]
    ty = max(th + 4, p1[1] - 4)
    cv2.rectangle(frame, (tx - 2, ty - th - 4), (tx + tw + 2, ty + 4), (0, 0, 0), -1)
    cv2.putText(
        frame, label, (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
    )


def draw_class_legend(
    frame: np.ndarray,
    class_names_to_show: list[str],
) -> None:
    """Compact swatch+label legend in the top-right corner."""
    width = frame.shape[1]
    x0, y0 = width - 220, 25
    for i, cname in enumerate(class_names_to_show):
        col = color_for(cname)
        yy = y0 + i * 22
        cv2.rectangle(frame, (x0, yy), (x0 + 18, yy + 14), col, -1)
        cv2.putText(
            frame, cname, (x0 + 24, yy + 13),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )


def run(
    video_path: Path,
    model_name: str,
    out_dir: Path,
    max_queue_kmh: float,
) -> None:
    if not video_path.exists():
        sys.exit(f"[full] video not found: {video_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"[full] could not open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    rois = QUEUE_ROIS
    print(f"[full] using {len(rois)} frozen ROI(s) and frozen mpp(y) calibration")

    model, used_model = load_model(model_name)
    name_to_id = {n.lower(): i for i, n in model.names.items()}
    allowed_ids = [
        name_to_id[n.lower()] for n in ALLOWED_CLASS_NAMES if n.lower() in name_to_id
    ]
    if not allowed_ids:
        sys.exit("[full] none of the requested classes exist in this model")
    available_class_names = [model.names[i] for i in allowed_ids]
    print(f"[full] classes: {available_class_names}")

    inference_conf = min(CONF_THRESHOLD, *PER_CLASS_CONF.values())

    video_out = out_dir / "annotated_full.mp4"
    writer = cv2.VideoWriter(
        str(video_out),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    det_path = out_dir / "detections.csv"
    queue_path = out_dir / "queue_log.csv"
    det_file = det_path.open("w", newline="")
    queue_file = queue_path.open("w", newline="")
    det_writer = csv.writer(det_file)
    queue_writer = csv.writer(queue_file)
    det_writer.writerow(
        ["frame", "track_id", "class_name", "x", "y", "speed_kmh",
         "roi_idx", "in_queue"]
    )
    # Per-ROI count + length only — ROIs are independent traffic lights,
    # so no aggregated total column.
    per_roi_count_cols = [f"roi{i}_count" for i in range(len(rois))]
    per_roi_len_cols = [f"roi{i}_length_m" for i in range(len(rois))]
    queue_writer.writerow(
        ["frame", "time_s"] + per_roi_count_cols + per_roi_len_cols
    )

    history: dict[int, deque] = {}
    speed_hist: dict[int, deque] = {}
    last_state: dict[int, dict] = {}

    def membership(gx: float, gy: float) -> int:
        for i, roi in enumerate(rois):
            if cv2.pointPolygonTest(roi, (gx, gy), False) >= 0:
                return i
        return -1

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

    frame_idx = 0
    try:
        for r in results_iter:
            frame = r.orig_img.copy()
            names = r.names

            # Translucent ROI fill + outline + label
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
                        dfr = frame_idx - f0
                        if dfr > 0:
                            dpx = ((gx - x0) ** 2 + (gy - y0) ** 2) ** 0.5
                            raw_speed_kmh = (dpx * mpp_at_y(gy)) / (dfr / fps) * 3.6

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

                    draw_detection(
                        frame, (x1, y1, x2, y2), cname, int(tid),
                        speed_kmh, in_queue, roi_idx, ghost=False,
                    )
                    det_writer.writerow(
                        [frame_idx, int(tid), cname,
                         f"{gx:.1f}", f"{gy:.1f}", f"{speed_kmh:.1f}",
                         roi_idx, int(in_queue)]
                    )

            # Carry-forward boxes that YOLO temporarily missed
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
                draw_detection(
                    frame, st["bbox"], st["class"], int(tid),
                    st["speed"], in_queue, roi_idx, ghost=True,
                )
                det_writer.writerow(
                    [frame_idx, int(tid), st["class"],
                     f"{gx:.1f}", f"{gy:.1f}", f"{st['speed']:.1f}",
                     roi_idx, int(in_queue)]
                )

            roi_lengths = [queue_length_meters(ys, mpp_at_y) for ys in roi_ys]
            t_s = frame_idx / fps
            queue_writer.writerow(
                [frame_idx, f"{t_s:.2f}"]
                + roi_counts + [f"{x:.1f}" for x in roi_lengths]
            )

            # Per-ROI HUD lines — each traffic light reported separately.
            cv2.putText(
                frame, f"t={t_s:.1f}s", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 5, cv2.LINE_AA,
            )
            cv2.putText(
                frame, f"t={t_s:.1f}s", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
            )
            for i, (c, l) in enumerate(zip(roi_counts, roi_lengths)):
                line = f"ROI{i}: queue={c}  length={l:.1f} m"
                yy = 80 + i * 36
                cv2.putText(frame, line, (20, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 6, cv2.LINE_AA)
                cv2.putText(frame, line, (20, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

            draw_class_legend(frame, available_class_names)

            writer.write(frame)
            frame_idx += 1
    finally:
        writer.release()
        det_file.close()
        queue_file.close()

    print(f"[full] done. model used: {used_model}")
    print(f"[full] frames processed: {frame_idx}")
    print(f"[full] annotated video : {video_out}")
    print(f"[full] detections CSV  : {det_path}")
    print(f"[full] queue CSV       : {queue_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified MVP: detect + track + speed + queue")
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--video", default=str(DEFAULT_VIDEO))
    p.add_argument("--out", default=str(HERE / "runs" / "full"))
    p.add_argument("--max-queue-kmh", type=float, default=MAX_QUEUE_KMH)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(Path(args.video), args.model, Path(args.out), args.max_queue_kmh)
