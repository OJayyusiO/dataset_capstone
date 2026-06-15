"""
Track vehicles and estimate their speed from a video.

Pipeline:
  1. Show first frame; you click two points whose real-world distance you know
     (e.g. across a lane ~3.5 m, or along a crosswalk).
  2. Type that distance (meters) at the prompt.
  3. Ultralytics tracking (ByteTrack) assigns stable IDs across frames.
  4. Per-track speed = (pixel displacement over a rolling window) * m/px
     * fps, converted to km/h.
  5. Writes:
        runs/tracked/annotated_speed.mp4   boxes + #id + class + km/h
        runs/tracked/speeds.csv            frame, track_id, class, x, y, kmh

Usage:
    python track_and_speed.py
    python track_and_speed.py --video other.mp4 --model yolo11s.pt
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

# Rolling-window length (frames) for the displacement→speed calculation.
# Larger = smoother, more lag. Smaller = snappier, jitterier.
SPEED_WINDOW_FRAMES = 10

# Median of the last N raw speeds per track — kills frame-to-frame wobble
# when a parked car's bbox jitters by a few pixels each detection.
SPEED_SMOOTHING_FRAMES = 7

# Stopped-state hysteresis: cars enter "stopped" when smoothed speed falls
# below ENTER, and only leave it again when smoothed speed rises above EXIT.
# This stops single-threshold flip-flop (0 → 2.1 → 0 → 2.3 → ...).
STOPPED_ENTER_KMH = 2.0
STOPPED_EXIT_KMH = 5.0

# When YOLO briefly misses a tracked vehicle, carry forward its last known
# bbox for this many frames so the visualization and any downstream counts
# don't flicker on/off.
DETECTION_GRACE_FRAMES = 15

# ---------------------------------------------------------------------------
# Calibration — frozen from measurements/redline.png (green=3.9 m, red=2.5 m).
#
# The reference picture had a green line (3.9 m, at video y≈411) and a red
# line (2.5 m, at video y≈180). We fit a linear m/px ramp between them so
# foreground vs background detections each get a sensible scale —
# a poor man's homography that handles perspective.
#
# To regenerate these numbers from a fresh screenshot, call
# measure_from_reference_image(...) below and copy the printed values here.
# ---------------------------------------------------------------------------
MPP_AT_Y_SLOPE = -1.735625e-04
MPP_AT_Y_INTERCEPT = 0.08284
MPP_AT_Y_CLAMP_LO = 180.0   # video y of the red (far) reference line
MPP_AT_Y_CLAMP_HI = 411.0   # video y of the green (near) reference line


def mpp_at_y(y: float) -> float:
    """Meters per video pixel as a function of the detection's bottom-y.
    Clamped to the calibrated range so cars below/above don't extrapolate
    to nonsense."""
    yc = max(MPP_AT_Y_CLAMP_LO, min(MPP_AT_Y_CLAMP_HI, y))
    return MPP_AT_Y_SLOPE * yc + MPP_AT_Y_INTERCEPT


# ---------------------------------------------------------------------------
# Calibration utility (NOT called at runtime). Kept so the user can re-derive
# the constants above from a fresh annotated screenshot. Run by hand, copy
# the printed values back into MPP_AT_Y_* above.
# ---------------------------------------------------------------------------
REFERENCE_IMAGE = HERE / "measurements" / "redline.png"
REFERENCE_GREEN_M = 3.9
REFERENCE_RED_M = 2.5

GREEN_HSV_LO = (40, 100, 130)
GREEN_HSV_HI = (70, 255, 255)
RED_HSV_LO_1, RED_HSV_HI_1 = (0, 120, 120), (15, 255, 255)
RED_HSV_LO_2, RED_HSV_HI_2 = (160, 120, 120), (179, 255, 255)


def _fit_largest_blob(mask: "np.ndarray") -> tuple[np.ndarray, np.ndarray, float] | None:
    """Return (p1, p2, mean_y) of the largest connected blob in a binary mask,
    via PCA over its pixels. None if no usable blob.
    """
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    blob = (labels == largest).astype(np.uint8)
    ys, xs = np.where(blob > 0)
    if len(xs) < 20:
        return None
    pts = np.column_stack([xs, ys]).astype(np.float64)
    mean = pts.mean(axis=0)
    centered = pts - mean
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    # Elementwise dot rather than `@` — BLAS raises spurious FP flags on
    # some shapes (macOS Accelerate), but the result is identical.
    proj = (centered * axis).sum(axis=1)
    p1 = mean + proj.min() * axis
    p2 = mean + proj.max() * axis
    return p1, p2, float(mean[1])


def measure_from_reference_image(
    image_path: Path,
    green_meters: float,
    red_meters: float,
    video_width: int,
) -> callable:
    """Detect green + red reference lines in the image; return a callable
    `mpp(y_video)` that gives meters-per-pixel as a linear function of the
    detection's vertical position in the video. Y outside the calibrated
    range is clamped to the nearest reference depth.
    """
    if not image_path.exists():
        sys.exit(f"[speed] reference image not found: {image_path}")
    img = cv2.imread(str(image_path))
    if img is None:
        sys.exit(f"[speed] could not read reference image: {image_path}")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, GREEN_HSV_LO, GREEN_HSV_HI)
    red_mask = cv2.inRange(hsv, RED_HSV_LO_1, RED_HSV_HI_1) | cv2.inRange(
        hsv, RED_HSV_LO_2, RED_HSV_HI_2
    )

    green = _fit_largest_blob(green_mask)
    red = _fit_largest_blob(red_mask)
    if green is None:
        sys.exit(f"[speed] couldn't find a green line in {image_path.name}")

    image_width = img.shape[1]
    scale = video_width / image_width

    def to_video(line, real_m):
        p1, p2, mean_y_img = line
        px_image = float(np.linalg.norm(p2 - p1))
        px_video = px_image * scale
        return real_m / px_video, mean_y_img * scale, px_image, p1, p2

    g_mpp, g_y, g_px, g_p1, g_p2 = to_video(green, green_meters)
    print(f"[speed] image→video scale: x{scale:.4f} ({image_width}px → {video_width}px)")
    print(f"[speed] GREEN line: {g_px:.1f} px ({green_meters} m) @ video y≈{g_y:.0f} → {g_mpp:.5f} m/px")

    if red is None:
        print("[speed] no red line found — falling back to constant m/px from green")
        mpp_fn = lambda _y: g_mpp  # noqa: E731
        endpoints = [(g_p1, g_p2)]
    else:
        r_mpp, r_y, r_px, r_p1, r_p2 = to_video(red, red_meters)
        print(f"[speed] RED   line: {r_px:.1f} px ({red_meters} m) @ video y≈{r_y:.0f} → {r_mpp:.5f} m/px")
        # Linear mpp(y) = a*y + b through (g_y, g_mpp) and (r_y, r_mpp).
        if abs(g_y - r_y) < 1.0:
            sys.exit("[speed] the two reference lines are at the same depth — can't fit perspective")
        a = (g_mpp - r_mpp) / (g_y - r_y)
        b = g_mpp - a * g_y
        y_lo, y_hi = (r_y, g_y) if r_y < g_y else (g_y, r_y)
        mpp_lo = a * y_lo + b
        mpp_hi = a * y_hi + b

        def mpp_fn(y: float) -> float:
            yc = max(y_lo, min(y_hi, y))
            return a * yc + b

        print(f"[speed] mpp(y) = {a:.6e} * y + {b:.5f}, clamped to y∈[{y_lo:.0f},{y_hi:.0f}] → mpp∈[{mpp_lo:.5f},{mpp_hi:.5f}]")
        endpoints = [(g_p1, g_p2), (r_p1, r_p2)]

    debug = img.copy()
    for p1, p2 in endpoints:
        cv2.line(debug, tuple(p1.astype(int)), tuple(p2.astype(int)), (255, 255, 0), 6)
        cv2.circle(debug, tuple(p1.astype(int)), 14, (255, 255, 0), -1)
        cv2.circle(debug, tuple(p2.astype(int)), 14, (255, 255, 0), -1)
    dbg_path = image_path.parent / f"{image_path.stem}_detected.png"
    cv2.imwrite(str(dbg_path), debug)
    print(f"[speed] sanity-check overlay: {dbg_path}")
    return mpp_fn


def run(
    video_path: Path,
    model_name: str,
    out_dir: Path,
) -> None:
    if not video_path.exists():
        sys.exit(f"[speed] video not found: {video_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"[speed] could not open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    print(f"[speed] using frozen calibration: mpp(y) = {MPP_AT_Y_SLOPE:.4e}*y + {MPP_AT_Y_INTERCEPT:.5f}, y∈[{MPP_AT_Y_CLAMP_LO:.0f},{MPP_AT_Y_CLAMP_HI:.0f}]")

    model, used_model = load_model(model_name)

    name_to_id = {n.lower(): i for i, n in model.names.items()}
    allowed_ids = [name_to_id[n.lower()] for n in ALLOWED_CLASS_NAMES if n.lower() in name_to_id]
    if not allowed_ids:
        sys.exit("[speed] none of the requested classes exist in this model")
    print(f"[speed] keeping classes: {[model.names[i] for i in allowed_ids]}")

    inference_conf = min(CONF_THRESHOLD, *PER_CLASS_CONF.values())

    video_out_path = out_dir / "annotated_speed.mp4"
    writer = cv2.VideoWriter(
        str(video_out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    csv_path = out_dir / "speeds.csv"
    csv_file = csv_path.open("w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "track_id", "class_name", "x", "y", "speed_kmh"])

    # Per-track ring of (frame_idx, x_ground, y_ground) used for speed calc.
    history: dict[int, deque] = {}
    # Per-track ring of recent raw speeds, for median smoothing.
    speed_hist: dict[int, deque] = {}
    # Per-track persistent state for carry-forward and hysteresis.
    # Keys: 'bbox', 'class', 'speed', 'stopped', 'last_frame'.
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

    def draw_box(frame, tid, cname, bbox, speed_kmh, ghost):
        x1, y1, x2, y2 = bbox
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        # Ghost (carry-forward) boxes are drawn slightly dimmer + dashed-ish.
        color = (120, 200, 120) if ghost else (0, 255, 0)
        thickness = 1 if ghost else 2
        cv2.rectangle(frame, p1, p2, color, thickness)
        label = f"#{int(tid)} {cname} {speed_kmh:.0f} km/h"
        cv2.putText(
            frame, label, (p1[0], max(12, p1[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
        )

    frame_idx = 0
    try:
        for r in results_iter:
            frame = r.orig_img.copy()
            names = r.names
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

                    # Bottom-center = where the vehicle meets the road. Most
                    # linear under a simple uniform-scale assumption.
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

                    # Hysteresis: once "stopped", stay stopped until we exceed
                    # the higher EXIT threshold. Prevents 1↔0 flapping.
                    prev = last_state.get(int(tid), {})
                    was_stopped = prev.get("stopped", False)
                    if was_stopped:
                        is_stopped = smoothed < STOPPED_EXIT_KMH
                    else:
                        is_stopped = smoothed < STOPPED_ENTER_KMH
                    speed_kmh = 0.0 if is_stopped else smoothed

                    last_state[int(tid)] = {
                        "bbox": (float(x1), float(y1), float(x2), float(y2)),
                        "class": cname,
                        "speed": speed_kmh,
                        "stopped": is_stopped,
                        "last_frame": frame_idx,
                    }
                    detected_this_frame.add(int(tid))

                    csv_writer.writerow(
                        [frame_idx, int(tid), cname, f"{gx:.1f}", f"{gy:.1f}", f"{speed_kmh:.1f}"]
                    )
                    draw_box(frame, tid, cname, (x1, y1, x2, y2), speed_kmh, ghost=False)

            # Carry-forward: tracks not detected this frame but still within
            # the grace period get re-emitted using their last known state.
            for tid, st in list(last_state.items()):
                if tid in detected_this_frame:
                    continue
                if frame_idx - st["last_frame"] > DETECTION_GRACE_FRAMES:
                    del last_state[tid]
                    continue
                bx1, by1, bx2, by2 = st["bbox"]
                gx = (bx1 + bx2) / 2
                gy = by2
                csv_writer.writerow(
                    [frame_idx, int(tid), st["class"], f"{gx:.1f}", f"{gy:.1f}", f"{st['speed']:.1f}"]
                )
                draw_box(frame, tid, st["class"], st["bbox"], st["speed"], ghost=True)

            writer.write(frame)
            frame_idx += 1
    finally:
        writer.release()
        csv_file.close()

    print(f"[speed] done. model used: {used_model}")
    print(f"[speed] frames processed: {frame_idx}")
    print(f"[speed] annotated video : {video_out_path}")
    print(f"[speed] speeds CSV      : {csv_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Track vehicles and estimate speed")
    p.add_argument("--model", default=MODEL_NAME, help="Model name or path (.pt)")
    p.add_argument("--video", default=str(DEFAULT_VIDEO), help="Path to input video")
    p.add_argument("--out", default=str(HERE / "runs" / "tracked"), help="Output dir")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(Path(args.video), args.model, Path(args.out))
