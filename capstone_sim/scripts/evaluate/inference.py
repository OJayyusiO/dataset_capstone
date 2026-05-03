"""
Real-World Inference Script

Runs YOLO + ByteTrack on a video file (or webcam/stream).
Produces an annotated output video with bounding boxes and track IDs.

Usage:
    python inference.py --model best.pt --source video.mp4
    python inference.py --model best.pt --source video.mp4 --output out.mp4
    python inference.py --model best.pt --source 0  # webcam
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from capstone_sim.scripts.utils.constants import CLASS_NAMES, CLASS_COLORS


def run_inference(model_path, source, output_path, conf, iou, show, line_width):
    model = YOLO(model_path)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Open source
    source_str = str(source)
    is_webcam = source_str.isdigit()
    cap = cv2.VideoCapture(int(source_str) if is_webcam else source_str)
    if not cap.isOpened():
        print(f"Error: Could not open {source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else 0

    print("=" * 60)
    print("Real-World Inference")
    print("=" * 60)
    print(f"Model:       {model_path}")
    print(f"Source:      {source}")
    print(f"Resolution:  {width}x{height} @ {fps:.1f} FPS")
    print(f"Frames:      {total_frames if total_frames > 0 else 'live stream'}")
    print(f"Output:      {output_path}")
    print(f"Device:      {device}")
    print(f"Conf/IoU:    {conf}/{iou}")
    print("=" * 60)
    print("\nProcessing... (press 'q' in window to stop)\n")

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_count = 0
    detection_count = 0
    track_ids_seen = set()
    start_time = time.time()
    frame_latencies = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run tracking
            t_start = time.perf_counter()
            results = model.track(
                source=frame,
                conf=conf,
                iou=iou,
                persist=True,
                tracker='bytetrack.yaml',
                verbose=False,
                device=device,
            )
            t_end = time.perf_counter()
            frame_latencies.append(t_end - t_start)

            # Draw detections
            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                detection_count += len(boxes)
                for j in range(len(boxes)):
                    xyxy = boxes.xyxy[j].cpu().numpy()
                    cls = int(boxes.cls[j].cpu().item())
                    conf_score = float(boxes.conf[j].cpu().item())
                    track_id = int(boxes.id[j].cpu().item()) if boxes.id is not None else -1

                    if track_id != -1:
                        track_ids_seen.add(track_id)

                    color = CLASS_COLORS.get(cls, (255, 255, 255))
                    x1, y1, x2, y2 = [int(v) for v in xyxy]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_width)

                    label = f"{CLASS_NAMES.get(cls, cls)} #{track_id} {conf_score:.0%}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (x1, y1 - label_size[1] - 6),
                                  (x1 + label_size[0], y1), color, -1)
                    cv2.putText(frame, label, (x1, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Stats overlay
            elapsed = time.time() - start_time
            current_fps = (frame_count + 1) / max(elapsed, 0.001)
            stats = f"Frame {frame_count} | {current_fps:.1f} FPS | Tracks: {len(track_ids_seen)}"
            cv2.putText(frame, stats, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)

            writer.write(frame)
            frame_count += 1

            if show:
                cv2.imshow('Inference', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Progress every 100 frames
            if frame_count % 100 == 0:
                avg_latency = np.mean(frame_latencies[-100:]) * 1000
                print(f"  Frame {frame_count}/{total_frames if total_frames else '?'} | "
                      f"{current_fps:.1f} FPS | "
                      f"Latency: {avg_latency:.1f}ms | "
                      f"Tracks: {len(track_ids_seen)}")

    except KeyboardInterrupt:
        print("\nStopped by user")

    finally:
        cap.release()
        writer.release()
        if show:
            cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    avg_latency_ms = np.mean(frame_latencies) * 1000 if frame_latencies else 0

    print("\n" + "=" * 60)
    print(f"Inference complete")
    print(f"  Frames processed:   {frame_count}")
    print(f"  Total detections:   {detection_count}")
    print(f"  Unique track IDs:   {len(track_ids_seen)}")
    print(f"  Avg FPS:            {frame_count / max(elapsed, 0.001):.1f}")
    print(f"  Avg latency:        {avg_latency_ms:.1f}ms")
    print(f"  Time elapsed:       {elapsed:.1f}s")
    print(f"  Output saved:       {output_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Run YOLO + ByteTrack on a video file')
    parser.add_argument('--model', type=str, required=True, help='Path to .pt model file')
    parser.add_argument('--source', type=str, required=True,
                        help='Path to video file, or "0" for webcam')
    parser.add_argument('--output', type=str, default=None,
                        help='Output video path (default: <source>_tracked.mp4)')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.5, help='IoU threshold')
    parser.add_argument('--show', action='store_true', help='Show live preview window')
    parser.add_argument('--line-width', type=int, default=2, help='Bounding box line width')
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        sys.exit(1)

    if args.source.isdigit():
        source = args.source
    else:
        source_path = Path(args.source)
        if not source_path.exists():
            print(f"Source not found: {source_path}")
            sys.exit(1)
        source = str(source_path)

    if args.output:
        output_path = Path(args.output)
    else:
        if args.source.isdigit():
            output_path = Path(f"webcam_tracked.mp4")
        else:
            src = Path(args.source)
            output_path = src.parent / f"{src.stem}_tracked.mp4"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_inference(str(model_path), source, output_path, args.conf, args.iou,
                  args.show, args.line_width)


if __name__ == '__main__':
    main()
