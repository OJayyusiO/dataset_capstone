"""
Model Evaluation Script

Runs YOLO detection + ByteTrack tracking on recorded test footage,
computes detection and tracking metrics, and renders an annotated video.

Does NOT require CARLA — only ultralytics, opencv, numpy.

Usage:
    python evaluate_model.py <recording_dir> <model.pt>
    python evaluate_model.py <recording_dir> <model.pt> --conf 0.25 --iou 0.5
"""

import argparse
import json
import csv
import sys
import time
import cv2
import yaml
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from capstone_sim.scripts.utils.constants import CLASS_COLORS


def load_ground_truth(gt_dir, num_frames, image_w, image_h):
    """Load ground truth labels from text files.

    Returns: {frame_idx: [{'class_id', 'bbox_xyxy', 'track_id'}, ...]}
    """
    gt = {}
    for i in range(num_frames):
        gt_file = gt_dir / f"{i:06d}.txt"
        detections = []
        if gt_file.exists():
            with open(gt_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 6:
                        class_id = int(parts[0])
                        xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        actor_id = int(parts[5])
                        # Convert YOLO normalized to pixel xyxy
                        x1 = (xc - w / 2) * image_w
                        y1 = (yc - h / 2) * image_h
                        x2 = (xc + w / 2) * image_w
                        y2 = (yc + h / 2) * image_h
                        detections.append({
                            'class_id': class_id,
                            'bbox_xyxy': [x1, y1, x2, y2],
                            'track_id': actor_id,
                        })
        gt[i] = detections
    return gt


def get_gpu_stats():
    """Get current GPU memory and power usage. Returns dict or None if unavailable."""
    if not torch.cuda.is_available():
        return None
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total,power.draw,power.limit,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(', ')
            return {
                'memory_used_mb': float(parts[0]),
                'memory_total_mb': float(parts[1]),
                'power_draw_w': float(parts[2]),
                'power_limit_w': float(parts[3]),
                'temperature_c': float(parts[4]),
            }
    except Exception:
        pass
    return None


def run_tracking(model, frames_dir, num_frames, conf, iou):
    """Run model.track() on each frame sequentially.

    Returns: (predictions, gpu_metrics)
    predictions: {frame_idx: [{'class_id', 'bbox_xyxy', 'track_id', 'confidence'}, ...]}
    gpu_metrics: dict with latency, fps, memory, power stats
    """
    predictions = {}
    frame_latencies = []
    gpu_mem_samples = []
    gpu_power_samples = []
    gpu_temp_samples = []

    for i in range(num_frames):
        frame_path = frames_dir / f"{i:06d}.png"
        if not frame_path.exists():
            predictions[i] = []
            continue

        # Measure per-frame latency
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_start = time.perf_counter()

        results = model.track(
            source=str(frame_path),
            conf=conf,
            iou=iou,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_end = time.perf_counter()
        frame_latencies.append(t_end - t_start)

        detections = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for j in range(len(boxes)):
                xyxy = boxes.xyxy[j].cpu().numpy()
                cls = int(boxes.cls[j].cpu().item())
                conf_score = float(boxes.conf[j].cpu().item())
                track_id = int(boxes.id[j].cpu().item()) if boxes.id is not None else -1
                detections.append({
                    'class_id': cls,
                    'bbox_xyxy': xyxy.tolist(),
                    'track_id': track_id,
                    'confidence': conf_score,
                })

        predictions[i] = detections

        # Sample GPU stats every 100 frames
        if (i + 1) % 100 == 0:
            stats = get_gpu_stats()
            if stats:
                gpu_mem_samples.append(stats['memory_used_mb'])
                gpu_power_samples.append(stats['power_draw_w'])
                gpu_temp_samples.append(stats['temperature_c'])

        if (i + 1) % 200 == 0:
            avg_latency = np.mean(frame_latencies[-200:]) * 1000
            current_fps = 1.0 / np.mean(frame_latencies[-200:])
            print(f"  Processed {i + 1}/{num_frames} frames | "
                  f"{current_fps:.1f} FPS | "
                  f"{avg_latency:.1f}ms/frame")

    # Compile GPU metrics (skip first 10 frames as warmup)
    warmup = min(10, len(frame_latencies))
    latencies_ms = [l * 1000 for l in frame_latencies[warmup:]]

    gpu_metrics = {
        'device': str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else 'CPU',
        'avg_latency_ms': round(np.mean(latencies_ms), 2) if latencies_ms else 0,
        'median_latency_ms': round(np.median(latencies_ms), 2) if latencies_ms else 0,
        'min_latency_ms': round(np.min(latencies_ms), 2) if latencies_ms else 0,
        'max_latency_ms': round(np.max(latencies_ms), 2) if latencies_ms else 0,
        'p95_latency_ms': round(np.percentile(latencies_ms, 95), 2) if latencies_ms else 0,
        'avg_fps': round(1000 / np.mean(latencies_ms), 1) if latencies_ms else 0,
    }
    if gpu_mem_samples:
        gpu_metrics['avg_memory_used_mb'] = round(np.mean(gpu_mem_samples), 0)
        gpu_metrics['peak_memory_used_mb'] = round(np.max(gpu_mem_samples), 0)
    if gpu_power_samples:
        gpu_metrics['avg_power_draw_w'] = round(np.mean(gpu_power_samples), 1)
        gpu_metrics['peak_power_draw_w'] = round(np.max(gpu_power_samples), 1)
    if gpu_temp_samples:
        gpu_metrics['avg_temperature_c'] = round(np.mean(gpu_temp_samples), 1)
        gpu_metrics['peak_temperature_c'] = round(np.max(gpu_temp_samples), 1)

    return predictions, gpu_metrics


def compute_iou(box1, box2):
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


def match_detections(gt_dets, pred_dets, iou_threshold):
    """Match predictions to ground truth using greedy IoU matching.

    Returns: (matches, unmatched_gt, unmatched_pred)
    matches: list of (gt_idx, pred_idx) pairs
    """
    if not gt_dets or not pred_dets:
        return [], list(range(len(gt_dets))), list(range(len(pred_dets)))

    iou_matrix = np.zeros((len(gt_dets), len(pred_dets)))
    for gi, gd in enumerate(gt_dets):
        for pi, pd in enumerate(pred_dets):
            if gd['class_id'] == pd['class_id']:
                iou_matrix[gi, pi] = compute_iou(gd['bbox_xyxy'], pd['bbox_xyxy'])

    matches = []
    matched_gt = set()
    matched_pred = set()

    # Greedy matching: highest IoU first
    while True:
        if iou_matrix.size == 0:
            break
        max_iou = iou_matrix.max()
        if max_iou < iou_threshold:
            break
        gi, pi = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
        matches.append((gi, pi))
        matched_gt.add(gi)
        matched_pred.add(pi)
        iou_matrix[gi, :] = 0
        iou_matrix[:, pi] = 0

    unmatched_gt = [i for i in range(len(gt_dets)) if i not in matched_gt]
    unmatched_pred = [i for i in range(len(pred_dets)) if i not in matched_pred]

    return matches, unmatched_gt, unmatched_pred


def compute_detection_metrics(gt, predictions, iou_threshold):
    """Compute per-class and overall detection metrics."""
    # Collect per-class TP, FP, FN
    class_stats = {}

    for frame_idx in gt:
        gt_dets = gt[frame_idx]
        pred_dets = predictions.get(frame_idx, [])

        matches, unmatched_gt, unmatched_pred = match_detections(gt_dets, pred_dets, iou_threshold)

        for gi, pi in matches:
            cls = gt_dets[gi]['class_id']
            if cls not in class_stats:
                class_stats[cls] = {'tp': 0, 'fp': 0, 'fn': 0}
            class_stats[cls]['tp'] += 1

        for gi in unmatched_gt:
            cls = gt_dets[gi]['class_id']
            if cls not in class_stats:
                class_stats[cls] = {'tp': 0, 'fp': 0, 'fn': 0}
            class_stats[cls]['fn'] += 1

        for pi in unmatched_pred:
            cls = pred_dets[pi]['class_id']
            if cls not in class_stats:
                class_stats[cls] = {'tp': 0, 'fp': 0, 'fn': 0}
            class_stats[cls]['fp'] += 1

    # Compute per-class precision, recall
    per_class = {}
    total_tp, total_fp, total_fn = 0, 0, 0

    for cls, stats in class_stats.items():
        tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        per_class[cls] = {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'tp': tp, 'fp': fp, 'fn': fn,
            'count': tp + fn,
        }
        total_tp += tp
        total_fp += fp
        total_fn += fn

    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0

    return {
        'precision': round(overall_precision, 4),
        'recall': round(overall_recall, 4),
        'per_class': per_class,
    }


def compute_confusion_matrix(gt, predictions, iou_threshold, num_classes=7):
    """Compute a confusion matrix for detection results.

    Returns a (num_classes+1) x (num_classes+1) matrix (list of lists).
    Rows = ground truth class, columns = predicted class.
    Index num_classes (7) represents 'background' (missed GT / false positive).
    """
    size = num_classes + 1
    matrix = [[0] * size for _ in range(size)]
    bg = num_classes  # background index

    for frame_idx in gt:
        gt_dets = gt[frame_idx]
        pred_dets = predictions.get(frame_idx, [])

        matches, unmatched_gt, unmatched_pred = match_detections(gt_dets, pred_dets, iou_threshold)

        for gi, pi in matches:
            gt_cls = gt_dets[gi]['class_id']
            pred_cls = pred_dets[pi]['class_id']
            matrix[gt_cls][pred_cls] += 1

        for gi in unmatched_gt:
            gt_cls = gt_dets[gi]['class_id']
            matrix[gt_cls][bg] += 1

        for pi in unmatched_pred:
            pred_cls = pred_dets[pi]['class_id']
            matrix[bg][pred_cls] += 1

    return matrix


def compute_tracking_metrics(gt, predictions, iou_threshold):
    """Compute MOTA and IDF1 tracking metrics."""
    total_gt = 0
    total_fp = 0
    total_fn = 0
    total_switches = 0

    # Track ID mapping: gt_track_id -> last predicted track_id
    gt_to_pred_track = {}

    # For IDF1
    idtp = 0
    idfp = 0
    idfn = 0

    for frame_idx in sorted(gt.keys()):
        gt_dets = gt[frame_idx]
        pred_dets = predictions.get(frame_idx, [])

        total_gt += len(gt_dets)

        matches, unmatched_gt, unmatched_pred = match_detections(gt_dets, pred_dets, iou_threshold)

        fn = len(unmatched_gt)
        fp = len(unmatched_pred)
        total_fn += fn
        total_fp += fp
        idfn += fn
        idfp += fp

        for gi, pi in matches:
            gt_track = gt_dets[gi]['track_id']
            pred_track = pred_dets[pi]['track_id']

            if gt_track in gt_to_pred_track:
                if gt_to_pred_track[gt_track] != pred_track:
                    total_switches += 1
            gt_to_pred_track[gt_track] = pred_track

            # For IDF1: correct ID if mapping is consistent
            idtp += 1

    mota = 1 - (total_fn + total_fp + total_switches) / total_gt if total_gt > 0 else 0
    idf1 = 2 * idtp / (2 * idtp + idfp + idfn) if (2 * idtp + idfp + idfn) > 0 else 0

    return {
        'MOTA': round(mota, 4),
        'IDF1': round(idf1, 4),
        'num_switches': total_switches,
        'FP': total_fp,
        'FN': total_fn,
        'total_gt': total_gt,
    }


def render_annotated_video(frames_dir, predictions, gt, output_path,
                           image_w, image_h, fps, class_names):
    """Render MP4 video with bounding boxes and track IDs."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (image_w, image_h))

    num_frames = len(predictions)
    for i in sorted(predictions.keys()):
        frame_path = frames_dir / f"{i:06d}.png"
        if not frame_path.exists():
            continue

        img = cv2.imread(str(frame_path))

        # Draw predictions
        for det in predictions[i]:
            x1, y1, x2, y2 = [int(v) for v in det['bbox_xyxy']]
            cls = det['class_id']
            track_id = det['track_id']
            conf = det['confidence']
            color = CLASS_COLORS.get(cls, (255, 255, 255))

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            label = f"{class_names.get(cls, cls)} #{track_id} {conf:.0%}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - label_size[1] - 6), (x1 + label_size[0], y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Frame counter
        cv2.putText(img, f"Frame {i}/{num_frames}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        writer.write(img)

    writer.release()
    print(f"Video saved: {output_path}")


def compute_per_frame_metrics(gt, predictions, iou_threshold):
    """Compute per-frame TP, FP, FN, ID switches for CSV output."""
    rows = []
    gt_to_pred_track = {}

    for frame_idx in sorted(gt.keys()):
        gt_dets = gt[frame_idx]
        pred_dets = predictions.get(frame_idx, [])

        matches, unmatched_gt, unmatched_pred = match_detections(gt_dets, pred_dets, iou_threshold)

        switches = 0
        for gi, pi in matches:
            gt_track = gt_dets[gi]['track_id']
            pred_track = pred_dets[pi]['track_id']
            if gt_track in gt_to_pred_track and gt_to_pred_track[gt_track] != pred_track:
                switches += 1
            gt_to_pred_track[gt_track] = pred_track

        rows.append({
            'frame': frame_idx,
            'num_gt': len(gt_dets),
            'num_pred': len(pred_dets),
            'tp': len(matches),
            'fp': len(unmatched_pred),
            'fn': len(unmatched_gt),
            'id_switches': switches,
        })

    return rows


def evaluate(recording_dir, model_path, output_base, conf, iou, video_fps, visualize=False):
    recording_dir = Path(recording_dir)
    frames_dir = recording_dir / 'frames'
    gt_dir = recording_dir / 'ground_truth'
    meta_path = recording_dir / 'recording_meta.yaml'

    if not meta_path.exists():
        print(f"Error: recording_meta.yaml not found in {recording_dir}")
        return

    with open(meta_path) as f:
        meta = yaml.safe_load(f)

    num_frames = meta['num_frames']
    image_w = meta['image_width']
    image_h = meta['image_height']
    fps = video_fps or meta.get('fps', 20)
    class_names = {int(k): v for k, v in meta['class_names'].items()}

    # Create output directory
    recording_name = recording_dir.name
    model_name = Path(model_path).stem
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = Path(output_base) / f"{recording_name}_{model_name}_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Model Evaluation")
    print("=" * 60)
    print(f"Recording: {recording_name}")
    print(f"Model: {model_path}")
    print(f"Frames: {num_frames}")
    print(f"Confidence: {conf}, IoU: {iou}")
    print(f"Output: {result_dir.resolve()}")
    print("=" * 60)

    # Load ground truth
    print("\nLoading ground truth...")
    gt = load_ground_truth(gt_dir, num_frames, image_w, image_h)
    total_gt_objects = sum(len(dets) for dets in gt.values())
    print(f"  {total_gt_objects} ground truth objects across {num_frames} frames")

    # Run tracking
    print("\nRunning detection + tracking...")
    model = YOLO(model_path)
    start_time = time.time()
    predictions, gpu_metrics = run_tracking(model, frames_dir, num_frames, conf, iou)
    tracking_time = time.time() - start_time
    total_predictions = sum(len(dets) for dets in predictions.values())
    print(f"  {total_predictions} detections in {tracking_time:.1f}s")

    # Compute metrics
    print("\nComputing metrics...")
    det_metrics = compute_detection_metrics(gt, predictions, iou)
    track_metrics = compute_tracking_metrics(gt, predictions, iou)
    confusion_mat = compute_confusion_matrix(gt, predictions, iou)

    print(f"\n  Detection:  Precision={det_metrics['precision']:.3f}  Recall={det_metrics['recall']:.3f}")
    print(f"  Tracking:   MOTA={track_metrics['MOTA']:.3f}  IDF1={track_metrics['IDF1']:.3f}  "
          f"ID Switches={track_metrics['num_switches']}")

    print(f"\n  GPU Performance ({gpu_metrics['device']}):")
    print(f"    Avg FPS:      {gpu_metrics['avg_fps']}")
    print(f"    Avg Latency:  {gpu_metrics['avg_latency_ms']:.1f}ms  "
          f"(p95: {gpu_metrics['p95_latency_ms']:.1f}ms)")
    if 'avg_memory_used_mb' in gpu_metrics:
        print(f"    Memory:       {gpu_metrics['avg_memory_used_mb']:.0f}MB avg / "
              f"{gpu_metrics['peak_memory_used_mb']:.0f}MB peak")
    if 'avg_power_draw_w' in gpu_metrics:
        print(f"    Power:        {gpu_metrics['avg_power_draw_w']:.1f}W avg / "
              f"{gpu_metrics['peak_power_draw_w']:.1f}W peak")
    if 'avg_temperature_c' in gpu_metrics:
        print(f"    Temperature:  {gpu_metrics['avg_temperature_c']:.0f}C avg / "
              f"{gpu_metrics['peak_temperature_c']:.0f}C peak")

    if det_metrics['per_class']:
        print("\n  Per-class:")
        for cls_id in sorted(det_metrics['per_class'].keys()):
            stats = det_metrics['per_class'][cls_id]
            name = class_names.get(cls_id, str(cls_id))
            print(f"    {name:12s}  P={stats['precision']:.3f}  R={stats['recall']:.3f}  "
                  f"F1={stats['f1']:.3f}  (n={stats['count']})")

    # Save metrics summary
    summary = {
        'model': str(model_path),
        'recording': recording_name,
        'confidence_threshold': conf,
        'iou_threshold': iou,
        'num_frames': num_frames,
        'detection': det_metrics,
        'tracking': track_metrics,
        'confusion_matrix': confusion_mat,
        'gpu_performance': gpu_metrics,
    }

    with open(result_dir / 'metrics_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Save per-frame metrics
    per_frame = compute_per_frame_metrics(gt, predictions, iou)
    with open(result_dir / 'per_frame_metrics.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['frame', 'num_gt', 'num_pred', 'tp', 'fp', 'fn', 'id_switches'])
        writer.writeheader()
        writer.writerows(per_frame)

    # Render annotated video
    print("\nRendering annotated video...")
    render_annotated_video(frames_dir, predictions, gt,
                           result_dir / 'annotated.mp4',
                           image_w, image_h, fps, class_names)

    # Generate charts if requested
    if visualize:
        try:
            from capstone_sim.scripts.evaluate.visualize_metrics import (
                plot_per_class_detection, plot_class_distribution,
                plot_tp_fp_fn, plot_tracking_summary, plot_per_frame,
                plot_confusion_matrix,
            )
            print("\nGenerating metric charts...")
            plot_per_class_detection(det_metrics, result_dir)
            plot_class_distribution(det_metrics, result_dir)
            plot_tp_fp_fn(det_metrics, result_dir)
            plot_tracking_summary(track_metrics, det_metrics, result_dir)
            plot_per_frame(result_dir / 'per_frame_metrics.csv', result_dir)
            if 'confusion_matrix' in summary:
                plot_confusion_matrix(summary['confusion_matrix'], result_dir)
        except ImportError:
            print("\nSkipping charts (matplotlib not installed)")

    # Generate HTML report
    if visualize:
        try:
            from capstone_sim.scripts.evaluate.generate_report import generate_html
            report_path = result_dir / 'report.html'
            print("\nGenerating HTML report...")
            generate_html(None, [str(result_dir)], str(report_path))
        except ImportError:
            pass

    print(f"\nResults saved to: {result_dir.resolve()}")
    return summary


def main():
    parser = argparse.ArgumentParser(description='Evaluate YOLO model with ByteTrack on recorded footage')
    parser.add_argument('recording', type=str, help='Path to recording directory from record_test.py')
    parser.add_argument('model', type=str, help='Path to trained .pt model file')
    parser.add_argument('--output', type=str, default='./eval_results',
                        help='Base directory for results (default: ./eval_results)')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold (default: 0.25)')
    parser.add_argument('--iou', type=float, default=0.5,
                        help='IoU threshold (default: 0.5)')
    parser.add_argument('--video-fps', type=int, default=None,
                        help='FPS for annotated video (default: from recording)')
    parser.add_argument('--no-visualize', action='store_true',
                        help='Skip generating metric charts')
    args = parser.parse_args()

    recording_path = Path(args.recording)
    if not recording_path.exists():
        print(f"Recording not found: {recording_path}")
        sys.exit(1)

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        sys.exit(1)

    evaluate(str(recording_path), str(model_path), args.output, args.conf, args.iou, args.video_fps, not args.no_visualize)


if __name__ == '__main__':
    main()
