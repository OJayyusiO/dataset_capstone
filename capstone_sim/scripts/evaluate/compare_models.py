"""
Model Comparison Tool

Compares multiple evaluation results on the same recording.
Generates a side-by-side annotated video and a speed/accuracy tradeoff chart.

Usage:
    python compare_models.py eval_result_dir1 eval_result_dir2 [eval_result_dir3 ...]
    python compare_models.py eval_results/  # compares all results in folder
"""

import argparse
import json
import sys
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from capstone_sim.scripts.utils.constants import CLASS_NAMES, CLASS_COLORS

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def load_eval_results(result_dirs):
    """Load metrics from multiple evaluation result directories."""
    results = []
    for d in result_dirs:
        d = Path(d)
        meta_path = d / 'metrics_summary.json'
        if not meta_path.exists():
            print(f"WARNING: Skipping {d.name} (no metrics_summary.json)")
            continue
        with open(meta_path) as f:
            data = json.load(f)
        data['_dir'] = d
        data['_name'] = Path(data.get('model', d.name)).stem
        results.append(data)
    return results


def render_comparison_video(results, output_path, fps=20):
    """Render side-by-side video comparing model detections."""
    # Find the recording frames directory
    # All results should be from the same recording
    recording_name = results[0].get('recording', '')
    num_frames = min(r['num_frames'] for r in results)
    num_models = len(results)

    # Load annotated videos
    video_paths = [r['_dir'] / 'annotated.mp4' for r in results]
    for vp in video_paths:
        if not vp.exists():
            print(f"ERROR: {vp} not found. Run evaluate_model.py first.")
            return

    caps = [cv2.VideoCapture(str(vp)) for vp in video_paths]

    # Get frame dimensions from first video
    frame_w = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Layout: stack horizontally, max 3 per row
    cols = min(num_models, 3)
    rows = (num_models + cols - 1) // cols

    # Scale down each panel to fit
    panel_w = min(frame_w, 640)
    scale = panel_w / frame_w
    panel_h = int(frame_h * scale)
    label_h = 30

    canvas_w = panel_w * cols
    canvas_h = (panel_h + label_h) * rows

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (canvas_w, canvas_h))

    frame_count = int(min(cap.get(cv2.CAP_PROP_FRAME_COUNT) for cap in caps))

    for i in range(frame_count):
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        for m, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                break

            # Resize
            panel = cv2.resize(frame, (panel_w, panel_h))

            # Position on canvas
            row = m // cols
            col = m % cols
            y_off = row * (panel_h + label_h)
            x_off = col * panel_w

            canvas[y_off:y_off + panel_h, x_off:x_off + panel_w] = panel

            # Model label
            label_y = y_off + panel_h
            cv2.rectangle(canvas, (x_off, label_y), (x_off + panel_w, label_y + label_h), (40, 40, 40), -1)
            model_name = results[m]['_name']
            mota = results[m].get('tracking', {}).get('MOTA', 0)
            prec = results[m].get('detection', {}).get('precision', 0)
            label = f"{model_name} | P={prec:.2f} MOTA={mota:.2f}"
            cv2.putText(canvas, label, (x_off + 5, label_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        writer.write(canvas)

        if (i + 1) % 500 == 0:
            print(f"  Rendered {i + 1}/{frame_count} frames")

    writer.release()
    for cap in caps:
        cap.release()
    print(f"Comparison video saved: {output_path}")


def plot_speed_accuracy(results, output_dir):
    """Plot speed vs accuracy tradeoff chart."""
    if plt is None:
        print("  Skipping speed/accuracy chart (matplotlib not installed)")
        return

    names = []
    fps_vals = []
    mota_vals = []
    precision_vals = []
    recall_vals = []

    for r in results:
        names.append(r['_name'])
        gpu = r.get('gpu_performance', {})
        fps_vals.append(gpu.get('avg_fps', r.get('inference_fps', 0)))
        mota_vals.append(r.get('tracking', {}).get('MOTA', 0))
        precision_vals.append(r.get('detection', {}).get('precision', 0))
        recall_vals.append(r.get('detection', {}).get('recall', 0))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = plt.cm.Set1(np.linspace(0, 1, len(names)))

    # FPS vs MOTA
    ax = axes[0]
    for i, name in enumerate(names):
        ax.scatter(fps_vals[i], mota_vals[i], s=150, c=[colors[i]], zorder=5)
        ax.annotate(name, (fps_vals[i], mota_vals[i]), textcoords='offset points',
                    xytext=(8, 8), fontsize=10, fontweight='bold')
    ax.set_xlabel('FPS (higher = faster)')
    ax.set_ylabel('MOTA (higher = better)')
    ax.set_title('Speed vs Tracking Accuracy')
    ax.grid(alpha=0.3)

    # FPS vs Precision
    ax = axes[1]
    for i, name in enumerate(names):
        ax.scatter(fps_vals[i], precision_vals[i], s=150, c=[colors[i]], zorder=5)
        ax.annotate(name, (fps_vals[i], precision_vals[i]), textcoords='offset points',
                    xytext=(8, 8), fontsize=10, fontweight='bold')
    ax.set_xlabel('FPS (higher = faster)')
    ax.set_ylabel('Precision (higher = better)')
    ax.set_title('Speed vs Precision')
    ax.grid(alpha=0.3)

    # FPS vs Recall
    ax = axes[2]
    for i, name in enumerate(names):
        ax.scatter(fps_vals[i], recall_vals[i], s=150, c=[colors[i]], zorder=5)
        ax.annotate(name, (fps_vals[i], recall_vals[i]), textcoords='offset points',
                    xytext=(8, 8), fontsize=10, fontweight='bold')
    ax.set_xlabel('FPS (higher = faster)')
    ax.set_ylabel('Recall (higher = better)')
    ax.set_title('Speed vs Recall')
    ax.grid(alpha=0.3)

    plt.suptitle('Speed / Accuracy Tradeoff', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'speed_accuracy_tradeoff.png', dpi=150)
    plt.close()
    print(f"  Saved speed_accuracy_tradeoff.png")


def plot_model_comparison_bars(results, output_dir):
    """Bar chart comparing all models across key metrics."""
    if plt is None:
        return

    names = [r['_name'] for r in results]
    metrics = {
        'Precision': [r.get('detection', {}).get('precision', 0) for r in results],
        'Recall': [r.get('detection', {}).get('recall', 0) for r in results],
        'MOTA': [r.get('tracking', {}).get('MOTA', 0) for r in results],
        'IDF1': [r.get('tracking', {}).get('IDF1', 0) for r in results],
    }

    x = np.arange(len(names))
    width = 0.8 / len(metrics)
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (metric_name, values) in enumerate(metrics.items()):
        bars = ax.bar(x + i * width, values, width, label=metric_name, color=colors[i])
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{val:.2f}', ha='center', fontsize=8)

    ax.set_ylabel('Score')
    ax.set_title('Model Comparison')
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(names)
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=150)
    plt.close()
    print(f"  Saved model_comparison.png")


def main():
    parser = argparse.ArgumentParser(description='Compare multiple model evaluation results')
    parser.add_argument('paths', nargs='+', help='Evaluation result directories or a parent folder')
    parser.add_argument('--output', type=str, default='./comparison_results',
                        help='Output directory (default: ./comparison_results)')
    parser.add_argument('--no-video', action='store_true',
                        help='Skip comparison video generation')
    parser.add_argument('--fps', type=int, default=20,
                        help='FPS for comparison video (default: 20)')
    args = parser.parse_args()

    # Collect result directories
    result_dirs = []
    for p in args.paths:
        p = Path(p)
        if p.is_dir() and (p / 'metrics_summary.json').exists():
            result_dirs.append(p)
        elif p.is_dir():
            # Look for subdirectories with metrics
            for sub in sorted(p.iterdir()):
                if sub.is_dir() and (sub / 'metrics_summary.json').exists():
                    result_dirs.append(sub)

    if len(result_dirs) < 2:
        print(f"Need at least 2 evaluation results to compare. Found {len(result_dirs)}.")
        sys.exit(1)

    results = load_eval_results(result_dirs)
    if len(results) < 2:
        print("Not enough valid results to compare.")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Model Comparison")
    print("=" * 60)
    print(f"Comparing {len(results)} models:")
    for r in results:
        gpu = r.get('gpu_performance', {})
        print(f"  {r['_name']:20s} | P={r['detection']['precision']:.3f} "
              f"R={r['detection']['recall']:.3f} "
              f"MOTA={r['tracking']['MOTA']:.3f} "
              f"FPS={gpu.get('avg_fps', r.get('inference_fps', 0)):.1f}")
    print(f"Output: {output_dir.resolve()}")
    print("=" * 60)

    # Generate charts
    print("\nGenerating charts...")
    plot_model_comparison_bars(results, output_dir)
    plot_speed_accuracy(results, output_dir)

    # Generate comparison video
    if not args.no_video:
        print("\nRendering comparison video...")
        render_comparison_video(results, output_dir / 'comparison.mp4', args.fps)

    # Save comparison summary
    summary = []
    for r in results:
        gpu = r.get('gpu_performance', {})
        summary.append({
            'model': r.get('model', ''),
            'name': r['_name'],
            'precision': r['detection']['precision'],
            'recall': r['detection']['recall'],
            'MOTA': r['tracking']['MOTA'],
            'IDF1': r['tracking']['IDF1'],
            'id_switches': r['tracking']['num_switches'],
            'avg_fps': gpu.get('avg_fps', r.get('inference_fps', 0)),
            'avg_latency_ms': gpu.get('avg_latency_ms', 0),
        })

    with open(output_dir / 'comparison_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_dir.resolve()}")


if __name__ == '__main__':
    main()
