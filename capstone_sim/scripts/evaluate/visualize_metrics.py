"""
Metrics Visualization Script

Generates charts from evaluation metrics_summary.json files.

Usage:
    python visualize_metrics.py <metrics_summary.json>
    python visualize_metrics.py <metrics_summary.json> --per-frame <per_frame_metrics.csv>
"""

import argparse
import json
import csv
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    print("Error: matplotlib is required. Install with: pip install matplotlib")
    sys.exit(1)

from capstone_sim.scripts.utils.constants import CLASS_NAMES, CLASS_COLORS_PLT


def plot_per_class_detection(det_metrics, output_dir):
    """Bar chart of per-class precision, recall, F1."""
    per_class = det_metrics['per_class']
    classes = sorted(per_class.keys(), key=lambda x: int(x))
    classes = [c for c in classes if per_class[c]['count'] > 0]

    names = [CLASS_NAMES.get(int(c), str(c)) for c in classes]
    precision = [per_class[c]['precision'] for c in classes]
    recall = [per_class[c]['recall'] for c in classes]
    f1 = [per_class[c]['f1'] for c in classes]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#3498db')
    bars2 = ax.bar(x, recall, width, label='Recall', color='#e74c3c')
    bars3 = ax.bar(x + width, f1, width, label='F1', color='#2ecc71')

    ax.set_ylabel('Score')
    ax.set_title('Per-Class Detection Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'per_class_detection.png', dpi=150)
    plt.close()
    print(f"  Saved per_class_detection.png")


def plot_class_distribution(det_metrics, output_dir):
    """Pie chart of ground truth class distribution."""
    per_class = det_metrics['per_class']
    classes = sorted(per_class.keys(), key=lambda x: int(x))
    classes = [c for c in classes if per_class[c]['count'] > 0]

    names = [CLASS_NAMES.get(int(c), str(c)) for c in classes]
    counts = [per_class[c]['count'] for c in classes]
    colors = [CLASS_COLORS_PLT.get(int(c), '#95a5a6') for c in classes]

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(counts, labels=names, autopct='%1.1f%%',
                                       colors=colors, startangle=90)
    ax.set_title('Ground Truth Class Distribution')
    plt.tight_layout()
    plt.savefig(output_dir / 'class_distribution.png', dpi=150)
    plt.close()
    print(f"  Saved class_distribution.png")


def plot_tp_fp_fn(det_metrics, output_dir):
    """Stacked bar chart of TP, FP, FN per class."""
    per_class = det_metrics['per_class']
    classes = sorted(per_class.keys(), key=lambda x: int(x))
    classes = [c for c in classes if per_class[c]['count'] > 0]

    names = [CLASS_NAMES.get(int(c), str(c)) for c in classes]
    tp = [per_class[c]['tp'] for c in classes]
    fp = [per_class[c]['fp'] for c in classes]
    fn = [per_class[c]['fn'] for c in classes]

    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, tp, label='True Positives', color='#2ecc71')
    ax.bar(x, fp, bottom=tp, label='False Positives', color='#e74c3c')
    ax.bar(x, fn, bottom=[t + f for t, f in zip(tp, fp)], label='False Negatives', color='#f39c12')

    ax.set_ylabel('Count')
    ax.set_title('Detection Results by Class (TP / FP / FN)')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'tp_fp_fn.png', dpi=150)
    plt.close()
    print(f"  Saved tp_fp_fn.png")


def plot_tracking_summary(track_metrics, det_metrics, output_dir):
    """Summary dashboard of tracking and detection metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # MOTA and IDF1 gauge
    ax = axes[0]
    metrics = {'MOTA': track_metrics['MOTA'], 'IDF1': track_metrics['IDF1']}
    names = list(metrics.keys())
    values = list(metrics.values())
    colors = ['#3498db' if v >= 0.5 else '#e74c3c' for v in values]
    bars = ax.barh(names, values, color=colors, height=0.5)
    ax.set_xlim(0, 1)
    ax.set_title('Tracking Metrics')
    ax.grid(axis='x', alpha=0.3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=12, fontweight='bold')

    # Overall detection
    ax = axes[1]
    det_names = ['Precision', 'Recall']
    det_values = [det_metrics['precision'], det_metrics['recall']]
    colors = ['#3498db', '#e74c3c']
    bars = ax.barh(det_names, det_values, color=colors, height=0.5)
    ax.set_xlim(0, 1)
    ax.set_title('Overall Detection')
    ax.grid(axis='x', alpha=0.3)
    for bar, val in zip(bars, det_values):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=12, fontweight='bold')

    # Tracking error breakdown
    ax = axes[2]
    error_names = ['FP', 'FN', 'ID Switches']
    error_values = [track_metrics['FP'], track_metrics['FN'], track_metrics['num_switches']]
    colors = ['#e74c3c', '#f39c12', '#9b59b6']
    bars = ax.barh(error_names, error_values, color=colors, height=0.5)
    ax.set_title('Tracking Errors')
    ax.grid(axis='x', alpha=0.3)
    for bar, val in zip(bars, error_values):
        ax.text(bar.get_width() + max(error_values) * 0.02, bar.get_y() + bar.get_height() / 2,
                f'{val}', va='center', fontsize=12, fontweight='bold')

    plt.suptitle('Evaluation Summary', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'evaluation_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved evaluation_summary.png")


def plot_confusion_matrix(confusion_matrix, output_dir):
    """Heatmap of the confusion matrix."""
    labels = ['car', 'ambulance', 'bus', 'truck', 'police_car', 'fire_truck', 'bike', 'background']
    matrix = np.array(confusion_matrix)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap='Blues')

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Ground Truth')
    ax.set_title('Confusion Matrix')

    # Annotate each cell with the count
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = matrix[i, j]
            color = 'white' if val > matrix.max() / 2 else 'black'
            ax.text(j, i, str(val), ha='center', va='center', color=color)

    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=150)
    plt.close()
    print(f"  Saved confusion_matrix.png")


def plot_per_frame(csv_path, output_dir):
    """Line plots of per-frame metrics over time."""
    frames, num_gt, num_pred, tp, fp, fn, switches = [], [], [], [], [], [], []

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append(int(row['frame']))
            num_gt.append(int(row['num_gt']))
            num_pred.append(int(row['num_pred']))
            tp.append(int(row['tp']))
            fp.append(int(row['fp']))
            fn.append(int(row['fn']))
            switches.append(int(row['id_switches']))

    # Smooth with rolling average for readability
    window = min(50, len(frames) // 10) if len(frames) > 100 else 1

    def smooth(data, w):
        if w <= 1:
            return data
        return np.convolve(data, np.ones(w) / w, mode='valid')

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # GT vs Predictions count
    ax = axes[0]
    ax.plot(frames, num_gt, alpha=0.3, color='#3498db')
    ax.plot(frames, num_pred, alpha=0.3, color='#e74c3c')
    if window > 1:
        ax.plot(smooth(frames, window), smooth(num_gt, window), color='#3498db', label='Ground Truth', linewidth=2)
        ax.plot(smooth(frames, window), smooth(num_pred, window), color='#e74c3c', label='Predictions', linewidth=2)
    else:
        ax.legend(['Ground Truth', 'Predictions'])
    ax.set_ylabel('Count')
    ax.set_title('Objects per Frame')
    ax.legend()
    ax.grid(alpha=0.3)

    # TP, FP, FN
    ax = axes[1]
    ax.plot(frames, tp, alpha=0.3, color='#2ecc71')
    ax.plot(frames, fp, alpha=0.3, color='#e74c3c')
    ax.plot(frames, fn, alpha=0.3, color='#f39c12')
    if window > 1:
        ax.plot(smooth(frames, window), smooth(tp, window), color='#2ecc71', label='TP', linewidth=2)
        ax.plot(smooth(frames, window), smooth(fp, window), color='#e74c3c', label='FP', linewidth=2)
        ax.plot(smooth(frames, window), smooth(fn, window), color='#f39c12', label='FN', linewidth=2)
    ax.set_ylabel('Count')
    ax.set_title('Detection Results per Frame')
    ax.legend()
    ax.grid(alpha=0.3)

    # ID Switches (cumulative)
    ax = axes[2]
    cumulative_switches = np.cumsum(switches)
    ax.plot(frames, cumulative_switches, color='#9b59b6', linewidth=2)
    ax.fill_between(frames, cumulative_switches, alpha=0.2, color='#9b59b6')
    ax.set_ylabel('Cumulative ID Switches')
    ax.set_xlabel('Frame')
    ax.set_title('ID Switches Over Time')
    ax.grid(alpha=0.3)

    plt.suptitle('Per-Frame Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'per_frame_metrics.png', dpi=150)
    plt.close()
    print(f"  Saved per_frame_metrics.png")


def main():
    parser = argparse.ArgumentParser(description='Visualize evaluation metrics')
    parser.add_argument('metrics', type=str, help='Path to metrics_summary.json')
    parser.add_argument('--per-frame', type=str, default=None,
                        help='Path to per_frame_metrics.csv (auto-detected if in same folder)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for charts (default: same folder as metrics)')
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        print(f"Error: {metrics_path} not found")
        sys.exit(1)

    with open(metrics_path) as f:
        data = json.load(f)

    output_dir = Path(args.output) if args.output else metrics_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect per-frame CSV
    per_frame_path = None
    if args.per_frame:
        per_frame_path = Path(args.per_frame)
    else:
        auto_path = metrics_path.parent / 'per_frame_metrics.csv'
        if auto_path.exists():
            per_frame_path = auto_path

    print("=" * 60)
    print("Generating Visualizations")
    print("=" * 60)
    print(f"Model: {data.get('model', 'unknown')}")
    print(f"Recording: {data.get('recording', 'unknown')}")
    print(f"Output: {output_dir.resolve()}")
    print()

    plot_per_class_detection(data['detection'], output_dir)
    plot_class_distribution(data['detection'], output_dir)
    plot_tp_fp_fn(data['detection'], output_dir)
    plot_tracking_summary(data['tracking'], data['detection'], output_dir)

    if 'confusion_matrix' in data:
        plot_confusion_matrix(data['confusion_matrix'], output_dir)

    if per_frame_path and per_frame_path.exists():
        plot_per_frame(per_frame_path, output_dir)
    else:
        print("  Skipping per-frame plots (no CSV found)")

    print(f"\nDone! Charts saved to {output_dir.resolve()}")


if __name__ == '__main__':
    main()
