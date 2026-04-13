"""
Dataset Analyzer

Analyzes a YOLO-format dataset for class distribution, imbalances,
empty labels, and image statistics.

Usage:
    python analyze_dataset.py path/to/dataset_output/
    python analyze_dataset.py path/to/dataset_output/ --visualize
"""

import argparse
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from capstone_sim.scripts.utils.constants import CLASS_NAMES

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def analyze(dataset_dir, visualize=False):
    dataset_dir = Path(dataset_dir)

    # Find label directories
    label_dirs = {}
    for split in ['train', 'val']:
        d = dataset_dir / 'labels' / split
        if d.exists():
            label_dirs[split] = d
    img_dirs = {}
    for split in ['train', 'val']:
        d = dataset_dir / 'images' / split
        if d.exists():
            img_dirs[split] = d

    if not label_dirs:
        print(f"Error: No label directories found in {dataset_dir}")
        return

    print("=" * 60)
    print("Dataset Analysis")
    print("=" * 60)
    print(f"Path: {dataset_dir.resolve()}")

    total_images = 0
    total_annotations = 0
    total_empty = 0
    overall_class_counts = Counter()
    overall_objects_per_image = []
    split_stats = {}

    for split, label_dir in label_dirs.items():
        label_files = sorted(label_dir.glob('*.txt'))
        num_images = len(label_files)
        total_images += num_images

        class_counts = Counter()
        empty_count = 0
        objects_per_image = []
        bbox_widths = []
        bbox_heights = []

        for lf in label_files:
            lines = lf.read_text().strip().split('\n')
            lines = [l for l in lines if l.strip()]

            if len(lines) == 0:
                empty_count += 1
                objects_per_image.append(0)
                continue

            objects_per_image.append(len(lines))
            for line in lines:
                parts = line.strip().split()
                if parts:
                    cls = int(parts[0])
                    class_counts[cls] += 1
                    if len(parts) >= 5:
                        bbox_widths.append(float(parts[3]))
                        bbox_heights.append(float(parts[4]))

        total_annotations += sum(class_counts.values())
        total_empty += empty_count
        overall_class_counts += class_counts
        overall_objects_per_image.extend(objects_per_image)

        split_stats[split] = {
            'images': num_images,
            'annotations': sum(class_counts.values()),
            'empty': empty_count,
            'class_counts': dict(class_counts),
            'avg_objects': sum(objects_per_image) / max(len(objects_per_image), 1),
            'bbox_widths': bbox_widths,
            'bbox_heights': bbox_heights,
        }

    # Print results
    print(f"\nOverall:")
    print(f"  Total images:      {total_images}")
    print(f"  Total annotations: {total_annotations}")
    print(f"  Empty images:      {total_empty} ({100*total_empty/max(total_images,1):.1f}%)")
    print(f"  Avg objects/image: {total_annotations/max(total_images-total_empty,1):.1f}")

    for split, stats in split_stats.items():
        print(f"\n  {split}:")
        print(f"    Images:      {stats['images']}")
        print(f"    Annotations: {stats['annotations']}")
        print(f"    Empty:       {stats['empty']}")

    # Class distribution
    print(f"\nClass Distribution:")
    total_ann = sum(overall_class_counts.values())
    max_count = max(overall_class_counts.values()) if overall_class_counts else 1
    min_count = min(overall_class_counts.values()) if overall_class_counts else 0

    for cls_id in sorted(CLASS_NAMES.keys()):
        count = overall_class_counts.get(cls_id, 0)
        name = CLASS_NAMES[cls_id]
        pct = 100 * count / max(total_ann, 1)
        bar = '#' * int(40 * count / max(max_count, 1))
        print(f"  {name:12s} {count:>7d} ({pct:5.1f}%) {bar}")

    # Imbalance warnings
    print(f"\nImbalance Check:")
    if max_count > 0 and min_count > 0:
        ratio = max_count / min_count
        print(f"  Max/min class ratio: {ratio:.1f}x")
        if ratio > 10:
            print(f"  WARNING: Severe imbalance (>{10}x). Consider adjusting spawn ratios.")
        elif ratio > 5:
            print(f"  WARNING: Moderate imbalance (>{5}x). May affect rare class performance.")
        else:
            print(f"  OK: Class balance is reasonable.")
    else:
        missing = [CLASS_NAMES[c] for c in CLASS_NAMES if overall_class_counts.get(c, 0) == 0]
        if missing:
            print(f"  WARNING: Missing classes: {missing}")

    # Empty image warnings
    empty_pct = 100 * total_empty / max(total_images, 1)
    if empty_pct > 20:
        print(f"\n  WARNING: {empty_pct:.0f}% empty images. Consider reducing warmup frames.")
    elif empty_pct > 5:
        print(f"\n  Note: {empty_pct:.0f}% empty images (background samples).")

    # Visualization
    if visualize and HAS_MPL:
        print("\nGenerating charts...")
        output_dir = dataset_dir
        _plot_analysis(overall_class_counts, overall_objects_per_image,
                       split_stats, output_dir)
    elif visualize and not HAS_MPL:
        print("\nSkipping charts (matplotlib not installed)")

    print(f"\nAnalysis complete.")


def _plot_analysis(class_counts, objects_per_image, split_stats, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Class distribution bar chart
    ax = axes[0, 0]
    classes = sorted(CLASS_NAMES.keys())
    names = [CLASS_NAMES[c] for c in classes]
    counts = [class_counts.get(c, 0) for c in classes]
    colors = ['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#00bcd4', '#ff5722', '#e91e63']
    ax.bar(names, counts, color=colors)
    ax.set_title('Class Distribution')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=45)
    for i, (n, c) in enumerate(zip(names, counts)):
        ax.text(i, c + max(counts) * 0.01, str(c), ha='center', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # 2. Objects per image histogram
    ax = axes[0, 1]
    non_empty = [o for o in objects_per_image if o > 0]
    if non_empty:
        ax.hist(non_empty, bins=range(0, max(non_empty) + 2), color='#3498db', edgecolor='white')
    ax.set_title('Objects per Image (non-empty)')
    ax.set_xlabel('Number of objects')
    ax.set_ylabel('Number of images')
    ax.grid(axis='y', alpha=0.3)

    # 3. Train/val split
    ax = axes[1, 0]
    split_names = list(split_stats.keys())
    split_counts = [split_stats[s]['images'] for s in split_names]
    split_colors = ['#3498db', '#e74c3c']
    ax.pie(split_counts, labels=[f"{s}\n({c} imgs)" for s, c in zip(split_names, split_counts)],
           colors=split_colors[:len(split_names)], autopct='%1.1f%%', startangle=90)
    ax.set_title('Train/Val Split')

    # 4. Bbox size distribution
    ax = axes[1, 1]
    all_widths = []
    all_heights = []
    for stats in split_stats.values():
        all_widths.extend(stats['bbox_widths'])
        all_heights.extend(stats['bbox_heights'])
    if all_widths:
        areas = [w * h for w, h in zip(all_widths, all_heights)]
        ax.hist(areas, bins=50, color='#2ecc71', edgecolor='white', alpha=0.8)
    ax.set_title('Bounding Box Area Distribution (normalized)')
    ax.set_xlabel('Area (w × h, normalized)')
    ax.set_ylabel('Count')
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Dataset Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'dataset_analysis.png', dpi=150)
    plt.close()
    print(f"  Saved dataset_analysis.png")


def main():
    parser = argparse.ArgumentParser(description='Analyze YOLO-format dataset')
    parser.add_argument('dataset', type=str, help='Path to dataset directory (with images/ and labels/)')
    parser.add_argument('--visualize', action='store_true', help='Generate analysis charts')
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: {dataset_path} not found")
        sys.exit(1)

    analyze(str(dataset_path), args.visualize)


if __name__ == '__main__':
    main()
