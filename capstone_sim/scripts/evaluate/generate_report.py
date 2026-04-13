"""
Training Report Generator

Generates an HTML report summarizing dataset, training, and evaluation results
for the capstone presentation.

Usage:
    python generate_report.py --dataset path/to/dataset_output --eval path/to/eval_results/run
    python generate_report.py --dataset path/to/dataset_output --eval eval1 eval2 --output report.html
"""

import argparse
import json
import sys
import base64
from pathlib import Path
from datetime import datetime


def image_to_base64(img_path):
    """Convert an image file to base64 for embedding in HTML."""
    if not img_path.exists():
        return None
    with open(img_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def load_eval(eval_dir):
    """Load evaluation metrics from a result directory."""
    eval_dir = Path(eval_dir)
    meta_path = eval_dir / 'metrics_summary.json'
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        data = json.load(f)
    data['_dir'] = eval_dir
    data['_name'] = Path(data.get('model', eval_dir.name)).stem
    return data


def generate_html(dataset_dir, eval_dirs, output_path):
    dataset_dir = Path(dataset_dir) if dataset_dir else None
    output_path = Path(output_path)

    evals = []
    for ed in eval_dirs:
        e = load_eval(ed)
        if e:
            evals.append(e)

    CLASS_NAMES = {0: 'car', 1: 'ambulance', 2: 'bus', 3: 'truck',
                   4: 'police_car', 5: 'fire_truck', 6: 'bike'}

    html = []
    html.append("""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Capstone Training Report</title>
<style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; color: #333; }
    h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
    h2 { color: #34495e; margin-top: 40px; }
    h3 { color: #555; }
    .card { background: white; border-radius: 8px; padding: 20px; margin: 15px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
    .metric { text-align: center; padding: 15px; background: #f8f9fa; border-radius: 6px; }
    .metric .value { font-size: 2em; font-weight: bold; color: #2c3e50; }
    .metric .label { font-size: 0.9em; color: #777; margin-top: 5px; }
    .good { color: #27ae60 !important; }
    .warn { color: #f39c12 !important; }
    .bad { color: #e74c3c !important; }
    table { width: 100%; border-collapse: collapse; margin: 10px 0; }
    th, td { padding: 10px 15px; text-align: left; border-bottom: 1px solid #eee; }
    th { background: #f8f9fa; font-weight: 600; }
    tr:hover { background: #f8f9fa; }
    img { max-width: 100%; border-radius: 6px; margin: 10px 0; }
    .img-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 15px; }
    .timestamp { color: #999; font-size: 0.85em; }
    .comparison-table td { text-align: center; }
    .comparison-table th { text-align: center; }
</style>
</head><body>""")

    html.append(f"<h1>Capstone Training Report</h1>")
    html.append(f'<p class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>')

    # Dataset section
    if dataset_dir:
        html.append("<h2>Dataset Overview</h2>")
        html.append('<div class="card">')

        # Check for analysis image
        analysis_img = dataset_dir / 'dataset_analysis.png'
        if analysis_img.exists():
            b64 = image_to_base64(analysis_img)
            html.append(f'<img src="data:image/png;base64,{b64}" alt="Dataset Analysis">')

        # Count images and labels
        total_train = len(list((dataset_dir / 'images' / 'train').glob('*.png'))) if (dataset_dir / 'images' / 'train').exists() else 0
        total_val = len(list((dataset_dir / 'images' / 'val').glob('*.png'))) if (dataset_dir / 'images' / 'val').exists() else 0

        html.append('<div class="metric-grid">')
        html.append(f'<div class="metric"><div class="value">{total_train + total_val}</div><div class="label">Total Images</div></div>')
        html.append(f'<div class="metric"><div class="value">{total_train}</div><div class="label">Training</div></div>')
        html.append(f'<div class="metric"><div class="value">{total_val}</div><div class="label">Validation</div></div>')
        html.append(f'<div class="metric"><div class="value">7</div><div class="label">Classes</div></div>')
        html.append('</div>')
        html.append('</div>')

    # Evaluation results
    if evals:
        if len(evals) == 1:
            html.append("<h2>Evaluation Results</h2>")
            e = evals[0]
            _render_single_eval(html, e, CLASS_NAMES)
        else:
            html.append("<h2>Model Comparison</h2>")
            _render_comparison(html, evals, CLASS_NAMES)

            for e in evals:
                html.append(f"<h2>Details: {e['_name']}</h2>")
                _render_single_eval(html, e, CLASS_NAMES)

    html.append("</body></html>")

    output_path.write_text('\n'.join(html), encoding='utf-8')
    print(f"Report saved: {output_path.resolve()}")


def _color_class(value, good=0.7, bad=0.4):
    if value >= good:
        return 'good'
    elif value >= bad:
        return 'warn'
    return 'bad'


def _render_single_eval(html, e, class_names):
    det = e.get('detection', {})
    trk = e.get('tracking', {})
    gpu = e.get('gpu_performance', {})

    html.append('<div class="card">')
    html.append(f'<h3>{e["_name"]}</h3>')

    # Key metrics
    html.append('<div class="metric-grid">')
    p = det.get('precision', 0)
    r = det.get('recall', 0)
    mota = trk.get('MOTA', 0)
    idf1 = trk.get('IDF1', 0)
    fps = gpu.get('avg_fps', e.get('inference_fps', 0))

    html.append(f'<div class="metric"><div class="value {_color_class(p)}">{p:.3f}</div><div class="label">Precision</div></div>')
    html.append(f'<div class="metric"><div class="value {_color_class(r)}">{r:.3f}</div><div class="label">Recall</div></div>')
    html.append(f'<div class="metric"><div class="value {_color_class(mota)}">{mota:.3f}</div><div class="label">MOTA</div></div>')
    html.append(f'<div class="metric"><div class="value {_color_class(idf1)}">{idf1:.3f}</div><div class="label">IDF1</div></div>')
    html.append(f'<div class="metric"><div class="value">{fps:.1f}</div><div class="label">FPS</div></div>')
    html.append(f'<div class="metric"><div class="value">{trk.get("num_switches", 0)}</div><div class="label">ID Switches</div></div>')
    html.append('</div>')

    # Per-class table
    per_class = det.get('per_class', {})
    if per_class:
        html.append('<h3>Per-Class Performance</h3>')
        html.append('<table><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>TP</th><th>FP</th><th>FN</th><th>Count</th></tr>')
        for cls_id in sorted(per_class.keys(), key=lambda x: int(x)):
            s = per_class[cls_id]
            name = class_names.get(int(cls_id), str(cls_id))
            html.append(f'<tr><td>{name}</td>'
                        f'<td class="{_color_class(s["precision"])}">{s["precision"]:.3f}</td>'
                        f'<td class="{_color_class(s["recall"])}">{s["recall"]:.3f}</td>'
                        f'<td class="{_color_class(s["f1"])}">{s["f1"]:.3f}</td>'
                        f'<td>{s["tp"]}</td><td>{s["fp"]}</td><td>{s["fn"]}</td>'
                        f'<td>{s["count"]}</td></tr>')
        html.append('</table>')

    # GPU performance
    if gpu:
        html.append('<h3>GPU Performance</h3>')
        html.append(f'<p>Device: {gpu.get("device", "N/A")} | '
                    f'Avg Latency: {gpu.get("avg_latency_ms", 0):.1f}ms | '
                    f'P95 Latency: {gpu.get("p95_latency_ms", 0):.1f}ms')
        if 'avg_memory_used_mb' in gpu:
            html.append(f' | Memory: {gpu["avg_memory_used_mb"]:.0f}MB avg')
        if 'avg_power_draw_w' in gpu:
            html.append(f' | Power: {gpu["avg_power_draw_w"]:.1f}W avg')
        html.append('</p>')

    # Embedded charts
    html.append('<div class="img-grid">')
    chart_names = ['evaluation_summary.png', 'per_class_detection.png', 'confusion_matrix.png',
                   'tp_fp_fn.png', 'per_frame_metrics.png', 'class_distribution.png']
    for chart in chart_names:
        chart_path = e['_dir'] / chart
        if chart_path.exists():
            b64 = image_to_base64(chart_path)
            html.append(f'<img src="data:image/png;base64,{b64}" alt="{chart}">')
    html.append('</div>')

    html.append('</div>')


def _render_comparison(html, evals, class_names):
    html.append('<div class="card">')

    # Summary table
    html.append('<table class="comparison-table">')
    html.append('<tr><th>Model</th><th>Precision</th><th>Recall</th><th>MOTA</th><th>IDF1</th><th>ID Switches</th><th>FPS</th></tr>')
    for e in evals:
        det = e.get('detection', {})
        trk = e.get('tracking', {})
        gpu = e.get('gpu_performance', {})
        fps = gpu.get('avg_fps', e.get('inference_fps', 0))
        html.append(f'<tr><td><strong>{e["_name"]}</strong></td>'
                    f'<td class="{_color_class(det.get("precision", 0))}">{det.get("precision", 0):.3f}</td>'
                    f'<td class="{_color_class(det.get("recall", 0))}">{det.get("recall", 0):.3f}</td>'
                    f'<td class="{_color_class(trk.get("MOTA", 0))}">{trk.get("MOTA", 0):.3f}</td>'
                    f'<td class="{_color_class(trk.get("IDF1", 0))}">{trk.get("IDF1", 0):.3f}</td>'
                    f'<td>{trk.get("num_switches", 0)}</td>'
                    f'<td>{fps:.1f}</td></tr>')
    html.append('</table>')
    html.append('</div>')


def main():
    parser = argparse.ArgumentParser(description='Generate HTML training report')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Path to dataset directory')
    parser.add_argument('--eval', nargs='+', required=True,
                        help='Path(s) to evaluation result directories')
    parser.add_argument('--output', type=str, default='report.html',
                        help='Output HTML file (default: report.html)')
    args = parser.parse_args()

    # Expand eval dirs
    eval_dirs = []
    for p in args.eval:
        p = Path(p)
        if p.is_dir() and (p / 'metrics_summary.json').exists():
            eval_dirs.append(p)
        elif p.is_dir():
            for sub in sorted(p.iterdir()):
                if sub.is_dir() and (sub / 'metrics_summary.json').exists():
                    eval_dirs.append(sub)

    if not eval_dirs:
        print("No evaluation results found.")
        sys.exit(1)

    generate_html(args.dataset, eval_dirs, args.output)


if __name__ == '__main__':
    main()
