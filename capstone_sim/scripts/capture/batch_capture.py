"""
Batch Dataset Capture

Runs capture_dataset.py on multiple scenario configs sequentially.
Each config can specify its own map, which is loaded automatically.

Usage:
    python batch_capture.py config1.yaml config2.yaml config3.yaml
    python batch_capture.py capstone_sim/configs/          # run all YAMLs in folder
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from capstone_sim.scripts.capture.capture_dataset import run_capture


def main():
    if len(sys.argv) < 2:
        print("Usage: python batch_capture.py <config.yaml> [config2.yaml ...] or <config_folder/>")
        sys.exit(1)

    # Collect config paths
    config_paths = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        if p.is_dir():
            yamls = sorted(p.glob('*.yaml'))
            config_paths.extend(yamls)
        elif p.is_file() and p.suffix in ('.yaml', '.yml'):
            config_paths.append(p)
        else:
            print(f"WARNING: Skipping {arg} (not a YAML file or directory)")

    if not config_paths:
        print("No config files found.")
        sys.exit(1)

    print("=" * 60)
    print("Batch Dataset Capture")
    print("=" * 60)
    print(f"Configs to run: {len(config_paths)}")
    for i, p in enumerate(config_paths):
        print(f"  [{i+1}] {p.name}")
    print("=" * 60)

    total_start = time.time()
    completed = 0
    failed = 0

    for i, config_path in enumerate(config_paths):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(config_paths)}] Running: {config_path.name}")
        print(f"{'='*60}")

        try:
            run_capture(config_path)
            completed += 1
        except KeyboardInterrupt:
            print("\n\nBatch interrupted by user.")
            print(f"Completed {completed}/{len(config_paths)} scenarios.")
            sys.exit(0)
        except Exception as e:
            print(f"\nERROR in {config_path.name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"Batch complete!")
    print(f"  Completed: {completed}/{len(config_paths)}")
    if failed:
        print(f"  Failed: {failed}")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
