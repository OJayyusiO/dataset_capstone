"""
Convert PNG frames to MP4 video.

Usage:
    python frames_to_video.py <frames_dir>
    python frames_to_video.py <frames_dir> --output out.mp4 --fps 20 --pattern 'cam0_*.png'
"""

import argparse
import sys
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(description='Convert PNG frames to MP4')
    parser.add_argument('frames_dir', type=str, help='Directory containing PNG frames')
    parser.add_argument('--output', type=str, default=None,
                        help='Output MP4 path (default: <frames_dir>/../recording.mp4)')
    parser.add_argument('--fps', type=int, default=20, help='Output FPS (default: 20)')
    parser.add_argument('--pattern', type=str, default='*.png',
                        help="Glob pattern for frames (default: '*.png'). "
                             "Use 'cam0_*.png' for multi-camera recordings.")
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    if not frames_dir.is_dir():
        print(f"Not a directory: {frames_dir}")
        sys.exit(1)

    frames = sorted(frames_dir.glob(args.pattern))
    if not frames:
        print(f"No frames found matching '{args.pattern}' in {frames_dir}")
        sys.exit(1)

    output = Path(args.output) if args.output else frames_dir.parent / 'recording.mp4'
    output.parent.mkdir(parents=True, exist_ok=True)

    first = cv2.imread(str(frames[0]))
    h, w = first.shape[:2]

    print(f"Frames:      {len(frames)}")
    print(f"Resolution:  {w}x{h}")
    print(f"FPS:         {args.fps}")
    print(f"Output:      {output.resolve()}")
    print()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output), fourcc, args.fps, (w, h))

    for i, f in enumerate(frames):
        img = cv2.imread(str(f))
        writer.write(img)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(frames)}")

    writer.release()
    print(f"\nDone: {output.resolve()}")


if __name__ == '__main__':
    main()
