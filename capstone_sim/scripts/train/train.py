"""
YOLOv11 Training Script

Trains a YOLOv11 model on a CARLA-generated YOLO-format dataset.

Usage:
    python train.py --data path/to/data.yaml
    python train.py --data path/to/data.yaml --model yolo11s.pt --epochs 150 --batch 8
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv11 on CARLA dataset')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to data.yaml file')
    parser.add_argument('--model', type=str, default='yolo11n.pt',
                        help='Pretrained model to start from (default: yolo11n.pt). '
                             'Options: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size (default: 16, reduce to 8 if out of VRAM)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Input image size (default: 640)')
    parser.add_argument('--name', type=str, default=None,
                        help='Name for this training run (default: auto-generated)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--cache', type=str, default='false',
                        choices=['true', 'false', 'disk'],
                        help='Cache mode: true (RAM, fastest, needs lots of RAM), '
                             'disk (decoded files on disk, medium speed), '
                             'false (no cache, default)')
    args = parser.parse_args()

    # Convert cache string to proper value
    cache_value = {'true': True, 'false': False, 'disk': 'disk'}[args.cache]

    data_path = Path(args.data).resolve()
    if not data_path.exists():
        print(f"Error: data.yaml not found at {data_path}")
        return

    # Set results directory in models/yolov11/runs
    project_dir = Path(__file__).resolve().parents[3] / 'capstone_sim' / 'models' / 'yolov11' / 'runs'

    if args.resume:
        print(f"Resuming training from: {args.resume}")
        model = YOLO(args.resume)
        results = model.train(resume=True)
    else:
        print("=" * 60)
        print("YOLOv11 Training")
        print("=" * 60)
        print(f"Model:    {args.model}")
        print(f"Dataset:  {data_path}")
        print(f"Epochs:   {args.epochs}")
        print(f"Batch:    {args.batch}")
        print(f"Img Size: {args.imgsz}")
        print(f"Output:   {project_dir}")
        print("=" * 60)

        model = YOLO(args.model)
        results = model.train(
            data=str(data_path),
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            project=str(project_dir),
            name=args.name,
            patience=20,
            save=True,
            save_period=10,
            device=0,
            workers=4,
            cache=cache_value,
            exist_ok=True,
        )

    print("\nTraining complete!")
    print(f"Results saved to: {project_dir}")
    print("\nTo run inference with your trained model:")
    print(f"  from ultralytics import YOLO")
    print(f"  model = YOLO('path/to/best.pt')")
    print(f"  results = model.predict('image.jpg')")


if __name__ == '__main__':
    main()
