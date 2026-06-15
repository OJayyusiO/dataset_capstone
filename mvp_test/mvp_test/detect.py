"""
YOLO video detection script.

Runs an Ultralytics YOLO model on a video file, writes:
  - an annotated .mp4 (boxes + labels burned in)
  - a per-frame detections .jsonl file (one line per frame)
  - a flat detections.csv for easy spreadsheet inspection

Usage (from this folder):
    pip install ultralytics opencv-python
    python detect.py

If you want a different model or video, edit MODEL_NAME / VIDEO_PATH below,
or pass them on the command line:
    python detect.py --model yolo11s.pt --video "my_clip.mov"
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# Defaults (edit if you want)
# --------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent

# The user asked for "yolo26s". That isn't a publicly released Ultralytics
# weight name as of writing — if Ultralytics can't fetch it, fall back to a
# known-good model. You can also point this at a local .pt path.
MODEL_NAME = "yolo26s.pt"
FALLBACK_MODELS = ["yolo11s.pt", "yolov8s.pt"]

# Default video: the .mov sitting next to this script.
DEFAULT_VIDEO = HERE / "Screen Recording 2026-05-15 at 14.05.05.mov"

# Inference settings
CONF_THRESHOLD = 0.25      # default minimum confidence to keep a detection
IOU_THRESHOLD = 0.45       # NMS IoU
DEVICE = None              # None = auto (CUDA/MPS/CPU). Or "cpu", "mps", "0".

# Per-class confidence overrides. Motorcycles are smaller and partially
# occluded more often, so we drop the bar to recover them.
PER_CLASS_CONF = {
    "motorcycle": 0.10,
}

# Only keep these classes. Names not present in the model are skipped with a
# warning — e.g. "fire truck" / "ambulance" aren't separate COCO classes, so
# stock YOLO weights will just label those vehicles as "truck" / "car".
ALLOWED_CLASS_NAMES = [
    "car",
    "truck",
    "bus",
    "motorcycle",
    "fire truck",
    "ambulance",
]


# --------------------------------------------------------------------------
def load_model(name: str):
    """Try the requested model, fall back to known-good weights if needed."""
    from ultralytics import YOLO

    tried = []
    candidates = [name] + [m for m in FALLBACK_MODELS if m != name]
    last_err: Exception | None = None
    for cand in candidates:
        try:
            print(f"[detect] loading model: {cand}")
            model = YOLO(cand)
            if cand != name:
                print(f"[detect] note: '{name}' was not available, using '{cand}' instead")
            return model, cand
        except Exception as e:  # noqa: BLE001
            tried.append(cand)
            last_err = e
            print(f"[detect] could not load {cand}: {e}")
    raise RuntimeError(
        f"Could not load any model. Tried: {tried}. Last error: {last_err}"
    )


def run(video_path: Path, model_name: str, out_dir: Path) -> None:
    import cv2  # noqa: F401  (ensures opencv is installed for Ultralytics video IO)

    if not video_path.exists():
        sys.exit(f"[detect] video not found: {video_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    model, used_model = load_model(model_name)

    # Resolve allowed class names → class IDs against this model's label map.
    name_to_id = {n.lower(): i for i, n in model.names.items()}
    allowed_ids: list[int] = []
    missing: list[str] = []
    for name in ALLOWED_CLASS_NAMES:
        cid = name_to_id.get(name.lower())
        if cid is None:
            missing.append(name)
        else:
            allowed_ids.append(cid)
    if missing:
        print(f"[detect] note: model has no class(es) for {missing} — skipping")
    if not allowed_ids:
        sys.exit("[detect] none of the requested classes exist in this model")
    print(f"[detect] keeping classes: {[model.names[i] for i in allowed_ids]}")

    # Run inference at the lowest active threshold so low-conf motorcycles
    # survive NMS; per-class thresholds are re-applied in the loop below.
    inference_conf = min(CONF_THRESHOLD, *PER_CLASS_CONF.values())

    print(f"[detect] running inference on: {video_path.name}")
    results_iter = model.predict(
        source=str(video_path),
        conf=inference_conf,
        iou=IOU_THRESHOLD,
        classes=allowed_ids,      # filter at inference — affects video + dumps
        device=DEVICE,
        save=True,                # write annotated video
        project=str(out_dir),     # output root
        name="annotated",         # subfolder name
        exist_ok=True,
        stream=True,              # iterate frames so we can dump labels too
        verbose=False,
    )

    jsonl_path = out_dir / "detections.jsonl"
    csv_path = out_dir / "detections.csv"

    frame_idx = 0
    total_dets = 0
    with jsonl_path.open("w") as jf, csv_path.open("w", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(
            ["frame", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"]
        )

        for r in results_iter:
            names = r.names  # {id: label}
            frame_record = {"frame": frame_idx, "detections": []}
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                cls_ids = r.boxes.cls.cpu().numpy().astype(int)
                for (x1, y1, x2, y2), cf_val, cid in zip(xyxy, confs, cls_ids):
                    cname = names.get(int(cid), str(int(cid)))
                    if float(cf_val) < PER_CLASS_CONF.get(cname, CONF_THRESHOLD):
                        continue
                    det = {
                        "class_id": int(cid),
                        "class_name": cname,
                        "confidence": float(cf_val),
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    }
                    frame_record["detections"].append(det)
                    writer.writerow(
                        [
                            frame_idx,
                            int(cid),
                            cname,
                            f"{float(cf_val):.4f}",
                            f"{float(x1):.2f}",
                            f"{float(y1):.2f}",
                            f"{float(x2):.2f}",
                            f"{float(y2):.2f}",
                        ]
                    )
                    total_dets += 1
            jf.write(json.dumps(frame_record) + "\n")
            frame_idx += 1

    annotated_dir = out_dir / "annotated"
    print(f"[detect] done. model used: {used_model}")
    print(f"[detect] frames processed : {frame_idx}")
    print(f"[detect] total detections : {total_dets}")
    print(f"[detect] annotated video  : {annotated_dir} (look for the .mp4/.avi inside)")
    print(f"[detect] per-frame JSONL  : {jsonl_path}")
    print(f"[detect] flat CSV         : {csv_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run YOLO on a video file")
    p.add_argument("--model", default=MODEL_NAME, help="Model name or path (.pt)")
    p.add_argument("--video", default=str(DEFAULT_VIDEO), help="Path to input video")
    p.add_argument(
        "--out", default=str(HERE / "runs"), help="Output directory root"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(Path(args.video), args.model, Path(args.out))
