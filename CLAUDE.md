# Project Context for Claude Code

> Read this on every new session before doing work. The full history is in `PROJECT_HISTORY.md`.

## Project at a glance

Vehicle detection + tracking system for traffic analytics, built on:
- **CARLA 0.9.16** — synthetic data generation
- **YOLOv11m** — trained on custom 7-class dataset (car, ambulance, bus, truck, police_car, fire_truck, bike)
- **ByteTrack** (via Ultralytics) — multi-object tracking
- **4-point homography** — pixel→world calibration for real-world metrics

**Current metrics:** mAP50 = 0.949, precision = 0.97, recall = 0.90, MOTA = 0.78, IDF1 = 0.88, 27.7 FPS on RTX 4060 Laptop.

**Current version:** V3.0 (analytics layer with speed + per-lane queue). Open PR on GitHub.

## Repository

- **Local path:** `C:\Users\omarj\Desktop\dataset_capstone`
- **GitHub:** `https://github.com/OJayyusiO/dataset_capstone` (private)
- **Remote name in local git:** `dataset_capstone` (not `origin`)
- **Branches in order of work:** `main` → V2 → V2.1 → V2.2 → V3.0

## Conda environment

```bash
conda activate capstone   # default env name
```

PyTorch with CUDA must be installed via PyTorch's index URL (`https://download.pytorch.org/whl/cu121`), not plain `pip install torch` which gives the CPU-only build.

## Folder structure

```
capstone_sim/
├── configs/                 # scenario YAMLs
├── analytics_configs/       # per-scenario calibration + lanes
├── analytics_runs/          # gitignored — per-run output folders
├── models/yolov11m/
│   ├── best.pt              # trained model
│   └── runs/                # training outputs
└── scripts/
    ├── capture/             # capture_dataset.py, record_test.py, setup_scenario.py, batch_capture.py
    ├── train/               # train.py
    ├── evaluate/            # evaluate_model.py, visualize_metrics.py, compare_models.py,
    │                          analyze_dataset.py, generate_report.py, inference.py
    ├── analytics/           # setup_analytics.py, traffic_analytics.py, live_analytics.py
    └── utils/               # constants.py, bbox.py, carla_helpers.py, switch_map.py,
                               visualize_spawns.py, visualize_traffic_lights.py,
                               create_spawn_points.py, frames_to_video.py
```

## Important design decisions (DO NOT undo these without asking)

1. **Custom dataset over pre-trained models** — YOLOE, VideoMT, and YOLOv12+COCO were tested and rejected because they can't reliably distinguish emergency vehicles (police_car vs car). The whole point of the CARLA pipeline is to fix this.

2. **CARLA synchronous mode** — every script that connects to CARLA uses `world.tick()` before reading actor state. Synchronous mode requires this; without it, things like camera positions read as `(0,0,0)`.

3. **MIN_VISIBILITY = 0.15** (in `utils/constants.py`) — lowered from 0.4 so the model learns partially visible vehicles at frame edges.

4. **Camera rotation via CARLA API** — `_camera_to_world_rotation` uses `carla.Transform.get_forward_vector()` etc. directly. Do not replace with manual pitch/yaw/roll math — CARLA's conventions are non-standard (positive pitch = nose up).

5. **Reference point for ground projection** — `(50% width, 85% height)` of bbox, not the very bottom. Avoids bumper artifacts on large vehicles. Constants are at the top of `traffic_analytics.py`.

6. **max_vehicles is a hard cap** — `compute_target_counts` in `carla_helpers.py` no longer forces at least 1 per class. Classes with target=0 are dropped.

7. **Auto-calibration for CARLA** — at the end of `record_test.py`, an `analytics_config.yaml` is automatically generated next to the recording with the homography matrix computed from camera intrinsics + extrinsics. Manual calibration is only for real-world video.

8. **Queue thresholds tunable** — `analytics_config.yaml` has `queue.speed_threshold_kmh` (default 7.2) and `queue.min_stationary_seconds` (default 2.0). Don't hardcode these.

9. **Stuck vehicle detection is position-based** — went back and forth between velocity and position; position-based with `stuck_check_interval: 600` (30 sec) and `stuck_threshold_m: 1.0` is the final design. Lenient enough for red-light queues, catches genuine deadlocks.

10. **Town06 is the default test scenario.** Use `Town6_1cam_test.yaml` for testing, not Town04. User asked for this explicitly.

## User preferences

- **Concise, technical responses.** No long preamble.
- **Hard caps and explicit thresholds** over magic defaults. Always make tunable values configurable in YAML.
- **Don't add features the user didn't ask for.** Especially: don't auto-add "would you like me to also..."
- **Don't undo their stylistic choices.** They keep emoji removed, comments minimal, etc.
- **Always check work before claiming complete.** They've caught silent failures multiple times.
- **Tests/demos should output saved data** — anything that runs the system should log to CSV + summary JSON, not just display.

## What's done (V3.0)

| Feature | Status |
|---|---|
| Synthetic dataset generation (multi-camera, weather cycling) | ✅ |
| YOLOv11m training pipeline (local + cloud) | ✅ |
| Detection + tracking evaluation (MOTA, IDF1, per-class metrics, HTML report) | ✅ |
| Real-world inference on MP4 / webcam / RTSP | ✅ |
| Camera calibration (auto for CARLA, manual 4-point for real video) | ✅ |
| Lane definition (interactive polygon picker) | ✅ |
| Speed per car (km/h, color-coded labels) | ✅ |
| Per-lane queue length (configurable speed + duration thresholds) | ✅ |
| Live analytics on CARLA simulation | ✅ |
| Stuck vehicle detection during long captures | ✅ |
| `analytics_runs/<scenario>_<timestamp>/` output: per_track.csv, per_lane_queue.csv, summary.json, optional MP4 | ✅ |

## What's NOT done (next priorities)

1. **Red-light violation detection** — needs forbidden line definition + light state input
2. **Highway entry traffic light counting** — needs entry zones + light state from CARLA (user has explicit permission to read light state directly from the simulator, not from camera vision)
3. **Collision detection** (stretch goal) — bbox overlap + speed drop; tunable, may be skipped

To build any of these, you need to first:
- Plumb light state per frame into the analytics pipeline (CARLA: read from traffic light actor; real video: schedule in YAML)
- Extend `setup_analytics.py` with a 2-point line picker (similar to existing polygon picker)

## Reports and presentations

- **`PROJECT_HISTORY.md`** — full chronological history (read this if asked about decisions)
- **`4992_ProgressReport_FILLED.docx`** — capstone progress report; most sections filled, search for `[TO FILL` for what still needs the user's input (team names, advisor info, gantt chart, budget, etc.)

## Don't do

- Don't push to `main`; always work on feature branches
- Don't force-push without explicit user permission
- Don't add `--no-verify` or skip hooks
- Don't run `gh` CLI — it isn't installed; user opens PRs manually via GitHub URL
- Don't use `git mv` blindly when files are already moved — check `git status` first
- Don't write Python helper scripts when the Edit tool suffices for small string replacements

## When user says "test it"

They mean run the script themselves and report back. Don't try to run training/evaluation jobs — they're long and need GPU+CARLA running locally.
