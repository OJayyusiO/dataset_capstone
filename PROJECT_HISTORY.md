# Capstone Project: Vehicle Detection & Tracking — Full Project History

A comprehensive documentation of the project from initial commit to current state, including all major decisions, challenges encountered, and how each was resolved.

---

## Project Goal

Build a vehicle detection and tracking system using:
- **CARLA simulator** to generate synthetic traffic data
- **YOLOv11** for object detection (7 classes: car, ambulance, bus, truck, police_car, fire_truck, bike)
- **ByteTrack** for multi-object tracking
- A complete pipeline from data capture → training → evaluation → inference on real video

---

## Phase -1: Pre-Trained Model Experimentation (February 19 — March 29, 2026)

> *This phase pre-dates any committed code in this repo. Reconstructed from the WSL Ubuntu environment at `/home/capstone/Capstone_Work/capstone_models/` and personal time-tracked notes.*

### The Initial Question: "Can we skip training entirely?"

Before committing to CARLA and custom datasets, the team explored whether **off-the-shelf pre-trained models** could solve the problem on real-world video footage. The hypothesis: maybe a powerful enough generic detector + tracker would work without us needing our own data.

This was running in parallel with the CARLA scenario work in the main repo. About **3 different model approaches** were tested over 6 weeks before settling on the custom-dataset path.

### Experiment 1: YOLOv12 + Fine-Tuning Research (Feb 19, 2026)

**What was discovered:**
- YOLOv12 with COCO pre-training detects cars, buses, trucks, motorcycles fine
- **COCO doesn't have an "emergency vehicle" class** — no ambulance, fire truck, or police car
- This was the first concrete realization that pre-trained models alone wouldn't work for the project's goals

**Action:** Started researching transfer learning and fine-tuning approaches.

### Experiment 1b: Transfer Learning Research (Feb 21, 2026)

Researched how to fine-tune a YOLO model on custom emergency vehicle data. The conclusion was that fine-tuning needed labeled data for emergency vehicles, which led to the next experiment...

### Experiment 2: VideoMT (March 17 & 21, 2026)

**March 17 attempt:**
- Set up Meta's VideoMT (Video Multi-Task) model
- **Failed: model only runs on Linux**, not Windows
- Workspace existed at `capstone_models/VidEoMT/` with full requirements.txt (transformers, pytorch-lightning, einops, submitit, scikit-image, shapely, h5py, pycocotools)

**March 21 retry:**
- **Migrated to Linux via WSL Ubuntu** with: `sudo apt install -y build-essential git cmake ninja-build pkg-config python3-dev python3-pip libgl1 libglib2.0-0`
- Got VideoMT setup working in Linux
- **Decided not to use VideoMT** after evaluating it
  - Too complex to adapt for traffic/vehicle detection
  - Designed for general video understanding, not detection-tracking
  - Massive dependency footprint relative to value

This Linux migration ended up being important — it became the workspace for all subsequent pre-trained model experiments.

### Experiment 3: YOLOE-26L Prompt-Based Detection (March 29, 2026)

**Goal:** YOLOE is an "open-vocabulary" YOLO variant — instead of detecting fixed classes, you give it text prompts like `"car"`, `"bus"`, `"ambulance"` and it detects whatever you ask for. **No training required.**

**What was built (in `capstone_models/yoloe/`):**
- `track_cars_yoloe.py` — Full inference pipeline with:
  - Prompt-based detection via `--prompts car bus ambulance` or `--prompts-file prompts.txt`
  - ByteTrack via Ultralytics' built-in tracking
  - Line-crossing counter for traffic flow stats
  - Live preview, video file, webcam, and RTSP stream support
  - CSV export of stats
- `prompts.txt`: `car`, `bus`, `pedestrian`, `ambulance`
- `mobileclip2_b.ts` (CLIP model for text prompt encoding)
- `vidgen.mp4` (test video)
- `runs/segment/` and `runs/track/` output folders

**Why this seemed promising:**
- Zero training required — direct path to a working demo
- Could detect any class on demand by changing prompts
- Same Ultralytics tracking infrastructure (would work later anyway)
- Much faster iteration cycle than training custom models
- Already supports the line-crossing analytics the team wanted

**Outcome:** YOLOE worked, but had fundamental limitations:
- **Couldn't reliably distinguish similar classes** — police cars and regular cars looked the same to the prompt-based detector. Fire truck vs truck was hit-or-miss.
- **Detection quality varied wildly** depending on prompt wording (`"police car"` vs `"police vehicle"` gave different results)
- **No control over class boundaries** — couldn't enforce that a sedan with a light bar must be police, not just car
- For a capstone that needed measurable per-class metrics, "good enough on common classes, weak on emergency vehicles" wasn't acceptable

### Experiment 4: YOLO26 (March 29, 2026)

Empty folder created at `capstone_models/yolo26/` — abandoned before any code was written. Likely planned to test the next iteration of YOLO but the team moved on after YOLOE results were conclusive.

### The Decision: Custom Dataset + CARLA

After ~6 weeks of testing pre-trained models on real footage:

**The verdict:** Pre-trained models were good baselines but couldn't reliably handle the specific class distinctions the project needed (emergency vehicles vs regular vehicles).

**The pivot:** Generate a custom dataset where every emergency vehicle is correctly labeled, train a model on it, then it would actually know the difference.

**Why CARLA over more real video:**
- Real footage requires manual labeling (extremely time-consuming for thousands of frames)
- Hard to get balanced class distributions in real video (emergency vehicles are rare in everyday traffic)
- CARLA gives free, perfect ground truth labels via 3D-to-2D bbox projection
- Can control weather, time of day, traffic density — impossible with pre-recorded footage
- Reproducible: anyone can re-generate the dataset

**The lesson:** Pre-trained models can answer "is there a vehicle here?" but not "what specific kind?" — for fine-grained classification, you need data tailored to your task.

This decision shaped everything that followed: the entire `dataset_capstone` repo exists because of this conclusion.

---

## Phase 0: Original Capstone_Work Repository (February — March 2026)

> *This phase pre-dates the current repo. Reconstructed from the original `Capstone_Work` repository (now deleted from GitHub but recovered from a private clone).*

### February 11, 2026 — Project Bootstrap
**Commits:** `1480008` clean installation of carla, `e17c4c3`/`b4c5d39`/`9907c36` README iterations

**Goal:** Get CARLA installed and document the install procedure.

**What was done:**
- Set up CARLA installation
- Set up conda environment
- Set up git and GitHub
- Initial repo structure with `capstone_sim/` folder
- `.gitignore` to exclude the 20GB CARLA simulator from version control
- `environment.yml` for conda dependencies
- README documenting how to install CARLA 0.9.16 and place it at `capstone_sim/CARLA_0.9.16/`

**Why these decisions:**
- CARLA simulator is too large for git → keep it local-only via gitignore
- Conda environment over plain pip → CARLA's Python API has specific version requirements and conda handles them more reliably

### February 12, 2026 — First Test Script

**What was done:**
- Created new branch for test sim (`testing_sim`)
- Installed Jupyter in conda environment
- Created first test script (basic CARLA connectivity check)

### February 15, 2026 — Custom Scenarios Tool

**What was done:**
- Iterated on the test script
- **Built the custom scenarios creation tool** — this became `setup_scenario.py`, the foundational interactive tool that survived through all later phases

### February 16, 2026 — Scenario System V1 Pushed to GitHub (`testing_sim` branch)
**Commit:** `2ef2822` — built scenario creator system, pre-made scenarios need to have positions and lightpoles set

This was the GitHub push of the work built over Feb 12-15.

**This was a massive 1,573-line addition** introducing the foundational architecture that survived (in heavily modified form) all the way to V2.

**What was built:**
- `setup_scenario.py` (161 lines) — Interactive tool: position spectator camera, pick a traffic light, save config to YAML
- `run_scenario.py` (298 lines) — **Original capture script** (later evolved into `capture_dataset.py`)
- `switch_map.py` (3 lines) — Quick-load a CARLA map by name
- `test_1.py` (185 lines) — Early standalone test demonstrating two cars stopping at a red light
- 7 pre-made Town05 scenarios (basic_day, heavy_traffic, morning_rush, evening_sunset, rainy_day, night_low_traffic, single_lane_simple)
- `scenario_config_example.yaml` documenting all available config fields
- `test_all_town05.bat` to batch-run all 7 scenarios
- `launch_carla_quality.bat` — Launch CARLA with `-quality-level=Epic` to fix "blob" rendering of vehicles
- `README_SCENARIO_SYSTEM.md` (236 lines) and `QUICK_REFERENCE.md` (95 lines) — extensive docs

**Key design choices that survived:**
- YAML-driven scenarios (still the foundation today)
- Camera-then-traffic-light selection flow in `setup_scenario.py` (still the same pattern)
- Spectator-based camera positioning (same UX today)
- Per-lane vehicle spawning at distances behind stop lines

**Challenge encountered: Vehicle quality looked terrible**
- **Symptom:** Cars rendered as low-poly "blobs" in CARLA
- **Cause:** CARLA defaults to low quality settings to maintain framerate
- **Solution:** `launch_carla_quality.bat` to launch with `-quality-level=Epic -ResX=1920 -ResY=1080`

### February 19, 2026 — YOLOv12 Tracking Pipeline (`models` branch)
**Commit:** `893d38e` — setup yolo12 tracking model

Set up YOLOv12, added tracking options, then began researching fine-tuning when realizing COCO doesn't include emergency vehicles (this realization led to the parallel pre-trained model experimentation in Phase -1).

**Goal:** Build a separate, more general-purpose tracking system for testing different MOT algorithms on intersection footage.

**What was built (in `capstone_models/yolov12/`):**
- `intersection_tracking.py` (472 lines) — Standalone tracker supporting **4 different MOT methods**:
  - `bytetrack` (default)
  - `botsort` (better re-identification in crowded scenes)
  - `centroid` (lightweight fallback, no external MOT backend — implemented from scratch using `CentroidTracker` class)
  - `gst_nvtracker` (NVIDIA DeepStream integration for production deployment)
- `README_TRACKING.md` (98 lines) — Documentation of all 4 tracking methods and when to use each
- `tracking_config_example.yaml` — Configurable tracking pipeline with:
  - Polygon-defined intersection zones (image-space coordinates)
  - Counting line for traffic flow analysis
  - Speed calibration via `meters_per_pixel`
  - Per-class filtering (`vehicle_classes: [car, truck, bus, motorcycle]`)
- `yolo12n.pt` — Pre-trained YOLOv12 nano weights
- `requirements.txt` — Tracking dependencies

**Key features built but later not carried forward:**
- **Speed estimation** via pixel displacement × meters_per_pixel
- **Line crossing counter** for measuring traffic flow
- **Intersection polygon** to filter detections inside a specific area
- **Per-track and scene-level metrics** (vehicle count over time, unique vehicles, line crossings)

**Why these were built:** The team initially planned to do real-time analytics (speed, flow rates) on top of detection. Later phases focused on getting detection itself working better first — these analytics could be re-introduced as post-processing in the future.

**Why YOLOv12 specifically:** Newer architecture than YOLOv8/v11 at the time. Has the same Ultralytics API so any model could be swapped in.

### March 12, 2026 — Training Pipeline Setup
**Commits:** `a0e6aa8` yolov12 training code, `01430cb` basic scenario, `198cf61`/`3ab6104` PR merges

**What was added (`capstone_models/yolov12/training/`):**
- `training_code.py` (131 lines) — YOLO training entry point
- `train_config.yaml` (43 lines) — Hyperparameters (epochs, batch, imgsz, device, etc.)
- `data_template.yaml` (11 lines) — Dataset definition template
- `extract_frames.py` (120 lines) — **Utility to extract frames from MP4 videos** for labeling
  - Supported `--target-fps`, `--max-frames`, `--jpeg-quality`, `--recursive`, `--overwrite`
- `requirements.txt` (3 lines) — Minimal training deps
- `README.md` (99 lines) — Full training workflow doc

**Key design choices:**
- Config-file-driven training (not pure CLI args) — easier to reproduce experiments
- Documented frame extraction utility — anticipated using real video as supplementary training data, not just CARLA captures
- Separate `dataset_root` from training code — keeps datasets out of the repo

### March 30, 2026 — Last Activity Before Migration
**Commits:** `04ddac8`/`efe743e` "claude" (cleanup commits)

**What happened:** The team began transitioning to a new repo structure. Two commits with placeholder names cleaned up the worktree before the repo was eventually deleted.

### Why the Repo Was Replaced

The original `Capstone_Work` repo had **two separate parallel architectures**:
- `capstone_sim/Code/` — Run scenarios and visualize them (no automatic data capture for training)
- `capstone_models/yolov12/` — Tracking and training, using external video files

These didn't connect. To train YOLO, you'd have to run a scenario, screen-record CARLA, extract frames with `extract_frames.py`, manually label them, then train. **No automated dataset capture pipeline existed.**

The new `dataset_capstone` repo was started in late March 2026 to fix this gap by building automatic YOLO-format dataset generation directly from CARLA scenarios (which became `capture_dataset.py`).

### What Was Carried Forward
From the original repo into `dataset_capstone`:
- `setup_scenario.py` interactive flow (camera + traffic light selection) — kept
- YAML-driven scenarios — kept (with major schema expansion)
- `switch_map.py` and `launch_carla_quality.bat` — kept
- `run_scenario.py` lane-based vehicle spawning logic — became the basis for `capture_dataset.py`
- 7 Town05 scenario configs — superseded by new multi-camera, weather-cycling configs

### What Was Abandoned
- Multi-method tracking (`botsort`, `centroid`, `gst_nvtracker`) — focused on just `bytetrack` for simplicity
- Speed estimation, line crossing, intersection polygons — deferred as post-processing features (still possible to add later)
- YOLOv12 → switched to YOLOv11 (more stable, better Ultralytics support at the time)
- `extract_frames.py` — no longer needed once `capture_dataset.py` generated YOLO-format data directly

---

## Phase 1: Initial Setup & Cleanup (March 31, 2026)

### Starting State
- Prototype scripts dumped in `capstone_sim/Code/`
- Single-camera, single-weather scenarios only
- Corrupted `environment.yml` (UTF-16 encoding artifacts)
- `.claude/` folder accidentally tracked in git
- Several test/scratch files mixed in with production code
- Stray `scenario_config.yaml` at repo root

### Actions Taken
1. **Cleaned up the repository** — Removed unused files: `test_1.py`, `better_test_system.py`, `QUICK_REFERENCE.md`, `run_scenario.py`, `launch_carla_quality.bat`
2. **Fixed environment.yml** — Replaced corrupted file with a clean conda environment specification
3. **Added .gitignore** — Excluded `.claude/`, `dataset_output/`, `CARLA_0.9.16/`, training runs
4. **Rewrote README** — Documented full setup procedure (CARLA install, conda env, usage)
5. **Made setup_scenario.py output configs to a sensible location** (next to the script)

### Why
The repo was mixing experimental scratch work with the actual workflow. A clean foundation was needed before adding real features.

---

## Phase 2: First Bug Fixes & Scenarios (April 3-4, 2026)

### Challenges & Solutions

**Challenge 1: Camera spawn returned `(0, 0, 0)` for `camera_location`**
- **Symptom:** Vehicles spawned correctly near the traffic light but `MAX_DISTANCE` filter rejected all of them because the camera was thought to be at origin
- **Cause:** In CARLA's synchronous mode, actor transforms aren't updated until `world.tick()` is called
- **Fix:** Added a `world.tick()` immediately after spawning the camera and before reading its location

**Challenge 2: Vehicles getting filtered out at edge of frame**
- **Symptom:** Trained model would "freak out" at vehicles partially off-screen
- **Cause:** `MIN_VISIBILITY = 0.4` threshold was dropping labels for vehicles 60% off-screen — model never learned what partially-visible vehicles looked like
- **Fix:** Lowered to `MIN_VISIBILITY = 0.15` so the model learned to detect partially visible vehicles

**Challenge 3: Spawn point selection was inflexible**
- **Symptom:** Vehicles spawning all over the map, sometimes nowhere near the camera
- **Solution:** Added two improvements:
  - `spawn_radius` filter — only use spawn points within N meters of camera/traffic light
  - `spawn_points` config field — manually specify spawn point indices (with range syntax: `["33-81"]`)
  - `radius_center` choice between `traffic_light` (default) or `camera`

**Other improvements:**
- Added `visualize_spawns.py` to draw numbered markers at all spawn points in CARLA
- Created multiple scenario configs across maps (Town01, Town03, Town06, Town10HD)

### Why These Decisions
The model only learns from what it sees. Edge-of-frame vehicles, missing classes, and stuck-in-corner spawning all degraded what we could teach it.

---

## Phase 3: YOLO Model Training (April 11, 2026)

### Actions Taken
1. **Added `train.py`** — Wrapper around Ultralytics YOLO API with sensible defaults
2. **Set up GPU training pipeline** — Initial training of yolo11n on captured dataset
3. **Documented training requirements** — Added `requirements.txt` with `ultralytics`, `torch`

### Challenges & Solutions

**Challenge: PyTorch installed without CUDA**
- **Symptom:** Training fell back to CPU, painfully slow
- **Cause:** Default `pip install torch` installs CPU-only version
- **Fix:** Documented the correct install: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`

**Challenge: VRAM ran out with yolo11m**
- **Cause:** RTX 4060 Laptop has only 8GB VRAM; yolo11m at batch 16 exceeded that
- **Fix:** Reduced batch size to 8 for medium model, kept 16 for nano

### Why
Training locally was the natural starting point. We discovered the hardware limitations of an RTX 4060 Laptop and made decisions accordingly (smaller models / smaller batches).

---

## Phase 4: Major Reorganization (April 12, 2026) — V2

### Why a Restructure Was Needed
After ~2 weeks of feature additions, the codebase had become messy:
- All scripts in one `Code/` folder
- YAMLs mixed with Python files
- Constants and helper functions duplicated across files
- `record_test.py` heavily depended on `capture_dataset.py` via fragile direct imports

### New Folder Structure
```
capstone_sim/
├── configs/                 # all YAML scenario configs
├── scripts/
│   ├── capture/            # capture_dataset.py, record_test.py, setup_scenario.py, batch_capture.py
│   ├── train/              # train.py
│   ├── evaluate/           # evaluate_model.py, visualize_metrics.py, etc.
│   └── utils/              # constants.py, bbox.py, carla_helpers.py, switch_map.py, visualize_spawns.py
├── models/
│   └── yolov11m/           # trained weights
└── environment.yml
```

### Key Refactor Decisions
1. **Created shared modules** in `scripts/utils/`:
   - `constants.py` — `CLASS_NAMES`, `BLUEPRINT_TO_CLASS`, `MAX_DISTANCE`, color palettes
   - `bbox.py` — projection matrix, 2D bbox computation, YOLO format conversion
   - `carla_helpers.py` — camera spawning, weather, vehicle spawning/despawning

2. **`sys.path.insert(0, ...)` pattern** — Every script that needs cross-module imports adds the project root to sys.path. Allows running scripts directly with `python capstone_sim/scripts/capture/capture_dataset.py`.

3. **Auto map switching** — Configs can include a `map` field; capture script switches automatically. Made batch capture possible.

### Why Not a Python Package?
A proper installable package (with `setup.py`) would be cleaner but adds complexity for a capstone project. The sys.path approach is pragmatic and works.

---

## Phase 5: Major Feature Additions (April 13, 2026)

This was a massive expansion adding multi-camera capture, weather cycling, batch processing, evaluation tools, and the test footage workflow.

### Multi-Camera Support

**Why:** A single camera angle limits dataset diversity. Real CCTV systems often have multiple viewpoints per intersection.

**Implementation:**
- `setup_scenario.py` now loops asking "Add another camera? (y/n)"
- YAML stores `cameras:` list (backwards-compatible with old `camera:` single field)
- `capture_dataset.py` and `record_test.py` spawn multiple cameras with separate image queues
- Each camera produces independently labeled frames (frame names include `cam0_`, `cam1_`, etc.)

**Decision: 2-3 cameras is the sweet spot.** 4+ cameras hits diminishing returns and slows CARLA significantly.

### Weather Cycling

**Why:** Real-world cameras face many weather conditions (day, night, rain, fog). Training on one condition produces a brittle model.

**Implementation:** Added `weather_cycle:` list of presets to the config. Capture script splits total frames evenly across presets, switching weather mid-scenario.

**Decision: 13 weather presets** covering day, night, sunrise, sunset, rain, fog, overcast, post-rain wet roads. Each gets ~540 frames out of 7,000.

**Decision: Don't cycle weather in test recordings.** ByteTrack metrics need consistent conditions to be meaningful — sudden weather changes would tank MOTA artificially.

### Batch Capture

**Why:** Running 5+ scenarios manually is tedious. Wanted to queue them up overnight.

**Implementation:** `batch_capture.py` accepts a folder of YAMLs and runs each sequentially. Combined with auto map switching, no manual intervention needed.

### Test Recording & Evaluation Pipeline

**Why:** Need a way to measure how well the trained model performs.

**Implementation:**
- **`record_test.py`** — Records sequential frames + ground truth labels with CARLA actor IDs (for tracking metrics)
- **`evaluate_model.py`** — Runs model + ByteTrack on recorded footage, computes:
  - Detection metrics: precision, recall per class
  - Tracking metrics: MOTA, IDF1, ID switches
  - GPU performance: FPS, latency, memory, power
- **`visualize_metrics.py`** — Generates charts (per-class bar chart, confusion matrix heatmap, per-frame line plots)
- **`compare_models.py`** — Side-by-side video comparison + speed/accuracy tradeoff chart
- **`analyze_dataset.py`** — Class distribution, imbalance warnings, bbox size analysis
- **`generate_report.py`** — Auto-generated HTML report for presentation

### Challenges & Solutions

**Challenge 1: Ctrl+C during capture left orphaned partial datasets**
- **Fix:** On `KeyboardInterrupt`, the script now identifies and deletes only the frames captured during this run (matched by `scenario_name` prefix)

**Challenge 2: Recording metadata wasn't saved when CARLA cleanup failed**
- **Fix:** Moved `recording_meta.yaml` save inside the `finally` block, **before** the cleanup operations that might error

**Challenge 3: `evaluate_model.py` was running but the model was on CPU**
- **Cause:** Same PyTorch CPU vs CUDA issue
- **Fix:** Documented the install order; added explicit GPU check

### Why These Tools
A capstone project needs to demonstrate **measurable results**, not just "I trained a model." The evaluation pipeline produces metrics that go directly into the presentation.

---

## Phase 6: Iteration on Data Quality (Late April / Early May 2026)

This was a long phase of trial-and-error with capture parameters.

### Key Discoveries

**Discovery 1: Empty/background frames help training**
- Problem: Vehicles all start spawned. Model never sees empty roads.
- Solution: Capture warmup frames as background images (no vehicles), giving model negative examples

**Discovery 2: Capture FPS choice depends on use case**
- Original plan: 2 FPS (max diversity per frame)
- Revised: 10 FPS (smoother sequences for ByteTrack evaluation)
- Final decision: 2 FPS for training (`capture_interval: 10`) — more diverse frames matter more than tracking smoothness during training. Tracking footage uses `record_test.py` separately at 20 FPS.

**Discovery 3: Force respawn prevents stuck-vehicle frames**
- Problem: With long captures, some vehicles get stuck (CARLA pathfinding fails for buses on tight corners). They produce identical frames forever.
- Solution: Added `force_respawn_interval` — every N frames, destroy ALL vehicles and respawn fresh ones

**Discovery 4: Weather changes need vehicle respawns too**
- Problem: When weather switched mid-scenario, the same stuck vehicles stayed stuck under the new weather, producing visually similar frames despite the changed conditions
- Solution: Weather cycling now triggers a full respawn

**Discovery 5: Parked vehicles in CARLA confused the model**
- Problem: Static parked cars on the side of the road produced ground truth labels but the model couldn't distinguish "always parked" from "stopped at red light"
- Solution: `world.unload_map_layer(carla.MapLayer.ParkedVehicles)` removes them entirely
- Caveat: Only works on `_Opt` versions of maps. For non-Opt maps, parked vehicles are baked into the geometry.

**Discovery 6: Class imbalance was hurting the model**
- First eval showed: police cars had 608 false positives with 0 real police cars in test scene; bus recall was 7.9%
- Solution: Rebalanced spawn ratios — boosted bus, police_car, fire_truck, bike; reduced car
- Result after retraining: bus recall jumped from 7.9% → 96%

### Default Parameters (Final)
After many iterations, settled on:
```yaml
simulation:
  total_frames: 7000      # ~540 per weather preset
  capture_interval: 10    # 2 FPS — maximum diversity
  warmup_frames: 60       # 3 sec of background images
  train_ratio: 0.8

spawn:
  max_vehicles: 25
  spawn_radius: 80.0
  respawn_interval: 40
  despawn_distance: 100.0
  force_respawn_interval: 200   # full reset every 10 sim seconds
  ratios:                       # rebalanced for rare classes
    car: 10, ambulance: 3, bus: 4, truck: 3
    police_car: 4, fire_truck: 2, bike: 5

traffic_light:
  red_time: 3.0      # short red — vehicles keep flowing
  green_time: 15.0   # long green — more throughput
  yellow_time: 2.0
```

---

## Phase 7: Cloud Training (May 2-3, 2026)

### Why Cloud
- Initial yolo11m training on RTX 4060 Laptop took 1+ hour for 2 epochs (later crashed with OOM)
- Estimated 25+ hours for 100 epochs locally
- Cloud GPUs (vast.ai) cost ~$0.50/hr — total of $1-2 for full training

### Provider: vast.ai with NVIDIA L40S 48GB
- 8x faster than RTX 4060 Laptop
- 48GB VRAM allows batch 96
- Cost: $0.47/hr

### Workflow Established
1. Zip dataset locally → upload to Google Drive
2. Use vast.ai's built-in Drive sync to pull dataset to instance
3. SSH in, install ultralytics, configure data.yaml path
4. Run training with `cache=ram` (huge speedup with 96GB available RAM)
5. Download `best.pt` via SCP
6. Destroy instance to stop billing

### Challenges & Solutions

**Challenge 1: Slow upload from home internet**
- 13GB dataset at 2.3 MB/s = 1.5 hours upload while paying for GPU
- **Fix:** Upload to Google Drive first (background), then sync from Drive to vast.ai (datacenter-to-datacenter is fast)

**Challenge 2: 16GB instance disk too small**
- Default vast.ai disk allocation was insufficient
- **Fix:** When renting, slide disk allocation to 50GB+

**Challenge 3: Ultralytics looking for dataset in wrong path**
- Ultralytics has a default `datasets_dir` setting that prepends to your data.yaml path
- **Fix:** `yolo settings datasets_dir=/` before training

**Challenge 4: VRAM OOM at epoch 17 with batch 96**
- Even 48GB filled up sometimes
- **Fix:** Resumed training with `batch=64` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

**Challenge 5: Training didn't resume because of missing settings**
- New ultralytics settings file was created on the instance, losing the `datasets_dir` setting
- **Fix:** Re-set the setting after the crash, then resume

### Final Training Results
- **39 epochs completed** (early stopping triggered)
- **Total time:** 1.3 hours on L40S
- **Final metrics:**
  - mAP50: **0.949**
  - mAP50-95: **0.865**
  - Precision: **0.97**
  - Recall: **0.897**
- **Per-class** all >0.87 mAP50, weakest was bike (0.87)

---

## Phase 8: Real-World Inference & V2.1 (May 3, 2026)

### Why
A trained model is useless if it can't be deployed. Needed a way to run on actual video files.

### Implementation
- **`inference.py`** — Takes any MP4 (or webcam) and runs YOLO + ByteTrack
- Output: annotated video + live FPS stats
- Performance: **27.7 FPS at 36ms latency** on RTX 4060 Laptop — fast enough for near-real-time CCTV processing

### Other V2.1 Additions
- **`create_spawn_points.py`** — Interactive tool to fly the spectator and capture road locations as custom spawn points
- **`visualize_traffic_lights.py`** — Draw numbered ID markers on all traffic lights
- **Custom spawn points config support** — `custom_spawn_points: [...]` in YAML
- **Traffic light timing control** — `red_time`, `green_time`, `yellow_time` per scenario
- **8 new multi-camera scenarios**

---

## Phase 9: Continued Training Attempt & Stuck Vehicle Detection (May 11, 2026) — V2.2

### Continued Training Attempt (didn't go well)
Tried to fine-tune `best.pt` further on the cloud (vast.ai L40S):

**Plan:** Continue training from existing `best.pt` with `imgsz=960` for better small object detection (cars/bikes were the weakest classes).

**What happened:**
- Ultralytics' auto-batch dropped from 64 → 32 due to higher VRAM use at imgsz=960
- `optimizer=auto` ignored the explicit `lr0=0.001`, falling back to `lr=0.01` (way too high for fine-tuning)
- Risk of unlearning the pre-trained weights was real
- **Decided to stop and accept the existing `best.pt`** rather than risk degrading the model

**Lessons for capstone:**
- Fine-tuning from a strong checkpoint requires explicit `optimizer=AdamW` + low `lr0=0.0001` + zero warmup
- Ultralytics' "auto" settings override what you pass — always be explicit
- A `best.pt` that's already at 0.95 mAP50 has very little room to improve via continued training; data improvements would be more effective

### New Feature: Stuck Vehicle Detection
**Why:** Even with `force_respawn_interval`, vehicles in CARLA could deadlock (a bus stopping a turning car) and produce many useless frames before the next forced respawn. Worse, in test recordings (long durations), a single deadlock could waste minutes of footage.

**Implementation in both `capture_dataset.py` and `record_test.py`:**
- New config fields:
  ```yaml
  spawn:
    stuck_check_interval: 600   # check every N frames (30 sec at 0.05s ticks)
    stuck_threshold_m: 1.0      # if vehicle moved < N meters since last check, kill it
  ```
- Tracks each vehicle's position between checks
- Kills any vehicle that didn't move at least `stuck_threshold_m` since last check
- Respawns to maintain target counts

**Iterations on the design:**
1. **First attempt: position-based** — Kept track of position over interval; failed because deadlocks weren't being caught aggressively enough
2. **Second attempt: velocity-based** — Counted consecutive frames with speed < 0.5 m/s; produced false positives (red light waits)
3. **Final: back to position-based with adjustable threshold** — Default `stuck_check_interval: 600` (30s) is lenient enough for traffic light queues but catches genuine deadlocks. User can tune for their use case.

**For queue length analytics:** The 30-second default ensures vehicles waiting at red lights don't get killed prematurely, preserving natural queue formation in the recording.

### Other V2.2 Additions
- **`frames_to_video.py`** — Quick utility to convert PNG frame sequences from `record_test.py` output into MP4 videos. Supports per-camera filtering for multi-camera recordings.
- **2 new scenario configs** — `Town04_1cam.yaml`, `Town6_1cam_test.yaml` for additional test footage

---

## Phase 10: Traffic Analytics System — V3.0 (May 12, 2026)

> *Beyond detection metrics: real-world traffic measurements built on top of the trained model.*

### Why
Detection metrics (precision, recall, MOTA) measure how well the model *sees* — but a real deployment needs **traffic analytics**: how fast are vehicles going? How many are queued? When the original capstone scope mentioned speed estimation and queue length, these became the demo features that show the system as a complete traffic monitoring product, not just a model.

### Architecture
A new `capstone_sim/scripts/analytics/` folder was added:
- `setup_analytics.py` — One-time configuration: camera calibration + lane definition
- `traffic_analytics.py` — Run analytics on a recorded video / frame sequence
- `live_analytics.py` — Same analytics live during a running CARLA simulation

Configurations saved per-scenario to `capstone_sim/analytics_configs/<scenario>.yaml`. Run outputs (CSVs + video) saved to `capstone_sim/analytics_runs/<scenario>_<timestamp>/`.

### Calibration

The hardest part of building real-world analytics: knowing what a "pixel" means in real meters. Two paths:

**For CARLA recordings (fully automatic):**
- Camera intrinsics + extrinsics are known exactly
- 4 sample pixels in the lower part of the image are projected onto the ground plane via ray-camera math
- `cv2.findHomography` builds the 3x3 image→world matrix
- **Zero user input needed** — happens automatically at end of `record_test.py`

**For real video (manual 4-point homography):**
- User clicks 4 corners of a known rectangle on the ground (e.g., lane markings, parking spot)
- Types the real-world dimensions (e.g., 3.5m wide × 12m long)
- Same homography computation
- This is the **industry standard** approach used by commercial traffic systems

### Speed Per Vehicle
- For each tracked vehicle, take a reference point on the bounding box (at 50% width, 85% height — slightly above the very bottom to avoid bumper artifacts)
- Apply homography → world (X, Y) in meters
- Speed = displacement / frame interval, smoothed over a 5-frame rolling average
- Color-coded labels in the video: gray (stopped), green (slow), yellow (normal), red (fast)

### Per-Lane Queue Counts

Interactive lane definition: click polygon corners around each lane, give it an ID. Saved to YAML.

Per-frame logic:
1. For each detection: is its speed below `speed_threshold_kmh` AND has it been below for `min_stationary_seconds`? (Both configurable in YAML)
2. If yes: which lane polygon is it in?
3. Increment that lane's queue count

Configurable thresholds in `analytics_config.yaml`:
```yaml
queue:
  speed_threshold_kmh: 7.2      # below this = "slow"
  min_stationary_seconds: 2.0   # must be slow this long to count as queued
```

The 2-second threshold prevents false positives from vehicles briefly slowing down (e.g., turning).

### Key Design Decisions

**Camera reference point for ground projection**
- Started with bbox bottom-center → gave incorrect lane assignment for large vehicles like fire trucks (the protruding bumper hit the ground at a different point than the truck's actual center)
- Final: 50% horizontal × 85% vertical of bbox — closer to the middle of the vehicle body, more stable for lane assignment

**Multi-pass calibration math**
- First attempt: manual rotation matrix using CARLA's pitch/yaw/roll — failed because CARLA uses non-standard rotation conventions (positive pitch = nose up)
- Fix: use `carla.Transform.get_forward_vector()` etc. directly — CARLA's API handles its own conventions correctly

**Per-scenario configs in dedicated folder**
- Originally tried saving `analytics_config.yaml` next to the scenario YAML — mixed configs with scenarios
- Final: separate `analytics_configs/` folder; each file named to match the scenario's stem (`Town6_1cam.yaml`)

**Two interfaces: live + recorded**
- `live_analytics.py` works on a running CARLA simulation — spawns traffic from scenario, computes analytics in real time, includes the full spawn lifecycle (respawn, despawn, stuck detection) from `record_test.py`
- `traffic_analytics.py` works on any recorded video or frame sequence — same outputs, different input

### Output Format

Each analytics run produces:
- **`live_analytics.mp4` / `analytics.mp4`** — Annotated video with speed labels and per-lane queue overlays
- **`per_track.csv`** — Every detection at every frame: `frame, track_id, class, world_x, world_y, speed_mps, speed_kmh`
- **`per_lane_queue.csv`** — `frame, lane_id_1_count, lane_id_2_count, ...`
- **`summary.json`** — Final stats: avg/max speed, max queue per lane, unique tracks, FPS

### Industry Comparison
The 4-point homography calibration is the same method used by commercial traffic systems like:
- Iteris VantageNext (lane occupancy + queue length)
- Econolite SPM (traffic light scheduling based on queue)
- Cisco Meraki MV traffic analytics

External calibrations (e.g., from camera mount specs or GPS drives) can be plugged in by writing the same homography matrix to `analytics_config.yaml`.

### Challenges & Solutions

**Challenge 1: CARLA's non-standard rotation conventions**
- Manual rotation math produced wrong-direction rays that didn't hit the ground plane
- Fix: use `carla.Transform.get_forward_vector()` directly

**Challenge 2: numpy types in YAML output**
- Initial dump produced `!!python/object/apply:numpy.core.multiarray.scalar` tags that couldn't be parsed back
- Fix: cast all numpy values to native Python `int`/`float` before serialization

**Challenge 3: Live mode wasn't following scenario rules**
- Initial `live_analytics.py` only did one spawn at startup, missing respawn/despawn/stuck behavior
- Fix: ported the full spawn lifecycle from `record_test.py`

**Challenge 4: Lane assignment was off for large vehicles**
- Bbox bottom-center put fire trucks in the wrong lane (front bumper protrudes left)
- Fix: shifted reference point to 50% × 85% of bbox

---

## Phase 11: Light State & Red-Light Violation Detection — V3.0 (June 2026)

> *Building on the analytics layer: traffic light state plumbing and the first violation detector.*

### Why
The original scope listed red-light violation detection and highway entry counting. Both depend on knowing the traffic light state per frame. The team has explicit permission to read light state directly from the CARLA simulator (treated as ground-truth signal infrastructure, not inferred from camera vision), which makes this tractable.

### Light State Plumbing
New shared module `scripts/utils/light_state.py` with a `LightStateProvider` that unifies three input modes behind one `state_at(frame_idx)` interface:
- **Live CARLA** — wraps a `carla.TrafficLight` actor, reads state on demand
- **Recorded** — reads a `light_states.csv` (frame, state) that `record_test.py` now logs every frame
- **Real video** — reads a manual `light_schedule:` in `analytics_config.yaml`

`record_test.py` now logs per-frame light state, and `traffic_analytics.py` / `live_analytics.py` draw a colored LIGHT indicator overlay.

**Robustness fix:** CARLA traffic light actor IDs are not stable across sessions. `live_analytics.py` tries the stored id first, then falls back to the traffic light nearest the camera. (The user also found their config simply had a stale id; the fallback remains as a safety net.)

### Forbidden Line Picker
Added Step 3 to `setup_analytics.py` — a 2-point line picker (reusing the calibration `PointPicker`) to define stop lines. Saved as `forbidden_lines: [{id, points}]`. `--redo-lines` flag wipes and redefines.

### Red-Light Violation Detection
`ViolationDetector` (in `traffic_analytics.py`, used by both analytics scripts):
- Tracks each vehicle's signed side of each line segment via cross product
- A violation fires when a vehicle's ground reference point flips sides AND the crossing is within the segment span (projection parameter in [-0.1, 1.1]) AND the light is red
- Each (track_id, line_id) is flagged once to avoid double-counting
- Logged to `violations.csv` (frame, track_id, line_id, light_state); count added to `summary.json` and the HUD
- Geometry was unit-tested before integration (crossing on red fires, on green doesn't, off-segment ignored, no double-count)

### `k` Toggle for Demos
CARLA autopilot vehicles obey traffic lights by default, so a normal run shows zero violations. Added a live `k` keypress in `live_analytics.py` that toggles all vehicles between ignoring and obeying red lights (via `tm.ignore_lights_percentage`), with newly-spawned vehicles inheriting the state. This makes the violation feature demonstrable on demand.

### Challenges & Solutions

**Challenge 5: `summary.json` lost on exit**
- The summary was written *after* the try/finally; when CARLA cleanup in `finally` threw (simulator state changed on quit), the summary write was skipped entirely
- Fix: moved the summary write inside `finally`, before the CARLA cleanup, wrapped in its own try/except — so the run stats are saved regardless of how the run ends

---

## Phase 12: Highway Entry Counting — V3.0 (June 2026)

> *The last of the original analytics deliverables, reusing the light state and polygon infrastructure already in place.*

### Why
Highway on-ramps are often metered by traffic lights. The deliverable: count how many vehicles enter each ramp/entry zone, broken down by the light state at the moment of entry — directly useful for ramp-metering analytics and for spotting vehicles entering on red.

### Implementation
Because light state (`LightStateProvider`) and polygon definition (`PolygonPicker`, used for lanes) already existed, this was mostly composition:
- `setup_analytics.py` — Step 4 adds an entry-zone polygon picker (`--redo-entry-zones`), saved as `entry_zones: [{id, polygon}]`
- `traffic_analytics.py` — `EntryCounter` counts each unique vehicle once per zone, on the frame it first transitions outside→inside, recording the light state at entry. Logs `entries.csv` (frame, track_id, zone_id, light_state); per-zone totals broken down by light go to `summary.json` and the overlay/HUD
- `live_analytics.py` — same, live

### Key Design Decisions
- **Count once per (track_id, zone)** — a vehicle entering is counted a single time even if detection jitter or re-entry occurs at the boundary, avoiding inflated counts
- **Entry edge detection** — outside→inside transition (not just "currently inside"), so a vehicle already in the zone at spawn isn't miscounted later
- **Grouped by light state** — `{zone: {total, by_light: {green, red, ...}}}` so "entered on red" is directly visible (the ramp-metering violation signal)

Geometry was unit-tested before integration: entry on green and red recorded correctly, no double-count on re-entry, summary totals accurate.

### Status
This completes all of the originally-scoped analytics features. Remaining: collision detection (stretch goal, may be skipped).

---

## Phase 13: Collision Detection (stretch goal) — V3.0 (June 2026)

> *The final stretch-goal feature. Heuristic, opt-in, and designed to resist the perspective false positives that plague camera-only collision detection.*

### Why
Camera-only collision detection is hard: perspective makes vehicles that are far apart in 3D appear to overlap in 2D, so naive bbox-overlap detection produces many false positives. The team treated this as an experimental stretch goal — useful if it works, acceptable to ship as opt-in.

### Implementation
`CollisionDetector` (in `traffic_analytics.py`, used by both analytics scripts) flags a pair of vehicles only when **all three** signals agree:
1. **Bbox overlap** — IoU ≥ threshold (they visually touch)
2. **World-space proximity** — ground-plane distance (via homography) within a few meters; this is the key guard against perspective false positives
3. **Sudden speed drop** — at least one vehicle decelerates sharply within a short window (a real impact causes abrupt deceleration)

Each unordered pair is flagged once. Opt-in via `--collisions`; thresholds tunable under `collision:` in the config. Logs `collisions.csv` (frame, track_a, track_b, world_dist_m); count in `summary.json` + HUD; involved vehicles drawn with red boxes + "COLLISION" marker.

### Key Design Decision
Using **world distance as a hard gate** (not just bbox IoU) is what makes this usable. Unit testing confirmed the guard: two boxes that overlap in the image but are 50m apart in world space are correctly ignored, while a genuine overlap + proximity + speed-drop is flagged.

### Status
All originally-scoped features plus the collision stretch goal are now complete. V3.0 delivers the full analytics suite: speed, per-lane queue, red-light violations, highway entry counting, and (experimental) collision detection.

---

## Final Test Results

Tested final model on `town02_test` recording (4,979 frames):

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Precision | 0.952 | 95% of predictions correct |
| Recall | 0.823 | Catches 82% of all vehicles |
| MOTA | 0.780 | Strong multi-object tracking accuracy |
| IDF1 | 0.882 | 88% of detections have correct identity |
| ID Switches | 47 | Down from 88 in earlier model |
| FPS | 27.7 | Near real-time on RTX 4060 Laptop |

**Per-class:** All classes >0.87 F1 except bikes (0.795)

---

## Key Lessons Learned

1. **Synchronous CARLA mode requires explicit `world.tick()`** before reading actor state.

2. **Class imbalance kills minority class performance.** Boosting spawn ratios for rare classes (police car, bus, fire truck) was more impactful than any other change.

3. **Edge-of-screen labels matter.** Lowering `MIN_VISIBILITY` from 0.4 to 0.15 fixed model "freaking out" at frame edges.

4. **Cloud GPUs are worth the $1-2** for serious training. Saved many hours of waiting on a laptop.

5. **Two FPS is enough for training; 20 FPS is needed for tracking eval.** These are different problems requiring different captures.

6. **Force respawn beats waiting for vehicles to unstick.** Don't trust CARLA pathfinding to recover from edge cases.

7. **Use `_Opt` map variants** — Same visual quality, but with toggleable layers (lets you remove parked vehicles).

8. **Auto-validation after capture catches problems early.** Running `analyze_dataset.py` immediately flags class imbalances and empty-frame issues before wasting hours on training.

9. **Track everything in evaluation output.** Confusion matrices, per-frame metrics, GPU performance — all useful for the capstone presentation.

10. **HTML reports beat slide-by-slide manual creation.** `generate_report.py` produces a self-contained presentation-ready document automatically.

---

## Final Project Structure

```
dataset_capstone/
├── README.md
├── PROJECT_HISTORY.md             # this file
├── .gitignore
└── capstone_sim/
    ├── environment.yml
    ├── configs/                   # 16 scenario YAMLs
    ├── models/yolov11m/
    │   └── best.pt                # final trained model
    ├── analytics_configs/         # per-scenario analytics setups (V3.0)
    ├── analytics_runs/            # gitignored — per-run output folders
    └── scripts/
        ├── capture/
        │   ├── capture_dataset.py
        │   ├── record_test.py
        │   ├── setup_scenario.py
        │   └── batch_capture.py
        ├── train/
        │   └── train.py
        ├── evaluate/
        │   ├── evaluate_model.py
        │   ├── visualize_metrics.py
        │   ├── compare_models.py
        │   ├── analyze_dataset.py
        │   ├── generate_report.py
        │   └── inference.py
        ├── analytics/             # V3.0
        │   ├── setup_analytics.py
        │   ├── traffic_analytics.py
        │   └── live_analytics.py
        └── utils/
            ├── constants.py
            ├── bbox.py
            ├── carla_helpers.py
            ├── switch_map.py
            ├── visualize_spawns.py
            ├── visualize_traffic_lights.py
            ├── create_spawn_points.py
            └── frames_to_video.py
```

---

## Git Branch History

### Original Repository: `Capstone_Work` (deleted, recovered from local clone)

| Branch | Purpose | Status |
|--------|---------|--------|
| `main` | Documentation + clean install of CARLA | Active |
| `testing_sim` | Scenario creator system (setup_scenario.py, run_scenario.py, Town05 configs) | Merged via PR #1 |
| `models` | YOLOv12 tracking + training pipeline | Merged via PR #2 |

### Current Repository: `dataset_capstone`

| Branch | Purpose | Status |
|--------|---------|--------|
| `main` | Stable baseline | Active |
| `big_changes` | First major feature batch (capture_dataset.py) | Merged (PR #1) |
| `model-training-and-testing` | YOLO training experiments | Standalone |
| `V2` | Restructure + multi-camera + evaluation | Merged (PR #2) |
| `V2.1` | Inference + custom spawn points + refinements | Merged |
| `V2.2` | Stuck vehicle detection + frames_to_video utility | Merged |
| `V3.0` | Traffic analytics: calibration, speed, queue per lane | Open PR |

---

## Full Project Timeline (chronological)

| Date | Event | Repo |
|------|-------|------|
| **2026-02-11** | Project bootstrapped — CARLA install, conda env, git/GitHub | Capstone_Work |
| **2026-02-12** | New `testing_sim` branch, Jupyter setup, first test script | Capstone_Work |
| **2026-02-15** | Custom scenario creation tool built (became `setup_scenario.py`) | Capstone_Work |
| **2026-02-16** | Pushed scenario system v1 to GitHub | Capstone_Work |
| **2026-02-19** | YOLOv12 tracking model + first realization that COCO lacks emergency vehicles | Capstone_Work |
| **2026-02-21** | Transfer learning / fine-tuning research | Capstone_Work |
| **2026-03-12** | Training pipeline (training_code.py, extract_frames.py, configs) committed | Capstone_Work |
| **2026-03-17** | Tried VideoMT on Windows — failed, model requires Linux | (local Windows) |
| **2026-03-21** | Migrated to Linux WSL, set up VideoMT — decided not to use it | WSL Ubuntu |
| **2026-03-29** | YOLOE prompt-based detection — works but can't distinguish emergency vehicles. Decision to pivot to CARLA + custom dataset | WSL Ubuntu |
| **2026-03-30** | Last activity on original repo before migration | Capstone_Work |
| **2026-03-31** | New repo `dataset_capstone` started — cleanup, env fixes, README rewrite | dataset_capstone |
| **2026-02-16** | Scenario creator system v1 (setup_scenario, run_scenario, Town05 scenarios) | Capstone_Work |
| **2026-02-19** | YOLOv12 tracking pipeline with 4 MOT methods (bytetrack/botsort/centroid/gst-nvtracker) | Capstone_Work |
| **2026-03-12** | Training pipeline (training_code.py, extract_frames.py, configs) | Capstone_Work |
| **2026-03-30** | Last activity on original repo before migration | Capstone_Work |
| **2026-03-31** | New repo `dataset_capstone` started — cleanup, env fixes, README rewrite | dataset_capstone |
| **2026-04-03** | Bug fixes (camera_location at origin), spawn radius, MIN_VISIBILITY tuning | dataset_capstone |
| **2026-04-04** | Spawn point visualizer + new scenarios | dataset_capstone |
| **2026-04-11** | YOLOv11 training script + initial training | dataset_capstone |
| **2026-04-12** | Major restructure into scripts/{capture,train,evaluate,utils}/ | dataset_capstone |
| **2026-04-13** | Multi-camera, weather cycling, batch capture, evaluation pipeline | dataset_capstone |
| **2026-05-02** | Cloud training on vast.ai L40S — 0.949 mAP50 | dataset_capstone |
| **2026-05-03** | Inference on real video, custom spawn points, traffic light viz, V2.1 PR | dataset_capstone |
| **2026-05-11** | Continued training attempt (abandoned), stuck vehicle detection, frames_to_video utility, V2.2 | dataset_capstone |
| **2026-05-12** | Traffic analytics system: calibration, speed per car, per-lane queue length, live + recorded modes, V3.0 | dataset_capstone |
| **2026-06-09** | Light state plumbing, forbidden-line picker, red-light violation detection, `k` demo toggle | dataset_capstone |
| **2026-06-09** | Highway entry zones + entry counting grouped by light state | dataset_capstone |
| **2026-06-09** | Collision detection (experimental, opt-in) — completes the analytics suite | dataset_capstone |
