# CARLA Dataset Capture System

A tool for generating YOLO-format object detection datasets using the CARLA simulator. It spawns configurable traffic scenarios at intersections and captures labeled images of vehicles (cars, ambulances, buses, trucks, police cars, fire trucks, and bikes).

---

## Project Structure

```
capstone_sim/
├── configs/              # scenario YAML configs
├── scripts/
│   ├── capture/          # capture_dataset.py, record_test.py, setup_scenario.py
│   ├── train/            # train.py
│   ├── evaluate/         # evaluate_model.py, visualize_metrics.py
│   └── utils/            # switch_map.py, visualize_spawns.py, shared modules
├── models/
│   └── yolov11/          # model weights and training runs
├── environment.yml
└── requirements.txt
```

---

## Prerequisites

- [CARLA 0.9.16](https://github.com/carla-simulator/carla/releases/tag/0.9.16/) (Windows)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/OJayyusiO/dataset_capstone.git
cd dataset_capstone
```

### 2. Install CARLA

Download the **CARLA 0.9.16 Windows** zip and extract it into the `capstone_sim/` folder:

```
capstone_sim/
├── configs/
├── scripts/
├── environment.yml
└── CARLA_0.9.16/
    ├── CarlaUE4.exe
    └── PythonAPI/
```

### 3. Create the conda environment

```bash
conda env create -f capstone_sim/environment.yml
conda activate capstone
```

To use a different environment name:

```bash
conda env create -f capstone_sim/environment.yml -n your_env_name
conda activate your_env_name
```

Install PyTorch with CUDA support and the remaining dependencies:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r capstone_sim/requirements.txt
```

### 4. Launch CARLA

For best visual quality:

```powershell
cd capstone_sim\CARLA_0.9.16
CarlaUE4.exe -quality-level=Epic -ResX=1920 -ResY=1080 -windowed
```

---

## Data Capture

### Step 1: Load a map

```bash
python capstone_sim/scripts/utils/switch_map.py Town05
```

### Step 2: Set up a scenario

```bash
python capstone_sim/scripts/capture/setup_scenario.py
```

This opens an interactive tool that:
1. Lets you fly the spectator camera to your desired viewpoint in CARLA
2. Captures the camera position when you press ENTER
3. Lists nearby traffic lights and lets you select one
4. Saves a config to `capstone_sim/configs/scenario_config.yaml`

**Spectator controls:**
- **WASD** - Move
- **Mouse** - Look around
- **Mouse Wheel** - Adjust movement speed (scroll down for slower, more precise movement)

### Step 2b: Pick spawn points (optional)

By default, vehicles spawn at any point within `spawn_radius` of the traffic light. To control exactly where vehicles spawn:

1. Visualize all spawn points on the map:
   ```bash
   python capstone_sim/scripts/utils/visualize_spawns.py
   ```
   This draws numbered markers in the CARLA viewport for 120 seconds.

2. Add the spawn point indices to your config:
   ```yaml
   spawn:
     spawn_points: ["33-81"]
   ```

### Step 3: Capture the dataset

```bash
python capstone_sim/scripts/capture/capture_dataset.py capstone_sim/configs/scenario_config.yaml
```

**Important:** Change the `scenario_name` field in the YAML before each capture run. The scenario name is used to name the output images, so reusing the same name will overwrite previous data.

---

## Configuration

Edit the generated YAML config in `capstone_sim/configs/` to customize:

**Weather:**
```yaml
weather:
  cloudiness: 40.0          # 0 (clear) to 100 (overcast)
  precipitation: 0.0        # 0 (dry) to 100 (heavy rain)
  sun_altitude_angle: 25.0  # -90 to 90 (negative = night)
  fog_density: 0.0          # 0 to 100
  wetness: 0.0              # 0 to 100
```

**Vehicle spawning:**
```yaml
spawn:
  max_vehicles: 30
  spawn_radius: 100.0       # meters from reference point
  radius_center: traffic_light  # or "camera"
  spawn_points: []           # specific indices, or ranges like ["33-81"]
  respawn_interval: 50
  despawn_distance: 150.0
  ratios:
    car: 15
    ambulance: 2
    bus: 2
    truck: 3
    police_car: 2
    fire_truck: 1
    bike: 4
```

**Simulation:**
```yaml
simulation:
  total_frames: 5000
  capture_interval: 2       # capture every N frames (2 = 10 FPS)
  warmup_frames: 100        # background images captured before vehicles spawn
  train_ratio: 0.8          # train/val split
```

---

## Output

The dataset is saved in YOLO format:

```
dataset_output/
├── data.yaml
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

**Detected classes:**

| ID | Class       |
|----|-------------|
| 0  | car         |
| 1  | ambulance   |
| 2  | bus         |
| 3  | truck       |
| 4  | police_car  |
| 5  | fire_truck  |
| 6  | bike        |

---

## Training

```bash
python capstone_sim/scripts/train/train.py --data path/to/dataset_output/data.yaml
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | (required) | Path to `data.yaml` in your dataset |
| `--model` | `yolo11n.pt` | Model size: `yolo11n.pt`, `yolo11s.pt`, `yolo11m.pt` |
| `--epochs` | `100` | Number of training epochs |
| `--batch` | `16` | Batch size (reduce to 8 for larger models) |
| `--imgsz` | `640` | Input image size |
| `--resume` | none | Path to checkpoint to resume training |

Results are saved to `capstone_sim/models/yolov11/runs/`.

---

## Evaluation

### 1. Record test footage

```bash
python capstone_sim/scripts/capture/record_test.py capstone_sim/configs/scenario_config.yaml
```

Saves sequential PNG frames and ground truth labels to `test_recordings/`. Record once, test multiple models.

| Flag | Default | Description |
|------|---------|-------------|
| `--output` | `./test_recordings` | Base directory for recordings |
| `--duration` | from config | Number of simulation frames |
| `--fps` | `20` | Recording FPS |

### 2. Evaluate a model

```bash
python capstone_sim/scripts/evaluate/evaluate_model.py test_recordings/<recording_dir> path/to/best.pt
```

Produces:
- `annotated.mp4` — Video with bounding boxes, track IDs, and class labels
- `metrics_summary.json` — Detection and tracking metrics (MOTA, IDF1, precision, recall)
- `per_frame_metrics.csv` — Per-frame breakdown

| Flag | Default | Description |
|------|---------|-------------|
| `--output` | `./eval_results` | Base directory for results |
| `--conf` | `0.25` | Confidence threshold |
| `--iou` | `0.5` | IoU threshold |

### 3. Visualize metrics

```bash
python capstone_sim/scripts/evaluate/visualize_metrics.py eval_results/<result_dir>/metrics_summary.json
```

Generates charts: per-class detection, class distribution, TP/FP/FN breakdown, tracking summary, and per-frame metrics over time.

### 4. Compare models

```bash
python capstone_sim/scripts/evaluate/evaluate_model.py test_recordings/my_test model_a.pt
python capstone_sim/scripts/evaluate/evaluate_model.py test_recordings/my_test model_b.pt
```

---

## Traffic Analytics (live + recorded)

Beyond detection metrics, the system can compute **real-world traffic metrics** like per-vehicle speed and per-lane queue length, using camera calibration to convert pixels to world meters.

### 1. Setup analytics (once per scenario / video)

For CARLA scenarios — calibration is fully automatic from camera intrinsics + extrinsics. You only need to define lanes:

```bash
python capstone_sim/scripts/analytics/setup_analytics.py capstone_sim/configs/Town6_1cam.yaml
```

This connects to CARLA, spawns the scenario camera, captures one frame, auto-calibrates, then walks through four steps: **(1) calibration**, **(2) lane polygons**, **(3) forbidden lines** (stop lines for red-light violations), **(4) highway entry zones** (for entry counting by light state). Saves to `capstone_sim/analytics_configs/<scenario>.yaml`.

For real video / CCTV — calibration is manual (click 4 corners of a known rectangle):

```bash
python capstone_sim/scripts/analytics/setup_analytics.py path/to/video.mp4
```

Source can also be a recording directory, webcam (`0`), or RTSP stream URL.

**Useful flags:**

| Flag | Description |
|------|-------------|
| `--recalibrate` | Redo calibration |
| `--redo-lanes` | Wipe existing lanes and define fresh |
| `--redo-lines` | Wipe existing forbidden lines and define fresh |
| `--redo-entry-zones` | Wipe existing highway entry zones and define fresh |
| `--manual` | Force manual calibration |

### 2. Run live analytics on CARLA

```bash
python capstone_sim/scripts/analytics/live_analytics.py capstone_sim/configs/Town6_1cam.yaml capstone_sim/models/yolov11m/best.pt
```

Spawns traffic from the scenario, runs detection + tracking + speed + per-lane queue counts + red-light violation detection + highway entry counting in real time. Reads the traffic light state directly from CARLA. Saves results to `capstone_sim/analytics_runs/<scenario>_<timestamp>/`.

| Flag | Description |
|------|-------------|
| `--save-video` | Save the annotated feed as MP4 |
| `--no-spawn` | Skip spawning vehicles (use if CARLA already has traffic) |
| `--conf` / `--iou` | Detection thresholds |

**Live keyboard controls:**

| Key | Action |
|-----|--------|
| `k` | Toggle all vehicles ignoring red lights (for demoing violations — CARLA vehicles obey lights by default) |
| `q` | Quit |

### 3. Run analytics on a recorded video

```bash
python capstone_sim/scripts/analytics/traffic_analytics.py test_recordings/<recording_dir> capstone_sim/models/yolov11m/best.pt
```

Same outputs as live mode, but on recorded footage.

### Outputs

Both live and recorded analytics save:
- `live_analytics.mp4` / `analytics.mp4` — Annotated video with speed labels, per-lane queue overlays, light indicator, and violation banners
- `per_track.csv` — Per-frame, per-track: track_id, class, world_x, world_y, speed
- `per_lane_queue.csv` — Per-frame queue count for each lane
- `violations.csv` — Each red-light violation: frame, track_id, line_id, light_state
- `entries.csv` — Each highway entry: frame, track_id, zone_id, light_state
- `summary.json` — Final stats: avg/max speed, max queue per lane, total violations, entry counts per zone (by light state), unique tracks, FPS

### Queue thresholds

Configurable in `analytics_config.yaml`:

```yaml
queue:
  speed_threshold_kmh: 7.2          # below this = "slow"
  min_stationary_seconds: 2.0        # must be slow for this long to count as queued
```

### Traffic light state & red-light violations

Light state is read directly from CARLA (live + recorded). For real video, define a manual schedule in `analytics_config.yaml`:

```yaml
light_schedule:
  - {frame: 0, state: red}
  - {frame: 150, state: green}
```

A vehicle that crosses a forbidden line (defined in setup step 3) while the light is red is logged as a violation. For CARLA recordings, `record_test.py` auto-logs per-frame light state to `light_states.csv`.

---

## Scripts Reference

| Script | Location | Purpose |
|--------|----------|---------|
| `switch_map.py` | `scripts/utils/` | Load a CARLA map by name |
| `visualize_spawns.py` | `scripts/utils/` | Draw numbered spawn point markers in CARLA |
| `visualize_traffic_lights.py` | `scripts/utils/` | Draw numbered traffic light markers in CARLA |
| `create_spawn_points.py` | `scripts/utils/` | Capture custom spawn points by flying spectator |
| `frames_to_video.py` | `scripts/utils/` | Convert PNG frame sequences to MP4 |
| `setup_scenario.py` | `scripts/capture/` | Position camera and select traffic light |
| `capture_dataset.py` | `scripts/capture/` | Capture YOLO-format dataset |
| `record_test.py` | `scripts/capture/` | Record test footage with ground truth |
| `batch_capture.py` | `scripts/capture/` | Run multiple scenario configs sequentially |
| `train.py` | `scripts/train/` | Train YOLOv11 model |
| `evaluate_model.py` | `scripts/evaluate/` | Run detection + tracking evaluation |
| `visualize_metrics.py` | `scripts/evaluate/` | Generate metric charts |
| `compare_models.py` | `scripts/evaluate/` | Side-by-side comparison of multiple models |
| `analyze_dataset.py` | `scripts/evaluate/` | Class distribution, imbalance warnings |
| `generate_report.py` | `scripts/evaluate/` | Auto-generated HTML report |
| `inference.py` | `scripts/evaluate/` | Run model on any MP4/webcam |
| `setup_analytics.py` | `scripts/analytics/` | Calibration + lane + forbidden-line + entry-zone definition |
| `traffic_analytics.py` | `scripts/analytics/` | Speed + queue + violations + entry counting on recorded video |
| `live_analytics.py` | `scripts/analytics/` | Speed + queue + violations + entry counting live on CARLA |
