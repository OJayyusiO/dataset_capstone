# CARLA Dataset Capture System

A tool for generating YOLO-format object detection datasets using the CARLA simulator. It spawns configurable traffic scenarios at intersections and captures labeled images of vehicles (cars, ambulances, buses, trucks, police cars, fire trucks, and bikes).

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

Download the **CARLA 0.9.16 Windows** zip and extract it into the `capstone_sim/` folder. Your directory should look like this:

```
capstone_sim/
├── Code/
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

### 4. Launch CARLA

For best visual quality, use the provided batch file:

```powershell
.\capstone_sim\Code\launch_carla_quality.bat
```

Or launch manually with quality flags:

```powershell
cd capstone_sim\CARLA_0.9.16
CarlaUE4.exe -quality-level=Epic -ResX=1920 -ResY=1080 -windowed
```

---

## Usage

There are three steps: (1) switch to the map you want, (2) set up a camera position and pick a traffic light, (3) capture the dataset.


### Step 1: Load a map

```bash
python capstone_sim/Code/switch_map.py Town05
```

Replace `Town05` with any CARLA map name.

### Step 2: Set up a scenario

```bash
python capstone_sim/Code/setup_scenario.py
```

This opens an interactive tool that:
1. Lets you fly the spectator camera to your desired viewpoint in CARLA
2. Captures the camera position when you press ENTER
3. Lists nearby traffic lights and lets you select one
4. Saves a `scenario_config.yaml` file with all settings

**Spectator controls:**
- **WASD** - Move
- **Mouse** - Look around
- **Mouse Wheel** - Adjust movement speed (scroll down for slower, more precise movement)

### Step 3: Capture the dataset

```bash
python capstone_sim/Code/capture_dataset.py scenario_config.yaml
```

This runs a long simulation and outputs a YOLO-format dataset with labeled images to the configured output directory (default: `./dataset_output`).

---

## Configuration

After running `setup_scenario.py`, edit the generated `scenario_config.yaml` to customize:

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
  spawn_radius: 100.0       # meters from traffic light
  respawn_interval: 50      # frames between respawn checks
  despawn_distance: 150.0   # despawn vehicles beyond this distance
  ratios:                   # relative spawn ratios per class
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
  capture_interval: 10      # capture every N frames
  warmup_frames: 100        # let traffic settle before capturing
  train_ratio: 0.8          # train/val split
```

---

## Output

The dataset is saved in YOLO format:

```
dataset_output/
├── data.yaml          # class definitions for training
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

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `switch_map.py` | Load a CARLA map by name |
| `setup_scenario.py` | Interactive tool to position camera and select traffic light |
| `capture_dataset.py` | Run simulation and capture YOLO-format dataset |
| `launch_carla_quality.bat` | Launch CARLA with high quality rendering settings |
