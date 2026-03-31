# CARLA Scenario Testing System

A flexible system for creating repeatable traffic light test scenarios in CARLA for object detection and tracking evaluation.

## 🚀 Quick Start: Run Pre-Made Scenarios

**Test all 7 Town05 scenarios at once:**
```powershell
.\capstone_sim\Code\test_all_town05.bat
```

**Available scenarios:**
- `town05_single_lane_simple` - ⭐ Easy: 2 vehicles, clear day
- `town05_basic_day` - ⭐⭐ Medium: 4 vehicles, clear day
- `town05_heavy_traffic` - ⭐⭐⭐ Hard: 8 vehicles, rush hour
- `town05_morning_rush` - ⭐⭐⭐ Hard: 6 vehicles, sunrise + fog
- `town05_evening_sunset` - ⭐⭐⭐ Hard: 4 vehicles, sunset
- `town05_rainy_day` - ⭐⭐⭐⭐ Very Hard: 4 vehicles, heavy rain
- `town05_night_low_traffic` - ⭐⭐⭐⭐ Very Hard: 3 vehicles, night

See [scenarios/README.md](scenarios/README.md) for complete details.

## Overview

This system has two parts:
1. **Setup Tool** (`setup_scenario.py`) - Captures camera position and creates config files
2. **Scenario Runner** (`run_scenario.py`) - Runs scenarios from config files

## Important: Launch CARLA with High Quality

**To fix "blob" cars and improve visual quality, launch CARLA with:**

```powershell
# Option 1: Use the provided batch file
.\capstone_sim\Code\launch_carla_quality.bat

# Option 2: Launch manually with quality flags
cd capstone_sim\CARLA_0.9.16
CarlaUE4.exe -quality-level=Epic -ResX=1920 -ResY=1080 -windowed
```

If you launch CARLA normally, vehicles will appear low-quality and blocky.

## Quick Start

### Option A: Use Pre-Made Scenarios (Fastest)

**For Town05 map (recommended):**

1. **Load Town05:**
   ```powershell
   python capstone_sim\Code\switch_map.py Town05
   ```

2. **Set up camera positions (one time per scenario):**
   ```powershell
   python capstone_sim\Code\setup_scenario.py
   # Position camera, select traffic light
   # Save output to: capstone_sim\Code\scenarios\town05_basic_day.yaml
   ```

3. **Run a scenario:**
   ```powershell
   python capstone_sim\Code\run_scenario.py capstone_sim\Code\scenarios\town05_basic_day.yaml
   ```

**Available pre-made scenarios** (in `scenarios/` folder):
- `town05_basic_day.yaml` - Clear weather, 2 lanes, moderate traffic
- `town05_heavy_traffic.yaml` - 8 vehicles, rush hour conditions
- `town05_morning_rush.yaml` - Sunrise lighting with fog
- `town05_evening_sunset.yaml` - Sunset, low light
- `town05_rainy_day.yaml` - Heavy rain, reduced visibility
- `town05_night_low_traffic.yaml` - Night time testing
- `town05_single_lane_simple.yaml` - Minimal test case

See [scenarios/README.md](scenarios/README.md) for full details.

### Option B: Create Custom Scenario

### 1. Create a Scenario Config

```powershell
python setup_scenario.py
```

This interactive tool will:
- Ask you to position the spectator camera in CARLA
- Capture the camera position and angle
- Show you nearby traffic lights
- Let you select which traffic light to use
- Generate a `scenario_config.yaml` file

**Spectator Controls (for precise positioning):**
- **WASD** - Move forward/back/left/right
- **Mouse** - Look around
- **Mouse Wheel DOWN** - Slow down movement speed (scroll down multiple times!)
- **Mouse Wheel UP** - Speed up movement speed
- **Right-click + Mouse** - Free look without moving

**TIP:** Scroll the mouse wheel DOWN 5-10 times before positioning for very precise control.

### 2. Edit the Config

Open `scenario_config.yaml` and customize:

```yaml
lanes:
  - lane_id: 0
    num_vehicles: 2                              # How many cars in this lane
    spawn_distances: [35.0, 50.0]                # Meters behind stop line
    vehicle_types: ["vehicle.tesla.model3", "vehicle.audi.tt"]

timing:
  red_seconds: 8.0      # How long red light lasts
  green_seconds: 8.0    # How long green light lasts

weather:
  cloudiness: 50.0      # 0-100
  precipitation: 20.0   # 0-100 (rain)
  sun_altitude_angle: 45.0
  fog_density: 10.0
  wetness: 30.0
```

### 3. Run the Scenario

```powershell
python run_scenario.py scenario_config.yaml
```

Press 'q' to quit early.

## Config File Reference

### Camera
- `location` - x, y, z coordinates (captured from spectator)
- `rotation` - pitch, yaw, roll angles (captured from spectator)
- `image_width`, `image_height` - Resolution

### Traffic Light
- `id` - CARLA actor ID (selected during setup)

### Lanes
Each lane entry:
- `lane_id` - Which stop waypoint/lane (0, 1, 2...)
- `num_vehicles` - Number of cars to spawn
- `spawn_distances` - List of distances from stop line (meters)
- `vehicle_types` - List of CARLA vehicle blueprints

### Timing
- `red_seconds` - Duration of red light phase
- `green_seconds` - Duration of green light phase
- `fixed_delta_seconds` - Simulation timestep (lower = smoother but slower)

### Weather
- `cloudiness` - 0 (clear) to 100 (overcast)
- `precipitation` - 0 (dry) to 100 (heavy rain)
- `sun_altitude_angle` - -90 to 90 (affects shadows/lighting)
- `fog_density` - 0 to 100
- `wetness` - 0 to 100 (road wetness)

### Traffic Manager
- `vehicle_speed_difference` - % slower than speed limit (higher = slower/safer)
- `ignore_lights_percentage` - 0 = obey all lights, 100 = ignore all

## Tips

**Making scenarios repeatable:**
- The system uses fixed random seed (42) for deterministic behavior
- Same config = same vehicle spawns, colors, and behavior

**If vehicles roll through red:**
- Increase `spawn_distances` (give more braking distance)
- Increase `vehicle_speed_difference` (make them drive slower)
- Increase `red_seconds` (more time to stop)

**Multiple lanes:**
- Traffic lights can control multiple lanes
- Add more lane entries in config to spawn cars in different lanes
- Each lane has its own vehicle count and spawn distances

**Performance:**
- Lower resolution for faster processing
- Increase `fixed_delta_seconds` to 0.1 for lower FPS but better stability
- Reduce `num_vehicles` if simulation is slow

## Fixing Visual Quality Issues

**Problem: Cars look like blobs/low quality**

This happens when CARLA is launched with default settings. Fix:

1. **Close CARLA if running**
2. **Relaunch with quality settings:**
   ```powershell
   cd capstone_sim\CARLA_0.9.16
   CarlaUE4.exe -quality-level=Epic -ResX=1920 -ResY=1080 -windowed
   ```
3. **Or use the provided batch file:**
   ```powershell
   .\capstone_sim\Code\launch_carla_quality.bat
   ```

**Quality levels (from lowest to highest):**
- `Low` - Fastest, blocky vehicles
- `Medium` - Better but still low detail
- `High` - Good balance
- `Epic` - High quality (recommended)
- `Cinematic` - Maximum quality (very slow)

**Additional quality flags:**
- `-dx12` - Use DirectX 12 (better performance on newer GPUs)
- `-ResX=2560 -ResY=1440` - Higher resolution (slower)
- `-benchmark -fps=30` - Lock FPS for consistency

## Example: Multi-Lane Intersection

```yaml
lanes:
  - lane_id: 0  # Straight lane
    num_vehicles: 3
    spawn_distances: [30.0, 45.0, 60.0]
    vehicle_types: ["vehicle.tesla.model3", "vehicle.audi.tt", "vehicle.dodge.charger_2020"]
  
  - lane_id: 1  # Turn lane
    num_vehicles: 2
    spawn_distances: [35.0, 50.0]
    vehicle_types: ["vehicle.toyota.prius", "vehicle.nissan.patrol"]
```

## Files

- `setup_scenario.py` - Interactive setup tool
- `run_scenario.py` - Scenario runner
- `scenario_config_example.yaml` - Example config with all options
- `scenario_config.yaml` - Generated config (created by setup tool)
