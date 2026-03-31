# Quick Reference Card

## Run Single Scenario
```powershell
python capstone_sim\Code\run_scenario.py capstone_sim\Code\scenarios\town05_basic_day.yaml
```

## Run All Scenarios
```powershell
.\capstone_sim\Code\test_all_town05.bat
```

## Create New Scenario
```powershell
# 1. Switch to desired map
python capstone_sim\Code\switch_map.py Town05

# 2. Position camera and capture
python capstone_sim\Code\setup_scenario.py

# 3. Output saved to scenario_config.yaml
# Edit or rename as needed
```

## Launch CARLA with Quality
```powershell
.\capstone_sim\Code\launch_carla_quality.bat
```

## Scenario Files Location
```
capstone_sim\Code\scenarios\
  ├── town05_single_lane_simple.yaml
  ├── town05_basic_day.yaml
  ├── town05_heavy_traffic.yaml
  ├── town05_morning_rush.yaml
  ├── town05_evening_sunset.yaml
  ├── town05_rainy_day.yaml
  └── town05_night_low_traffic.yaml
```

## Common Config Edits

### Change number of vehicles
```yaml
lanes:
  - lane_id: 0
    num_vehicles: 3  # Change this
```

### Change weather
```yaml
weather:
  cloudiness: 50.0        # 0-100
  precipitation: 30.0     # 0-100 (rain)
  sun_altitude_angle: 20.0  # -90 to 90
```

### Change timing
```yaml
timing:
  red_seconds: 10.0
  green_seconds: 10.0
```

### Make vehicles drive slower
```yaml
traffic_manager:
  vehicle_speed_difference: 40.0  # Higher = slower (% below speed limit)
```

## Controls During Scenario
- **Q** - Quit scenario early
- **Close window** - Also quits

## Troubleshooting

**Cars look like blobs:**
- Restart CARLA with `launch_carla_quality.bat`

**Cars roll through red light:**
- Increase `vehicle_speed_difference` (e.g., 40.0)
- Increase `spawn_distances` (e.g., [40.0, 55.0])
- Increase `red_seconds` (e.g., 15.0)

**Scenario too slow/fast:**
- Change `red_seconds` and `green_seconds`

**Wrong traffic light:**
- Edit `traffic_light: id:` in YAML file
- Or run `setup_scenario.py` again

**Camera in wrong position:**
- Edit `camera: location:` and `rotation:` in YAML
- Or run `setup_scenario.py` again
