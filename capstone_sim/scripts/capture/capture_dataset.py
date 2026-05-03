"""
CARLA Dataset Capture Script

Runs a long-duration traffic simulation at a user-configured camera position
near a traffic light, continuously spawning vehicles with configurable class
ratios, and captures YOLO-format training data.

Usage:
    1. Run setup_scenario.py to create a config YAML
    2. Edit the YAML to set weather, spawn ratios, duration, etc.
    3. python capture_dataset.py scenario_config.yaml
"""

import carla
import yaml
import sys
import queue
import cv2
import numpy as np
import random
import argparse
import math
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from capstone_sim.scripts.utils.constants import (
    CLASS_NAMES, CLASS_NAME_TO_ID, BLUEPRINT_TO_CLASS,
    MAX_DISTANCE,
)
from capstone_sim.scripts.utils.bbox import (
    build_projection_matrix, get_2d_bbox, bbox_to_yolo,
)
from capstone_sim.scripts.utils.carla_helpers import (
    spawn_camera, apply_weather,
    get_available_blueprints, build_class_blueprint_map,
    compute_target_counts, spawn_to_fill, despawn_far_vehicles,
)


# =============================================================================
# OUTPUT / FILE I/O
# =============================================================================

def setup_output_dirs(output_dir):
    """Create the YOLO dataset directory structure."""
    dirs = {
        'images_train': output_dir / 'images' / 'train',
        'images_val': output_dir / 'images' / 'val',
        'labels_train': output_dir / 'labels' / 'train',
        'labels_val': output_dir / 'labels' / 'val',
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def save_frame(image_array, labels, frame_id, split, dirs):
    """Save an image and its YOLO label file."""
    img_path = dirs[f'images_{split}'] / f'{frame_id}.png'
    lbl_path = dirs[f'labels_{split}'] / f'{frame_id}.txt'

    cv2.imwrite(str(img_path), image_array)
    with open(lbl_path, 'w') as f:
        for label in labels:
            f.write(label + '\n')


def write_data_yaml(output_dir):
    """Write the ultralytics-compatible data.yaml file."""
    yaml_content = f"""path: {output_dir.resolve().as_posix()}
train: images/train
val: images/val

names:
  0: car
  1: ambulance
  2: bus
  3: truck
  4: police_car
  5: fire_truck
  6: bike
"""
    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"Wrote {yaml_path}")


# =============================================================================
# MAIN CAPTURE LOOP
# =============================================================================

def run_capture(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Extract config sections
    # Support both 'cameras' (list) and legacy 'camera' (single dict)
    if 'cameras' in config:
        camera_configs = config['cameras']
    else:
        camera_configs = [config['camera']]

    weather_config = config.get('weather', {})
    weather_cycle = config.get('weather_cycle')
    sim_config = config.get('simulation', {})
    spawn_config = config.get('spawn', {})
    output_config = config.get('output', {})

    total_frames = sim_config.get('total_frames', 5000)
    capture_interval = sim_config.get('capture_interval', 10)
    warmup_frames = sim_config.get('warmup_frames', 100)
    train_ratio = sim_config.get('train_ratio', 0.8)
    fixed_delta = sim_config.get('fixed_delta_seconds', 0.05)

    max_vehicles = spawn_config.get('max_vehicles', 30)
    respawn_interval = spawn_config.get('respawn_interval', 50)
    despawn_distance = spawn_config.get('despawn_distance', 150.0)
    force_respawn_interval = spawn_config.get('force_respawn_interval', 0)
    ratios = spawn_config.get('ratios', {
        'car': 15, 'ambulance': 2, 'bus': 2, 'truck': 3,
        'police_car': 2, 'fire_truck': 1, 'bike': 4
    })

    num_cameras = len(camera_configs)
    output_dir = Path(output_config.get('directory', './dataset_output'))
    dirs = setup_output_dirs(output_dir)

    # Build projection matrix per camera
    cam_params = []
    for cc in camera_configs:
        w = cc.get('image_width', 1280)
        h = cc.get('image_height', 720)
        f = cc.get('fov', 70)
        cam_params.append({'w': w, 'h': h, 'fov': f, 'K': build_projection_matrix(w, h, f)})

    target_counts = compute_target_counts(ratios, max_vehicles)

    # Print summary
    expected_captures = max(0, (total_frames - warmup_frames)) // capture_interval * num_cameras
    print("=" * 60)
    print("CARLA Dataset Capture")
    print("=" * 60)
    print(f"Scenario: {config.get('scenario_name', 'unnamed')}")
    print(f"Output: {output_dir.resolve()}")
    print(f"Cameras: {num_cameras}")
    print(f"Frames: {total_frames} total, {warmup_frames} warmup")
    print(f"Capture every {capture_interval} frames -> ~{expected_captures} images")
    print(f"Train/Val split: {train_ratio:.0%}/{1-train_ratio:.0%}")
    print(f"Max vehicles: {max_vehicles}, spawn radius: {spawn_config.get('spawn_radius', 100.0)}m")
    print(f"Spawn ratios: {ratios}")
    print(f"Target counts: {{{', '.join(f'{CLASS_NAMES[k]}: {v}' for k, v in target_counts.items())}}}")
    if weather_cycle:
        print(f"Weather: cycling through {len(weather_cycle)} presets")
    else:
        print(f"Weather: sun={weather_config.get('sun_altitude_angle', 45)}, "
              f"cloud={weather_config.get('cloudiness', 10)}, "
              f"rain={weather_config.get('precipitation', 0)}, "
              f"fog={weather_config.get('fog_density', 0)}")
    print("=" * 60)

    # Connect to CARLA
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    # Switch map if specified in config
    target_map = config.get('map')
    if target_map:
        current_map = client.get_world().get_map().name.split('/')[-1]
        if current_map != target_map:
            print(f"Loading map {target_map}...")
            client.load_world(target_map)
            time.sleep(5)  # Wait for map to load

    world = client.get_world()

    # Remove parked vehicles from the map (they confuse the model)
    world.unload_map_layer(carla.MapLayer.ParkedVehicles)

    bp_lib = world.get_blueprint_library()
    all_spawn_points = world.get_map().get_spawn_points()

    # Enable synchronous mode
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = fixed_delta
    settings.substepping = True
    settings.max_substep_delta_time = 0.01
    settings.max_substeps = 10
    world.apply_settings(settings)

    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_random_device_seed(42)
    random.seed(42)
    np.random.seed(42)
    tm_port = traffic_manager.get_port()

    # Get available blueprints and group by class
    available_bps = get_available_blueprints(bp_lib)
    class_bps = build_class_blueprint_map(available_bps)

    missing_classes = [CLASS_NAMES[cid] for cid in target_counts if cid not in class_bps]
    if missing_classes:
        print(f"WARNING: No blueprints found for: {missing_classes}")

    current_vehicles = []  # list of (actor, class_id)
    actor_list = []

    try:
        # Apply initial weather
        if weather_cycle:
            weather_presets = weather_cycle
            # Calculate frames per weather preset (excluding warmup)
            capture_frames = total_frames - warmup_frames
            frames_per_weather = max(1, capture_frames // len(weather_presets))
            apply_weather(world, weather_presets[0])
            print(f"Weather preset 1/{len(weather_presets)} applied (switching every {frames_per_weather} frames)")
        else:
            weather_presets = None
            frames_per_weather = 0
            apply_weather(world, weather_config)
            print("Weather applied")

        # Spawn cameras
        cameras = []
        image_queues = []
        camera_locations = []
        for cam_idx, cc in enumerate(camera_configs):
            q = queue.Queue()
            cam = spawn_camera(world, bp_lib, cc)
            actor_list.append(cam)
            cam.listen(q.put)
            cameras.append(cam)
            image_queues.append(q)

        world.tick()  # Tick so CARLA updates camera transforms
        for cam_idx, cam in enumerate(cameras):
            loc = cam.get_transform().location
            camera_locations.append(loc)
            print(f"Camera {cam_idx} spawned at ({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f})")

        # Find traffic light and filter spawn points nearby
        spawn_radius = spawn_config.get('spawn_radius', 100.0)
        light_id = config.get('traffic_light', {}).get('id')
        light_location = None
        if light_id:
            traffic_lights = list(world.get_actors().filter('traffic.traffic_light'))
            selected_light = next((l for l in traffic_lights if l.id == light_id), None)
            if selected_light:
                light_location = selected_light.get_transform().location
                print(f"Traffic light {light_id} found at ({light_location.x:.1f}, {light_location.y:.1f})")
                # Apply custom traffic light timings if specified
                light_config = config.get('traffic_light', {})
                if 'red_time' in light_config:
                    selected_light.set_red_time(light_config['red_time'])
                if 'green_time' in light_config:
                    selected_light.set_green_time(light_config['green_time'])
                if 'yellow_time' in light_config:
                    selected_light.set_yellow_time(light_config['yellow_time'])
                if any(k in light_config for k in ['red_time', 'green_time', 'yellow_time']):
                    print(f"  Timings: red={selected_light.get_red_time():.1f}s "
                          f"green={selected_light.get_green_time():.1f}s "
                          f"yellow={selected_light.get_yellow_time():.1f}s")
            else:
                print(f"WARNING: Traffic light {light_id} not found, using camera location for spawn filtering")

        # Reference location for spawn radius filtering and despawning
        radius_center = spawn_config.get('radius_center', 'traffic_light')
        if radius_center == 'camera':
            reference_location = camera_locations[0]
        else:
            reference_location = light_location if light_location else camera_locations[0]

        # Filter spawn points: use explicit list if provided, otherwise radius-based
        raw_spawn_points = spawn_config.get('spawn_points')
        # Parse spawn points - supports integers and "start-end" range strings
        selected_indices = []
        if raw_spawn_points:
            for entry in raw_spawn_points:
                if isinstance(entry, int):
                    selected_indices.append(entry)
                elif isinstance(entry, str) and '-' in entry:
                    parts = entry.split('-')
                    start, end = int(parts[0]), int(parts[1])
                    selected_indices.extend(range(start, end + 1))
        # Add custom spawn points from config
        custom_spawns = spawn_config.get('custom_spawn_points', [])
        custom_transforms = []
        for cs in custom_spawns:
            loc = carla.Location(x=cs['x'], y=cs['y'], z=cs.get('z', 0.5))
            rot = carla.Rotation(yaw=cs.get('yaw', 0.0))
            custom_transforms.append(carla.Transform(loc, rot))

        if selected_indices or custom_transforms:
            spawn_points = []
            for idx in selected_indices:
                if 0 <= idx < len(all_spawn_points):
                    spawn_points.append(all_spawn_points[idx])
                else:
                    print(f"WARNING: Spawn point index {idx} out of range (0-{len(all_spawn_points)-1})")
            spawn_points.extend(custom_transforms)
            print(f"Using {len(spawn_points)} spawn points "
                  f"({len(selected_indices)} from map, {len(custom_transforms)} custom)")
        else:
            ref_label = 'camera' if radius_center == 'camera' else ('traffic light' if light_location else 'camera')
            spawn_points = [
                sp for sp in all_spawn_points
                if sp.location.distance(reference_location) <= spawn_radius
            ]
            if len(spawn_points) == 0:
                print(f"WARNING: No spawn points within {spawn_radius}m, using all spawn points")
                spawn_points = all_spawn_points
            else:
                print(f"Using {len(spawn_points)}/{len(all_spawn_points)} spawn points within {spawn_radius}m of {ref_label}")

        frame_counter = 0
        captured_count = 0
        vehicles_spawned = False
        start_time = time.time()

        print(f"\nStarting simulation ({total_frames} frames)...")
        print(f"Warmup: {warmup_frames} frames (capturing background images with no vehicles)")
        print("Press Ctrl+C to stop early\n")

        for frame in range(total_frames):
            world.tick()

            # Drain all image queues (keep only latest per camera)
            latest_images = [None] * num_cameras
            for ci, q in enumerate(image_queues):
                try:
                    while True:
                        latest_images[ci] = q.get_nowait()
                except queue.Empty:
                    pass

            # Capture background images during warmup (no vehicles present)
            if frame < warmup_frames:
                if (frame % capture_interval == 0):
                    for ci, raw_img in enumerate(latest_images):
                        if raw_img is None:
                            continue
                        p = cam_params[ci]
                        img_data = np.array(raw_img.raw_data)
                        img = img_data.reshape((p['h'], p['w'], 4))[:, :, :3].copy()
                        split = 'train' if random.random() < train_ratio else 'val'
                        scene_name = config.get('scenario_name', 'scene')
                        cam_label = f"cam{ci}" if num_cameras > 1 else ""
                        frame_id = f"{scene_name}_{cam_label}_{frame_counter:06d}" if cam_label else f"{scene_name}_{frame_counter:06d}"
                        save_frame(img, [], frame_id, split, dirs)
                        frame_counter += 1
                        captured_count += 1
                if frame == warmup_frames - 1:
                    print(f"Warmup complete ({warmup_frames} frames, {captured_count} background images captured)")
                continue

            # Spawn vehicles after warmup
            if not vehicles_spawned:
                spawned = spawn_to_fill(world, bp_lib, tm_port, class_bps,
                                        target_counts, current_vehicles, spawn_points)
                print(f"Spawned {spawned} vehicles")
                for v, cls_id in current_vehicles:
                    if cls_id == 6:  # bikes go slower
                        traffic_manager.vehicle_percentage_speed_difference(v, 50.0)
                    else:
                        traffic_manager.vehicle_percentage_speed_difference(v, 30.0)
                vehicles_spawned = True

            # Force respawn: kill all vehicles and respawn to clear stuck/repetitive scenes
            frames_since_warmup = frame - warmup_frames
            if force_respawn_interval > 0 and frames_since_warmup > 0 and frames_since_warmup % force_respawn_interval == 0:
                for actor, _ in current_vehicles:
                    if actor.is_alive:
                        actor.destroy()
                current_vehicles.clear()
                world.tick()
                spawned = spawn_to_fill(world, bp_lib, tm_port, class_bps,
                                        target_counts, current_vehicles, spawn_points)
                for v, cls_id in current_vehicles:
                    if cls_id == 6:
                        traffic_manager.vehicle_percentage_speed_difference(v, 50.0)
                    else:
                        traffic_manager.vehicle_percentage_speed_difference(v, 30.0)
                print(f"  Forced respawn at frame {frame}: {spawned} fresh vehicles")

            # Respawn cycle: despawn far vehicles, spawn new ones to fill
            elif frame > 0 and frame % respawn_interval == 0:
                removed = despawn_far_vehicles(current_vehicles, reference_location,
                                               despawn_distance)
                spawned = spawn_to_fill(world, bp_lib, tm_port, class_bps,
                                        target_counts, current_vehicles, spawn_points)
                for v, cls_id in current_vehicles:
                    try:
                        if cls_id == 6:
                            traffic_manager.vehicle_percentage_speed_difference(v, 50.0)
                        else:
                            traffic_manager.vehicle_percentage_speed_difference(v, 30.0)
                    except RuntimeError:
                        pass

            # Weather cycling — despawn all vehicles and respawn fresh ones on weather change
            if weather_presets and frames_per_weather > 0:
                frames_since_warmup = frame - warmup_frames
                preset_idx = min(frames_since_warmup // frames_per_weather, len(weather_presets) - 1)
                if frames_since_warmup > 0 and frames_since_warmup % frames_per_weather == 0 and preset_idx < len(weather_presets):
                    apply_weather(world, weather_presets[preset_idx])
                    # Destroy all vehicles and respawn to clear stuck ones
                    for actor, _ in current_vehicles:
                        if actor.is_alive:
                            actor.destroy()
                    current_vehicles.clear()
                    world.tick()  # Let CARLA process the destroys
                    spawned = spawn_to_fill(world, bp_lib, tm_port, class_bps,
                                            target_counts, current_vehicles, spawn_points)
                    for v, cls_id in current_vehicles:
                        if cls_id == 6:
                            traffic_manager.vehicle_percentage_speed_difference(v, 50.0)
                        else:
                            traffic_manager.vehicle_percentage_speed_difference(v, 30.0)
                    print(f"  Weather preset {preset_idx + 1}/{len(weather_presets)} applied, respawned {spawned} vehicles")

            # Capture at interval
            if (frame - warmup_frames) % capture_interval != 0:
                continue

            # Process each camera
            for ci, raw_img in enumerate(latest_images):
                if raw_img is None:
                    continue

                p = cam_params[ci]
                img_data = np.array(raw_img.raw_data)
                img = img_data.reshape((p['h'], p['w'], 4))[:, :, :3].copy()

                # Compute YOLO labels for this camera
                labels = []
                for vehicle, class_id in current_vehicles:
                    if not vehicle.is_alive:
                        continue
                    dist = vehicle.get_transform().location.distance(camera_locations[ci])
                    if dist > MAX_DISTANCE:
                        continue
                    bbox = get_2d_bbox(vehicle, cameras[ci], p['K'], p['w'], p['h'])
                    if bbox is None:
                        continue
                    yolo_label = bbox_to_yolo(bbox, class_id, p['w'], p['h'])
                    labels.append(yolo_label)

                # Skip frames with no detections (background images are captured during warmup)
                if len(labels) == 0:
                    continue

                # Train/val split
                split = 'train' if random.random() < train_ratio else 'val'
                scene_name = config.get('scenario_name', 'scene')
                cam_label = f"cam{ci}" if num_cameras > 1 else ""
                frame_id = f"{scene_name}_{cam_label}_{frame_counter:06d}" if cam_label else f"{scene_name}_{frame_counter:06d}"
                save_frame(img, labels, frame_id, split, dirs)
                frame_counter += 1
                captured_count += 1

            # Progress update every 50 captures
            if captured_count > 0 and captured_count % 50 == 0:
                elapsed = time.time() - start_time
                fps = frame / max(elapsed, 0.001)
                alive_count = sum(1 for v, _ in current_vehicles if v.is_alive)
                print(f"  Frame {frame}/{total_frames} | "
                      f"Captured: {captured_count} | "
                      f"Vehicles: {alive_count} | "
                      f"{fps:.1f} sim fps")

        print(f"\nSimulation complete ({total_frames} frames)")

        interrupted = False

    except KeyboardInterrupt:
        interrupted = True
        print("\n\nStopped by user")

    finally:
        # Cleanup CARLA
        for cam in cameras:
            cam.stop()
        for actor in actor_list:
            if actor.is_alive:
                actor.destroy()
        for actor, _ in current_vehicles:
            if actor.is_alive:
                actor.destroy()

        # Restore settings
        world.apply_settings(original_settings)
        traffic_manager.set_synchronous_mode(False)

    if interrupted:
        # Delete frames captured during this run
        print(f"Cleaning up {captured_count} frames from interrupted capture...")
        for split in ['train', 'val']:
            for subdir in ['images', 'labels']:
                d = dirs.get(f'{subdir}_{split}')
                if d and d.exists():
                    scene_name = config.get('scenario_name', 'scene')
                    for f in d.glob(f'{scene_name}_*'):
                        f.unlink()
        print("Cleanup complete. No data saved from this run.")
    else:
        # Write data.yaml
        write_data_yaml(output_dir)

        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"Dataset capture complete!")
        print(f"Total captured frames: {captured_count}")
        print(f"Time elapsed: {elapsed/60:.1f} minutes")
        print(f"Output: {output_dir.resolve()}")
        print("=" * 60)

        # Auto-validate dataset
        try:
            from capstone_sim.scripts.evaluate.analyze_dataset import analyze
            print()
            analyze(str(output_dir))
        except ImportError:
            pass


def main():
    parser = argparse.ArgumentParser(description='Capture YOLO dataset from CARLA scenario')
    parser.add_argument('config', type=str, help='Path to scenario config YAML file')
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    try:
        run_capture(config_path)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
