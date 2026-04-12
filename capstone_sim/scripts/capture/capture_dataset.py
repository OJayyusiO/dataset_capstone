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
    camera_config = config['camera']
    weather_config = config.get('weather', {})
    sim_config = config.get('simulation', {})
    spawn_config = config.get('spawn', {})
    output_config = config.get('output', {})

    total_frames = sim_config.get('total_frames', 5000)
    capture_interval = sim_config.get('capture_interval', 2)
    warmup_frames = sim_config.get('warmup_frames', 100)
    train_ratio = sim_config.get('train_ratio', 0.8)
    fixed_delta = sim_config.get('fixed_delta_seconds', 0.05)

    max_vehicles = spawn_config.get('max_vehicles', 30)
    respawn_interval = spawn_config.get('respawn_interval', 50)
    despawn_distance = spawn_config.get('despawn_distance', 150.0)
    ratios = spawn_config.get('ratios', {
        'car': 15, 'ambulance': 2, 'bus': 2, 'truck': 3,
        'police_car': 2, 'fire_truck': 1, 'bike': 4
    })

    image_w = camera_config.get('image_width', 1280)
    image_h = camera_config.get('image_height', 720)
    fov = camera_config.get('fov', 70)

    output_dir = Path(output_config.get('directory', './dataset_output'))
    dirs = setup_output_dirs(output_dir)
    K = build_projection_matrix(image_w, image_h, fov)

    target_counts = compute_target_counts(ratios, max_vehicles)

    # Print summary
    expected_captures = max(0, (total_frames - warmup_frames)) // capture_interval
    print("=" * 60)
    print("CARLA Dataset Capture")
    print("=" * 60)
    print(f"Scenario: {config.get('scenario_name', 'unnamed')}")
    print(f"Output: {output_dir.resolve()}")
    print(f"Frames: {total_frames} total, {warmup_frames} warmup")
    print(f"Capture every {capture_interval} frames -> ~{expected_captures} images")
    print(f"Train/Val split: {train_ratio:.0%}/{1-train_ratio:.0%}")
    print(f"Max vehicles: {max_vehicles}, spawn radius: {spawn_config.get('spawn_radius', 100.0)}m")
    print(f"Spawn ratios: {ratios}")
    print(f"Target counts: {{{', '.join(f'{CLASS_NAMES[k]}: {v}' for k, v in target_counts.items())}}}")
    print(f"Weather: sun={weather_config.get('sun_altitude_angle', 45)}, "
          f"cloud={weather_config.get('cloudiness', 10)}, "
          f"rain={weather_config.get('precipitation', 0)}, "
          f"fog={weather_config.get('fog_density', 0)}")
    print("=" * 60)

    # Connect to CARLA
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

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

    image_queue = queue.Queue()
    current_vehicles = []  # list of (actor, class_id)
    actor_list = []

    try:
        # Apply weather
        apply_weather(world, weather_config)
        print("Weather applied")

        # Spawn camera
        camera = spawn_camera(world, bp_lib, camera_config)
        actor_list.append(camera)
        camera.listen(image_queue.put)
        world.tick()  # Tick so CARLA updates the camera transform
        camera_location = camera.get_transform().location
        print(f"Camera spawned at ({camera_location.x:.1f}, {camera_location.y:.1f}, {camera_location.z:.1f})")

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
            else:
                print(f"WARNING: Traffic light {light_id} not found, using camera location for spawn filtering")

        # Reference location for spawn radius filtering and despawning
        radius_center = spawn_config.get('radius_center', 'traffic_light')
        if radius_center == 'camera':
            reference_location = camera_location
        else:
            reference_location = light_location if light_location else camera_location

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
        if selected_indices:
            spawn_points = []
            for idx in selected_indices:
                if 0 <= idx < len(all_spawn_points):
                    spawn_points.append(all_spawn_points[idx])
                else:
                    print(f"WARNING: Spawn point index {idx} out of range (0-{len(all_spawn_points)-1})")
            print(f"Using {len(spawn_points)} manually selected spawn points")
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

            # Drain image queue (keep only latest)
            latest_image = None
            try:
                while True:
                    latest_image = image_queue.get_nowait()
            except queue.Empty:
                pass

            # Capture background images during warmup (no vehicles present)
            if frame < warmup_frames:
                if (frame % capture_interval == 0) and latest_image is not None:
                    img_data = np.array(latest_image.raw_data)
                    img = img_data.reshape((image_h, image_w, 4))[:, :, :3].copy()
                    split = 'train' if random.random() < train_ratio else 'val'
                    scene_name = config.get('scenario_name', 'scene')
                    frame_id = f"{scene_name}_{frame_counter:06d}"
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

            # Respawn cycle: despawn far vehicles, spawn new ones to fill
            if frame > 0 and frame % respawn_interval == 0:
                removed = despawn_far_vehicles(current_vehicles, reference_location,
                                               despawn_distance)
                spawned = spawn_to_fill(world, bp_lib, tm_port, class_bps,
                                        target_counts, current_vehicles, spawn_points)
                # Configure new vehicles
                for v, cls_id in current_vehicles:
                    try:
                        if cls_id == 6:
                            traffic_manager.vehicle_percentage_speed_difference(v, 50.0)
                        else:
                            traffic_manager.vehicle_percentage_speed_difference(v, 30.0)
                    except RuntimeError:
                        pass  # vehicle may have been destroyed between check and configure

            # Capture at interval
            if (frame - warmup_frames) % capture_interval != 0:
                continue

            if latest_image is None:
                continue

            # Convert image
            img_data = np.array(latest_image.raw_data)
            img = img_data.reshape((image_h, image_w, 4))[:, :, :3].copy()

            # Compute YOLO labels
            labels = []
            for vehicle, class_id in current_vehicles:
                if not vehicle.is_alive:
                    continue
                dist = vehicle.get_transform().location.distance(camera_location)
                if dist > MAX_DISTANCE:
                    continue
                bbox = get_2d_bbox(vehicle, camera, K, image_w, image_h)
                if bbox is None:
                    continue
                yolo_label = bbox_to_yolo(bbox, class_id, image_w, image_h)
                labels.append(yolo_label)

            # Skip frames with no detections (background images are captured during warmup)
            if len(labels) == 0:
                if frame % 100 == 0:
                    alive_count = sum(1 for v, _ in current_vehicles if v.is_alive)
                    print(f"  Frame {frame}: no vehicles in view (alive: {alive_count})")
                continue

            # Train/val split
            split = 'train' if random.random() < train_ratio else 'val'
            scene_name = config.get('scenario_name', 'scene')
            frame_id = f"{scene_name}_{frame_counter:06d}"
            save_frame(img, labels, frame_id, split, dirs)
            frame_counter += 1
            captured_count += 1

            # Progress update every 50 captures
            if captured_count % 50 == 0:
                elapsed = time.time() - start_time
                fps = frame / max(elapsed, 0.001)
                alive_count = sum(1 for v, _ in current_vehicles if v.is_alive)
                print(f"  Frame {frame}/{total_frames} | "
                      f"Captured: {captured_count} | "
                      f"Vehicles: {alive_count} | "
                      f"{fps:.1f} sim fps")

        print(f"\nSimulation complete ({total_frames} frames)")

    except KeyboardInterrupt:
        print("\n\nStopped by user")

    finally:
        # Cleanup
        camera.stop()
        for actor in actor_list:
            if actor.is_alive:
                actor.destroy()
        for actor, _ in current_vehicles:
            if actor.is_alive:
                actor.destroy()

        # Restore settings
        world.apply_settings(original_settings)
        traffic_manager.set_synchronous_mode(False)

    # Write data.yaml
    write_data_yaml(output_dir)

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Dataset capture complete!")
    print(f"Total captured frames: {captured_count}")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    print(f"Output: {output_dir.resolve()}")
    print("=" * 60)

    # Print label distribution
    print("\nLabel distribution:")
    for split in ['train', 'val']:
        label_dir = output_dir / 'labels' / split
        counts = {i: 0 for i in CLASS_NAMES}
        total_labels = 0
        for label_file in label_dir.glob('*.txt'):
            with open(label_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls = int(parts[0])
                        counts[cls] = counts.get(cls, 0) + 1
                        total_labels += 1
        print(f"  {split}: {total_labels} annotations")
        for cls_id, name in CLASS_NAMES.items():
            print(f"    {name}: {counts.get(cls_id, 0)}")


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
