"""
CARLA Test Footage Recorder

Records sequential frames and ground truth labels from a CARLA scenario
for use in model evaluation. Uses the same YAML config format as capture_dataset.py.

The output can be reused across multiple model evaluations.

Usage:
    python record_test.py scenario_config.yaml
    python record_test.py scenario_config.yaml --duration 2000 --fps 20
"""

import carla
import yaml
import sys
import queue
import cv2
import numpy as np
import random
import argparse
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from capstone_sim.scripts.utils.constants import (
    CLASS_NAMES, CLASS_NAME_TO_ID, MAX_DISTANCE,
)
from capstone_sim.scripts.utils.bbox import (
    build_projection_matrix, get_2d_bbox, bbox_to_yolo,
)
from capstone_sim.scripts.utils.carla_helpers import (
    spawn_camera, apply_weather,
    get_available_blueprints, build_class_blueprint_map, compute_target_counts,
    spawn_to_fill, despawn_far_vehicles,
)


def get_gt_labels(current_vehicles, camera, K, image_w, image_h, camera_location):
    """Compute ground truth labels for all visible vehicles.

    Returns list of strings: 'class_id x_center y_center w h actor_id'
    """
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
        # Append CARLA actor ID for tracking ground truth
        labels.append(f"{yolo_label} {vehicle.id}")
    return labels


def run_recording(config_path, output_base, duration, fps):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    camera_config = config['camera']
    weather_config = config.get('weather', {})
    spawn_config = config.get('spawn', {})

    image_w = camera_config.get('image_width', 1280)
    image_h = camera_config.get('image_height', 720)
    fov = camera_config.get('fov', 70)

    fixed_delta = 1.0 / fps
    total_frames = duration
    warmup_frames = config.get('simulation', {}).get('warmup_frames', 100)

    max_vehicles = spawn_config.get('max_vehicles', 30)
    respawn_interval = spawn_config.get('respawn_interval', 50)
    despawn_distance = spawn_config.get('despawn_distance', 150.0)
    ratios = spawn_config.get('ratios', {
        'car': 15, 'ambulance': 2, 'bus': 2, 'truck': 3,
        'police_car': 2, 'fire_truck': 1, 'bike': 4
    })

    target_counts = compute_target_counts(ratios, max_vehicles)
    K = build_projection_matrix(image_w, image_h, fov)

    # Create output directory
    scenario_name = config.get('scenario_name', 'test')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    recording_dir = Path(output_base) / f"{scenario_name}_{timestamp}"
    frames_dir = recording_dir / 'frames'
    gt_dir = recording_dir / 'ground_truth'
    frames_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CARLA Test Footage Recorder")
    print("=" * 60)
    print(f"Scenario: {scenario_name}")
    print(f"Output: {recording_dir.resolve()}")
    print(f"Frames: {total_frames} ({warmup_frames} warmup)")
    print(f"FPS: {fps} (fixed_delta={fixed_delta:.4f})")
    print(f"Max vehicles: {max_vehicles}")
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

    available_bps = get_available_blueprints(bp_lib)
    class_bps = build_class_blueprint_map(available_bps)

    image_queue = queue.Queue()
    current_vehicles = []
    actor_list = []

    try:
        apply_weather(world, weather_config)
        print("Weather applied")

        camera = spawn_camera(world, bp_lib, camera_config)
        actor_list.append(camera)
        camera.listen(image_queue.put)
        world.tick()
        camera_location = camera.get_transform().location
        print(f"Camera at ({camera_location.x:.1f}, {camera_location.y:.1f}, {camera_location.z:.1f})")

        # Find traffic light and set reference location
        spawn_radius = spawn_config.get('spawn_radius', 100.0)
        light_id = config.get('traffic_light', {}).get('id')
        light_location = None
        if light_id:
            traffic_lights = list(world.get_actors().filter('traffic.traffic_light'))
            selected_light = next((l for l in traffic_lights if l.id == light_id), None)
            if selected_light:
                light_location = selected_light.get_transform().location
                print(f"Traffic light {light_id} at ({light_location.x:.1f}, {light_location.y:.1f})")

        radius_center = spawn_config.get('radius_center', 'traffic_light')
        if radius_center == 'camera':
            reference_location = camera_location
        else:
            reference_location = light_location if light_location else camera_location

        # Filter spawn points
        raw_spawn_points = spawn_config.get('spawn_points')
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
            print(f"Using {len(spawn_points)} manually selected spawn points")
        else:
            spawn_points = [
                sp for sp in all_spawn_points
                if sp.location.distance(reference_location) <= spawn_radius
            ]
            if len(spawn_points) == 0:
                spawn_points = all_spawn_points
            print(f"Using {len(spawn_points)}/{len(all_spawn_points)} spawn points")

        vehicles_spawned = False
        frame_counter = 0
        start_time = time.time()

        print(f"\nRecording {total_frames} frames...")
        print("Press Ctrl+C to stop early\n")

        for frame in range(total_frames):
            world.tick()

            latest_image = None
            try:
                while True:
                    latest_image = image_queue.get_nowait()
            except queue.Empty:
                pass

            # Warmup: record frames but don't spawn vehicles yet
            if frame < warmup_frames:
                if latest_image is not None:
                    img_data = np.array(latest_image.raw_data)
                    img = img_data.reshape((image_h, image_w, 4))[:, :, :3].copy()
                    cv2.imwrite(str(frames_dir / f"{frame_counter:06d}.png"), img)
                    with open(gt_dir / f"{frame_counter:06d}.txt", 'w') as f:
                        pass  # Empty label file — no vehicles yet
                    frame_counter += 1
                if frame == warmup_frames - 1:
                    print(f"Warmup complete ({warmup_frames} frames)")
                continue

            # Spawn vehicles after warmup
            if not vehicles_spawned:
                spawned = spawn_to_fill(world, bp_lib, tm_port, class_bps,
                                        target_counts, current_vehicles, spawn_points)
                print(f"Spawned {spawned} vehicles")
                for v, cls_id in current_vehicles:
                    if cls_id == 6:
                        traffic_manager.vehicle_percentage_speed_difference(v, 50.0)
                    else:
                        traffic_manager.vehicle_percentage_speed_difference(v, 30.0)
                vehicles_spawned = True

            # Respawn cycle
            if frame > 0 and frame % respawn_interval == 0:
                despawn_far_vehicles(current_vehicles, reference_location, despawn_distance)
                spawn_to_fill(world, bp_lib, tm_port, class_bps,
                              target_counts, current_vehicles, spawn_points)
                for v, cls_id in current_vehicles:
                    try:
                        if cls_id == 6:
                            traffic_manager.vehicle_percentage_speed_difference(v, 50.0)
                        else:
                            traffic_manager.vehicle_percentage_speed_difference(v, 30.0)
                    except RuntimeError:
                        pass

            if latest_image is None:
                continue

            # Save frame
            img_data = np.array(latest_image.raw_data)
            img = img_data.reshape((image_h, image_w, 4))[:, :, :3].copy()
            cv2.imwrite(str(frames_dir / f"{frame_counter:06d}.png"), img)

            # Save ground truth
            labels = get_gt_labels(current_vehicles, camera, K, image_w, image_h, camera_location)
            with open(gt_dir / f"{frame_counter:06d}.txt", 'w') as f:
                for label in labels:
                    f.write(label + '\n')

            frame_counter += 1

            # Progress
            if frame_counter % 200 == 0:
                elapsed = time.time() - start_time
                sim_fps = frame / max(elapsed, 0.001)
                alive = sum(1 for v, _ in current_vehicles if v.is_alive)
                print(f"  Frame {frame_counter}/{total_frames} | "
                      f"Vehicles: {alive} | "
                      f"GT objects: {len(labels)} | "
                      f"{sim_fps:.1f} sim fps")

        print(f"\nRecording complete ({frame_counter} frames saved)")

    except KeyboardInterrupt:
        print("\n\nStopped by user")

    finally:
        # Save recording metadata first (before cleanup which may fail)
        meta = {
            'scenario_config': str(Path(config_path).name),
            'scenario_name': scenario_name,
            'recording_date': datetime.now().isoformat(),
            'num_frames': frame_counter,
            'image_width': image_w,
            'image_height': image_h,
            'fov': fov,
            'fps': fps,
            'fixed_delta_seconds': fixed_delta,
            'class_names': CLASS_NAMES,
        }
        with open(recording_dir / 'recording_meta.yaml', 'w') as f:
            yaml.dump(meta, f, default_flow_style=False, sort_keys=False)

        camera.stop()
        for actor in actor_list:
            if actor.is_alive:
                actor.destroy()
        for actor, _ in current_vehicles:
            if actor.is_alive:
                actor.destroy()
        world.apply_settings(original_settings)
        traffic_manager.set_synchronous_mode(False)

    elapsed = time.time() - start_time
    print(f"\nSaved to: {recording_dir.resolve()}")
    print(f"Frames: {frame_counter}")
    print(f"Time: {elapsed/60:.1f} minutes")


def main():
    parser = argparse.ArgumentParser(description='Record test footage from CARLA scenario')
    parser.add_argument('config', type=str, help='Path to scenario config YAML')
    parser.add_argument('--output', type=str, default='./test_recordings',
                        help='Base directory for recordings (default: ./test_recordings)')
    parser.add_argument('--duration', type=int, default=None,
                        help='Number of simulation frames (default: from config total_frames)')
    parser.add_argument('--fps', type=int, default=20,
                        help='Recording FPS (default: 20)')
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    # Get duration from config if not specified
    duration = args.duration
    if duration is None:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        duration = cfg.get('simulation', {}).get('total_frames', 2000)

    run_recording(str(config_path), args.output, duration, args.fps)


if __name__ == '__main__':
    main()
