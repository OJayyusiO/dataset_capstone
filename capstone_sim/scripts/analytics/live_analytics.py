"""
Live Traffic Analytics on a running CARLA simulation.

Spawns a camera (and optionally traffic) from a scenario YAML, runs YOLO + ByteTrack
+ analytics in real time, and displays an annotated live feed.

Auto-computes calibration from the camera's known intrinsics + extrinsics —
no manual setup needed.

Saves per-track speed CSV, per-lane queue CSV, summary JSON, and (optionally)
the annotated video to a timestamped folder under capstone_sim/analytics_runs/.

Usage:
    python live_analytics.py <scenario.yaml> <model.pt>
    python live_analytics.py scenario.yaml best.pt --no-spawn
    python live_analytics.py scenario.yaml best.pt --save-video
"""

import argparse
import csv
import json
import queue
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import carla
import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from capstone_sim.scripts.utils.constants import CLASS_NAMES, CLASS_COLORS
from capstone_sim.scripts.utils.carla_helpers import (
    spawn_camera, apply_weather, get_available_blueprints,
    build_class_blueprint_map, compute_target_counts,
    spawn_to_fill, despawn_far_vehicles,
)
from capstone_sim.scripts.utils.light_state import LightStateProvider, draw_light_indicator
from capstone_sim.scripts.analytics.setup_analytics import auto_calibrate_from_carla
from capstone_sim.scripts.analytics.traffic_analytics import (
    SpeedTracker, QueueTracker, draw_detection, pixel_to_world,
    compute_queue_counts, draw_lanes_overlay,
    vehicle_ground_point, ViolationDetector, draw_forbidden_lines,
    EntryCounter, draw_entry_zones,
    CollisionDetector, draw_collisions,
    DEFAULT_QUEUE_SPEED_KMH, DEFAULT_QUEUE_MIN_STATIONARY_SECONDS,
    DEFAULT_COLLISION_IOU, DEFAULT_COLLISION_SPEED_DROP_KMH,
    DEFAULT_COLLISION_WORLD_DIST_M, DEFAULT_COLLISION_WINDOW_SECONDS,
)

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics not installed. Run: pip install ultralytics")
    sys.exit(1)


def build_calibration_from_scenario(scenario_config, camera_index=0):
    """Build a temporary recording_meta.yaml-style dict and reuse auto_calibrate."""
    if 'cameras' in scenario_config:
        camera_configs = scenario_config['cameras']
    else:
        camera_configs = [scenario_config['camera']]

    cameras_meta = []
    for ci, cc in enumerate(camera_configs):
        cameras_meta.append({
            'index': ci,
            'location': cc['location'],
            'rotation': cc['rotation'],
            'image_width': cc.get('image_width', 1280),
            'image_height': cc.get('image_height', 720),
            'fov': cc.get('fov', 70),
        })

    # auto_calibrate_from_carla expects a file path, so write a temp file
    import tempfile
    meta_dict = {'cameras': cameras_meta}
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tf:
        yaml.dump(meta_dict, tf)
        tmp_path = Path(tf.name)
    try:
        cal = auto_calibrate_from_carla(tmp_path, camera_index=camera_index)
    finally:
        tmp_path.unlink()
    return cal


def run(scenario_path, model_path, save_video, spawn_traffic, conf, iou,
        analytics_config_path, output_dir, detect_collisions=False):
    scenario_path = Path(scenario_path)
    with open(scenario_path) as f:
        scenario = yaml.safe_load(f)

    # Output folder: timestamped subdir under analytics_runs/
    if output_dir is None:
        analytics_runs = Path(__file__).resolve().parents[2] / 'analytics_runs'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = analytics_runs / f"{scenario_path.stem}_{timestamp}"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-find analytics config — look in capstone_sim/analytics_configs/<scenario>.yaml
    if analytics_config_path is None:
        analytics_dir = Path(__file__).resolve().parents[2] / 'analytics_configs'
        candidate = analytics_dir / f"{scenario_path.stem}.yaml"
        if candidate.exists():
            analytics_config_path = candidate

    lanes = []
    queue_cfg = {}
    forbidden_lines = []
    entry_zones = []
    collision_cfg = {}
    if analytics_config_path:
        with open(analytics_config_path) as f:
            ac = yaml.safe_load(f)
        lanes = ac.get('lanes', [])
        queue_cfg = ac.get('queue', {})
        forbidden_lines = ac.get('forbidden_lines', [])
        entry_zones = ac.get('entry_zones', [])
        collision_cfg = ac.get('collision', {})
        if 'calibration' in ac:
            calibration = ac['calibration']
            print(f"Using calibration + {len(lanes)} lane(s) from {analytics_config_path}")
            H = np.array(calibration['homography_matrix'])
        else:
            calibration = build_calibration_from_scenario(scenario, camera_index=0)
            H = np.array(calibration['homography_matrix'])
    else:
        # No analytics config — auto-calibrate, no lanes
        print("No analytics_config.yaml found next to scenario. Run setup_analytics.py first to define lanes.")
        calibration = build_calibration_from_scenario(scenario, camera_index=0)
        if calibration is None:
            print("Auto-calibration failed (camera may not see the ground).")
            sys.exit(1)
        H = np.array(calibration['homography_matrix'])

    queue_speed_kmh = queue_cfg.get('speed_threshold_kmh', DEFAULT_QUEUE_SPEED_KMH)
    queue_min_seconds = queue_cfg.get('min_stationary_seconds', DEFAULT_QUEUE_MIN_STATIONARY_SECONDS)
    violation_detector = ViolationDetector(forbidden_lines)
    entry_counter = EntryCounter(entry_zones)
    collision_detector = None  # built after fps is known

    # Camera params (use first camera)
    if 'cameras' in scenario:
        cam_cfg = scenario['cameras'][0]
    else:
        cam_cfg = scenario['camera']
    image_w = cam_cfg.get('image_width', 1280)
    image_h = cam_cfg.get('image_height', 720)
    fps = 1.0 / scenario.get('simulation', {}).get('fixed_delta_seconds', 0.05)

    # Now that we know fps, build the queue tracker
    queue_tracker = QueueTracker(queue_speed_kmh, queue_min_seconds, fps)
    if detect_collisions:
        collision_detector = CollisionDetector(
            iou_threshold=collision_cfg.get('iou_threshold', DEFAULT_COLLISION_IOU),
            speed_drop_kmh=collision_cfg.get('speed_drop_kmh', DEFAULT_COLLISION_SPEED_DROP_KMH),
            world_dist_m=collision_cfg.get('world_distance_m', DEFAULT_COLLISION_WORLD_DIST_M),
            window_seconds=collision_cfg.get('window_seconds', DEFAULT_COLLISION_WINDOW_SECONDS),
            fps=fps,
        )

    print("=" * 60)
    print("Live Traffic Analytics on CARLA")
    print("=" * 60)
    print(f"Scenario:    {scenario_path}")
    print(f"Model:       {model_path}")
    print(f"Resolution:  {image_w}x{image_h} @ {fps:.1f} FPS")
    print(f"Calibration: {calibration['mode']}")
    print(f"Lanes:       {len(lanes)}")
    print(f"Queue:       slower than {queue_speed_kmh:.1f} km/h for {queue_min_seconds:.1f}+ sec")
    print(f"Output:      {output_dir.resolve()}")
    if save_video:
        print(f"Save video:  enabled")
    print("=" * 60)
    print("\nPress 'q' in the preview window to stop.\n")

    model = YOLO(model_path)
    speed_tracker = SpeedTracker(fps=fps, homography=H)

    # Connect to CARLA
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    target_map = scenario.get('map')
    if target_map:
        current_map = client.get_world().get_map().name.split('/')[-1]
        if current_map != target_map:
            print(f"Loading map {target_map}...")
            client.load_world(target_map)
            time.sleep(5)

    world = client.get_world()
    world.unload_map_layer(carla.MapLayer.ParkedVehicles)
    bp_lib = world.get_blueprint_library()

    # Synchronous mode
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / fps
    world.apply_settings(settings)
    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)
    random.seed(42)

    actor_list = []
    current_vehicles = []
    image_queue = queue.Queue()
    writer = None

    try:
        apply_weather(world, scenario.get('weather', {}))
        camera = spawn_camera(world, bp_lib, cam_cfg)
        actor_list.append(camera)
        camera.listen(image_queue.put)
        world.tick()

        # Find the scenario's traffic light for live light-state readout.
        # CARLA assigns traffic light actor IDs at world load, and they are NOT
        # guaranteed stable across sessions. So: try the stored id first, then
        # fall back to the traffic light nearest the camera (almost always the
        # one controlling the intersection in view).
        light_provider = LightStateProvider(mode='none')
        light_id = scenario.get('traffic_light', {}).get('id')
        all_lights = list(world.get_actors().filter('traffic.traffic_light'))
        selected_light = None
        if light_id:
            selected_light = next((l for l in all_lights if l.id == light_id), None)
            if selected_light:
                print(f"Reading live state from traffic light {light_id}")
        if selected_light is None and all_lights:
            cam_loc = camera.get_transform().location
            selected_light = min(
                all_lights,
                key=lambda l: l.get_transform().location.distance(cam_loc),
            )
            print(f"Traffic light id {light_id} not found this session; "
                  f"using nearest light (id {selected_light.id}) to the camera instead")
        if selected_light is not None:
            light_provider = LightStateProvider.from_carla(selected_light)

        if save_video:
            video_path = output_dir / 'live_analytics.mp4'
            writer = cv2.VideoWriter(str(video_path),
                                     cv2.VideoWriter_fourcc(*'mp4v'),
                                     fps, (image_w, image_h))

        # CSV loggers
        csv_path = output_dir / 'per_track.csv'
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['frame', 'track_id', 'class', 'world_x', 'world_y', 'speed_mps', 'speed_kmh'])

        queue_csv_file = None
        queue_csv_writer = None
        if lanes:
            queue_csv_path = output_dir / 'per_lane_queue.csv'
            queue_csv_file = open(queue_csv_path, 'w', newline='')
            queue_csv_writer = csv.writer(queue_csv_file)
            queue_csv_writer.writerow(['frame'] + [lane['id'] for lane in lanes])

        violation_csv_file = None
        violation_csv_writer = None
        if forbidden_lines:
            violation_csv_path = output_dir / 'violations.csv'
            violation_csv_file = open(violation_csv_path, 'w', newline='')
            violation_csv_writer = csv.writer(violation_csv_file)
            violation_csv_writer.writerow(['frame', 'track_id', 'line_id', 'light_state'])

        entry_csv_file = None
        entry_csv_writer = None
        if entry_zones:
            entry_csv_path = output_dir / 'entries.csv'
            entry_csv_file = open(entry_csv_path, 'w', newline='')
            entry_csv_writer = csv.writer(entry_csv_file)
            entry_csv_writer.writerow(['frame', 'track_id', 'zone_id', 'light_state'])

        collision_csv_file = None
        collision_csv_writer = None
        if collision_detector:
            collision_csv_path = output_dir / 'collisions.csv'
            collision_csv_file = open(collision_csv_path, 'w', newline='')
            collision_csv_writer = csv.writer(collision_csv_file)
            collision_csv_writer.writerow(['frame', 'track_a', 'track_b', 'world_dist_m'])

        # Stats trackers for summary
        max_queue_per_lane = defaultdict(int)
        speed_samples = []  # all per-detection speeds in m/s
        total_violations = 0
        total_entries = 0
        total_collisions = 0

        # Set up spawn lifecycle from scenario YAML (matches record_test.py)
        spawn_config = scenario.get('spawn', {})
        ratios = spawn_config.get('ratios', {'car': 10})
        max_vehicles = spawn_config.get('max_vehicles', 20)
        respawn_interval = spawn_config.get('respawn_interval', 50)
        despawn_distance = spawn_config.get('despawn_distance', 150.0)
        force_respawn_interval = spawn_config.get('force_respawn_interval', 0)
        stuck_check_interval = spawn_config.get('stuck_check_interval', 600)
        stuck_threshold_m = spawn_config.get('stuck_threshold_m', 1.0)

        target_counts = compute_target_counts(ratios, max_vehicles)
        available_bps = get_available_blueprints(bp_lib)
        class_bps = build_class_blueprint_map(available_bps)
        all_spawn_points = world.get_map().get_spawn_points()

        # Resolve which spawn points to use (custom list + map indices + radius fallback)
        cam_loc = camera.get_transform().location
        radius_center = spawn_config.get('radius_center', 'camera')
        # For live mode, default reference is the camera (no traffic_light_id by default)
        reference_location = cam_loc

        raw_spawn_points = spawn_config.get('spawn_points')
        selected_indices = []
        if raw_spawn_points:
            for entry in raw_spawn_points:
                if isinstance(entry, int):
                    selected_indices.append(entry)
                elif isinstance(entry, str) and '-' in entry:
                    parts = entry.split('-')
                    selected_indices.extend(range(int(parts[0]), int(parts[1]) + 1))

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
            spawn_points.extend(custom_transforms)
            print(f"Using {len(spawn_points)} spawn points "
                  f"({len(selected_indices)} from map, {len(custom_transforms)} custom)")
        else:
            spawn_radius = spawn_config.get('spawn_radius', 80.0)
            spawn_points = [sp for sp in all_spawn_points
                            if sp.location.distance(reference_location) <= spawn_radius] or all_spawn_points
            print(f"Using {len(spawn_points)} spawn points within {spawn_radius}m of camera")

        # Initial spawn
        if spawn_traffic:
            spawn_to_fill(world, bp_lib, tm.get_port(), class_bps,
                          target_counts, current_vehicles, spawn_points)
            for v, cls_id in current_vehicles:
                if cls_id == 6:
                    tm.vehicle_percentage_speed_difference(v, 50.0)
                else:
                    tm.vehicle_percentage_speed_difference(v, 30.0)
            print(f"Spawned {len(current_vehicles)} vehicles")

        cv2.namedWindow('Live Analytics', cv2.WINDOW_NORMAL)
        frame_idx = 0
        start_time = time.time()
        track_ids_seen = set()
        last_check_positions = {}  # for stuck detection
        ignore_lights = False      # toggled with 'k' — makes vehicles run red lights

        def apply_ignore_lights(pct):
            for v, _ in current_vehicles:
                if v.is_alive:
                    try:
                        tm.ignore_lights_percentage(v, pct)
                    except RuntimeError:
                        pass

        print("Controls: 'k' = toggle vehicles ignoring red lights, 'q' = quit")

        while True:
            world.tick()

            # --- Spawn lifecycle (only if spawn_traffic enabled) ---
            if spawn_traffic:
                # Force respawn: kill all and respawn
                if force_respawn_interval > 0 and frame_idx > 0 and frame_idx % force_respawn_interval == 0:
                    for actor, _ in current_vehicles:
                        if actor.is_alive:
                            actor.destroy()
                    current_vehicles.clear()
                    last_check_positions.clear()
                    world.tick()
                    spawn_to_fill(world, bp_lib, tm.get_port(), class_bps,
                                  target_counts, current_vehicles, spawn_points)
                    for v, cls_id in current_vehicles:
                        if cls_id == 6:
                            tm.vehicle_percentage_speed_difference(v, 50.0)
                        else:
                            tm.vehicle_percentage_speed_difference(v, 30.0)
                    print(f"  Forced respawn: {len(current_vehicles)} fresh vehicles")

                # Normal respawn cycle: despawn far, spawn new
                elif frame_idx > 0 and frame_idx % respawn_interval == 0:
                    despawn_far_vehicles(current_vehicles, reference_location, despawn_distance)
                    spawn_to_fill(world, bp_lib, tm.get_port(), class_bps,
                                  target_counts, current_vehicles, spawn_points)
                    for v, cls_id in current_vehicles:
                        try:
                            if cls_id == 6:
                                tm.vehicle_percentage_speed_difference(v, 50.0)
                            else:
                                tm.vehicle_percentage_speed_difference(v, 30.0)
                        except RuntimeError:
                            pass

                # Stuck vehicle check
                if stuck_check_interval > 0 and frame_idx > 0 and frame_idx % stuck_check_interval == 0:
                    alive = []
                    stuck_count = 0
                    for actor, cls_id in current_vehicles:
                        if not actor.is_alive:
                            continue
                        current_pos = actor.get_transform().location
                        actor_id = actor.id
                        if actor_id in last_check_positions:
                            distance_moved = current_pos.distance(last_check_positions[actor_id])
                            if distance_moved < stuck_threshold_m:
                                actor.destroy()
                                stuck_count += 1
                                del last_check_positions[actor_id]
                                continue
                        last_check_positions[actor_id] = current_pos
                        alive.append((actor, cls_id))
                    current_vehicles.clear()
                    current_vehicles.extend(alive)
                    if stuck_count > 0:
                        spawn_to_fill(world, bp_lib, tm.get_port(), class_bps,
                                      target_counts, current_vehicles, spawn_points)
                        for v, cls_id in current_vehicles:
                            try:
                                if cls_id == 6:
                                    tm.vehicle_percentage_speed_difference(v, 50.0)
                                else:
                                    tm.vehicle_percentage_speed_difference(v, 30.0)
                            except RuntimeError:
                                pass
                        print(f"  Removed {stuck_count} stuck vehicle(s)")

            # If "ignore lights" is on, keep newly spawned vehicles running reds too
            if ignore_lights:
                apply_ignore_lights(100.0)

            # Drain queue, keep latest
            latest = None
            try:
                while True:
                    latest = image_queue.get_nowait()
            except queue.Empty:
                pass
            if latest is None:
                continue

            img_data = np.array(latest.raw_data)
            frame = img_data.reshape((image_h, image_w, 4))[:, :, :3].copy()

            results = model.track(
                source=frame, conf=conf, iou=iou,
                persist=True, tracker='bytetrack.yaml', verbose=False,
            )

            frame_detections = []
            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                for j in range(len(boxes)):
                    xyxy = boxes.xyxy[j].cpu().numpy()
                    cls = int(boxes.cls[j].cpu().item())
                    conf_score = float(boxes.conf[j].cpu().item())
                    track_id = int(boxes.id[j].cpu().item()) if boxes.id is not None else -1
                    if track_id == -1:
                        continue
                    track_ids_seen.add(track_id)
                    speed_mps = speed_tracker.update(frame_idx, track_id, xyxy)
                    ref_x, ref_y = vehicle_ground_point(xyxy)
                    world_pos = pixel_to_world(H, ref_x, ref_y) or (0, 0)
                    frame_detections.append({
                        'point': (ref_x, ref_y),
                        'speed_mps': speed_mps,
                        'track_id': track_id,
                        'bbox': [float(v) for v in xyxy],
                        'world': world_pos,
                    })
                    draw_detection(frame, xyxy, cls, track_id, conf_score, speed_mps * 3.6)

                    # Log per-track CSV row
                    csv_writer.writerow([
                        frame_idx, track_id, CLASS_NAMES.get(cls, cls),
                        round(world_pos[0], 3), round(world_pos[1], 3),
                        round(speed_mps, 3), round(speed_mps * 3.6, 1),
                    ])
                    speed_samples.append(speed_mps)

            # Annotate detections with is_queued (uses speed + min-stationary-frames)
            queue_tracker.annotate(frame_detections)

            # Per-lane queue counts overlay + CSV
            if lanes:
                queue_counts = compute_queue_counts(lanes, frame_detections)
                draw_lanes_overlay(frame, lanes, queue_counts)
                if queue_csv_writer:
                    queue_csv_writer.writerow([frame_idx] + [queue_counts.get(l['id'], 0) for l in lanes])
                for lid, count in queue_counts.items():
                    if count > max_queue_per_lane[lid]:
                        max_queue_per_lane[lid] = count

            # Current light state for this frame
            light_state = light_provider.state_at(frame_idx) if light_provider.available else 'unknown'

            # Red-light violations
            violations_now = []
            if forbidden_lines:
                violations_now = violation_detector.check(frame_detections, light_state, frame_idx)
                for v in violations_now:
                    total_violations += 1
                    if violation_csv_writer:
                        violation_csv_writer.writerow([v['frame'], v['track_id'], v['line_id'], light_state])
                    print(f"  VIOLATION: track #{v['track_id']} crossed {v['line_id']} on red (frame {frame_idx})")
                draw_forbidden_lines(frame, forbidden_lines, active_violation=bool(violations_now))

            # Highway entry counting
            if entry_zones:
                entries_now = entry_counter.check(frame_detections, light_state, frame_idx)
                for e in entries_now:
                    total_entries += 1
                    if entry_csv_writer:
                        entry_csv_writer.writerow([e['frame'], e['track_id'], e['zone_id'], e['light_state']])
                draw_entry_zones(frame, entry_zones, entry_counter)

            # Collision detection (experimental)
            collisions_now = []
            if collision_detector:
                collisions_now = collision_detector.check(frame_detections, frame_idx)
                for c in collisions_now:
                    total_collisions += 1
                    if collision_csv_writer:
                        collision_csv_writer.writerow([c['frame'], c['track_a'], c['track_b'], c['world_dist_m']])
                    print(f"  COLLISION: tracks #{c['track_a']} & #{c['track_b']} "
                          f"({c['world_dist_m']}m apart, frame {frame_idx})")
                draw_collisions(frame, frame_detections, collisions_now)

            # Traffic light indicator
            if light_provider.available:
                draw_light_indicator(frame, light_state)

            # Violation banner
            if violations_now:
                vtxt = f"RED-LIGHT VIOLATION x{len(violations_now)}"
                (tw, th), _ = cv2.getTextSize(vtxt, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                cv2.rectangle(frame, (image_w // 2 - tw // 2 - 12, 50),
                              (image_w // 2 + tw // 2 + 12, 50 + th + 16), (0, 0, 255), -1)
                cv2.putText(frame, vtxt, (image_w // 2 - tw // 2, 50 + th + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            elapsed = time.time() - start_time
            real_fps = (frame_idx + 1) / max(elapsed, 0.001)
            hud = f"Frame {frame_idx}  |  {real_fps:.1f} FPS  |  Tracks: {len(track_ids_seen)}  |  Violations: {total_violations}"
            if entry_zones:
                hud += f"  |  Entries: {total_entries}"
            if collision_detector:
                hud += f"  |  Collisions: {total_collisions}"
            cv2.putText(frame, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)

            # "Ignore lights" status badge (toggled with 'k')
            if ignore_lights:
                badge = "IGNORE LIGHTS: ON (k)"
                (bw, bh), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (10, 40), (10 + bw + 12, 40 + bh + 14), (0, 0, 200), -1)
                cv2.putText(frame, badge, (16, 40 + bh + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow('Live Analytics', frame)
            if writer is not None:
                writer.write(frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('k'):
                ignore_lights = not ignore_lights
                apply_ignore_lights(100.0 if ignore_lights else 0.0)
                print(f"  [k] Vehicles {'IGNORING' if ignore_lights else 'OBEYING'} traffic lights")

            frame_idx += 1

    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        # Close data files first so they're flushed no matter what
        if writer is not None:
            writer.release()
        csv_file.close()
        if queue_csv_file:
            queue_csv_file.close()
        if violation_csv_file:
            violation_csv_file.close()
        if entry_csv_file:
            entry_csv_file.close()
        if collision_csv_file:
            collision_csv_file.close()

        # Write summary JSON BEFORE the CARLA cleanup (which can throw if the
        # simulator state changed), so the run summary is always saved.
        try:
            elapsed = time.time() - start_time
            avg_speed_mps = float(np.mean(speed_samples)) if speed_samples else 0.0
            max_speed_mps = float(np.max(speed_samples)) if speed_samples else 0.0
            summary = {
                'scenario': str(scenario_path),
                'model': str(model_path),
                'run_started': datetime.fromtimestamp(start_time).isoformat(),
                'duration_seconds': round(elapsed, 1),
                'frames_processed': frame_idx,
                'inference_fps': round(frame_idx / max(elapsed, 0.001), 1),
                'unique_tracks': len(track_ids_seen),
                'total_detections': len(speed_samples),
                'avg_speed_kmh': round(avg_speed_mps * 3.6, 1),
                'max_speed_kmh': round(max_speed_mps * 3.6, 1),
                'max_queue_per_lane': dict(max_queue_per_lane),
                'total_red_light_violations': total_violations,
                'entry_counts': entry_counter.summary(),
                'total_collisions': total_collisions,
                'calibration_mode': calibration.get('mode'),
                'num_lanes': len(lanes),
                'num_forbidden_lines': len(forbidden_lines),
                'num_entry_zones': len(entry_zones),
            }
            with open(output_dir / 'summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"\nDone — {frame_idx} frames in {elapsed:.1f}s ({frame_idx / max(elapsed, 0.001):.1f} FPS)")
            print(f"  Unique tracks: {len(track_ids_seen)}")
            print(f"  Avg speed: {summary['avg_speed_kmh']} km/h, Max: {summary['max_speed_kmh']} km/h")
            if max_queue_per_lane:
                for lid, count in max_queue_per_lane.items():
                    print(f"  Max queue in {lid}: {count}")
            if forbidden_lines:
                print(f"  Red-light violations: {total_violations}")
            if entry_zones:
                print(f"  Entry counts: {entry_counter.summary()}")
            if collision_detector:
                print(f"  Collisions detected: {total_collisions}")
            print(f"\nResults saved to: {output_dir.resolve()}")
        except Exception as e:
            print(f"  Warning: could not write summary.json: {e}")

        # CARLA cleanup last (any failure here no longer loses the summary)
        cv2.destroyAllWindows()
        for actor in actor_list:
            if actor.is_alive:
                actor.destroy()
        for actor, _ in current_vehicles:
            if actor.is_alive:
                actor.destroy()
        world.apply_settings(original_settings)
        tm.set_synchronous_mode(False)


def main():
    parser = argparse.ArgumentParser(description='Live traffic analytics on a CARLA simulation')
    parser.add_argument('scenario', type=str, help='Path to scenario YAML (with camera + map)')
    parser.add_argument('model', type=str, help='Path to .pt YOLO model')
    parser.add_argument('--save-video', action='store_true',
                        help='Save the annotated live feed as live_analytics.mp4 in the run folder')
    parser.add_argument('--no-spawn', action='store_true',
                        help='Skip spawning vehicles (use if CARLA already has traffic running)')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--iou', type=float, default=0.5)
    parser.add_argument('--config', type=str, default=None,
                        help='Path to analytics_config.yaml with predefined lanes')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: capstone_sim/analytics_runs/<scenario>_<timestamp>/)')
    parser.add_argument('--collisions', action='store_true',
                        help='Enable experimental collision detection (bbox overlap + world proximity + speed drop)')
    args = parser.parse_args()

    if not Path(args.scenario).exists():
        print(f"Scenario not found: {args.scenario}")
        sys.exit(1)
    if not Path(args.model).exists():
        print(f"Model not found: {args.model}")
        sys.exit(1)
    if args.config and not Path(args.config).exists():
        print(f"Analytics config not found: {args.config}")
        sys.exit(1)

    spawn_traffic = not args.no_spawn
    run(args.scenario, args.model, args.save_video, spawn_traffic,
        args.conf, args.iou, args.config, args.output, args.collisions)


if __name__ == '__main__':
    main()
