"""CARLA simulation helper functions for spawning vehicles, cameras, and weather."""

import carla
import random

from capstone_sim.scripts.utils.constants import BLUEPRINT_TO_CLASS, CLASS_NAME_TO_ID


def spawn_camera(world, bp_lib, camera_config):
    """Spawn an RGB camera from config."""
    cam_loc = carla.Location(
        x=camera_config['location']['x'],
        y=camera_config['location']['y'],
        z=camera_config['location']['z']
    )
    cam_rot = carla.Rotation(
        pitch=camera_config['rotation']['pitch'],
        yaw=camera_config['rotation']['yaw'],
        roll=camera_config['rotation']['roll']
    )
    cam_tf = carla.Transform(cam_loc, cam_rot)

    cam_bp = bp_lib.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', str(camera_config.get('image_width', 1280)))
    cam_bp.set_attribute('image_size_y', str(camera_config.get('image_height', 720)))
    cam_bp.set_attribute('fov', str(camera_config.get('fov', 70)))
    cam_bp.set_attribute('motion_blur_intensity', '0.0')
    cam_bp.set_attribute('exposure_mode', 'manual')
    cam_bp.set_attribute('exposure_compensation', '-1.5')
    cam_bp.set_attribute('exposure_min_bright', '0.5')
    cam_bp.set_attribute('exposure_max_bright', '2.0')
    cam_bp.set_attribute('gamma', '2.2')
    cam_bp.set_attribute('lens_flare_intensity', '0.1')
    cam_bp.set_attribute('bloom_intensity', '0.3')

    camera = world.spawn_actor(cam_bp, cam_tf)
    return camera


def apply_weather(world, weather_config):
    """Apply weather settings from config."""
    weather = carla.WeatherParameters(
        cloudiness=weather_config.get('cloudiness', 10.0),
        precipitation=weather_config.get('precipitation', 0.0),
        sun_altitude_angle=weather_config.get('sun_altitude_angle', 45.0),
        fog_density=weather_config.get('fog_density', 0.0),
        wetness=weather_config.get('wetness', 0.0)
    )
    world.set_weather(weather)


def get_available_blueprints(bp_lib):
    """Check which blueprints from our mapping actually exist in this CARLA build."""
    available = {}
    for bp_name, class_id in BLUEPRINT_TO_CLASS.items():
        results = bp_lib.filter(bp_name)
        if len(results) > 0:
            available[bp_name] = class_id
    return available


def build_class_blueprint_map(available_bps):
    """Group available blueprints by class_id."""
    class_bps = {}
    for bp_name, class_id in available_bps.items():
        class_bps.setdefault(class_id, []).append(bp_name)
    return class_bps


def compute_target_counts(ratios, max_vehicles):
    """Convert ratio dict to target vehicle counts per class."""
    total_ratio = sum(ratios.values())
    if total_ratio == 0:
        return {}
    targets = {}
    for class_name, ratio in ratios.items():
        class_id = CLASS_NAME_TO_ID.get(class_name)
        if class_id is not None:
            count = round(ratio / total_ratio * max_vehicles)
            if count > 0:
                targets[class_id] = count

    # If rounding caused total to exceed max_vehicles, trim the largest classes
    while sum(targets.values()) > max_vehicles:
        largest = max(targets, key=targets.get)
        targets[largest] -= 1
        if targets[largest] == 0:
            del targets[largest]
    return targets


def spawn_to_fill(world, bp_lib, tm_port, class_bps, target_counts,
                  current_vehicles, spawn_points):
    """Spawn vehicles to reach target counts per class."""
    current_counts = {}
    alive = []
    for actor, class_id in current_vehicles:
        if actor.is_alive:
            current_counts[class_id] = current_counts.get(class_id, 0) + 1
            alive.append((actor, class_id))

    current_vehicles.clear()
    current_vehicles.extend(alive)

    spawn_list = []
    for class_id, target in target_counts.items():
        current = current_counts.get(class_id, 0)
        needed = target - current
        bps = class_bps.get(class_id, [])
        if not bps or needed <= 0:
            continue
        for i in range(needed):
            bp_name = bps[i % len(bps)]
            spawn_list.append((bp_name, class_id))

    random.shuffle(spawn_list)

    occupied_locations = [
        actor.get_transform().location for actor, _ in current_vehicles
        if actor.is_alive
    ]
    min_spawn_gap = 8.0

    available_sp = []
    for sp in spawn_points:
        too_close = False
        for occ in occupied_locations:
            if sp.location.distance(occ) < min_spawn_gap:
                too_close = True
                break
        if not too_close:
            available_sp.append(sp)

    random.shuffle(available_sp)

    spawned = 0
    sp_idx = 0
    for bp_name, class_id in spawn_list:
        if sp_idx >= len(available_sp):
            break

        bp = bp_lib.find(bp_name)
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)

        actor = world.try_spawn_actor(bp, available_sp[sp_idx])
        sp_idx += 1

        if actor is not None:
            actor.set_autopilot(True, tm_port)
            current_vehicles.append((actor, class_id))
            occupied_locations.append(actor.get_transform().location)
            spawned += 1

    return spawned


def despawn_far_vehicles(current_vehicles, reference_location, despawn_distance):
    """Remove vehicles that are too far from the reference point."""
    removed = 0
    alive = []
    for actor, class_id in current_vehicles:
        if not actor.is_alive:
            continue
        dist = actor.get_transform().location.distance(reference_location)
        if dist > despawn_distance:
            actor.destroy()
            removed += 1
        else:
            alive.append((actor, class_id))

    current_vehicles.clear()
    current_vehicles.extend(alive)
    return removed
