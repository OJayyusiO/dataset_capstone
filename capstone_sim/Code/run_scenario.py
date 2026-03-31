"""
CARLA Scenario Runner
Runs a traffic light test scenario from a YAML config file.

Usage:
    python run_scenario.py scenario_config.yaml
"""

import carla
import yaml
import sys
import queue
import cv2
import numpy as np
import random
import argparse
from pathlib import Path


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def apply_weather(world, weather_config):
    weather = carla.WeatherParameters(
        cloudiness=weather_config.get('cloudiness', 10.0),
        precipitation=weather_config.get('precipitation', 0.0),
        sun_altitude_angle=weather_config.get('sun_altitude_angle', 45.0),
        fog_density=weather_config.get('fog_density', 0.0),
        wetness=weather_config.get('wetness', 0.0)
    )
    world.set_weather(weather)
    print(f"✓ Weather applied: cloudiness={weather.cloudiness}, sun_alt={weather.sun_altitude_angle}")


def spawn_camera(world, bp_lib, camera_config):
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
    
    # Quality improvements
    cam_bp.set_attribute('fov', '70')  # Narrower FOV for better detail (was 90)
    cam_bp.set_attribute('motion_blur_intensity', '0.0')  # Disable motion blur for sharpness
    cam_bp.set_attribute('exposure_mode', 'manual')
    cam_bp.set_attribute('exposure_compensation', '-1.5')  # Reduce brightness/overexposure
    cam_bp.set_attribute('exposure_min_bright', '0.5')
    cam_bp.set_attribute('exposure_max_bright', '2.0')
    cam_bp.set_attribute('gamma', '2.2')  # Standard gamma
    cam_bp.set_attribute('lens_flare_intensity', '0.1')  # Minimal lens flare
    cam_bp.set_attribute('bloom_intensity', '0.3')  # Reduce bloom (glow)
    
    camera = world.spawn_actor(cam_bp, cam_tf)
    print(f"✓ Camera spawned at ({cam_loc.x:.1f}, {cam_loc.y:.1f}, {cam_loc.z:.1f})")
    return camera


def get_traffic_light(world, light_id):
    traffic_lights = list(world.get_actors().filter('traffic.traffic_light'))
    light = next((l for l in traffic_lights if l.id == light_id), None)
    if not light:
        print(f"❌ Traffic light {light_id} not found.")
        return None, None
    
    stop_wps = light.get_stop_waypoints()
    if not stop_wps:
        print(f"❌ Traffic light {light_id} has no stop waypoints.")
        return None, None
    
    print(f"✓ Traffic light {light_id} controls {len(stop_wps)} lane(s)")
    return light, stop_wps


def spawn_vehicles_in_lane(world, bp_lib, stop_wp, lane_config, tm_port, tm_config):
    """Spawn vehicles in a single lane behind the stop line."""
    vehicles = []
    num_vehicles = lane_config.get('num_vehicles', 0)
    spawn_distances = lane_config.get('spawn_distances', [])
    vehicle_types = lane_config.get('vehicle_types', ['vehicle.tesla.model3'])
    
    # Ensure we have enough distances and types
    if len(spawn_distances) < num_vehicles:
        spawn_distances = spawn_distances + [spawn_distances[-1] + 10.0] * (num_vehicles - len(spawn_distances))
    if len(vehicle_types) < num_vehicles:
        vehicle_types = vehicle_types * ((num_vehicles // len(vehicle_types)) + 1)
    
    for i in range(num_vehicles):
        distance = spawn_distances[i]
        wp_back = stop_wp.previous(distance)
        if not wp_back:
            print(f"  ⚠ Could not find spawn point {distance}m back in lane {lane_config.get('lane_id', 0)}")
            continue
        
        spawn_tf = wp_back[0].transform
        spawn_tf.location.z += 0.5
        
        vehicle_bp_name = vehicle_types[i]
        try:
            vehicle_bp = bp_lib.find(vehicle_bp_name)
        except:
            print(f"  ⚠ Vehicle blueprint '{vehicle_bp_name}' not found, using default")
            vehicle_bp = bp_lib.filter('vehicle.*')[0]
        
        # Random color
        if vehicle_bp.has_attribute('color'):
            color = random.choice(vehicle_bp.get_attribute('color').recommended_values)
            vehicle_bp.set_attribute('color', color)
        
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_tf)
        if vehicle:
            vehicle.set_autopilot(True, tm_port)
            vehicles.append(vehicle)
        else:
            print(f"  ⚠ Failed to spawn vehicle {i+1} in lane {lane_config.get('lane_id', 0)}")
    
    return vehicles


def configure_traffic_manager(traffic_manager, vehicles, tm_config):
    """Apply traffic manager settings to all vehicles."""
    speed_diff = tm_config.get('vehicle_speed_difference', 30.0)
    ignore_lights = tm_config.get('ignore_lights_percentage', 0.0)
    
    for vehicle in vehicles:
        traffic_manager.ignore_lights_percentage(vehicle, ignore_lights)
        traffic_manager.vehicle_percentage_speed_difference(vehicle, speed_diff)


def run_scenario(config_path):
    config = load_config(config_path)
    actor_list = []
    image_queue = queue.Queue()
    
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    
    # Force high quality rendering and disable LOD culling
    world.unload_map_layer(carla.MapLayer.ParkedVehicles)
    
    bp_lib = world.get_blueprint_library()
    
    timing = config.get('timing', {})
    fixed_delta = timing.get('fixed_delta_seconds', 0.05)
    red_seconds = timing.get('red_seconds', 8.0)
    green_seconds = timing.get('green_seconds', 8.0)
    
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = fixed_delta

    settings.substepping = True
    settings.max_substep_delta_time = 0.01  # Calculate physics every 0.01s (100 Hz)
    settings.max_substeps = 10


    world.apply_settings(settings)
    
    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_random_device_seed(42)
    random.seed(42)
    
    try:
        print(f"\nRunning scenario: {config.get('scenario_name', 'unnamed')}")
        print("=" * 60)
        
        # Weather
        apply_weather(world, config.get('weather', {}))
        
        # Camera
        camera = spawn_camera(world, bp_lib, config['camera'])
        actor_list.append(camera)
        camera.listen(image_queue.put)
        
        # Traffic light
        light_id = config['traffic_light']['id']
        selected_light, stop_wps = get_traffic_light(world, light_id)
        if not selected_light:
            return
        
        # Spawn vehicles per lane
        all_vehicles = []
        lanes_config = config.get('lanes', [])
        
        for lane_cfg in lanes_config:
            lane_id = lane_cfg.get('lane_id', 0)
            if lane_id >= len(stop_wps):
                print(f"  ⚠ Lane {lane_id} does not exist for this traffic light")
                continue
            
            print(f"\nSpawning vehicles in lane {lane_id}...")
            vehicles = spawn_vehicles_in_lane(
                world, bp_lib, stop_wps[lane_id], lane_cfg,
                traffic_manager.get_port(),
                config.get('traffic_manager', {})
            )
            all_vehicles.extend(vehicles)
            actor_list.extend(vehicles)
            print(f"  ✓ Spawned {len(vehicles)} vehicle(s) in lane {lane_id}")
        
        # Configure traffic manager
        configure_traffic_manager(traffic_manager, all_vehicles, config.get('traffic_manager', {}))
        
        # Set initial light state
        selected_light.set_state(carla.TrafficLightState.Red)
        selected_light.freeze(True)
        
        print("\n" + "=" * 60)
        print("SCENARIO RUNNING")
        print("=" * 60)
        print("Phase 1: RED (vehicles approach and stop)")
        
        red_frames = int(red_seconds / fixed_delta)
        green_frames = int(green_seconds / fixed_delta)
        total_frames = red_frames + green_frames
        
        img_h = config['camera'].get('image_height', 720)
        img_w = config['camera'].get('image_width', 1280)
        
        for frame in range(total_frames):
            if frame == red_frames:
                print("Phase 2: GREEN (vehicles proceed)")
                selected_light.set_state(carla.TrafficLightState.Green)
            
            status = "RED" if frame < red_frames else "GREEN"
            color = (0, 0, 255) if frame < red_frames else (0, 255, 0)
            
            world.tick()
            
            try:
                image = image_queue.get(timeout=2.0)
                img_data = np.array(image.raw_data)
                img = img_data.reshape((img_h, img_w, 4))[:, :, :3].copy()
                
                # UI overlay - minimal size
                cv2.rectangle(img, (5, 5), (135, 40), (0, 0, 0), -1)
                cv2.putText(img, f"{status}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                cv2.putText(img, f"F:{frame}/{total_frames}", (10, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
                
                cv2.imshow("CARLA Scenario", img)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    print("\nUser aborted (pressed 'q')")
                    break
            except queue.Empty:
                print("\n⚠ Camera feed timeout")
                break
        
        print("\n✓ Scenario complete")
        
    finally:
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        
        for actor in actor_list:
            if actor.is_alive:
                actor.destroy()
        
        cv2.destroyAllWindows()
        print("✓ Cleaned up")


def main():
    parser = argparse.ArgumentParser(description='Run CARLA traffic scenario from config')
    parser.add_argument('config', type=str, help='Path to scenario config YAML file')
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)
    
    try:
        run_scenario(config_path)
    except KeyboardInterrupt:
        print("\n\n✓ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
