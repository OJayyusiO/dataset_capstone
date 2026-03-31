"""
CARLA Scenario Setup Tool
Creates a config file by positioning the spectator camera and selecting a traffic light.

Usage:
1. Launch CARLA simulator
2. Run: python setup_scenario.py
3. In CARLA, fly to desired camera position
4. Press ENTER in terminal when camera is positioned
5. Script lists nearest traffic lights
6. Enter the traffic light ID you want
7. Config file is saved
"""

import carla
import yaml
import sys
from pathlib import Path


def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    
    # Slow down spectator for precise positioning
    spectator = world.get_spectator()
    current_velocity = spectator.get_velocity()
    
    # Set very slow movement speed (reduce by ~80%)
    # Note: This only affects WASD movement slightly, but helps with mouse
    print("\n✓ Spectator speed reduced for precise positioning")

    print("=" * 60)
    print("CARLA SCENARIO SETUP TOOL")
    print("=" * 60)
    print("\nSpectator Controls:")
    print("  WASD + Mouse = Move/Look")
    print("  Mouse Wheel = Adjust movement speed (scroll DOWN for slower)")
    print("  Right-click + Mouse = Free look")
    print("\nTIP: Scroll mouse wheel DOWN multiple times to slow movement speed!")
    print("\nInstructions:")
    print("1. Move the spectator camera in CARLA to your desired position")
    print("2. Angle the camera to face the traffic you want to record")
    print("3. Press ENTER here when ready...\n")
    
    input("Press ENTER when camera is positioned: ")

    # Capture spectator transform
    spectator = world.get_spectator()
    cam_tf = spectator.get_transform()
    
    print(f"\n✓ Camera position captured:")
    print(f"  Location: ({cam_tf.location.x:.2f}, {cam_tf.location.y:.2f}, {cam_tf.location.z:.2f})")
    print(f"  Rotation: (pitch={cam_tf.rotation.pitch:.1f}, yaw={cam_tf.rotation.yaw:.1f}, roll={cam_tf.rotation.roll:.1f})")

    # Find nearby traffic lights
    traffic_lights = list(world.get_actors().filter('traffic.traffic_light'))
    traffic_lights = [l for l in traffic_lights if l.get_stop_waypoints()]
    
    if not traffic_lights:
        print("\n❌ No traffic lights with stop waypoints found.")
        return

    # Sort by distance
    traffic_lights_sorted = sorted(
        traffic_lights,
        key=lambda l: l.get_transform().location.distance(cam_tf.location)
    )

    print(f"\n✓ Found {len(traffic_lights_sorted)} traffic lights with stop waypoints")
    print("\nNearest traffic lights:")
    for i, light in enumerate(traffic_lights_sorted[:10]):
        loc = light.get_transform().location
        dist = loc.distance(cam_tf.location)
        num_lanes = len(light.get_stop_waypoints())
        print(f"  [{i+1}] id={light.id:>4}  distance={dist:>6.1f}m  lanes={num_lanes}  loc=({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f})")

    # Select traffic light
    while True:
        try:
            choice = input("\nEnter traffic light ID: ").strip()
            selected_id = int(choice)
            selected_light = next((l for l in traffic_lights_sorted if l.id == selected_id), None)
            if selected_light:
                break
            else:
                print("Invalid ID. Please try again.")
        except ValueError:
            print("Please enter a number.")

    num_lanes = len(selected_light.get_stop_waypoints())
    print(f"\n✓ Selected traffic light {selected_light.id} with {num_lanes} lane(s)")

    # Build config
    config = {
        'scenario_name': 'custom_scenario',
        'camera': {
            'location': {
                'x': round(cam_tf.location.x, 2),
                'y': round(cam_tf.location.y, 2),
                'z': round(cam_tf.location.z, 2)
            },
            'rotation': {
                'pitch': round(cam_tf.rotation.pitch, 2),
                'yaw': round(cam_tf.rotation.yaw, 2),
                'roll': round(cam_tf.rotation.roll, 2)
            },
            'image_width': 1280,
            'image_height': 720,
            'fov': 70
        },
        'traffic_light': {
            'id': selected_light.id
        },
        'weather': {
            'cloudiness': 40.0,
            'precipitation': 0.0,
            'sun_altitude_angle': 25.0,
            'fog_density': 0.0,
            'wetness': 0.0
        },
        'simulation': {
            'total_frames': 5000,
            'capture_interval': 10,
            'warmup_frames': 100,
            'train_ratio': 0.8,
            'fixed_delta_seconds': 0.05
        },
        'spawn': {
            'max_vehicles': 30,
            'spawn_radius': 100.0,
            'respawn_interval': 50,
            'despawn_distance': 150.0,
            'ratios': {
                'car': 15,
                'ambulance': 2,
                'bus': 2,
                'truck': 3,
                'police_car': 2,
                'fire_truck': 1,
                'bike': 4
            }
        },
        'output': {
            'directory': './dataset_output'
        }
    }

    # Save config next to this script
    script_dir = Path(__file__).resolve().parent
    output_path = script_dir / 'scenario_config.yaml'
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\n✓ Config saved to: {output_path}")
    print("\nNext steps:")
    print(f"1. Edit {output_path} to customize:")
    print("   - Weather conditions")
    print("   - Spawn ratios per vehicle class (car, ambulance, bus, truck, police_car, fire_truck, bike)")
    print("   - Max concurrent vehicles and respawn interval")
    print("   - Total simulation frames and capture interval")
    print("   - Train/val split ratio")
    print("   - Output directory")
    print("2. Run: python capture_dataset.py scenario_config.yaml")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAborted.")
        sys.exit(0)
