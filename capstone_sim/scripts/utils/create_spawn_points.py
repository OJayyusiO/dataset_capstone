"""
CARLA Custom Spawn Point Creator

Lets you fly the spectator camera and create custom spawn points
on the road below your position. Outputs a YAML file with the points.

Usage:
    python create_spawn_points.py
    python create_spawn_points.py --output my_spawns.yaml
"""

import carla
import yaml
import sys
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Create custom spawn points by flying the spectator')
    parser.add_argument('--output', type=str, default='custom_spawn_points.yaml',
                        help='Output YAML file (default: custom_spawn_points.yaml)')
    args = parser.parse_args()

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    carla_map = world.get_map()

    print("=" * 60)
    print("CARLA Custom Spawn Point Creator")
    print("=" * 60)
    print("\nFly the spectator camera to where you want a spawn point.")
    print("The script will snap to the nearest road below you.")
    print("\nControls:")
    print("  ENTER - Capture spawn point at current location")
    print("  d     - Delete last spawn point")
    print("  q     - Save and quit")
    print()

    spawn_points = []

    while True:
        cmd = input(f"[{len(spawn_points)} points] Press ENTER to capture, 'd' to delete last, 'q' to save+quit: ").strip().lower()

        if cmd == 'q':
            break
        elif cmd == 'd':
            if spawn_points:
                removed = spawn_points.pop()
                print(f"  Removed point at ({removed['x']:.1f}, {removed['y']:.1f})")
            else:
                print("  No points to remove")
            continue

        # Capture spectator location
        spectator = world.get_spectator()
        spec_loc = spectator.get_transform().location

        # Snap to nearest driving lane waypoint
        waypoint = carla_map.get_waypoint(spec_loc, project_to_road=True,
                                           lane_type=carla.LaneType.Driving)
        if waypoint is None:
            print(f"  No road found near ({spec_loc.x:.1f}, {spec_loc.y:.1f}). Move to a road.")
            continue

        wp_tf = waypoint.transform
        spawn = {
            'x': round(wp_tf.location.x, 2),
            'y': round(wp_tf.location.y, 2),
            'z': round(wp_tf.location.z + 0.5, 2),  # slight Z offset to avoid ground clipping
            'yaw': round(wp_tf.rotation.yaw, 2),
        }
        spawn_points.append(spawn)
        print(f"  + Spawn point #{len(spawn_points)}: ({spawn['x']}, {spawn['y']}, {spawn['z']}) yaw={spawn['yaw']}")

        # Visualize the spawn point in CARLA
        world.debug.draw_string(
            wp_tf.location + carla.Location(z=2.0),
            f"#{len(spawn_points)}",
            life_time=60.0,
            color=carla.Color(0, 255, 0)
        )
        world.debug.draw_point(
            wp_tf.location + carla.Location(z=0.5),
            size=0.2,
            life_time=60.0,
            color=carla.Color(0, 255, 0)
        )

    if not spawn_points:
        print("\nNo spawn points captured.")
        return

    output_path = Path(args.output)
    # Write each spawn point in flow style for easy copy-paste into scenario configs
    with open(output_path, 'w') as f:
        f.write("custom_spawn_points:\n")
        for sp in spawn_points:
            f.write(f"  - {{x: {sp['x']}, y: {sp['y']}, z: {sp['z']}, yaw: {sp['yaw']}}}\n")

    print(f"\nSaved {len(spawn_points)} spawn points to: {output_path.resolve()}")
    print("\nTo use in a scenario config, copy the 'custom_spawn_points' section into the spawn block:")
    print("\nspawn:")
    print("  custom_spawn_points:")
    for sp in spawn_points[:3]:
        print(f"    - {{x: {sp['x']}, y: {sp['y']}, z: {sp['z']}, yaw: {sp['yaw']}}}")
    if len(spawn_points) > 3:
        print(f"    # ... and {len(spawn_points) - 3} more")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(0)
