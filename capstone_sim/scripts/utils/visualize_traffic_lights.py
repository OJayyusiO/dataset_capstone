"""
CARLA Traffic Light Visualizer

Draws numbered markers at all traffic lights in the CARLA simulator viewport.
Use this to identify traffic light IDs for the scenario config.

Usage:
    python visualize_traffic_lights.py
"""

import carla
import sys


def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    traffic_lights = list(world.get_actors().filter('traffic.traffic_light'))
    traffic_lights = [l for l in traffic_lights if l.get_stop_waypoints()]

    if not traffic_lights:
        print("No traffic lights with stop waypoints found on this map.")
        return

    print(f"Drawing {len(traffic_lights)} traffic lights (visible for 120 seconds)")
    print("Fly around in CARLA to see the numbered markers.\n")

    for light in traffic_lights:
        loc = light.get_transform().location
        num_lanes = len(light.get_stop_waypoints())

        # Draw ID label above the traffic light
        world.debug.draw_string(
            loc + carla.Location(z=3.0),
            f"ID:{light.id}",
            life_time=120.0,
            color=carla.Color(255, 255, 0)
        )

        # Draw lane count below ID
        world.debug.draw_string(
            loc + carla.Location(z=2.0),
            f"{num_lanes} lane(s)",
            life_time=120.0,
            color=carla.Color(200, 200, 200)
        )

        # Draw point at the traffic light
        world.debug.draw_point(
            loc + carla.Location(z=0.5),
            size=0.15,
            life_time=120.0,
            color=carla.Color(255, 0, 0)
        )

        # Draw lines to stop waypoints
        for wp in light.get_stop_waypoints():
            world.debug.draw_line(
                loc + carla.Location(z=0.5),
                wp.transform.location + carla.Location(z=0.5),
                thickness=0.05,
                life_time=120.0,
                color=carla.Color(0, 255, 0)
            )

    print("Traffic lights:")
    for light in sorted(traffic_lights, key=lambda l: l.id):
        loc = light.get_transform().location
        lanes = len(light.get_stop_waypoints())
        print(f"  ID={light.id:>4}  lanes={lanes}  loc=({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f})")

    print(f"\nDone. Markers visible in CARLA for 120 seconds.")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(0)
