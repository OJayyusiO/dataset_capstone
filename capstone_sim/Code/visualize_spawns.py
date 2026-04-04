"""
CARLA Spawn Point Visualizer

Draws numbered markers at all spawn points in the CARLA simulator viewport.
Use this to identify spawn point indices for the scenario config.

Usage:
    python visualize_spawns.py
"""

import carla
import sys


def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    spawn_points = world.get_map().get_spawn_points()

    print(f"Drawing {len(spawn_points)} spawn points (visible for 120 seconds)")
    print("Fly around in CARLA to see the numbered markers.\n")

    for i, sp in enumerate(spawn_points):
        world.debug.draw_string(
            sp.location + carla.Location(z=1.0),
            str(i),
            life_time=120.0,
            color=carla.Color(255, 0, 0)
        )
        world.debug.draw_point(
            sp.location + carla.Location(z=0.5),
            size=0.1,
            life_time=120.0,
            color=carla.Color(0, 255, 0)
        )

    print("Spawn points:")
    for i, sp in enumerate(spawn_points):
        loc = sp.location
        print(f"  [{i}] ({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f})")

    print(f"\nDone. Markers visible in CARLA for 120 seconds.")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(0)
