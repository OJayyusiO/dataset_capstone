import carla

client = carla.Client('localhost', 2000)
world = client.get_world()
spawn_points = world.get_map().get_spawn_points()

for i, sp in enumerate(spawn_points):
    world.debug.draw_string(
        sp.location + carla.Location(z=1.0),
        str(i),
        life_time=60.0,
        color=carla.Color(255, 0, 0)
    )
    world.debug.draw_point(
        sp.location + carla.Location(z=0.5),
        size=0.1,
        life_time=60.0,
        color=carla.Color(0, 255, 0)
    )

print(f"Drew {len(spawn_points)} spawn points (visible for 60 seconds)")
