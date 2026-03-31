import carla
import queue
import cv2
import numpy as np
import random
#set map to town05
 
# Scenario configuration
TARGET_LIGHT_ID = None  # Set after listing lights, e.g., 123
USE_NEAREST_TO_SPECTATOR = True
CAR1_BACK_DISTANCE = 35.0
CAR2_BACK_DISTANCE = 45.0
RED_SECONDS = 10.0
GREEN_SECONDS = 15.0
FIXED_DELTA_SECONDS = 0.05
IMAGE_W = 1280
IMAGE_H = 720


def select_traffic_light(world):
    traffic_lights = list(world.get_actors().filter('traffic.traffic_light'))
    if not traffic_lights:
        return None, None

    traffic_lights = [l for l in traffic_lights if l.get_stop_waypoints()]
    if not traffic_lights:
        return None, None

    if TARGET_LIGHT_ID is not None:
        light = next((l for l in traffic_lights if l.id == TARGET_LIGHT_ID), None)
        if not light:
            return None, None
        return light, light.get_stop_waypoints()

    if USE_NEAREST_TO_SPECTATOR:
        spectator_loc = world.get_spectator().get_transform().location
        traffic_lights_sorted = sorted(
            traffic_lights,
            key=lambda l: l.get_transform().location.distance(spectator_loc)
        )
        print("Nearest traffic lights to spectator (id, location):")
        for l in traffic_lights_sorted[:5]:
            loc = l.get_transform().location
            print(f"  id={l.id} loc=({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f})")
        light = traffic_lights_sorted[0]
        return light, light.get_stop_waypoints()

    random.shuffle(traffic_lights)
    light = traffic_lights[0]
    return light, light.get_stop_waypoints()


def make_camera_transform(light, waypoint):
    light_tf = light.get_transform()
    if waypoint:
        lane_yaw = waypoint.transform.rotation.yaw
        cam_yaw = (lane_yaw + 180.0) % 360.0
    else:
        cam_yaw = (light_tf.rotation.yaw + 180.0) % 360.0

    cam_loc = carla.Location(
        light_tf.location.x,
        light_tf.location.y,
        light_tf.location.z + 6.0
    )
    cam_rot = carla.Rotation(pitch=-20.0, yaw=cam_yaw, roll=0.0)
    return carla.Transform(cam_loc, cam_rot)


def main():
    actor_list = []
    image_queue = queue.Queue()

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    car_map = world.get_map()
    bp_lib = world.get_blueprint_library()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = FIXED_DELTA_SECONDS

    settings.substepping = True
    settings.max_substep_delta_time = 0.01  # Calculate physics every 0.01s (100 Hz)
    settings.max_substeps = 10

    world.apply_settings(settings)

    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_random_device_seed(42)
    random.seed(42)

    try:
        selected_light, stop_wps = select_traffic_light(world)
        if not selected_light or not stop_wps:
            print("No traffic light found.")
            return

        stop_wp = stop_wps[0]
        wp_back_1 = stop_wp.previous(CAR1_BACK_DISTANCE)
        wp_back_2 = stop_wp.previous(CAR2_BACK_DISTANCE)
        if not wp_back_1 or not wp_back_2:
            print("Could not find spawn points behind the traffic light.")
            return

        spawn_point_1 = wp_back_1[0].transform
        spawn_point_1.location.z += 0.5
        spawn_point_2 = wp_back_2[0].transform
        spawn_point_2.location.z += 0.5

        print(f"Using traffic light id: {selected_light.id}")

        cam_tf = make_camera_transform(selected_light, stop_wp)
        cam_bp = bp_lib.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(IMAGE_W))
        cam_bp.set_attribute('image_size_y', str(IMAGE_H))
        camera = world.spawn_actor(cam_bp, cam_tf)
        actor_list.append(camera)
        camera.listen(image_queue.put)

        bp1 = bp_lib.find('vehicle.tesla.model3')
        bp1.set_attribute('color', '255,0,0')
        car1 = world.try_spawn_actor(bp1, spawn_point_1)
        if car1:
            actor_list.append(car1)

        bp2 = bp_lib.find('vehicle.audi.tt')
        bp2.set_attribute('color', '0,0,255')
        car2 = world.try_spawn_actor(bp2, spawn_point_2)
        if car2:
            actor_list.append(car2)

        if car1:
            car1.set_autopilot(True, traffic_manager.get_port())
            traffic_manager.ignore_lights_percentage(car1, 0.0)
            traffic_manager.vehicle_percentage_speed_difference(car1, 30.0)
        if car2:
            car2.set_autopilot(True, traffic_manager.get_port())
            traffic_manager.ignore_lights_percentage(car2, 0.0)
            traffic_manager.vehicle_percentage_speed_difference(car2, 30.0)

        selected_light.set_state(carla.TrafficLightState.Red)
        selected_light.freeze(True)
        print("Phase 1: RED (approach and stop)")

        red_frames = int(RED_SECONDS / FIXED_DELTA_SECONDS)
        green_frames = int(GREEN_SECONDS / FIXED_DELTA_SECONDS)

        for frame in range(red_frames + green_frames):
            if frame == red_frames:
                print("Phase 2: GREEN (go)")
                selected_light.set_state(carla.TrafficLightState.Green)

            status = "RED" if frame < red_frames else "GREEN"
            color = (0, 0, 255) if frame < red_frames else (0, 255, 0)

            world.tick()

            try:
                image = image_queue.get(timeout=2.0)
                i = np.array(image.raw_data)
                img = i.reshape((IMAGE_H, IMAGE_W, 4))[:, :, :3].copy()
                cv2.rectangle(img, (10, 10), (350, 120), (0, 0, 0), -1)
                cv2.putText(img, f"Light: {status}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
                cv2.putText(img, f"Frame: {frame}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
                cv2.imshow("Traffic Light Scenario", img)
                if cv2.waitKey(1) == ord('q'):
                    break
            except queue.Empty:
                break

    finally:
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        for a in actor_list:
            a.destroy()
        cv2.destroyAllWindows()
        print("Cleaned up.")


if __name__ == '__main__':
    main()