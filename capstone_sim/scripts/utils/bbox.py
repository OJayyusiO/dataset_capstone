"""3D-to-2D bounding box projection utilities."""

import math
import numpy as np

from capstone_sim.scripts.utils.constants import MIN_VISIBILITY, MIN_BBOX_SIDE, MIN_BBOX_AREA


def build_projection_matrix(w, h, fov):
    """Build camera intrinsic matrix from image dimensions and FOV."""
    focal = w / (2.0 * math.tan(fov * math.pi / 360.0))
    K = np.array([
        [focal, 0.0,   w / 2.0],
        [0.0,   focal, h / 2.0],
        [0.0,   0.0,   1.0],
    ])
    return K


def get_image_point(world_point, K, world_to_camera):
    """Project a single 3D world point to 2D image coordinates."""
    point_world = np.array([world_point.x, world_point.y, world_point.z, 1.0])
    point_cam = world_to_camera @ point_world

    # UE4 coords -> standard camera coords
    point_camera = np.array([point_cam[1], -point_cam[2], point_cam[0]])

    depth = point_camera[2]
    if depth <= 0:
        return None, None, depth

    img_point = K @ point_camera
    u = img_point[0] / img_point[2]
    v = img_point[1] / img_point[2]
    return u, v, depth


def get_2d_bbox(vehicle, camera, K, image_w, image_h):
    """Compute 2D bounding box for a vehicle as seen from a camera.

    Returns (x_min, y_min, x_max, y_max) in pixel coordinates, or None.
    """
    world_to_camera = np.array(camera.get_transform().get_inverse_matrix())

    bb = vehicle.bounding_box
    verts = bb.get_world_vertices(vehicle.get_transform())

    us, vs = [], []
    behind_count = 0
    for vert in verts:
        u, v, depth = get_image_point(vert, K, world_to_camera)
        if depth <= 0:
            behind_count += 1
            continue
        us.append(u)
        vs.append(v)

    if len(us) == 0:
        return None
    if behind_count > 4:
        return None

    raw_x_min, raw_x_max = min(us), max(us)
    raw_y_min, raw_y_max = min(vs), max(vs)
    raw_area = max(0, raw_x_max - raw_x_min) * max(0, raw_y_max - raw_y_min)

    x_min = max(0, raw_x_min)
    y_min = max(0, raw_y_min)
    x_max = min(image_w, raw_x_max)
    y_max = min(image_h, raw_y_max)

    if x_min >= x_max or y_min >= y_max:
        return None

    clamped_area = (x_max - x_min) * (y_max - y_min)
    if raw_area > 0 and (clamped_area / raw_area) < MIN_VISIBILITY:
        return None
    if (x_max - x_min) < MIN_BBOX_SIDE or (y_max - y_min) < MIN_BBOX_SIDE:
        return None
    if clamped_area < MIN_BBOX_AREA:
        return None

    return (x_min, y_min, x_max, y_max)


def bbox_to_yolo(bbox, class_id, image_w, image_h):
    """Convert pixel bbox to YOLO format: class_id x_center y_center w h (normalized)."""
    x_min, y_min, x_max, y_max = bbox
    x_center = ((x_min + x_max) / 2.0) / image_w
    y_center = ((y_min + y_max) / 2.0) / image_h
    w = (x_max - x_min) / image_w
    h = (y_max - y_min) / image_h
    return f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
