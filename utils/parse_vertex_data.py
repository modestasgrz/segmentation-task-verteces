import cv2
import numpy as np
import json
from typing import Any, Dict, List, Tuple
from scipy.spatial import ConvexHull


def parse_vertex_data_from_json_file(data_file_path: str) -> List[Dict[str, Any]]:

    with open(data_file_path, "r") as f:
        vertex_data = [str.strip(entry) for entry in f.read()[2:-2].split("]\n[")]

    for i in range(len(vertex_data)):
        single_splited_vertex_data = vertex_data[i].split(",\n")
        batch_part = (
            single_splited_vertex_data[0]
            .replace("[", "{")
            .replace("]", "}")
            .replace("=", '" : "')
        )
        indexes_part = ",\n".join(single_splited_vertex_data[1:])
        single_vertex_data = "{" + ",\n".join([batch_part, indexes_part]) + "}"
        single_vertex_json_data = json.loads(single_vertex_data)
        vertex_data[i] = single_vertex_json_data

    return vertex_data


def get_vertex_coordinates(
    single_mask_vertex_data: Dict[str, Any],
    img_shape: Tuple[int, int] = (512, 512),  # height, width
) -> List[List[int]]:

    return [
        [int(value[0]), int(value[1])]
        for key, value in single_mask_vertex_data["indexes"].items()
        if value[0] >= 0
        and value[0] <= img_shape[1]
        and value[1] >= 0
        and value[1] <= img_shape[0]
    ]


def order_coordinates(vertex_coordinates: List[List[int]]) -> List[Any]:

    if len(vertex_coordinates) < 3:
        return vertex_coordinates  # Not enough points to form a polygon
    hull = ConvexHull(vertex_coordinates)
    ordered_coords = [vertex_coordinates[i] for i in hull.vertices]
    return ordered_coords


def parse_img_filename(single_mask_vertex_data: Dict[str, Any]) -> str:

    camera_angle = single_mask_vertex_data["batch"]["camera"]
    img_nr = single_mask_vertex_data["batch"]["image"]

    if camera_angle in ["Front", "Angle", "Side"]:
        pass
    elif str.lower(camera_angle) == "camera_front":
        camera_angle = "Front"
    elif str.lower(camera_angle) == "camera_45":
        camera_angle = "Angle"
    elif str.lower(camera_angle) == "camera_side":
        camera_angle = "Side"
    else:
        raise ValueError("Unrecognized camera angle refrence = {}".format(camera_angle))

    img_filename = f"Render_{img_nr}_{camera_angle}.png"
    return img_filename


def create_mask(
    vertex_coordinates: List[List[int]],
    img_shape: Tuple[int, int] = (512, 512),  # height, width
) -> np.ndarray:

    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    if len(vertex_coordinates) == 0:
        return mask
    points = np.array([vertex_coordinates], dtype=np.int32)
    cv2.fillPoly(mask, [points], 1)
    mask = mask[
        ::-1, :
    ]  # Mask for some reason is displayed upside down - here it's being rotated 180 degrees
    return mask


def get_vertex_points_img(
    vertex_coordinates: List[List[int]],
    img_shape: Tuple[int, int] = (512, 512),  # height, width
) -> np.ndarray:

    image = np.zeros((img_shape[0], img_shape[1], 3), dtype=np.uint8)
    for coord in vertex_coordinates:
        x, y = int(coord[0]), int(coord[1])
        cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
    return image


def get_polygon_draw_points_img(
    vertex_coordinates: List[List[int]], img_shape: Tuple[int, int],
) -> np.ndarray:

    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    points = np.array(vertex_coordinates, dtype=np.int32)

    # Draw each point and connect lines step-by-step
    for i in range(len(points)):
        cv2.circle(mask, (points[i][0], points[i][1]), 3, 1, -1)
        if i > 0:
            cv2.line(
                mask,
                (points[i - 1][0], points[i - 1][1]),
                (points[i][0], points[i][1]),
                1,
                1,
            )
    return mask


if __name__ == "__main__":

    vertex_data = parse_vertex_data_from_json_file(
        data_file_path="data/parameter set/batch01/vertex_3k.txt"
    )

    image_shape = (512, 512)
    vertex_coordinates = get_vertex_coordinates(vertex_data[0], img_shape=image_shape)
    ordered_vertex_coordinates = order_coordinates(vertex_coordinates)
    mask = create_mask(ordered_vertex_coordinates, img_shape=image_shape)

    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("TkAgg")

    plt.imshow(mask, cmap="gray")
    plt.show()
