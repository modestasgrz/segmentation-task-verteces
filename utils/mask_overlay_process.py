import os
import copy
import cv2
import numpy as np
from typing import Tuple, Dict, Any
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")

try:
    from utils.varying_verteces_identification import identify_varying_verteces_adaptive
    from utils.parse_vertex_data import (
        get_vertex_coordinates,
        order_coordinates,
        create_mask,
        parse_img_filename,
    )
    from utils.image_io import load_img
    from utils.train_data_offsets import (
        find_mask_center_offset,
        find_mask_scale_difference,
    )
except ImportError:
    from varying_verteces_identification import identify_varying_verteces_adaptive
    from parse_vertex_data import (
        get_vertex_coordinates,
        order_coordinates,
        create_mask,
        parse_img_filename,
    )
    from image_io import load_img
    from train_data_offsets import find_mask_center_offset, find_mask_scale_difference


def overlay_mask_on_image(
    img: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (255, 255, 255),
    alpha: float = 0.5,
) -> np.ndarray:

    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    colored_mask = np.zeros_like(img, dtype=np.uint8)
    # Apply color to all channels
    for i in range(3):
        colored_mask[:, :, i] = np.where(mask > 0, color[i], 0)

    overlay = cv2.addWeighted(img, 1 - alpha, colored_mask, alpha, 0)

    return overlay


def create_mask_from_verteces_dict(
    single_mask_vertex_data: Dict[str, Any],
    img_shape: Tuple[int, int] = (512, 512),
    convex_order_coordinates: bool = False,
) -> np.ndarray:

    vertex_coordinates = get_vertex_coordinates(
        single_mask_vertex_data, img_shape=img_shape
    )
    if convex_order_coordinates:
        vertex_coordinates = order_coordinates(vertex_coordinates)
    return create_mask(vertex_coordinates, img_shape=img_shape)


def get_mask_overlay_process(
    single_mask_vertex_data: Dict[str, Any],
    batch_dir_path: str,
    img_shape: Tuple[int, int] = (512, 512),
    convex_order_coordinates: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    mask = create_mask_from_verteces_dict(
        single_mask_vertex_data, img_shape, convex_order_coordinates
    )
    img_filename = parse_img_filename(single_mask_vertex_data)
    img = load_img(
        os.path.join(batch_dir_path, "Rendering", img_filename), remove_alpha=True
    )
    img = cv2.resize(img, img_shape)
    overlayed_img = overlay_mask_on_image(img, mask)
    return img, mask, overlayed_img


def display_single_mask_vertex_data(
    single_mask_vertex_data: Dict[str, Any],
    batch_dir_path: str,
    img_shape: Tuple[int, int] = (512, 512),
    convex_order_coordinates: bool = False,
):

    img, mask, overlayed_img = get_mask_overlay_process(
        single_mask_vertex_data, batch_dir_path, img_shape, convex_order_coordinates
    )

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(img)
    ax[1].imshow(mask)
    ax[2].imshow(overlayed_img)

    plt.show()


def apply_mask_overlay_process(
    reference_mask_vertex_data: Dict[str, Any],
    target_mask_vertex_data: Dict[str, Any],
    batch_dir_path: str,
    img_shape: Tuple[int, int] = (512, 512),
    num_std_devs: float = 5,
    convex_order_coordinates: bool = False,
    only_varying_verteces: bool = True,
    apply_train_data_offset_correction: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    reference_mask_vertex_data_values = copy.copy(reference_mask_vertex_data)
    target_mask_vertex_data_values = copy.copy(target_mask_vertex_data)

    if apply_train_data_offset_correction:

        reference_mask = create_mask_from_verteces_dict(
            reference_mask_vertex_data, img_shape=img_shape
        )
        target_mask = create_mask_from_verteces_dict(
            target_mask_vertex_data, img_shape=img_shape
        )

        reference_offX, reference_offY = find_mask_center_offset(reference_mask)
        target_offX, target_offY = find_mask_center_offset(target_mask)
        scale = find_mask_scale_difference(reference_mask, target_mask)

        reference_mask_vertex_data_values["indexes"] = {
            key: [
                (value[0] - reference_offX) * scale,
                (value[1] - reference_offY) * scale,
            ]
            for key, value in reference_mask_vertex_data_values["indexes"].items()
        }
        target_mask_vertex_data_values["indexes"] = {
            key: [(value[0] - target_offX) * scale, (value[1] - target_offY) * scale]
            for key, value in target_mask_vertex_data_values["indexes"].items()
        }

    if only_varying_verteces:
        varying_single_vertex_data = identify_varying_verteces_adaptive(
            reference_mask_vertex_data_values["indexes"],
            target_mask_vertex_data_values["indexes"],
            num_std_devs=num_std_devs,
        )

        target_mask_vertex_data_values["indexes"] = varying_single_vertex_data

    if apply_train_data_offset_correction:
        target_mask_vertex_data_values["indexes"] = {
            key: [(value[0] + target_offX) / scale, (value[1] + target_offY) / scale]
            for key, value in target_mask_vertex_data_values["indexes"].items()
        }

    return get_mask_overlay_process(
        target_mask_vertex_data_values,
        batch_dir_path,
        img_shape,
        convex_order_coordinates=convex_order_coordinates,
    )
