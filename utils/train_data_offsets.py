import cv2
import numpy as np

from typing import Tuple


def find_center_of_the_mask(mask: np.ndarray) -> Tuple[float, float]:

    M = cv2.moments(mask)
    if M["m00"] != 0:
        cX = M["m10"] / M["m00"]
        cY = M["m01"] / M["m00"]
    else:
        cX, cY = 0, 0  # Default value if the mask is empty
        print("Default value used")

    return cX, cY


def find_mask_center_offset(mask: np.ndarray) -> Tuple[float, float]:

    cX, cY = find_center_of_the_mask(mask)
    oX, oY = mask.shape[1] / 2, mask.shape[0] / 2

    offX, offY = cX - oX, cY - oY
    return offX, offY


def find_mask_scale_difference(
    reference_mask: np.ndarray, target_mask: np.ndarray,
) -> float:

    assert np.max(reference_mask) <= 1 and np.max(target_mask) <= 1

    reference_mask = np.ceil(reference_mask)
    target_mask = np.ceil(target_mask)

    reference_mask_size = np.sum(reference_mask)
    target_mask_size = np.sum(target_mask)

    return reference_mask_size / target_mask_size


def retake_face_img(img: np.ndarray, scale: float) -> np.ndarray:

    target_height, target_width = img.shape[:2]

    rescaled_height = int(target_height * scale)
    rescaled_width = int(target_width * scale)

    img = cv2.resize(
        img, (rescaled_width, rescaled_height), interpolation=cv2.INTER_LINEAR
    )

    # Calculate the necessary padding or cropping
    pad_height = max(target_height - rescaled_height, 0)
    pad_width = max(target_width - rescaled_width, 0)

    crop_height = max(rescaled_height - target_height, 0)
    crop_width = max(rescaled_width - target_width, 0)

    # Padding
    if pad_height > 0 or pad_width > 0:
        top = pad_height // 2
        bottom = pad_height - top
        left = pad_width // 2
        right = pad_width - left

        # Pad the img
        padded_img = np.pad(
            img, ((top, bottom), (left, right), (0, 0)), mode="constant"
        )
        return padded_img

    # Cropping
    elif crop_height > 0 or crop_width > 0:
        top = crop_height // 2
        bottom = top + target_height
        left = crop_width // 2
        right = left + target_width

        # Crop the img
        cropped_img = img[top:bottom, left:right]
        return cropped_img

    # If no padding or cropping needed
    else:
        return img
