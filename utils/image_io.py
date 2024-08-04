from PIL import Image
import numpy as np


def load_img(img_path: str, remove_alpha: bool = False) -> np.ndarray:

    img = np.array(Image.open(img_path))
    if remove_alpha:
        try:
            img = img[:, :, :3]
        except:
            pass

    return img


def save_img(img: np.ndarray, save_path: str):

    Image.fromarray(img).save(save_path)
