import os

from utils.mask_overlay_process import apply_mask_overlay_process
from utils.image_io import save_img
from utils.train_data_annotations import parse_train_data_annotations

ANNOTATIONS_FILE_PATH = (
    "data\\Annotations\\train_set-1-full.csv"  # * Write annotations csv file path
)
BATCH_DIR_PATH = "data\\train set\\batch01"  # * Write data batch dir path
TEMP_RESULTS_DIR_PATH = "temp_results\\train_batch"  # * Write results dir path

IMG_SHAPE = (512, 512)
NUM_STD_DEVS = 2  # 5 for parameter sets

vertex_data = parse_train_data_annotations(ANNOTATIONS_FILE_PATH)

os.makedirs(TEMP_RESULTS_DIR_PATH, exist_ok=True)
for vertex_entry_index in range(len(vertex_data) - 3):

    try:
        if vertex_entry_index < 3:

            reference_mask_vertex_data = vertex_data[vertex_entry_index + 3]
            target_mask_vertex_data = vertex_data[vertex_entry_index]
            (img, mask, overlayed_img) = apply_mask_overlay_process(
                reference_mask_vertex_data,
                target_mask_vertex_data,
                batch_dir_path=BATCH_DIR_PATH,
                img_shape=IMG_SHAPE,
                num_std_devs=NUM_STD_DEVS,
                convex_order_coordinates=True,
                only_varying_verteces=True,
                apply_train_data_offset_correction=True,
            )

            save_img(
                overlayed_img,
                save_path=os.path.join(
                    TEMP_RESULTS_DIR_PATH, f"{vertex_entry_index}.png"
                ),
            )

        reference_mask_vertex_data = vertex_data[vertex_entry_index]
        target_mask_vertex_data = vertex_data[vertex_entry_index + 3]
        (img, mask, overlayed_img) = apply_mask_overlay_process(
            reference_mask_vertex_data,
            target_mask_vertex_data,
            batch_dir_path=BATCH_DIR_PATH,
            img_shape=IMG_SHAPE,
            num_std_devs=NUM_STD_DEVS,
            convex_order_coordinates=True,
            only_varying_verteces=True,
            apply_train_data_offset_correction=True,
        )
        save_img(
            overlayed_img,
            save_path=os.path.join(
                TEMP_RESULTS_DIR_PATH, f"{vertex_entry_index + 3}.png"
            ),
        )
    except Exception as e:
        print(str(e))
        continue
