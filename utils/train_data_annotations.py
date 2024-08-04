import json
import pandas as pd
from typing import Dict, Any, List


def read_train_data_annotations_csv(filepath: str,) -> pd.DataFrame:

    return pd.read_csv(filepath, index_col=0)


def parse_train_data_annotations_from_df(
    annotations_df: pd.DataFrame,
) -> List[Dict[str, Any]]:

    train_data_verteces = []
    for _, row in annotations_df.iterrows():
        single_verteces_entry_dict = {"batch": {}, "indexes": {}}
        row_data_json = json.loads(row.to_json())
        single_verteces_entry_dict["batch"]["batch"] = str(row_data_json["batch_id"])
        single_verteces_entry_dict["batch"][
            "camera"
        ] = f"{row_data_json['camera_type']}"
        single_verteces_entry_dict["batch"]["image"] = str(row_data_json["id"])
        indexes = list(row_data_json.items())[9:]
        for i in range(0, len(indexes), 2):
            single_verteces_entry_dict["indexes"][indexes[i][0].split("_")[0]] = [
                indexes[i][1],
                indexes[i + 1][1],
            ]

        train_data_verteces.append(single_verteces_entry_dict)
        single_verteces_entry_dict["dome_rotation"] = {
            "x": row_data_json["dome rotation x"],
            "y": row_data_json["dome rotation y"],
            "z": row_data_json["dome rotation z"],
        }

    return train_data_verteces


def parse_train_data_annotations(filepath: str,) -> List[Dict[str, Any]]:

    annotations = read_train_data_annotations_csv(filepath)
    return parse_train_data_annotations_from_df(annotations)


if __name__ == "__main__":

    annotations = read_train_data_annotations_csv(
        "data\\Annotations\\train_set-1-full.csv"
    )
    annotations = parse_train_data_annotations_from_df(annotations)
    print(len(annotations))
    with open("insight.txt", "w+") as f:
        json.dump(annotations[4], f, indent=4)
