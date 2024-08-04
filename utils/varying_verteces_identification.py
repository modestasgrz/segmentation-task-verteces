import numpy as np
from typing import Dict, List


def identify_varying_verteces(
    target_verteces: Dict[str, List[float]],
    reference_verteces: Dict[str, List[float]],
    threshold: float = 4,
) -> Dict[str, List[float]]:

    varying_verteces = {}
    for key in reference_verteces:
        if key in target_verteces:
            reference_coords = np.array(reference_verteces[key])
            target_coords = np.array(target_verteces[key])
            distance = np.linalg.norm(reference_coords - target_coords)
            if distance > threshold:
                varying_verteces[key] = target_coords
    return varying_verteces


def calculate_vertex_coordinates_distances(
    target_verteces: Dict[str, List[float]], reference_verteces: Dict[str, List[float]],
) -> Dict[str, List[float]]:

    distances = []
    for key in reference_verteces:
        if key in target_verteces:
            reference_coords = np.array(reference_verteces[key])
            target_coords = np.array(target_verteces[key])
            distance = np.linalg.norm(reference_coords - target_coords)
            distances.append(distance)
    return np.array(distances)


def identify_varying_verteces_adaptive(
    target_verteces: Dict[str, List[float]],
    reference_verteces: Dict[str, List[float]],
    num_std_devs: float = 5,
) -> Dict[str, List[float]]:

    distances = calculate_vertex_coordinates_distances(
        target_verteces, reference_verteces
    )
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    threshold = mean_distance + num_std_devs * std_distance

    varying_verteces = {}
    for key in reference_verteces:
        if key in target_verteces:
            reference_coords = np.array(reference_verteces[key])
            target_coords = np.array(target_verteces[key])
            distance = np.linalg.norm(reference_coords - target_coords)
            if distance > threshold:
                varying_verteces[key] = target_coords
    return varying_verteces
