import numpy as np
from enum import Enum


class DegradationType(Enum):
    NOISE = "noise"
    # TODO: Add other types of degradation such as missing pixels.


def degrade_image(img: np.ndarray, degradation_config: dict, degradation_type=DegradationType.NOISE) -> np.ndarray:
    if degradation_type == DegradationType.NOISE:
        np.random.seed(41)  # TODO: remove it
        return img + np.random.normal(loc=0, scale=degradation_config["noise_sigma"], size=img.shape)
    else:
        raise NotImplementedError("Only noise is available for the degradation method.")
