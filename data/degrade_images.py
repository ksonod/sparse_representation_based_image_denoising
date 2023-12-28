import numpy as np
from enum import Enum


class DegradationType(Enum):
    NOISE = "noise"
    # TODO: Add other types of degradation such as missing pixels.


def degrade_image(img: np.ndarray, degradation_config: dict) -> np.ndarray:
    """
    Degrading an input image with a specified method.
    :param img: Clean input image.
    :param degradation_config: Config for degradation
    :return: numpy array of a degraded image
    """
    if degradation_config["degradation_type"] == DegradationType.NOISE:
        if "random_seed" in degradation_config:
            np.random.seed(degradation_config["random_seed"])
        return img + np.random.normal(
            loc=0, scale=degradation_config["noise_sigma"], size=img.shape
        )  # add random noise
    else:
        raise NotImplementedError("Only noise is available for the degradation method.")
