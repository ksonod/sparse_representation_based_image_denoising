import numpy as np
from enum import Enum

class DegradationType(Enum):
    add_random_noise = 0
    missing_pixels = 1

# TODO: Add missing pixels
class DegradingImage:
    def __init__(self, sigma=20, rand_seed=1,
                 degradation_type=DegradationType.add_random_noise):

        self.sigma = sigma
        self.rand_seed = rand_seed
        self.degradation_type = degradation_type

        np.random.seed(self.rand_seed)

    def degradation(self, img):
        if self.degradation_type == DegradationType.add_random_noise:
            return img + np.random.normal(loc=0, scale=self.sigma, size=img.shape)

