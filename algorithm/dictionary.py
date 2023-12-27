import numpy as np
from enum import Enum


class DictionaryType(Enum):
    DCT = "dct"
    # TODO: Add different dictionaries


class Dictionary:
    def __init__(self, dictionary_type: DictionaryType, patch_size):
        self.dictionary_type = dictionary_type

        if patch_size[0] != patch_size[1]:
            raise ValueError('A patch should be square.')
        else:
            self.patch_size = patch_size

    def get_dictionary(self) -> np.ndarray:

        if self.dictionary_type == DictionaryType.DCT:
            patch_n = self.patch_size[0]  # patch size

            dic = np.zeros(self.patch_size)

            for k in range(patch_n):
                if k > 0:
                    coeff = 1 / np.sqrt(2 * patch_n)
                else:  # k==0
                    coeff = 0.5 / np.sqrt(patch_n)
                dic[:, k] = 2 * coeff * np.cos((0.5 + np.arange(patch_n)) * k * np.pi / patch_n)

            # Create the DCT for both axes
            return np.kron(dic, dic)
        else:
            # TODO: Add different dictionaries
            raise NotImplementedError("Choose a different dictionary")

