import numpy as np
from enum import Enum


class DictionaryType(Enum):
    DCT = "discrete_cosine_transform"
    # TODO: Add different dictionaries


class Dictionary:
    def __init__(self, dictionary_type: DictionaryType, patch_size):
        self.dictionary_type = dictionary_type

        if patch_size[0] != patch_size[1]:
            raise ValueError('A patch should be square.')
        else:
            self.patch_size = patch_size

        self.defined_dictionary = None

    def build_dictionary(self):
        """
        A dictionary can be built with this method.

        - DCT: Direct Cosine Transform. In the current implementation, each column (or row) corresponds to a flattened
         DCT basis.
        """
        if self.dictionary_type == DictionaryType.DCT:  #
            n = self.patch_size[0]

            temp_dict = np.zeros(self.patch_size)

            for k in range(n):
                if k > 0:
                    coeff = 1 / np.sqrt(2 * n)
                else:  # k==0
                    coeff = 0.5 / np.sqrt(n)
                temp_dict[:, k] = 2 * coeff * np.cos((0.5 + np.arange(n)) * k * np.pi / n)

            self.defined_dictionary = np.kron(temp_dict, temp_dict)
        else:
            # TODO: Add different dictionaries
            raise NotImplementedError("Choose a different dictionary")

