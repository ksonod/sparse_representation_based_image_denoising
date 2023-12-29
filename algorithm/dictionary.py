import numpy as np
import matplotlib.pyplot as plt
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
        self.learned_dictionary = None

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

    def show_dictionary(self):
        """
        This function converts each atom stored in a column of a dictionary into a 2D-shape atom for visualization and
        shows the 2D atoms.
        """

        def show_dict(input_dictionary: np.ndarray, fig_title: str):
            dict_idx = 0
            plt.figure()
            plt.title(fig_title)
            plt.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

            for idx_row in range(int(input_dictionary.shape[0] / self.patch_size[0])):
                for idx_col in range(int(input_dictionary.shape[1]/self.patch_size[1])):

                    dict_atom = input_dictionary[:, dict_idx].reshape(self.patch_size)  # reorganized atom

                    plt.subplot(*self.patch_size, dict_idx + 1)
                    plt.pcolormesh(dict_atom, cmap="gray")
                    plt.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

                    dict_idx += 1
            plt.subplots_adjust(wspace=0, hspace=0)

        if self.defined_dictionary is not None:
            show_dict(self.defined_dictionary, "Original dictionary")

        if self.learned_dictionary is not None:
            show_dict(self.learned_dictionary, "Learned dictionary")
