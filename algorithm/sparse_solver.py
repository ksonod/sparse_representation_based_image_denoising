import numpy as np
from skimage.util import view_as_windows
from typing import Tuple
from algorithm import statistics
from algorithm.dictionary import Dictionary


class SparseSolver:
    def __init__(
            self,
            enable_dictionary_learning: bool,
            num_learning_iterations: int,
            img: np.ndarray,
            dictionary: Dictionary,
            epsilon: float,
            verbose: bool
    ):
        self.enable_dictionary_learning = enable_dictionary_learning
        self.num_learning_iterations = num_learning_iterations
        self.img = img
        self.dictionary = dictionary
        self.epsilon = epsilon
        self.verbose = verbose

    def __call__(self) -> np.ndarray:
        image_patches = self.create_image_patches(stride=1)
        self.dictionary.build_dictionary()  # Dictionary

        if self.enable_dictionary_learning:
            new_dictionary, mean_error, mean_cardinality = self.unitary_dictionary_learning(
                image_patches,
                self.dictionary.defined_dictionary,
                self.num_learning_iterations,
                self.epsilon
            )
        else:
            new_dictionary = np.copy(self.dictionary.defined_dictionary)

        reconst_img_patches, _ = self.batch_thresholding(
            D=new_dictionary, patches=image_patches, epsilon=self.epsilon
        )
        reconst_img = self.col2img(reconst_img_patches, self.dictionary.patch_size, self.img.shape)

        return reconst_img

    def create_image_patches(self, stride=1, flatten=True) -> np.ndarray:
        """
        Create patches from the h x w image using a sliding window.
        :param: stride: Stride value (pixel) when using the sliding window method
        :return: Flattened patches with the shape of (patch_size[0]*patch_size[1]) x num_patches
        """

        patches = view_as_windows(
            arr_in=self.img, window_shape=self.dictionary.patch_size, step=stride
        )
        if flatten:
            patches = patches.reshape(
                -1, self.dictionary.patch_size[0] * self.dictionary.patch_size[1]
            ).T
        return patches

    @staticmethod
    def batch_thresholding(D: np.array, patches: np.ndarray, epsilon: float) -> Tuple[np.ndarray, np.ndarray]:
        """
            With the error-constraint thresholding pursuit algorithm, this batch_thresholding function solves:
                argmin_{alpha_i} sum_i || alpha_i ||_0
                s.t.  || y_i - D alpha_i ||_2**2  <= epsilon**2 for all i,
            where D is a dictionary (square shape, n x n), y_i are the input signals with the length of n, and epsilon
            is the acceptable residual error.

            :param D: numpy array dictionary (n x n).
            :param patches: numpy array.
            :param epsilon: allowed residual error.
            :return X, A: Reconstructed signal X (= D A) and the coefficients A of the dictionary D.
        """

        num_atoms = D.shape[1]  # number of atoms
        num_patches = patches.shape[1]  # number of patches

        inner_prod = np.matmul(D.T, patches)

        residual_sq = inner_prod ** 2
        sorted_residual = np.sort(residual_sq, axis=0)
        sorted_residual_idx = np.argsort(residual_sq, axis=0)
        accumulated_residual = np.cumsum(sorted_residual, axis=0)
        above_thr_idx = (accumulated_residual > epsilon ** 2)  # indices of elements above a threshold

        col_sub = np.tile(np.arange(num_patches).reshape(-1, 1), num_atoms).T

        mat_inds_to_keep = sorted_residual_idx[above_thr_idx]
        col_sub_to_keep = col_sub[above_thr_idx]

        A = np.zeros((num_atoms, num_patches))
        A[mat_inds_to_keep, col_sub_to_keep] = inner_prod[mat_inds_to_keep, col_sub_to_keep]

        X = np.matmul(D, A)  # Reconstruction of restored image patches using dictionary and determined coefficients.

        return X, A

    @staticmethod
    def col2img(img_patches: np.ndarray, img_patch_size: Tuple, img_size: Tuple) -> np.ndarray:
        """
        This method transforms image patches into an image by averaging overlapping patches.
        :param img_patches: 2D np.ndarray with the shape of p x q, where p is the size of the flatten image patch and q
                            is the number of patches. Each column contains a flatten image patch.
        :param img_patch_size: Tuple. This is an original image patch size (m x n) that will be used to transform a
                               column (p x 1) in img_patches into a single 2D image of m x n.
        :param img_size: Final image size M x N.
        :return: img: Transformed image M x N.
        """

        numerator_img = np.zeros((img_size[0], img_size[1]))
        denominator_img = np.zeros((img_size[0], img_size[1]))

        for i in range(img_size[0] - img_patch_size[0] + 1):
            for j in range(img_size[1] - img_patch_size[1] + 1):
                # rebuild current patch
                num_of_curr_patch = i * (img_size[1] - img_patch_size[1] + 1) + (j + 1)
                last_row = i + img_patch_size[0]
                last_col = j + img_patch_size[1]
                curr_patch = img_patches[:, num_of_curr_patch - 1]
                curr_patch = np.reshape(curr_patch, (img_patch_size[0], img_patch_size[1]))
                numerator_img[i:last_row, j:last_col] = numerator_img[i:last_row, j:last_col] + curr_patch
                denominator_img[i:last_row, j:last_col] = denominator_img[i:last_row, j:last_col] + \
                                                          np.ones(curr_patch.shape)
        img = numerator_img / denominator_img

        return img

    def unitary_dictionary_learning(
            self,
            Y: np.ndarray,
            D_init: np.ndarray,
            num_iterations: int,
            pursuit_param: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        This function trains a unitary dictionary via procrustes analysis. The trained dictionary can be obtained by:
          D = argmin_D || Y - DA ||_F^2 s.t. D'D = I,
        where A is a matrix that contains all the estimated coefficients and Y contains training examples.

        :param Y: A matrix containing training patches in its columns.
        :param D_init: Initial unitary dictionary
        :param num_iterations: Number of updating the dictionary.
        :param pursuit_param: Criterion for stopping the pursuit algorithm.
        :return: D, mean_error, mean_cardinality: Trained dictionary, average representation error, and number of
                 non-zero elements.
        """

        mean_error = np.zeros(num_iterations)
        mean_cardinality = np.zeros(num_iterations)

        D = np.copy(D_init)

        # Procrustes analysis
        for i in range(num_iterations):
            [X, A] = self.batch_thresholding(D, Y, pursuit_param)

            if self.verbose:
                print(f"Iteration {i + 1}: ", end=" ")
            mean_error[i], mean_cardinality[i] = statistics.calculate_statistics(X, Y, A, self.verbose)

            [U, _, V] = np.linalg.svd(np.matmul(A, Y.T))
            D = np.matmul(V.T, U.T)

        return D, mean_error, mean_cardinality
