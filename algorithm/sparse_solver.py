import numpy as np
import numpy.matlib
from skimage.util import view_as_windows
from algorithm import compute_stat

class SparseSolver:
    def __init__(self, dictionary_learning=True, num_learning_iterations=20):
        self.dictionary_learning = dictionary_learning
        self.num_learning_iterations = num_learning_iterations

    def __call__(self, img, dict, sparseland_model):
        patches = self.create_overlapping_patches(img, sparseland_model["patch_size"])

        if self.dictionary_learning:
            dict, mean_error, mean_cardinality = self.unitary_dictionary_learning(patches, dict,
                                                                                  self.num_learning_iterations,
                                                                                  sparseland_model["epsilon"])

        [est_patches, est_coeffs] = self.batch_thresholding(dict, patches, sparseland_model["epsilon"])
        reconst_img = self.col2im(est_patches, sparseland_model["patch_size"], img.shape)

        return reconst_img

    def create_overlapping_patches(self, img, patch_size):
        h, w = img.shape

        # number of patches in x and y directions
        x_patch_num = w - patch_size[1] + 1
        y_patch_num = h - patch_size[0] + 1

        # creating overlapping patches (y_patch_num, x_patch_num, patch_size[1], patch_size[0])
        patches = view_as_windows(img, patch_size, step=1)

        # flattened patches
        flattened_patches = patches.reshape(x_patch_num * y_patch_num, np.prod(patch_size)).T

        return flattened_patches

    def batch_thresholding(self, dict, patches, epsilon):
        # BATCH_THRESHOLDING solves the pursuit problem via the error-constraint thresholding pursuit.
        # It solves the following problem:
        #   min_{alpha_i} \sum_i || alpha_i ||_0
        #                  s.t.  ||y_i - D alpha_i||_2**2 \leq epsilon**2 for all i,
        # where D is a dictionary of size n X n, y_i are the input signals of
        # length n (being the columns of the matrix Y) and epsilon stands
        # for the allowed residual error.
        #
        # The solution is returned in the matrix A, containing the representations
        # of the patches as its columns, along with the denoised signals
        # given by  X = DA.

        # Get the number of atoms
        num_atoms = dict.shape[1]

        # Get the number of patches
        N = patches.shape[1]

        # Compute the inner products between the dictionary atoms and the input patches
        inner_products = np.matmul(dict.T, patches)

        # Compute epsilon**2, which is the square residual error allowed per patch
        epsilon_sq = epsilon ** 2

        # Compute the square value of each entry in 'inner_products' matrix
        residual_sq = inner_products ** 2

        # Sort each column in 'residual_sq' matrix in ascending order
        mat_sorted = np.sort(residual_sq, axis=0)
        mat_inds = np.argsort(residual_sq, axis=0)

        # Compute the cumulative sums for each column of 'mat_sorted' and save the result in the matrix 'accumulate_residual'
        accumulate_residual = np.cumsum(mat_sorted, axis=0)

        # Compute the indices of the dominant coefficients that we want to keep
        inds_to_keep = (accumulate_residual > epsilon_sq)

        # Allocate a matrix of size n X N to save the sparse vectors
        A = np.zeros((num_atoms, N))

        # In what follows we compute the location of each non-zero to be assigned
        # to the matrix of the sparse vectors A. To this end, we need to map
        # 'mat_inds' to a linear subscript format. The mapping will be done using
        # Matlab's 'sub2ind' function.

        # TODO: Replace matlib
        # Create a repetition of the column index for all rows
        col_sub = np.matlib.repmat(np.arange(N), num_atoms, 1)

        # Map the entries in 'inds_to_keep' to their corresponding locations
        # in 'mat_inds' and 'col_sub'.
        mat_inds_to_keep = mat_inds[inds_to_keep]
        col_sub_to_keep = col_sub[inds_to_keep]

        # Assign to 'A' the coefficients in 'inner_products' using
        # the precomputed 'mat_inds_to_keep' and 'col_sub_to_keep'
        A[mat_inds_to_keep, col_sub_to_keep] = inner_products[mat_inds_to_keep, col_sub_to_keep]

        # Reconstruct the patches using 'A' matrix
        X = np.matmul(dict, A)

        return X, A

    # TODO: Edit and clean the code
    def col2im(self, patches, patch_size, im_size):
        # COL_TO_IM Rearrange matrix columns into an image of size MXN
        #
        # Inputs:
        #  patches - A matrix of size p * q, where p is the patch flatten size (height * width = m * n), and q is number of patches.
        #  patch_size - The size of the patch [height width] = [m n]
        #  im_size    - The size of the image we aim to build [height width] = [M N]
        #
        # Output:
        #  im - The reconstructed image, computed by returning the patches in
        #       'patches' to their original locations, followed by a
        #       patch-averaging over the overlaps

        num_im = np.zeros((im_size[0], im_size[1]))
        denom_im = np.zeros((im_size[0], im_size[1]))

        for i in range(im_size[0] - patch_size[0] + 1):
            for j in range(im_size[1] - patch_size[1] + 1):
                # rebuild current patch
                num_of_curr_patch = i * (im_size[1] - patch_size[1] + 1) + (j + 1)
                last_row = i + patch_size[0]
                last_col = j + patch_size[1]
                curr_patch = patches[:, num_of_curr_patch - 1]
                curr_patch = np.reshape(curr_patch, (patch_size[0], patch_size[1]))

                # update 'num_im' and 'denom_im' w.r.t. 'curr_patch'
                num_im[i:last_row, j:last_col] = num_im[i:last_row, j:last_col] + curr_patch
                denom_im[i:last_row, j:last_col] = denom_im[i:last_row, j:last_col] + np.ones(curr_patch.shape)

        # Averaging
        im = num_im / denom_im

        return im

    def unitary_dictionary_learning(self, Y, D_init, num_iterations, pursuit_param):
        # UNITARY_DICTIONARY_LEARNING Train a unitary dictionary via
        # Procrustes analysis.
        #
        # Inputs:
        #   Y              - A matrix that contains the training patches
        #                    (as vectors) as its columns
        #   D_init         - Initial UNITARY dictionary
        #   num_iterations - Number of dictionary updates
        #   pursuit_param  - The stopping criterion for the pursuit algorithm
        #
        # Outputs:
        #   D          - The trained UNITARY dictionary
        #   mean_error - A vector, containing the average representation error,
        #                computed per iteration and averaged over the total
        #                training examples
        #   mean_cardinality - A vector, containing the average number of nonzeros,
        #                      computed per iteration and averaged over the total
        #                      training examples

        # Allocate a vector that stores the average representation
        # error per iteration
        mean_error = np.zeros(num_iterations)

        # Allocate a vector that stores the average cardinality per iteration
        mean_cardinality = np.zeros(num_iterations)

        # Set the dictionary to be D_init
        D = np.copy(D_init)

        # Run the Procrustes analysis algorithm for num_iterations
        for i in range(num_iterations):
            # Compute the representation of each noisy patch
            [X, A] = self.batch_thresholding(D, Y, pursuit_param)

            # Compute and display the statistics
            print('Iter %02d: ' % (i + 1), end=" ")
            mean_error[i], mean_cardinality[i] = compute_stat.compute_stat(X, Y, A)

            # Update the dictionary via Procrustes analysis.
            # Solve D = argmin_D || Y - DA ||_F^2 s.t. D'D = I,
            # where 'A' is a matrix that contains all the estimated coefficients,
            # and 'Y' contains the training examples. Use the Procrustes algorithm.
            # Write your code here... D = ???

            [U, _, V] = np.linalg.svd(np.matmul(A, Y.T))
            D = np.matmul(V.T, U.T)
            # not np.matmul(V, U.T). See https://stackoverflow.com/questions/50930899/svd-command-in-python-v-s-matlab

        return D, mean_error, mean_cardinality
