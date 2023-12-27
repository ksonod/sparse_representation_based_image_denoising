import numpy as np
import math

def compute_psnr(y_original, y_estimated):
    # COMPUTE_PSNR Computes the PSNR between two images
    #
    # Input:
    #  y_original  - The original image
    #  y_estimated - The estimated image
    #
    # Output:
    #  psnr_val - The Peak Signal to Noise Ratio (PSNR) score

    y_original = np.reshape(y_original, (-1))
    y_estimated = np.reshape(y_estimated, (-1))

    # Compute the dynamic range
    # Write your code here... dynamic_range = ????
    dynamic_range = 255.0

    # Compute the Mean Squared Error (MSE)
    # Write your code here... mse_val = ????
    mse_val = (1 / len(y_original)) * np.sum((y_original - y_estimated) ** 2)

    # Compute the PSNR
    # Write your code here... psnr_val = ????
    psnr_val = 10 * math.log10(dynamic_range ** 2 / mse_val)

    return psnr_val

# TODO: Clean and modify it
def compute_stat(est_patches, orig_patches, est_coeffs):
    # COMPUTE_STAT Compute and print usefull statistics of the pursuit and
    # learning procedures
    #
    # Inputs:
    #  est_patches  - A matrix, containing the recovered patches as its columns
    #  orig_patches - A matrix, containing the original patches as its columns
    #  est_coeffs   - A matrix, containing the estimated representations,
    #                 leading to est_patches, as its columns
    #
    # Outputs:
    #  residual_error  - Average Mean Squared Error (MSE) per pixel
    #  avg_cardinality - Average number of nonzeros that is used to represent
    #                    each patch
    #

    # Compute the Mean Square Error per patch
    MSE_per_patch = np.sum((est_patches - orig_patches) ** 2, axis=0)

    # Compute the average
    residual_error = np.mean(MSE_per_patch) / np.shape(orig_patches)[0]

    # Compute the average number of non-zeros
    avg_cardinality = np.sum(np.abs(est_coeffs) > 10 ** (-10)) / np.shape(est_coeffs)[1]

    # Display the results
    print('Residual error %2.2f, Average cardinality %2.2f' % (residual_error, avg_cardinality))

    return residual_error, avg_cardinality
