import numpy as np
from typing import Tuple


def calculate_psnr(y_original: np.ndarray, y_tested: np.ndarray, dynamic_range=255) -> float:
    """
    This function calculates peak signal-to-noise (PSNR) using original and another images.
    :param y_original: Original image.
    :param y_tested: Image to be tested.
    :param dynamic_range: Maximum signal value.
    :return: PSNR
    """

    mse_val = np.sum((y_original - y_tested) ** 2) / y_original.size
    psnr_val = 10 * np.log10(dynamic_range ** 2 / mse_val)
    return psnr_val


def calculate_statistics(
        est_img_patches: np.ndarray, original_img_patches: np.ndarray, est_coeffs: np.ndarray, verbose: bool
) -> Tuple[float, float]:
    """
    This function calculates some useful statistics.
    :param est_img_patches: estimated image patches in its columns
    :param original_img_patches: original image patches in its columns
    :param est_coeffs: estimated coefficients
    :param verbose:
    :return: residual_error, avg_cardinality: Mean squared error averaged over pixels ahd patches and average number of
                                              non-zero elements.
    """

    non_zero_threshold = 10 ** (-10)

    # Averaging over pixels in a patch and all the patches.
    residual_error = np.mean((est_img_patches - original_img_patches) ** 2)

    # Averaging over all the patches.
    avg_cardinality = np.mean(np.sum(np.abs(est_coeffs) > non_zero_threshold, axis=0))

    if verbose:
        print(f"Residual error = {residual_error: .2f} -- Average cardinality = {avg_cardinality: .2f}")

    return residual_error, avg_cardinality
