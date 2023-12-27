import numpy as np
from data.degrade_images import DegradationType, degrade_image
from PIL import Image
import matplotlib.pyplot as plt
from algorithm.dictionary import Dictionary, DictionaryType
from algorithm.sparse_solver import SparseSolver
from algorithm.compute_stat import compute_psnr
from pathlib import Path


INPUT_FILE = {
    "file_path": Path("./data/sample_image/image.png")
}

CONFIG = {
    "image_degradation": {
        "noise_sigma": 20
    },
    "sparse_model": {
        "patch_size": (10, 10),  # The patch must be a square
        "initial_dict": DictionaryType.dct_dictionary,
        "dictionary_learning": True,  # False means a predefined dictionary will be used.
        "num_learning_iterations": 20  # number of learning iterations
    }
}

CONFIG["sparse_model"]["epsilon"] = np.sqrt(1.1) * CONFIG["sparse_model"]["patch_size"][0] * CONFIG["image_degradation"]["noise_sigma"]


def run_scripts(input_file: dict, config:dict):
    # Load an image
    img = np.array(Image.open(input_file["file_path"]))
    img = img[:256, 251:251+256]  # TODO: Remove it because it exists only for test purposes.

    # Degrading an image
    degraded_img = degrade_image(img, config["image_degradation"])

    # Get a dictionary
    dictionary = Dictionary(dictionary_type=config["sparse_model"]["initial_dict"])
    dict = dictionary.get_dictionary(config["sparse_model"]["patch_size"])

    sparse_solver = SparseSolver(dictionary_learning=config["sparse_model"]["dictionary_learning"],
                                 num_learning_iterations=config["sparse_model"]["num_learning_iterations"])
    reconst_img = sparse_solver(degraded_img, dict, config["sparse_model"])

    # data visualization
    plt.figure(figsize=(14, 4))
    plt.subplot(131)
    plt.imshow(img, "gray")
    plt.axis("off")
    plt.title("Original")

    plt.subplot(132)
    plt.imshow(degraded_img, "gray")
    plt.axis("off")
    plt.title("degraded img. PSNR={:.3f}".format(compute_psnr(img, degraded_img)))

    plt.subplot(133)
    plt.imshow(reconst_img, "gray")
    plt.axis("off")
    plt.title("reconstrucred img. PSNR={:.3f}".format(compute_psnr(img, reconst_img)))

    plt.show()


if __name__ == "__main__":
    run_scripts(input_file=INPUT_FILE, config=CONFIG)
