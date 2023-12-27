import numpy as np
from data.degrade_images import degrade_image
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
        "initial_dict": DictionaryType.DCT,
        "dictionary_learning": False,  # False means a predefined dictionary will be used.
        "num_learning_iterations": 20  # number of learning iterations
    }
}


def run_scripts(input_file: dict, config: dict):

    if "epsilon" not in config:
        config["sparse_model"]["epsilon"] = \
            np.sqrt(1.1) * config["sparse_model"]["patch_size"][0] * config["image_degradation"]["noise_sigma"]

    # Load an image
    img = np.array(Image.open(input_file["file_path"]))
    img = img[:256, 251:251+256]  # TODO: Remove it because it exists only for test purposes.
    degraded_img = degrade_image(img, config["image_degradation"])

    dictionary = Dictionary(
        dictionary_type=config["sparse_model"]["initial_dict"], patch_size=config["sparse_model"]["patch_size"]
    )

    sparse_solver = SparseSolver(
        dictionary_learning=config["sparse_model"]["dictionary_learning"],
        num_learning_iterations=config["sparse_model"]["num_learning_iterations"],
        img=degraded_img,
        dictionary=dictionary,
        epsilon=config["sparse_model"]["epsilon"]
    )
    reconstructed_img = sparse_solver()

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
    plt.imshow(reconstructed_img, "gray")
    plt.axis("off")
    plt.title("reconstrucred img. PSNR={:.3f}".format(compute_psnr(img, reconstructed_img)))

    plt.show()


if __name__ == "__main__":
    run_scripts(input_file=INPUT_FILE, config=CONFIG)
