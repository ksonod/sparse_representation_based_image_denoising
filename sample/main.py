import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from data.degrade_images import degrade_image, DegradationType
from algorithm.dictionary import Dictionary, DictionaryType
from algorithm.sparse_solver import SparseSolver
from algorithm.statistics import calculate_psnr


INPUT_FILE = {
    "file_path": Path("./data/sample_image/image.png")  # Gray scale image
}

CONFIG = {
    "image_degradation": {
        "noise_sigma": 25,
        "random_seed": 41,
        "degradation_type": DegradationType.NOISE
    },
    "sparse_model": {
        "patch_size": (10, 10),  # The patch must be a square
        "initial_dict": DictionaryType.DCT,
        "enable_dictionary_learning": False,  # False means a predefined dictionary will be used.
        # "num_learning_iterations": 30,  # number of learning iterations
        # "epsilon": 210,
        "verbose": True,
    },
}


def check_config(config: dict):
    if "epsilon" not in config:
        config["sparse_model"]["epsilon"] = \
            np.sqrt(1.1) * config["sparse_model"]["patch_size"][0] * config["image_degradation"]["noise_sigma"]

    if not config["sparse_model"]["enable_dictionary_learning"]:
        config["sparse_model"]["num_learning_iterations"] = 0


def run_scripts(input_file: dict, config: dict):

    check_config(config)

    # Load an image
    img = np.array(Image.open(input_file["file_path"]))
    degraded_img = degrade_image(img, config["image_degradation"])

    dictionary = Dictionary(
        dictionary_type=config["sparse_model"]["initial_dict"], patch_size=config["sparse_model"]["patch_size"]
    )

    sparse_solver = SparseSolver(
        enable_dictionary_learning=config["sparse_model"]["enable_dictionary_learning"],
        num_learning_iterations=config["sparse_model"]["num_learning_iterations"],
        img=degraded_img,
        dictionary=dictionary,
        epsilon=config["sparse_model"]["epsilon"],
        verbose=config["sparse_model"]["verbose"]
    )
    reconstructed_img = sparse_solver()

    if config["sparse_model"]["verbose"]:
        dictionary.show_dictionary()

    # Data visualization
    plt.figure(figsize=(14, 4))
    plt.subplot(131)
    plt.imshow(img, "gray")
    plt.axis("off")
    plt.title("Original image")

    plt.subplot(132)
    plt.imshow(degraded_img, "gray")
    plt.axis("off")
    plt.title("Degraded image PSNR={:.3f}".format(calculate_psnr(img, degraded_img)))

    plt.subplot(133)
    plt.imshow(reconstructed_img, "gray")
    plt.axis("off")
    plt.title("Reconstrucred image. PSNR={:.3f}".format(calculate_psnr(img, reconstructed_img)))

    plt.show()


if __name__ == "__main__":
    run_scripts(input_file=INPUT_FILE, config=CONFIG)
