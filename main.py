import numpy as np
import imageio
from degrading_image_tools import DegradationType, DegradingImage
import matplotlib.pyplot as plt
from dictionary import Dictionary, DictionaryType
from sparse_solver import SparseSolver
from compute_stat import compute_psnr

#################
image_data_path = "/Users/kotarosonoda/Repository/sparseland_for_image_processing.git/sample_data/barbara.png"

config_image_degradation = {
    "image_degradation_type": DegradationType.add_random_noise,
    "noise_sigma": 20, "random_seed": 1
}

sparseland_model = {
    "patch_size": (10, 10),  # The patch must be a square
    "initial_dict": DictionaryType.dct_dictionary,
    "dictionary_learning": True,  # False means a predefined dictionary will be used.
    "num_learning_iterations": 20  # number of learning iterations
}
sparseland_model["epsilon"] = np.sqrt(1.1) * sparseland_model["patch_size"][0] * config_image_degradation["noise_sigma"]

#################

# Load an image
img = imageio.imread(image_data_path)
img = img[:256, 251:251+256]

# Degrading an image
degrading_image = DegradingImage(sigma=config_image_degradation["noise_sigma"],
                                 rand_seed=config_image_degradation["random_seed"],
                                 degradation_type=config_image_degradation["image_degradation_type"])
degraded_img = degrading_image.degradation(img)

# Get a dictionary
dictionary = Dictionary(dictionary_type=sparseland_model["initial_dict"])
dict = dictionary.get_dictionary(sparseland_model["patch_size"])

sparse_solver = SparseSolver(dictionary_learning=sparseland_model["dictionary_learning"],
                             num_learning_iterations=sparseland_model["num_learning_iterations"])
reconst_img = sparse_solver(degraded_img, dict, sparseland_model)


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
