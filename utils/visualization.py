import numpy as np
import matplotlib.pyplot as plt

# normalize saliency map to [0,1]
def normalize_saliency(saliency):
    saliency = saliency - saliency.min()
    saliency = saliency / (saliency.max() + 1e-8)
    return saliency

# display original image and saliency map side by side
def plot_saliency(original_image, saliency_map, title="Saliency Map"):
    saliency_map = normalize_saliency(saliency_map)
    plt.figure(figsize=(10,4))

    # original image
    plt.subplot(1,2,1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis("off")

    # saliency map
    plt.subplot(1, 2, 2)
    plt.imshow(saliency_map, cmap="hot")
    plt.title(title)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# Overlay saliency map on original image.
def overlay_saliency(original_image, saliency_map, alpha=0.5):
    saliency_map = normalize_saliency(saliency_map)

    plt.figure(figsize=(5, 5))
    plt.imshow(original_image)
    plt.imshow(saliency_map, cmap="jet", alpha=alpha)
    plt.axis("off")
    plt.title("Overlay Visualization")
    plt.show()