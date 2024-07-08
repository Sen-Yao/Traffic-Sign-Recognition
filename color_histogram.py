import cv2
import numpy as np


def improved_color_histogram(image, num_bins_r, num_bins_g, num_bins_b):
    """
    Compute the improved color histogram feature for the given image.

    Parameters:
    - image: np.ndarray, input RGB image.
    - num_bins_r: int, number of bins for the red channel.
    - num_bins_g: int, number of bins for the green channel.
    - num_bins_b: int, number of bins for the blue channel.

    Returns:
    - feature_vector: np.ndarray, the improved color histogram feature vector.
    """
    # Ensure the image is in RGB format
    if image.shape[2] != 3:
        raise ValueError("Input image must be an RGB image")

    # Compute histograms for each channel
    hist_r = cv2.calcHist([image], [0], None, [num_bins_r], [0, 256])
    hist_g = cv2.calcHist([image], [1], None, [num_bins_g], [0, 256])
    hist_b = cv2.calcHist([image], [2], None, [num_bins_b], [0, 256])

    # Normalize the histograms
    hist_r /= np.sum(hist_r)
    hist_g /= np.sum(hist_g)
    hist_b /= np.sum(hist_b)

    # Concatenate the histograms to form the improved feature vector
    feature_vector = np.concatenate((hist_r, hist_g, hist_b)).flatten()

    return feature_vector