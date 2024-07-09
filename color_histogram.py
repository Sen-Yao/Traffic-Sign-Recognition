import cv2
import numpy as np
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
    image = np.array(image)
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

    # Create an empty feature vector
    feature_vector = np.zeros((num_bins_r, num_bins_g, num_bins_b))

    # Fill the feature vector with the normalized histograms
    for i in range(num_bins_r):
        for j in range(num_bins_g):
            for k in range(num_bins_b):
                feature_vector[i, j, k] = hist_r[i] * hist_g[j] * hist_b[k]

    return feature_vector.flatten()


def compute_hog_features(images, feature_vector, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    """
    Compute the HOG features for the given image and concatenate them with the provided feature vector.

    Parameters:
    - image: np.ndarray, input RGB image.
    - feature_vector: np.ndarray, the improved color histogram feature vector.
    - orientations: int, number of orientation bins for the HOG descriptor.
    - pixels_per_cell: tuple of int, size of the cell in pixels.
    - cells_per_block: tuple of int, number of cells in each block.

    Returns:
    - combined_feature_vector: np.ndarray, the combined feature vector with color histogram and HOG features.
    """
    hog_features_list = []
    for image in images:
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.resize(gray_image, (48, 48))
        # Normalize the image using Gamma correction
        normalized_image = np.power(image / 255.0, 1.0 / 2.2)

        # Compute HOG features
        hog_features = hog(normalized_image, orientations=orientations, pixels_per_cell=pixels_per_cell,
                           cells_per_block=cells_per_block, block_norm='L2-Hys', visualize=False, feature_vector=True)
        hog_features_list.append(hog_features)
    feature_vector = np.array(feature_vector)
    hog_features_list = np.array(hog_features_list)
    # Concatenate the improved color histogram features with the HOG features
    print("shape", feature_vector.shape, hog_features_list.shape)
    combined_feature_vector = np.concatenate((feature_vector, hog_features_list), axis=1)

    return combined_feature_vector


def pca_dimension_reduction(feature_matrix, n_components):
    """
    Perform PCA to reduce the dimension of the feature matrix.

    Parameters:
    - feature_matrix: np.ndarray, the combined feature matrix from color histogram and HOG features.
    - n_components: int, the number of principal components to keep.

    Returns:
    - reduced_feature_matrix: np.ndarray, the feature matrix after PCA dimensionality reduction.
    """
    # Standardize the feature matrix
    scaler = StandardScaler()
    feature_matrix_standardized = scaler.fit_transform(feature_matrix)

    # Perform PCA
    pca = PCA(n_components=n_components)
    reduced_feature_matrix = pca.fit_transform(feature_matrix_standardized)

    return reduced_feature_matrix