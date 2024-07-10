import cv2
import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift
from tqdm import tqdm
from skimage.feature import hog
from skimage.filters import gabor_kernel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from color_histogram import improved_color_histogram, compute_hog_features, pca_dimension_reduction


def extract_hog_features(X, orientations=8, pixels_per_cell=(10, 10), cells_per_block=(1, 1)):
    X_features = []
    for x in X:
        # resize them to 48x48 pixels.
        temp_x = cv2.resize(x, (48, 48))
        # convert the images to grayscale using the cvtColor function in opencv
        temp_x = cv2.cvtColor(temp_x, cv2.COLOR_BGR2GRAY)

        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # temp_x = clahe.apply(temp_x)
        # 高斯模糊
        blurred_img = cv2.GaussianBlur(temp_x, (11, 11), 0)

        # 增强对比度
        enhanced_img = cv2.addWeighted(temp_x, 1.5, blurred_img, -0.5, 0)

        # extract hog features
        x_feature = hog(enhanced_img, orientations=orientations, pixels_per_cell=pixels_per_cell,
                        cells_per_block=cells_per_block, visualize=False)
        X_features.append(x_feature)
    return X_features


def compute_power_spectrum(image):
    # Convert image to float32 and compute the power spectrum in the Fourier domain
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    power_spectrum = np.abs(fshift)**2
    return power_spectrum


def apply_gabor_filters(img, filters):
    filtered_images = []
    for kern in filters:
        filtered = cv2.filter2D(img, cv2.CV_8UC3, np.real(kern))
        filtered_images.append(filtered)
    return filtered_images


def extract_gist_descriptor(img, filters, num_blocks):

    gabor_responses = apply_gabor_filters(img, filters)
    height, width = img.shape
    block_size = (height // num_blocks, width // num_blocks)
    gist_descriptor = []
    for gabor_img in gabor_responses:
        for i in range(num_blocks):
            for j in range(num_blocks):
                block = gabor_img[i * block_size[0]:(i + 1) * block_size[0],
                        j * block_size[1]:(j + 1) * block_size[1]]
                gist_descriptor.append(block.mean())
    return gist_descriptor


def extract_gist_features(X, orientations=8, image_size=(64, 64), num_blocks=4):
    X_features = []
    # Define Gabor filters
    gabor_filters = []
    for theta in np.linspace(0, np.pi, orientations):
        for frequency in [0.1, 0.2, 0.3, 0.4]:
            gabor_filters.append(gabor_kernel(frequency, theta=theta))
    for x in tqdm(X, desc="Extracting GIST features"):
        # Resize images to the specified size
        temp_x = cv2.resize(x, image_size)
        temp_x = cv2.cvtColor(temp_x, cv2.COLOR_BGR2GRAY)
        # Initialize a list to hold GIST descriptors for all channels
        gist_descriptor = []

        # Process each channel separately
        # Generate outer BB by removing 5 pixels
        outer_bb = temp_x[5:-5, 5:-5]
        outer_bb = cv2.resize(outer_bb, image_size)

        # Generate inner BB by removing additional 10 pixels
        inner_bb = outer_bb[10:-10, 10:-10]
        inner_bb = cv2.resize(inner_bb, image_size)

        # Extract GIST descriptors for both BBs
        gist_outer = extract_gist_descriptor(outer_bb, gabor_filters, num_blocks)
        gist_inner = extract_gist_descriptor(inner_bb, gabor_filters, num_blocks)

        # Concatenate the GIST descriptors for this channel
        gist_descriptor.extend(gist_outer + gist_inner)

        # Append the final GIST descriptor for this image to the list of features
        X_features.append(gist_descriptor)

    return np.array(X_features)


def color_histogram_extractor(X):
    num_bins_r = 8
    num_bins_g = 8
    num_bins_b = 8
    print("Processing the histogram")
    color_histogram = improved_color_histogram(X, num_bins_r, num_bins_g, num_bins_b)
    gist_features = extract_gist_features(X)
    implement_features = np.hstack((color_histogram, gist_features))
    return implement_features
