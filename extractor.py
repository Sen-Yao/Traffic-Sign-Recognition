import cv2
import numpy as np
from tqdm import tqdm
from skimage.feature import hog
from skimage.filters import gabor_kernel


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


def extract_gist_features(X, orientations=8, pixels_per_cell=(10, 10), cells_per_block=(1, 1), image_size=(64, 64),
                          num_blocks=4):
    X_features = []

    # Define Gabor filters
    gabor_filters = []
    for theta in np.linspace(0, np.pi, orientations):
        for frequency in [0.1, 0.2, 0.3, 0.4]:
            gabor_filters.append(gabor_kernel(frequency, theta=theta))

    def apply_gabor_filters(img, filters):
        filtered_images = []
        for kern in filters:
            filtered = cv2.filter2D(img, cv2.CV_8UC3, np.real(kern))
            filtered_images.append(filtered)
        return filtered_images

    for x in tqdm(X, desc="Extracting GIST features"):
        # Resize images to the specified size
        temp_x = cv2.resize(x, image_size)
        # Convert the images to grayscale
        temp_x = cv2.cvtColor(temp_x, cv2.COLOR_BGR2GRAY)

        # Apply Gabor filters
        gabor_responses = apply_gabor_filters(temp_x, gabor_filters)

        # Divide the image into blocks and compute the mean response in each block
        height, width = temp_x.shape
        block_size = (height // num_blocks, width // num_blocks)
        gist_descriptor = []

        for gabor_img in gabor_responses:
            for i in range(num_blocks):
                for j in range(num_blocks):
                    block = gabor_img[i * block_size[0]:(i + 1) * block_size[0],
                            j * block_size[1]:(j + 1) * block_size[1]]
                    gist_descriptor.append(block.mean())

        X_features.append(gist_descriptor)

    return np.array(X_features)