import cv2
from skimage.feature import hog


def extract_hog_features(X, orientations=8, pixels_per_cell=(10, 10), cells_per_block=(1, 1)):
    X_features = []
    for x in X:
        # resize them to 48x48 pixels.
        temp_x = cv2.resize(x, (48, 48))
        # convert the images to grayscale using the cvtColor function in opencv
        temp_x = cv2.cvtColor(temp_x, cv2.COLOR_BGR2GRAY)
        # extract hog features
        x_feature = hog(temp_x, orientations=orientations, pixels_per_cell=pixels_per_cell,
                        cells_per_block=cells_per_block, visualize=False)
        X_features.append(x_feature)
    return X_features
