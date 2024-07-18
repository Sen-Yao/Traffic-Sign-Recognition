import cv2
import glob
import os
import pandas as pd
from tqdm import tqdm


def read_ctsd_dataset(dataset_path, dataset_name):
    X = []
    y = []
    image_paths = glob.glob(f"{dataset_path}/{dataset_name}/images/*.png", recursive=True)

    for i in tqdm(image_paths, desc="Reading images"):
        label = i.split("images")[1][1:4]
        y.append(int(str(label)))
        # read the images using opencv and append them to list X
        img = cv2.imread(i)
        img = cv2.resize(img, (48, 48))
        X.append(img)

    return X, y


def read_gtsrb_dataset(dataset_path, dataset_name):
    train_X = []
    test_X = []
    train_y = []
    test_y = []

    # Read Train set
    train_folder_path = os.path.join(dataset_path, dataset_name)
    train_csv_path = os.path.join(dataset_path, dataset_name, 'Train.csv')
    train_df = pd.read_csv(train_csv_path)

    for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Reading training images"):
        # Extract image path and label
        img_path = os.path.join(train_folder_path, row['Path'])
        label = row['ClassId']

        # Read the image using OpenCV
        img = cv2.imread(img_path)

        # Crop the image based on ROI coordinates
        x1, y1, x2, y2 = row['Roi.X1'], row['Roi.Y1'], row['Roi.X2'], row['Roi.Y2']
        img_cropped = img[y1:y2, x1:x2]

        # Resize the image to 48x48
        img_resized = cv2.resize(img_cropped, (48, 48))

        # Append the image and label to the respective lists
        train_X.append(img_resized)
        train_y.append(int(label))

    # Read Test set
    test_csv_path = os.path.join(dataset_path, dataset_name, 'Test.csv')
    test_df = pd.read_csv(test_csv_path)

    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Reading testing images"):
        # Extract image path and label
        img_path = os.path.join(dataset_path, dataset_name, row['Path'])
        label = row['ClassId']

        # Read the image using OpenCV
        img = cv2.imread(img_path)

        # Crop the image based on ROI coordinates
        x1, y1, x2, y2 = row['Roi.X1'], row['Roi.Y1'], row['Roi.X2'], row['Roi.Y2']
        img_cropped = img[y1:y2, x1:x2]

        # Resize the image to 48x48
        img_resized = cv2.resize(img_cropped, (48, 48))

        # Append the image and label to the respective lists
        test_X.append(img_resized)
        test_y.append(int(label))

    return train_X, test_X, train_y, test_y