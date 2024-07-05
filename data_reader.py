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
        y.append(label)
        # read the images using opencv and append them to list X
        img = cv2.imread(i)
        X.append(img)

    return X, y


def read_gtsrb_dataset(dataset_path, dataset_name):
    train_X = []
    test_X = []
    train_y = []
    test_y = []

    # Read Train set
    train_folder_path = os.path.join(dataset_path, dataset_name, 'Train')
    class_folders = [f for f in os.listdir(train_folder_path) if os.path.isdir(os.path.join(train_folder_path, f))]

    for class_folder in tqdm(class_folders, desc="Reading training images"):
        class_path = os.path.join(train_folder_path, class_folder)
        label = class_folder
        for img_path in glob.glob(f"{class_path}/*.png"):
            # Read the images using OpenCV and append them to the list
            img = cv2.imread(img_path)
            train_X.append(img)
            train_y.append(label)

    # Read Test set
    test_csv_path = os.path.join(dataset_path, dataset_name, 'Test.csv')
    test_df = pd.read_csv(test_csv_path)

    # Traverse through all png images in the Test directory
    test_image_dir = os.path.join(dataset_path, dataset_name, 'Test')
    test_image_paths = [os.path.join(test_image_dir, f"{index:05d}.png") for index in range(len(test_df))]

    for img_path, (index, row) in tqdm(zip(test_image_paths, test_df.iterrows()), total=len(test_df),
                                       desc="Reading testing images"):
        # Read the image using OpenCV
        img = cv2.imread(img_path)
        test_X.append(img)

        # Append the corresponding label from the CSV file
        label = row['ClassId']
        test_y.append(str(label))

    return train_X, test_X, train_y, test_y
