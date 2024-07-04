import cv2
import glob


def read_ctsd_dataset(dataset_path, dataset_name):
    X = []
    y = []
    for i in glob.glob(f"{dataset_path}/{dataset_name}/images/*.png", recursive=True):
        label = i.split("images")[1][1:4]
        y.append(label)
        # read the images using opencv and append them to list X
        img = cv2.imread(i)
        X.append(img)
    return X, y
