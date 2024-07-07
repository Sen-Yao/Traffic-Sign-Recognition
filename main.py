# main.py
import argparse
import time
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from data_reader import read_ctsd_dataset, read_gtsrb_dataset
from extractor import extract_hog_features, extract_gist_features


def main():
    parser = argparse.ArgumentParser(description="Train an SVM classifier on a dataset with HOG features")

    # Change the dataset_path to point to the unzipped Dataset_1/images folder in your computer.
    parser.add_argument('--dataset_path', type=str, default='Dataset', help='Path to the dataset')
    parser.add_argument('--dataset_name', type=str, default='GTSRB',  help='Name of the dataset')
    parser.add_argument('--feature_extractor', type=str, default='GIST', help='Feature extraction method (default: hog)')
    parser.add_argument('--classifier', type=str, default='svm', help='Classifier to use (default: svm)')

    args = parser.parse_args()

    X = []
    y = []
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    # Train set and Test set have been split?
    done_train_test_split = False

    if args.dataset_name == 'CTSD':
        print("Reading CTSD Dataset...")
        X, y = read_ctsd_dataset(args.dataset_path, args.dataset_name)
    elif args.dataset_name == 'GTSRB':
        print("Reading GTSRB Dataset...")
        done_train_test_split = True
        X_train, X_test, y_train, y_test = read_gtsrb_dataset(args.dataset_path, args.dataset_name)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")

    if args.feature_extractor == 'HOG':
        print("Extracting HOG features...")
        if done_train_test_split:
            X_train = extract_hog_features(X_train)
            X_test = extract_hog_features(X_test)
        else:
            X = extract_hog_features(X)
    elif args.feature_extractor == 'GIST':
        print("Extracting GIST features...")
        if done_train_test_split:
            X_train = extract_gist_features(X_train)
            X_test = extract_gist_features(X_test)
        else:
            X = extract_gist_features(X)
    else:
        raise ValueError(f"Unsupported feature extractor: {args.feature_extractor}")

    if not done_train_test_split:
        # split X_features and y into training and testing sets
        # Use a 80-20 split and make sure to shuffle the samples.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    # Use the sklearn SVM package to train a classifier using x_train and y_train.
    # 选择并训练分类器
    if args.classifier == 'svm':
        clf = SVC(kernel='linear')
        print("Training SVM classifier...")
    elif args.classifier == 'knn':
        clf = KNeighborsClassifier(n_neighbors=5)
        print("Training KNN classifier with k =", 5)
    else:
        raise ValueError(f"Unsupported classifier: {args.classifier}")
    t1 = time.time()
    print("Training start at", time.ctime(t1))
    clf.fit(X_train, y_train)
    print("Training time =", time.time()-t1)
    print("Predicting...")
    # Use the x_test and y_test to evaluate the classifier and print the accuracy value.
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
