# main.py
import argparse
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from data_reader import read_ctsd_dataset
from extractor import extract_hog_features


def main():
    parser = argparse.ArgumentParser(description="Train an SVM classifier on a dataset with HOG features")

    # Change the dataset_path to point to the unzipped Dataset_1/images folder in your computer.
    parser.add_argument('--dataset_path', type=str, default='Dataset', help='Path to the dataset')
    parser.add_argument('--dataset_name', type=str, default='CTSD',  help='Name of the dataset')
    parser.add_argument('--feature_extractor', type=str, default='hog', help='Feature extraction method (default: hog)')
    parser.add_argument('--classifier', type=str, default='svm', help='Classifier to use (default: svm)')

    args = parser.parse_args()

    if args.dataset_name == 'CTSD':
        print("Reading CTSD Dataset...")
        X, y = read_ctsd_dataset(args.dataset_path, args.dataset_name)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")

    if args.feature_extractor == 'hog':
        print("Extracting hog features...")
        X_features = extract_hog_features(X)
    else:
        raise ValueError(f"Unsupported feature extractor: {args.feature_extractor}")

    # split X_features and y into training and testing sets
    # Use a 80-20 split and make sure to shuffle the samples.
    X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42, shuffle=True)

    # Use the sklearn SVM package to train a classifier using x_train and y_train.
    if args.classifier == 'svm':
        clf = SVC(kernel='linear')
        print("Training SVM classifier...")
    else:
        raise ValueError(f"Unsupported classifier: {args.classifier}")

    clf.fit(X_train, y_train)

    # Use the x_test and y_test to evaluate the classifier and print the accuracy value.
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
