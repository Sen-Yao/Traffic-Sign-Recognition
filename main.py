# main.py
import argparse
import time
import os
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from sklearn.decomposition import PCA

from data_reader import read_ctsd_dataset, read_gtsrb_dataset
from extractor import extract_hog_features, extract_gist_features, color_histogram_extractor
from MLP import MLP, create_data_loaders, train, test


def main():
    parser = argparse.ArgumentParser(description="Train an SVM classifier on a dataset with HOG features")

    # Change the dataset_path to point to the unzipped Dataset_1/images folder in your computer.
    parser.add_argument('--dataset_path', type=str, default='Dataset', help='Path to the dataset')

    parser.add_argument('--dataset_name', type=str, default='GTSRB',  help='Name of the dataset')
    parser.add_argument('--feature_extractor', type=str, default='color_histogram', help='Feature extraction method (default: hog)')

    parser.add_argument('--classifier', type=str, default='mlp', help='Classifier to use (default: svm)')

    args = parser.parse_args()

    X = []
    y = []
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    # Train set and Test set have been split?
    done_train_test_split = False

    if args.dataset_name == 'GTSRB-test':
        print("Reading GTSRB-test Dataset...")
        X, y = read_ctsd_dataset(args.dataset_path, args.dataset_name)
    elif args.dataset_name == 'CTSD':
        print("Reading CTSD Dataset...")
        X, y = read_ctsd_dataset(args.dataset_path, args.dataset_name)
        print(X[0].shape)
    elif args.dataset_name == 'GTSRB':
        print("Reading GTSRB Dataset...")
        done_train_test_split = True
        X_train, X_test, y_train, y_test = read_gtsrb_dataset(args.dataset_path, args.dataset_name)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")

    features_dir = 'features'
    os.makedirs(features_dir, exist_ok=True)

    if args.feature_extractor == 'HOG':
        print("Extracting HOG features...")
        if done_train_test_split:
            X_train = extract_hog_features(X_train)
            X_test = extract_hog_features(X_test)
        else:
            X = extract_hog_features(X)
    elif args.feature_extractor == 'GIST':
        print("Extracting GIST features...")
        train_features_path = os.path.join(features_dir, f'{args.dataset_name}_train_gist_features.pkl')
        test_features_path = os.path.join(features_dir, f'{args.dataset_name}_test_gist_features.pkl')
        if done_train_test_split:

            print("Extracting GIST features...")
            train_features_path = os.path.join(features_dir, f'{args.dataset_name}_train_gist_features.pkl')
            test_features_path = os.path.join(features_dir, f'{args.dataset_name}_test_gist_features.pkl')
            if done_train_test_split:
                if os.path.exists(train_features_path) and os.path.exists(test_features_path):
                    X_train = joblib.load(train_features_path)
                    X_test = joblib.load(test_features_path)
                    print("Loaded precomputed GIST features.")
                else:
                    pca = PCA(n_components=1024)
                    X_train = extract_gist_features(X_train)
                    X_train = pca.fit_transform(X_train)
                    X_test = extract_gist_features(X_test)
                    X_test = pca.transform(X_test)
                    joblib.dump(X_train, train_features_path)
                    joblib.dump(X_test, test_features_path)
            else:
                features_path = os.path.join(features_dir, f'{args.dataset_name}_gist_features.pkl')
                if os.path.exists(features_path):
                    X = joblib.load(features_path)
                    print("Loaded precomputed GIST features.")
                else:
                    X = extract_gist_features(X)
                    joblib.dump(X, features_path)
        else:
            X = extract_gist_features(X)
    elif args.feature_extractor == 'color_histogram':
        print("Extracting color histogram features...")
        if done_train_test_split:
            train_features_path = os.path.join(features_dir, f'{args.dataset_name}_color_train_features.pkl')
            test_features_path = os.path.join(features_dir, f'{args.dataset_name}_color_test_features.pkl')
            if done_train_test_split:
                if os.path.exists(train_features_path) and os.path.exists(test_features_path):
                    X_train = joblib.load(train_features_path)
                    X_test = joblib.load(test_features_path)
                    print("Loaded precomputed color features.")
                else:
                    pca = PCA(n_components=512)
                    X_train = color_histogram_extractor(X_train)
                    print('X_train:', X_train.shape)
                    print('processing PCA on X_train')
                    X_train = pca.fit_transform(X_train)
                    print('X_train:', X_train.shape)
                    X_test = color_histogram_extractor(X_test)
                    print('X_test:', X_test.shape)
                    print('processing PCA on X_test')
                    X_test = pca.transform(X_test)
                    print('X_test:', X_test.shape)
                    joblib.dump(X_train, train_features_path)
                    joblib.dump(X_test, test_features_path)
            else:
                features_path = os.path.join(features_dir, f'{args.dataset_name}_color_features.pkl')
                if os.path.exists(features_path):
                    X = joblib.load(features_path)
                    print("Loaded precomputed color features.")
                else:
                    X = color_histogram_extractor(X)
                    joblib.dump(X, features_path)
        else:
            X = color_histogram_extractor(X)
    else:
        raise ValueError(f"Unsupported feature extractor: {args.feature_extractor}")

    if not done_train_test_split:
        # split X_features and y into training and testing sets
        # Use a 80-20 split and make sure to shuffle the samples.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    # 选择并训练分类器
    if args.classifier == 'svm':
        # clf = SVC(kernel='linear', verbose=True)
        clf = SVC(kernel='linear')
        print("Training SVM classifier...")
    elif args.classifier == 'knn':
        clf = KNeighborsClassifier(n_neighbors=5)
        print("Training KNN classifier with k =", 5)
    elif args.classifier == 'mlp':
        input_size = X_train.shape[1]
        hidden_size = 128
        output_size = len(np.unique(y_train))
        epochs = 15
        batch_size = 64
        learning_rate = 0.0001

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = MLP(input_size, hidden_size, output_size).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        train_loader, test_loader = create_data_loaders(X_train, y_train, X_test, y_test, batch_size)

        train(model, criterion, optimizer, train_loader, test_loader, epochs)
        test(model, test_loader)
    else:
        raise ValueError(f"Unsupported classifier: {args.classifier}")

    if args.classifier != 'mlp':
        # Traditional Machine Learning
        train_loader, test_loader = X_train, X_test
        t1 = time.time()
        print("Training start at", time.ctime(t1))
        clf.fit(train_loader, y_train)
        print("Training time =", time.time() - t1)
        print("Predicting...")
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
