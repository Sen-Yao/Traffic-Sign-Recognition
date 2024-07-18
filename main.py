# main.py
import argparse
import time
import os
from collections import defaultdict

import joblib
import numpy as np
import torch
import cv2
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier

import  matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample

from data_reader import read_ctsd_dataset, read_gtsrb_dataset
from extractor import extract_hog_features, extract_gist_features, color_histogram_extractor, color_histogram_CNN_extractor, cnn_preprocess
from MLP import MLP, create_data_loaders, train, test, test_bagging
from graph_embedding import graph_embedding_lda
from CNN import CNN, cnn_train, cnn_create_data_loaders


def main():
    parser = argparse.ArgumentParser(description="Train an SVM classifier on a dataset with HOG features")

    # Change the dataset_path to point to the unzipped Dataset_1/images folder in your computer.
    parser.add_argument('--dataset_path', type=str, default='Dataset', help='Path to the dataset')

    parser.add_argument('--dataset_name', type=str, default='GTSRB',  help='Name of the dataset')
    parser.add_argument('--feature_extractor', type=str, default='color_histogram_cnn', help='Feature extraction method (default: hog)')

    parser.add_argument('--classifier', type=str, default='svm', help='Classifier to use (default: svm)')
    parser.add_argument('--ensemble', type=str, default='none', help='Ensemble Learning to use (default: none)')

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
        print(X[0].shape)
    elif args.dataset_name == 'GTSRB':
        print("Reading GTSRB Dataset...")
        done_train_test_split = True
        X_train, X_test, y_train, y_test = read_gtsrb_dataset(args.dataset_path, args.dataset_name)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")

    features_dir = 'features'
    os.makedirs(features_dir, exist_ok=True)
    if args.feature_extractor == 'none':
        pass
    elif args.feature_extractor == 'HOG':
        print("Extracting HOG features...")
        if done_train_test_split:
            X_train = extract_hog_features(X_train)
            X_test = extract_hog_features(X_test)
        else:
            X = extract_hog_features(X)
    elif args.feature_extractor == 'GIST':
        print("Extracting GIST features...")
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
            cnn_model_features_path = os.path.join(features_dir, f'{args.dataset_name}_color_model.pkl')
            if done_train_test_split:
                if os.path.exists(train_features_path) and os.path.exists(test_features_path):
                    X_train = joblib.load(train_features_path)
                    X_test = joblib.load(test_features_path)
                    print("Loaded precomputed color features.")
                else:

                    color_pca = PCA(n_components=32)
                    hog_pca = PCA(n_components=64)
                    gist_pca = PCA(n_components=64)
                    print("Start Extracting CNN features")
                    X_train = color_histogram_extractor(X_train, color_pca=color_pca, hog_pca=hog_pca,
                                                            gist_pca=gist_pca, train=True)
                    print('X_train:', X_train.shape)
                    X_test = color_histogram_extractor(X_test, color_pca=color_pca, hog_pca=hog_pca,
                                                           gist_pca=gist_pca, train=False)
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
    elif args.feature_extractor == 'color_histogram_cnn':
        print("Extracting color histogram with CNN features...")
        if done_train_test_split:
            train_features_path = os.path.join(features_dir, f'{args.dataset_name}_color-cnn_train_features.pkl')
            test_features_path = os.path.join(features_dir, f'{args.dataset_name}_color-cnn_test_features.pkl')
            cnn_model_features_path = os.path.join(features_dir, f'{args.dataset_name}_color-cnn_model.pkl')
            if done_train_test_split:
                if os.path.exists(train_features_path) and os.path.exists(test_features_path):
                    X_train = joblib.load(train_features_path)
                    X_test = joblib.load(test_features_path)
                    print("Loaded precomputed color features.")
                else:
                    input_width_and_height = 48 - 2 * 8
                    if os.path.exists(cnn_model_features_path):
                        cnn_model = CNN(input_width_and_height, input_width_and_height, num_classes=len(np.unique(y_train)))
                        cnn_model.load_state_dict(torch.load(cnn_model_features_path))
                        cnn_model.eval()
                        print("Loaded pre-trained CNN model")
                    else:
                        print("Start Training CNN")
                        reshape_X_train = cnn_preprocess(X_train)
                        reshape_X_test = cnn_preprocess(X_test)

                        cnn_model = CNN(input_width_and_height, input_width_and_height, num_classes=len(np.unique(y_train)))
                        batch_size = 64
                        epochs = 5
                        learning_rate = 0.001
                        train_loader = cnn_create_data_loaders(reshape_X_train, y_train, batch_size)
                        test_loader = cnn_create_data_loaders(reshape_X_test, y_test, batch_size)
                        criterion = nn.CrossEntropyLoss()
                        optimizer = optim.Adam(cnn_model.parameters(), lr=learning_rate)

                        # 训练模型
                        cnn_train(cnn_model, criterion, optimizer, train_loader, test_loader, epochs)

                        # 保存模型
                        torch.save(cnn_model.state_dict(), cnn_model_features_path)
                        print(f"CNN model saved to {cnn_model_features_path}")

                    color_pca = PCA(n_components=32)
                    hog_pca = PCA(n_components=64)
                    gist_pca = PCA(n_components=64)
                    cnn_pca = PCA(n_components=256)
                    print("Start Extracting CNN features")
                    X_train = color_histogram_CNN_extractor(X_train, cnn_model, color_pca=color_pca, hog_pca=hog_pca, gist_pca=gist_pca, cnn_pca=cnn_pca, train=True)
                    print('X_train:', X_train.shape)
                    X_test = color_histogram_CNN_extractor(X_test, cnn_model, color_pca=color_pca, hog_pca=hog_pca, gist_pca=gist_pca, cnn_pca=cnn_pca, train=False)
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

    elif args.feature_extractor == 'shape':
        X = color_histogram_extractor(X)
        print("Extracting shape features...")
        if done_train_test_split:
            train_features_path = os.path.join(features_dir, f'{args.dataset_name}_shape_train_features.pkl')
            test_features_path = os.path.join(features_dir, f'{args.dataset_name}_shape_test_features.pkl')
            if os.path.exists(train_features_path) and os.path.exists(test_features_path):
                X_train = joblib.load(train_features_path)
                X_test = joblib.load(test_features_path)
                print("Loaded precomputed features.")
            else:
                X_train = color_histogram_extractor(X_train)
                print('X_train:', X_train.shape)
                X_test = color_histogram_extractor(X_test)
                print('X_test:', X_test.shape)
                joblib.dump(X_train, train_features_path)
                joblib.dump(X_test, test_features_path)
    elif args.feature_extractor == 'graph':
        print("Extracting graph features...")
        if done_train_test_split:
            train_features_path = os.path.join(features_dir, f'{args.dataset_name}_graph_train_features.pkl')
            test_features_path = os.path.join(features_dir, f'{args.dataset_name}_graph_test_features.pkl')
            if done_train_test_split:
                if os.path.exists(train_features_path) and os.path.exists(test_features_path):
                    X_train = joblib.load(train_features_path)
                    X_test = joblib.load(test_features_path)
                    print("Loaded precomputed graph features.")
                else:
                    pca = PCA(n_components=512)
                    X_train = color_histogram_extractor(X_train)
                    print('X_train:', X_train.shape)
                    print('processing PCA on X_train')
                    X_train = pca.fit_transform(X_train)
                    print('X_train:', X_train.shape, 'y_train', np.array(y_train).shape)
                    X_train = graph_embedding_lda(X_train, np.array(y_train))
                    print('X_train:', X_train.shape)
                    X_test = color_histogram_extractor(X_test)
                    print('X_test:', X_test.shape)
                    print('processing PCA on X_test')
                    X_test = pca.transform(X_test)
                    print('X_test:', X_test.shape)
                    X_test = graph_embedding_lda(X_test, y_test)
                    print('X_test:', X_test.shape)
                    joblib.dump(X_train, train_features_path)
                    joblib.dump(X_test, test_features_path)
            else:
                features_path = os.path.join(features_dir, f'{args.dataset_name}_graph_features.pkl')
                if os.path.exists(features_path):
                    X = joblib.load(features_path)
                    print("Loaded precomputed graph features.")
                else:
                    X = color_histogram_extractor(X)
                    joblib.dump(X, features_path)
                    X = graph_embedding_lda(X, y)
        else:
            X = color_histogram_extractor(X)
            X = graph_embedding_lda(X, y)

    else:
        raise ValueError(f"Unsupported feature extractor: {args.feature_extractor}")

    if not done_train_test_split:
        # split X_features and y into training and testing sets
        # Use a 80-20 split and make sure to shuffle the samples.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    # Default: SVM
    clf = SVC(kernel='linear')
    # 选择并训练分类器
    if args.classifier == 'svm':
        clf = SVC(kernel='linear')
        print("Training SVM classifier...")
    elif args.classifier == 'GaussianNB':
        clf = GaussianNB()
        print("Training Gaussian Naive Bayes classifier...")
    elif args.classifier == 'knn':
        clf = KNeighborsClassifier(n_neighbors=5)
        print("Training KNN classifier with k =", 5)
    elif args.classifier == 'decision_tree':
        clf = DecisionTreeClassifier(random_state=42)
        print("Training Decision Tree classifier...")
    elif args.classifier == 'random_forest':
        clf = RandomForestClassifier(random_state=42)
        print("Training random forest classifier...")
    elif args.classifier == 'mlp':
        if args.ensemble == 'none':
            input_size = X_train.shape[1]
            hidden_size = 128
            output_size = len(np.unique(y_train))
            epochs = 8
            batch_size = 64
            learning_rate = 0.0001

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model = MLP(input_size, hidden_size, output_size).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            train_loader, test_loader = create_data_loaders(X_train, y_train, X_test, y_test, batch_size)

            train(model, criterion, optimizer, train_loader, test_loader, epochs)
            test(model, test_loader)
        elif args.ensemble == 'Bagging':
            input_size = X_train.shape[1]
            hidden_size = 128
            output_size = len(np.unique(y_train))
            epochs = 8
            batch_size = 64
            learning_rate = 0.0001

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            n_estimators = 5
            train_loader, test_loader = create_data_loaders(X_train, y_train, X_test, y_test, batch_size)
            models = []
            # Bagging for MLP
            for _ in range(n_estimators):
                model = MLP(input_size, hidden_size, output_size).to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)

                X_resampled, y_resampled = resample(X_train, y_train)
                train_loader_resampled = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(torch.tensor(X_resampled, dtype=torch.float32),
                                                   torch.tensor(y_resampled, dtype=torch.long)),
                    batch_size=batch_size, shuffle=True
                )
                train(model, criterion, optimizer, train_loader_resampled, test_loader, epochs)
                models.append(model)
            test_bagging(models, test_loader)
    else:
        raise ValueError(f"Unsupported classifier: {args.classifier}")

    # Start Training
    if args.ensemble == 'none' or args.classifier == 'mlp':
        pass
    elif args.ensemble == 'Bagging':
        if args.classifier != 'mlp':
            print("Using Bagging")
            clf = BaggingClassifier(estimator=clf, n_estimators=10, random_state=42)

    elif args.ensemble == "AdaBoost":
        print("Using AdaBoost")
        clf = AdaBoostClassifier(estimator=clf, n_estimators=50, random_state=42)
    else:
        raise ValueError(f"Unsupported ensemble learning: {args.ensemble}")
    # Not using Ensemble Learning
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

        # 统计并打印分类错误的测试集图片
        error_stats = defaultdict(int)
        for i in range(len(y_test)):
            if y_test[i] != y_pred[i]:
                print(f"Image {i}: True label = {y_test[i]}, Predicted label = {y_pred[i]}")
                error_stats[(y_test[i], y_pred[i])] += 1

        # 按降序排列并打印误分类情况
        sorted_error_stats = sorted(error_stats.items(), key=lambda item: item[1], reverse=True)
        print("\nMisclassification counts (sorted):")
        for (true_label, pred_label), count in sorted_error_stats:
            print(f"True label {true_label} was misclassified as {pred_label}: {count} times")

if __name__ == "__main__":
    main()
