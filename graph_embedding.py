import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.linalg import svd


def graph_embedding(feature_matrix, labels):
    """
    Perform graph embedding on the feature matrix.

    Parameters:
    - feature_matrix: List[List[float]], the input feature matrix where each row is a data sample.
    - labels: List[int], the class labels for each data sample.

    Returns:
    - new_feature_matrix: np.ndarray, the transformed feature matrix.
    """
    # Convert the input feature matrix to a NumPy array
    X = np.array(feature_matrix)
    n_samples, n_features = X.shape

    # Center the data
    X_centered = StandardScaler(with_std=False).fit_transform(X)

    # Create the weight matrix W
    W = np.zeros((n_samples, n_samples))
    unique_labels = np.unique(labels)

    for label in unique_labels:
        indices = np.where(labels == label)[0]
        M_l = len(indices)
        for i in indices:
            for j in indices:
                if i != j:
                    W[i, j] = 1 / M_l

    # Solve the eigenproblem
    X_centered_T = X_centered.T
    S_b = X_centered_T @ W @ X_centered
    S_w = X_centered_T @ X_centered

    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.pinv(S_w) @ S_b)

    # Sort eigenvectors by eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Select the top c-1 eigenvectors (c is the number of unique classes)
    c = len(unique_labels)
    selected_eigenvectors = eigenvectors[:, :c - 1]

    # Transform the data
    new_feature_matrix = X_centered @ selected_eigenvectors

    return new_feature_matrix