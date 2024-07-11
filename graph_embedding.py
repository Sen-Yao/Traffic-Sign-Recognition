import numpy as np


def graph_embedding_lda(X, y):
    """
    Perform LDA with graph embedding on the given dataset.

    Parameters:
    X (numpy.ndarray): 2D array of shape (n_samples, n_features) representing the features.
    y (numpy.ndarray): 1D array of shape (n_samples,) representing the class labels.

    Returns:
    numpy.ndarray: Transformed feature matrix.
    """
    # Ensure X and y are numpy arrays
    X = np.asarray(X)
    y = np.asarray(y)

    # Center the data
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean

    # Compute the weight matrix W
    n_samples = X.shape[0]
    W = np.zeros((n_samples, n_samples))
    classes = np.unique(y)
    for c in classes:
        class_indices = np.where(y == c)[0]
        n_class_samples = class_indices.size
        for i in class_indices:
            for j in class_indices:
                W[i, j] = 1 / n_class_samples

    # Perform SVD
    U, R, Vt = np.linalg.svd(X_centered, full_matrices=False)
    V = Vt.T  # V is the right singular vectors
    print(U.shape, W.shape, V.shape, R.shape)
    # Convert the problem to a smaller dimensional space
    R_inv = np.linalg.inv(np.diag(R))
    S = U.T @ W @ U
    R_inv_S = R_inv @ S @ R_inv

    # Solve the eigenproblem in the smaller space
    eigvals, eigvecs = np.linalg.eig(R_inv_S)

    # Sort eigenvectors by eigenvalues in descending order
    sorted_indices = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, sorted_indices]

    # Select the top c-1 eigenvectors
    c = len(classes)
    eigvecs = eigvecs[:, :c - 1]

    # Transform the data
    transformed_data = X_centered @ V @ eigvecs

    return transformed_data

# Example usage:
# X = np.array([[...], [...], ...])
# y = np.array([...])
# transformed_X = graph_embedding_lda(X, y)
