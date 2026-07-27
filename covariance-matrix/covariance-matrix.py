import numpy as np

def covariance_matrix(X):
    """
    Compute the sample covariance matrix from dataset X.
    Args:
        X: list[list[float]] | np.ndarray with shape (N, D)
    Returns:
        np.ndarray of shape (D, D) with covariance values,
        or None if invalid input (N < 2 or not 2D).
    """
    # Convert to NumPy array
    X = np.asarray(X, dtype=float)

    # Validate shape
    if X.ndim != 2:
        return None
    N, D = X.shape
    if N < 2:
        return None

    # Step 1: Center the data
    mu = X.mean(axis=0)
    X_centered = X - mu

    # Step 2: Compute covariance matrix
    cov = (X_centered.T @ X_centered) / (N - 1)

    return cov
