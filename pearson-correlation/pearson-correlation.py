import numpy as np

def pearson_correlation(X):
    """
    Compute Pearson correlation matrix from dataset X.
    Args:
        X: list[list[float]] | np.ndarray with shape (N, D)
    Returns:
        np.ndarray of shape (D, D) with correlation values,
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

    # Step 2: Covariance matrix
    cov = (X_centered.T @ X_centered) / (N - 1)

    # Step 3: Standard deviations
    sigma = np.sqrt(np.diag(cov))

    # Step 4: Normalize covariance to correlation
    denom = np.outer(sigma, sigma)
    with np.errstate(divide='ignore', invalid='ignore'):
        corr = cov / denom

    return corr
