import numpy as np

def zscore_standardize(X, axis=0, eps=1e-12):
    """
    Standardize features to zero mean and unit variance.
    If 2D and axis=0 (default), standardize per column.
    Returns np.ndarray (float).
    """
    X = np.asarray(X, dtype=float)
    mean = np.mean(X, axis=axis, keepdims=True)
    std = np.std(X, axis=axis, keepdims=True)
    return (X - mean) / (std + eps)
