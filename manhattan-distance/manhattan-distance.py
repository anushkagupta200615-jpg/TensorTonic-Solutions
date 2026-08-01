import numpy as np

def manhattan_distance(x, y) -> float:
    """
    Compute the Manhattan (L1) distance between two vectors.

    Parameters:
        x, y (array-like): equal-length vectors (lists or NumPy arrays)

    Returns:
        float: Manhattan distance
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    return float(np.sum(np.abs(x - y)))
