import numpy as np

def _entropy(y):
    """
    Helper: Compute Shannon entropy (base 2) for labels y.
    """
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0

def information_gain(y, split_mask):
    """
    Compute Information Gain of a binary split on labels y.
    Uses the _entropy() helper above.
    """
    y = np.asarray(y)
    split_mask = np.asarray(split_mask, dtype=bool)

    # Parent entropy
    H_parent = _entropy(y)

    # Left and right subsets
    y_left = y[split_mask]
    y_right = y[~split_mask]

    n_left, n_right = len(y_left), len(y_right)
    N = n_left + n_right

    # Edge case: if one side is empty, IG = 0
    if n_left == 0 or n_right == 0:
        return 0.0

    # Weighted child entropy
    H_children = (n_left / N) * _entropy(y_left) + (n_right / N) * _entropy(y_right)

    # Information Gain
    return float(H_parent - H_children)
