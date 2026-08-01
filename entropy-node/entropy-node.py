import numpy as np

def entropy_node(y):
    """
    Compute Shannon entropy of class labels in a node.

    Parameters:
        y (array-like): Class labels for samples in the node

    Returns:
        float: Entropy H(S) in bits
    """
    y = np.asarray(y)
    if y.size == 0:
        return 0.0

    # Count occurrences of each class
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()

    # Stable log formula: 0 * log2(0) = 0
    entropy_value = -np.sum(p * np.log2(p, where=(p > 0)))

    return float(entropy_value)
