import numpy as np

def cross_entropy_loss(y_true, y_pred) -> float:
    """
    Compute average cross-entropy loss for multi-class classification.

    Parameters:
        y_true (array-like): shape (N,) - true class labels (integers)
        y_pred (array-like): shape (N,C) - predicted probabilities per class

    Returns:
        float: average cross-entropy loss
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=float)

    # Check dimensions
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same number of samples")

    if y_true.size == 0:
        return 0.0

    # Gather predicted probabilities for the correct classes
    correct_probs = y_pred[np.arange(len(y_true)), y_true]

    # Compute cross-entropy loss
    loss = -np.mean(np.log(correct_probs))

    return float(loss)
