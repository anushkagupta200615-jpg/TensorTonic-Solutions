import numpy as np

def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.

    Parameters:
        y_true (list/array): true class labels
        y_pred (list/array): predicted class labels

    Returns:
        float: micro F1 score in [0,1]
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")

    n = y_true.shape[0]
    if n == 0:
        return 0.0

    # True positives: correct predictions
    TP = np.sum(y_true == y_pred)

    # In single-label multi-class, every misclassification contributes one FP and one FN
    FP = n - TP
    FN = FP

    # Apply micro-F1 formula
    if TP == 0:
        return 0.0
    return float(2 * TP / (2 * TP + FP + FN))
