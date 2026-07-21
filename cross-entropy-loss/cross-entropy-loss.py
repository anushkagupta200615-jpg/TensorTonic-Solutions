import numpy as np

def cross_entropy_loss(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same number of samples")
    correct_probs = y_pred[np.arange(len(y_true)), y_true]
    loss = -np.mean(np.log(correct_probs))

    return float(loss)