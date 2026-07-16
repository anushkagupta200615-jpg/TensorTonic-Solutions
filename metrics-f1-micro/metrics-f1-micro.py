def f1_micro(y_true, y_pred) -> float:

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    
    TP = sum(yt == yp for yt, yp in zip(y_true, y_pred))

    FP = len(y_true) - TP
    FN = FP

    if TP == 0:
        return 0.0
    return float(2 * TP / (2 * TP + FP + FN))
