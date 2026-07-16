import numpy as np

def manhattan_distance(x, y):

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    return float(np.sum(np.abs(x - y)))
