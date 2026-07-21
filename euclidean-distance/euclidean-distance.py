import numpy as np

def euclidean_distance(x, y):
    x=np.asarray(x, dtype=float)
    y=np.asarray(y, dtype=float)
    if x.shape!=y.shape:
        raise ValueError("x and y must have the same shape")
    return float(np.sqrt(np.sum((x-y)**2)))