import numpy as np

def leaky_relu(x, alpha=0.1):
    x = np.asarray(x, dtype=float)
    return np.where(x >= 0, x, alpha * x)
