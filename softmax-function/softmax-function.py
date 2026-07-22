import numpy as np

def softmax(x):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        
        x_max = np.max(x)
        e = np.exp(x - x_max)
        return e / np.sum(e)
    elif x.ndim == 2:
       
        x_max = np.max(x, axis=1, keepdims=True)
        e = np.exp(x - x_max)
        return e / np.sum(e, axis=1, keepdims=True)
    else:
        raise ValueError("Input must be 1D or 2D array")
