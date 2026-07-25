import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    
    try:
        X = np.asarray(matrix, dtype=float)
        if X.ndim != 2:
            return None

        if norm_type == 'l1':
            norms = np.sum(np.abs(X), axis=axis, keepdims=True)
        elif norm_type == 'l2':
            norms = np.sqrt(np.sum(X**2, axis=axis, keepdims=True))
        elif norm_type == 'max':
            norms = np.max(np.abs(X), axis=axis, keepdims=True)
        else:
            return None

        
        norms = np.where(norms == 0, 1, norms)

        return X / norms

    except Exception:
        return None
