import numpy as np

def dropout(x, p=0.5, rng=None):
    if rng is None:
        a=np.random.random(size=np.shape(x))
    else:
        a=rng.random(size=np.shape(x))
        keep_mask = a > p
        scale = 1.0 / (1.0 - p) if p < 1.0 else 0.0
        dropout_pattern = keep_mask.astype(float) * scale
        output = np.array(x, dtype=float) * dropout_pattern
    
    return output, dropout_pattern
        