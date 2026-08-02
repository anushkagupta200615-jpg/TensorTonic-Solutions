import numpy as np

def global_avg_pool(x):
    """
    Compute global average pooling over spatial dimensions.

    Supports:
        (C, H, W)  -> (C,)
        (N, C, H, W) -> (N, C)

    Parameters:
        x (np.ndarray): input tensor

    Returns:
        np.ndarray: pooled output (float64)
    """
    x = np.asarray(x, dtype=float)

    if x.ndim == 3:  # (C, H, W)
        C, H, W = x.shape
        return x.reshape(C, H * W).mean(axis=1).astype(float)

    elif x.ndim == 4:  # (N, C, H, W)
        N, C, H, W = x.shape
        return x.reshape(N, C, H * W).mean(axis=2).astype(float)

    else:
        raise ValueError("Input must have shape (C,H,W) or (N,C,H,W)")
