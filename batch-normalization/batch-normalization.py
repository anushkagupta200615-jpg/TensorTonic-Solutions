import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    Normalize → Scale → Shift
    """
    # Ensure inputs are NumPy arrays
    x = np.asarray(x, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    beta = np.asarray(beta, dtype=float)

    if x.ndim == 2:  # (N, D)
        mean = x.mean(axis=0)
        var = x.var(axis=0)
        x_hat = (x - mean) / np.sqrt(var + eps)
        out = gamma * x_hat + beta

    elif x.ndim == 4:  # (N, C, H, W)
        mean = x.mean(axis=(0, 2, 3), keepdims=True)
        var = x.var(axis=(0, 2, 3), keepdims=True)
        x_hat = (x - mean) / np.sqrt(var + eps)
        out = gamma.reshape(1, -1, 1, 1) * x_hat + beta.reshape(1, -1, 1, 1)

    else:
        raise ValueError("Input must be 2D (N,D) or 4D (N,C,H,W)")

    return out
