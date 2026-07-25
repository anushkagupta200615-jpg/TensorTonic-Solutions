import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean") -> float:
    
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    y = np.asarray(y)

   
    if a.shape != b.shape:
        raise ValueError("Embeddings a and b must have the same shape")
    if y.ndim != 1 or y.shape[0] != a.shape[0]:
        raise ValueError("y must be 1D with length equal to number of samples")
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("y must contain only 0 or 1")

   
    d = np.linalg.norm(a - b, axis=1)

    
    pos_loss = y * (d ** 2)
    neg_loss = (1 - y) * (np.maximum(0, margin - d) ** 2)
    losses = pos_loss + neg_loss

   
    if reduction == "mean":
        return float(losses.mean())
    elif reduction == "sum":
        return float(losses.sum())
    else:
        raise ValueError("reduction must be 'mean' or 'sum'")
