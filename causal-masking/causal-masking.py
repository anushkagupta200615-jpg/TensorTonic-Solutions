import numpy as np

def apply_causal_mask(scores, mask_value=-1e9):
    T = scores.shape[-1]
    mask = np.triu(np.ones((T, T)), k=1).astype(bool)
    masked_scores = scores.copy().astype(float)
    masked_scores[..., mask] = mask_value
    return masked_scores
