import numpy as np

def apply_causal_mask(scores, mask_value=-1e9):
  
    scores = np.asarray(scores, dtype=float)
    T = scores.shape[-1]

  
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)

   
    masked_scores = np.where(mask, mask_value, scores)

    return masked_scores
