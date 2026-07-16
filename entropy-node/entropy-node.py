import numpy as np

def entropy_node(y):
  
    y = np.asarray(y)
    if y.size == 0:
        return 0.0

   
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()

    entropy_value = -np.sum(p * np.log2(p, where=(p > 0)))

    return float(entropy_value)
