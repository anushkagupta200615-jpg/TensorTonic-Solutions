import numpy as np

def auc(fpr, tpr):
   
    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)

    if fpr.shape != tpr.shape:
        raise ValueError("fpr and tpr must have the same shape")

    
    dx = np.diff(fpr)
    avg_y = (tpr[:-1] + tpr[1:]) / 2.0
    area = np.sum(dx * avg_y)

    return float(area)
