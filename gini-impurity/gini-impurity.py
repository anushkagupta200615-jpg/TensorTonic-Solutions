import numpy as np

def gini_impurity(y_left, y_right):

    def gini(y):
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()
        return 1.0 - np.sum(p ** 2)

    n_left, n_right = len(y_left), len(y_right)
    n_total = n_left + n_right
    if n_total == 0:
        return 0.0

    gini_left = gini(y_left)
    gini_right = gini(y_right)

    weighted_gini = (n_left / n_total) * gini_left + (n_right / n_total) * gini_right
    return float(weighted_gini)
