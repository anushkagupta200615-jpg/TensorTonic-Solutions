import numpy as np

def silhouette_score(X, labels):
    """
    Compute the mean Silhouette Score for given points and cluster labels.
    X: np.ndarray of shape (n_samples, n_features)
    labels: np.ndarray of shape (n_samples,)
    Returns: float
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    n = len(X)

    
    diff = X[:, None, :] - X[None, :, :]
    D = np.sqrt(np.sum(diff**2, axis=-1))  # (n, n)

    s = np.zeros(n)
    for i in range(n):
        same = labels == labels[i]
        other = labels != labels[i]

        if np.sum(same) > 1:
            a_i = np.sum(D[i, same]) / (np.sum(same) - 1)
        else:
            a_i = 0.0

       
        b_i = np.inf
        for c in np.unique(labels[other]):
            mask = labels == c
            b_i = min(b_i, np.mean(D[i, mask]))

        s[i] = (b_i - a_i) / max(a_i, b_i)

    return float(np.mean(s))
