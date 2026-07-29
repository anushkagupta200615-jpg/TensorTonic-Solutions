import numpy as np

def knn_distance(X_train, X_test, k):
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)

    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Train and test must have same feature dimension")

    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    dists = np.sqrt(((X_test[:, None, :] - X_train[None, :, :]) ** 2).sum(axis=2))
    sorted_idx = np.argsort(dists, axis=1)

    if k > n_train:
        pad = -1 * np.ones((n_test, k - n_train), dtype=int)
        return np.hstack([sorted_idx[:, :n_train], pad])
    else:
        return sorted_idx[:, :k]
