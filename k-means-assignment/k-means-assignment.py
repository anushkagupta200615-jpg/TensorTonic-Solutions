import numpy as np

def k_means_assignment(points, centroids):
    """
    Assign each point to the nearest centroid using squared Euclidean distance.

    Args:
        points: list[list[float]] or np.ndarray of shape (n_points, d)
        centroids: list[list[float]] or np.ndarray of shape (n_centroids, d)

    Returns:
        list[int] of length n_points with cluster indices
    """
    points = np.asarray(points, dtype=float)      # (n, d)
    centroids = np.asarray(centroids, dtype=float)  # (k, d)

    # Compute squared distances between each point and each centroid
    diff = points[:, None, :] - centroids[None, :, :]
    sq_dist = np.sum(diff**2, axis=-1)  # shape (n_points, n_centroids)

    # Assign each point to the nearest centroid
    assignments = np.argmin(sq_dist, axis=1)

    return assignments.tolist()
