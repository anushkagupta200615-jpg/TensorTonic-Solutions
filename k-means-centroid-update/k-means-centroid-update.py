import numpy as np

def k_means_centroid_update(points, assignments, k):
    """
    Compute new centroids as the mean of assigned points.

    Args:
        points: list[list[float]] or np.ndarray of shape (n_points, d)
        assignments: list[int] or np.ndarray of shape (n_points,)
        k: int - number of clusters

    Returns:
        list[list[float]] of shape (k, d) with new centroid positions
    """
    points = np.asarray(points, dtype=float)        # (n, d)
    assignments = np.asarray(assignments, dtype=int)  # (n,)
    d = points.shape[1]

    new_centroids = []
    for cluster_id in range(k):
        cluster_points = points[assignments == cluster_id]
        if len(cluster_points) > 0:
            centroid = cluster_points.mean(axis=0)
        else:
            centroid = np.zeros(d)  # empty cluster → zero vector
        new_centroids.append(centroid.tolist())

    return new_centroids
