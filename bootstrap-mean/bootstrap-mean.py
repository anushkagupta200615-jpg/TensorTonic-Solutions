import numpy as np

def bootstrap_mean(x, n_bootstrap=1000, ci=0.95, rng=None):
    """
    Bootstrap estimate of mean and confidence interval.

    Parameters
    ----------
    x : array-like, shape (N,)
        Input observations
    n_bootstrap : int
        Number of bootstrap samples
    ci : float
        Confidence level (e.g., 0.95 for 95%)
    rng : np.random.Generator or None
        Random number generator for reproducibility

    Returns
    -------
    boot_means : np.ndarray, shape (n_bootstrap,)
        Bootstrap sample means
    lower : float
        Lower bound of confidence interval
    upper : float
        Upper bound of confidence interval
    """
    x = np.asarray(x)
    N = len(x)
    if rng is None:
        rng = np.random.default_rng()

    # Generate bootstrap samples (indices)
    indices = rng.integers(0, N, size=(n_bootstrap, N))
    samples = x[indices]

    # Compute means for each bootstrap sample
    boot_means = samples.mean(axis=1)

    # Confidence interval bounds
    alpha = 1 - ci
    lower, upper = np.quantile(boot_means, [alpha/2, 1 - alpha/2])

    return boot_means, float(lower), float(upper)
