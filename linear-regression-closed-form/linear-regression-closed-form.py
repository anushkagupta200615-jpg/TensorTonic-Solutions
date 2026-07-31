import numpy as np

def linear_regression_closed_form(X, y):
    """
    Compute the optimal weight vector using the normal equation:
        w = (X^T X)^(-1) X^T y

    Args:
        X: list[list[float]] or np.ndarray of shape (n, d)
        y: list[float] or np.ndarray of shape (n,)

    Returns:
        np.ndarray of shape (d,) with regression weights
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    # Compute normal equation
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    Xt_y = X.T @ y
    w = XtX_inv @ Xt_y

    return w
