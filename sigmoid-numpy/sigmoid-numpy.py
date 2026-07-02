import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function with safe handling.
    """
    x = np.array(x, dtype=float)   # ensure NumPy array
    return 1 / (1 + np.exp(-x))

# --- Test Cases ---
x1 = [0, 2, -2]
print("Test 1:", sigmoid(x1))  # [0.5, 0.88079708, 0.11920292]

x2 = np.array([10, -10])
print("Test 2:", sigmoid(x2))  # [0.9999546, 0.0000454]

# Using np.allclose for validation
expected = np.array([0.5, 0.88079708, 0.11920292])
print("Validation:", np.allclose(sigmoid(x1), expected))
