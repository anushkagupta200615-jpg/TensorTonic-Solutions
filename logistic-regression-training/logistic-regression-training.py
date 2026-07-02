import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    X = np.array(X, dtype=float)   # ensure NumPy array
    y = np.array(y, dtype=float)   # ensure NumPy array
    N, D = X.shape
    
    # Initialize parameters
    w = np.zeros(D)
    b = 0.0
    
    for step in range(steps):
        # Forward pass
        z = np.dot(X, w) + b
        p = _sigmoid(z)
        
        # Compute gradients
        dw = np.dot(X.T, (p - y)) / N
        db = np.mean(p - y)
        
        # Update parameters
        w -= lr * dw
        b -= lr * db
        
        # Optional: monitor loss
        if step % 100 == 0:
            loss = -np.mean(y*np.log(p+1e-9) + (1-y)*np.log(1-p+1e-9))
            print(f"Step {step}, Loss: {loss:.4f}")
    
    return w, b

# --- Example Usage ---
X = np.array([[0],[1],[2],[3]])
y = np.array([0,0,1,1])
w, b = train_logistic_regression(X, y, lr=0.1, steps=500)

print("Learned parameters:", w, b)

# Predictions
preds = _sigmoid(np.dot(X, w) + b)
print("Predictions:", preds)
print("Accuracy:", np.mean((preds >= 0.5) == y))
