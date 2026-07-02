import numpy as np

def sigmoid(x):
    
    x = np.array(x, dtype=float)  
    return 1 / (1 + np.exp(-x))


x1 = [0, 2, -2]
print("Test 1:", sigmoid(x1))  

x2 = np.array([10, -10])
print("Test 2:", sigmoid(x2)) 


expected = np.array([0.5, 0.88079708, 0.11920292])
print("Validation:", np.allclose(sigmoid(x1), expected))
