import numpy as np

def matrix_trace(A):
   
    A = np.asarray(A)
    
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Input must be a square matrix")

    n = A.shape[0]
   
    diag_indices = np.arange(n)
    return A[diag_indices, diag_indices].sum()
