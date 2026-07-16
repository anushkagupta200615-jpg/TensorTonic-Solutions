import numpy as np

def apply_homogeneous_transform(T, points):

    points = np.atleast_2d(points)  
    
   
    ones = np.ones((points.shape[0], 1))
    points_h = np.hstack([points, ones])  
    
    
    transformed_h = (T @ points_h.T).T  
   
    transformed = transformed_h[:, :3]
    
   
    if transformed.shape[0] == 1:
        return transformed[0]
    return transformed
