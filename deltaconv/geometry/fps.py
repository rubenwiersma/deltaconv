import numpy as np
import warnings
from deltaconv_bindings import geodesicFPS

def geodesic_fps(points, n_samples):
    ## Validate input
    if n_samples > points.shape[0]:
        warnings.warn("Number of samples is larger than number of points.")
    if type(points) is not np.ndarray:
        raise ValueError("`points` should be a numpy array")
    if (len(points.shape) != 2) or (points.shape[1] != 3):
        raise ValueError("`points` should have shape (V,3), shape is " + str(points.shape))
    
    ## Call the main algorithm from the bindings
    sample_id = geodesicFPS(points, n_samples)

    ## Return the result
    return sample_id.squeeze()