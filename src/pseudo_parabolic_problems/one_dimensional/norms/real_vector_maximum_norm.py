import numpy as np

from numpy import ndarray

def real_vector_maximum_norm(x: ndarray, 
                             y: ndarray) -> float:

    r"""Computes the maximum norm error between x,y \in \RR^n:

        ||x-y||_{\infty} = \max_{i=1,...,n} |x_i-y_i|
        
    Args:
        x (ndarray): Array x.
        y (ndarray): Array y.

    Returns:
        float: The maximum norm error between x and y.    
    """ 

    return(np.linalg.norm(x-y, ord = np.inf))   # type:ignore