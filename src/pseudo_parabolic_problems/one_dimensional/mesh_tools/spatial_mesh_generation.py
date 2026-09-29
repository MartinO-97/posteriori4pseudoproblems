import numpy as np

from numpy import ndarray

def equidistant_mesh(a: float, 
                     b: float, 
                     N: int) -> tuple[ndarray, ndarray]:
    
    r"""
    Generation of a equidistant spatial mesh. 

    Given an interval [a,b], an equdistant mesh of N+1 points is generated, i.e. 

    a = x_0 < x_1 < ... < x_{N-1} < x_N = b,    x_i-x_{i-1} = h, i=1,...,N.

    Furthermore, the spatial step sizes are computed
    
    Args:
        a (float): Starting point of [a,b].
        b (float): End point of [a,b].
        N (int): Number of subintervals (N+1 is number of mesh points).

    Returns:
        tuple[ndarray, ndarray]:
            - Delta (ndarray): An equidistant mesh of [a,b].
            - h (ndarray): Spatial step sizes.     
    """    

    Delta = np.linspace(a, b, N+1, True)   
    h = Delta[1:] - Delta[:-1]
    
    return(Delta, h)


def _bakhvalov_shishkin_mesh():

    """
    """