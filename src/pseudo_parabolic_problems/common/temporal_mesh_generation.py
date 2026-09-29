import numpy as np
from numpy import ndarray

def equidistant_mesh(T: float, 
                     M: int) -> tuple[ndarray, ndarray]:
    
    r"""
    Generation of a equidistant temporal mesh. 

    Given a time T, an equdistant mesh of M+1 points of [0,T] is generated, i.e. 

    0 = t_0 < t_1 < ... < t_{M-1} < t_M = T,    t_j-t_{j-1} = \tau, j=1,...,M.

    Furthermore, the spatial step sizes are computed.
    
    Parameters
    ---------
    T: float 
        Final time T.
    
    M: int 
        Number of subintervals (M+1 is number of mesh points).

    Returns
    -------
    omega_t: ndarray 
        An equidistant mesh of [0,T].
    
    Tau: ndarray 
        Temporal step sizes.     
    """    

    omega_t = np.linspace(0, T, M+1, True)   
    Tau = omega_t[1:] - omega_t[:-1]
    
    return(omega_t, Tau)