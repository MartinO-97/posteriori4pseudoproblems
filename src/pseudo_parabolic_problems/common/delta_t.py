import numpy as np
from numpy import ndarray

def delta_t(v_j: ndarray, 
            v_j_s: ndarray, 
            tau: float) -> ndarray:
    
    r"""
    Computation of the divided difference \delta^1_t v^j, for some function v:[0,T] \to H^1_0(\Omega):

    delta^1_t v^j = (v^j - v^{j-1})/tau
    
   
    Paramters
    ---------
    v_j: ndarray 
        Array, associtated with v^j
    
    v_j_s: ndarray 
        Array, associtated with v^{j-1}
    
    tau: float 
        The value tau.

    Returns
    -------
    :ndarray 
        \delta^1_t v^j.
        
    """    
    
    delta = (v_j - v_j_s)/tau
    
    return(delta)