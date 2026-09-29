import numpy as np

from numpy import ndarray
from ..basis_functions.p1_shape_functions import *

def p1_shape_functions_evaluation_1d(x: ndarray) -> tuple[ndarray, ndarray]:
    
    r"""
    Evaluation of the shape functions \psi_L and \psi_R at the nodes x of 
    the Gauss-Lobatto quadrature formula. 

    Parameters
    ----------
    x: ndarray 
        Nodes of the Gauss-Lobatto formula.

    Returns
    -------
    psi_l_v: ndarray
        \psi_L evaluated at x.
    
    psi_r_v: ndarray 
        \psi_R evaluated at x.         
    """ 

    # Evaluation
    psi_l_v = psi_L(x)
    psi_r_v = psi_R(x) 
                    
    return(psi_l_v, psi_r_v)    