import numpy as np

from numpy import ndarray
from ..basis_functions.legendre_polynomials import *

def legendre_evaluation(x: ndarray, 
                        r: int) -> tuple[ndarray, ndarray, ndarray]:
    
    r"""
    Evaluation of the integrated Legendre, the Legendre and the derivatives of the Legendre 
    polynomials at the nodes x of the Gauss-Lobatto quadrature formula. 

    Parameters
    ----------
    x: ndarray 
        Nodes of the Gauss-Lobatto formula.
    
    r: int 
        The integrated Legendre polynomials N_1,...,N_r, the Legendre polynomials P_1,...,P_r 
        and the derivatives of the Legendre polynomials P'_1,...,P'_r are considered.

    Returns
    -------
    Le_N_G: ndarray 
        N_1,...,N_r evaluated at x.
    
    Le_P_G: ndarray 
        P_1,...,P_r evaluated at x.
    
    Le_P_G_der: ndarray 
        P'_1,...,P'_r evaluated at x.             
    """ 

    # Initializations of the arrays, where the evaluations are stored
    Le_N_G = np.zeros((r,np.shape(x)[0]))
    Le_P_G = np.zeros((r,np.shape(x)[0]))
    Le_P_G_der = np.zeros((r,np.shape(x)[0]))

    # Evaluation
    for l in range(0,r):
            Le_N_G[l,1:-1] = integrated_legendre(l+1,x[1:-1])
            Le_P_G[l,:] = legendre(l+1,x)
            Le_P_G_der[l,:] = derivative_legendre(l+1,x)
                    
    return(Le_N_G, Le_P_G, Le_P_G_der)    