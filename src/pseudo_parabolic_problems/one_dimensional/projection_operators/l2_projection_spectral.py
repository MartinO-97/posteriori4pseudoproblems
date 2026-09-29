import numpy as np
import numpy.linalg as lg

from numpy import ndarray
from typing import Callable
from ..mesh_tools.interval_mapp import interval_mapp as int_mapp

def l2_projection_spectral(func_u: (Callable[[ndarray], ndarray]),
                           a: float,
                           b: float,
                           mass: ndarray,
                           x: ndarray,
                           w: ndarray,
                           le_n_g: ndarray) -> ndarray:

    r"""
    Computation of the L^2 projection of a given function into the ansatz
    space V^0_h \subset H^1_0(a,b) of a spectral Galerin method:
    
        (U,v) = (u,v)   for all v \in V^0_h.

    Args:
        func_u (Callable[[ndarray], ndarray]): The function u.
        a (float): The starting point of the spatial interval [a,b].
        b (float): The end point of the spatial interval [a,b].
        mass (ndarray): The mass matrix with respect to V^0_h.
        x (ndarray): The nodes of the Gauss-Lobatto quadrature formula.
        w (ndarray): The weights of the Gauss-Lobatto quadrature formula.
        le_n_g (ndarray): The integrated Legendre polynomials N_1, ..., N_r, evaluated at x.        

    Returns:
        ndarray: The L^2 projection U of u.
    """
    
    # Evaluate u(\chi^{-1}()) in the Gauss Lobatto nodes, where \chi:[a,b] \to [-1,1] 
    # is an affine-linear transformation
    u_value = func_u(int_mapp(x, np.array([-1,1]), np.array([a,b])))    # type: ignore    
    
    return lg.solve(mass,(b-a)/2*np.dot(le_n_g, u_value * w))