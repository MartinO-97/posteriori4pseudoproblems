import numpy as np

from numpy import ndarray
from typing import Callable
from ...norms.h1_norm_function_fem import h1_norm_function_fem as h1_norm
from ...approximate_solutions import fem_solution_gauss_lobatto as fem_solution
from ....common.eta_S_1 import eta_S_1

def eta_init(sol_vector: ndarray, 
             func_u0: Callable[[ndarray], ndarray],
             func_der_u0: Callable[[ndarray], ndarray],
             C_a: float, 
             c_a: float, 
             c_c: float,
             T: float,
             psi_r_v: ndarray,
             psi_l_v: ndarray, 
             le_n_g: ndarray | None,
             le_p_g: ndarray | None,
             r: int,
             x: ndarray, w: ndarray,
             h: ndarray, 
             Delta: ndarray,
             N: int) -> float:
    
    r"""
    The component \eta_{init} of the estimator \eta_{H^1_0}.
    
    Args:
        sol_vector(ndarray): Coefficients of u^j_h.
        func_u0(Callable[[ndarray], ndarray] | None): The initial condition 'u_0'.
        func_der_u0(Callable[[ndarray], ndarray] | None): The first derivative of initial condition 'u_0'.
        C_a (float): Boundedness constant of the bilinear form a.
        c_a (float): Coercivity constant of the bilinear form a.
        c_c (float): 'Coercivity' constant of the bilinear form c.
        T (float): The final time T.
        psi_r_v (ndarray): The shape functions \psi_R, evaluated at the Gauss-Lobatto nodes.
        psi_l_v (ndarray): The shape functions \psi_L, evaluated at the Gauss-Lobatto nodes.
        le_n_g (ndarray): The integrated Legendre Polynomials N_l, l=1,...,r, evaluated at the Gauss-Lobatto nodes.
        le_p_g (ndarray): The Legendre Polynomials P_l, l=1,...,r, evaluated at the Gauss-Lobatto nodes.
        r (int): r+1 is the maximum degree of the considered integrated Legendre polynomials. 
        x (ndarray): The nodes of the GL formula.
        w (ndarray): The weights of the GL formula.
        h (ndarray): The spatial step sizes.
        Delta (ndarray): The mesh points of the spatial mesh of [a,b]. 
        N (int): Number of spatial subintervals.         
    
    Returns:
        float: The component eta_{init}.

    """    
    
    # u^0_h and the derivative of u^0_h
    func_u0_h = fem_solution.approximate_solution_intervals(sol_vector, psi_r_v, psi_l_v, le_n_g, r, N)
    func_der_u0_h = fem_solution.derivative_approximate_solution_intervals(sol_vector, le_p_g, np.shape(x)[0], r, N, h)

    value = eta_S_1(C_a, c_a, c_c, T) * h1_norm(func_u0, func_u0_h, func_der_u0, func_der_u0_h, x, w, h, Delta)

    return(value)