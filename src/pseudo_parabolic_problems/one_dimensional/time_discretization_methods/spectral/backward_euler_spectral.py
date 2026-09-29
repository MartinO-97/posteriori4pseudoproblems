import numpy as np
import numpy.linalg as lg

from numpy import ndarray
from typing import Callable
from ...mesh_tools.interval_mapp import interval_mapp as int_mapp

def backward_euler_spectral(x : ndarray, 
                            w : ndarray, 
                            tau_j : float, 
                            t : float, 
                            func_f: Callable[[ndarray, float], ndarray], 
                            sol_vector_s : ndarray, 
                            matrix_L : ndarray, 
                            matrix_M : ndarray, 
                            a : float, 
                            b : float, 
                            le_n_g : ndarray) -> ndarray: 
    
    r"""
    Backward Euler method for spectral Galerkin discretization in a spatial domain [a,b].

    This function computes an approximation U^j of the solution u(t_j)
    to a (pseudo-)parabolic equation of the form

    L \partial_t u(t) + M u(t) = f(t),

    using the implicit Backward Euler time discretization:

    L (U^j - U^{j-1}) / \tau_j + M U^j = f(t_j),

    where U^{j-1} approximates u(t_{j-1}).

    Args:
        x (ndarray): Nodes of the Gauss-Lobatto quadrature rule.
        w (ndarray): Weights of the Gauss-Lobatto quadrature rule.
        tau_j (float): Time step size \tau_j.
        t (float): Current time level t_j.
        func_f (callable): Right-hand side function f(x, t).
        sol_vector_s (ndarray): Coefficient vector from the previous time step U^{j-1}.
        matrix_L (ndarray): Matrix representation of operator L.
        matrix_M (ndarray): Matrix representation of operator M.
        a (float): Left endpoint of the spatial domain.
        b (float): Right endpoint of the spatial domain.
        le_n_g (ndarray): Evaluations of integrated Legendre polynomials at quadrature points x.

    Returns:
        ndarray: Coefficients of the computed approximate solution U^j.
    """
    
    # the array [-1,1]
    ref_array = np.array([-1,1])
        
    # the coefficient matrix for the backward Euler method   
    coef_matrix = np.add(np.divide(matrix_L, tau_j), matrix_M) 
    
    # construction of RHS 
    f_j = func_f(int_mapp(x, ref_array, np.array([a, b])), t)
    
    f_v0 = np.multiply(f_j, w)
    
    result_vector = np.add(np.dot(le_n_g, f_v0), np.dot(np.divide(matrix_L, tau_j), sol_vector_s))
    
    # computing the coefficients of the approximate solution
    sol_vector = lg.solve(coef_matrix, result_vector)
        
    return(sol_vector)
