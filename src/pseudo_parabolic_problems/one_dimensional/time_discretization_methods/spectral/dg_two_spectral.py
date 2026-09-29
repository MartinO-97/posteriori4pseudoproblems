import numpy as np

from numpy import ndarray
from typing import Callable
from numpy.linalg import solve
from ...mesh_tools.interval_mapp import interval_mapp as int_mapp

def dg_two_spectral(x : ndarray, 
                    w : ndarray, 
                    tau_j : float, 
                    t : float, 
                    func_f: Callable[[ndarray, float], ndarray], 
                    sol_vector_s : ndarray, 
                    matrix_L : ndarray, 
                    matrix_M : ndarray, 
                    a : float, 
                    b : float, 
                    le_n_g : ndarray,
                    dim_V: int) -> ndarray: 
    
    r"""
    dG(2)-method for a spectral Galerkin discretization in a spatial domain [a,b].

    This function computes an approximation U^j of the solution u(t_j)
    to a (pseudo-)parabolic equation of the form

    L \partial_t u(t) + M u(t) = f(t),

    using the dG(2) method.

    Args:
        x (ndarray): Nodes of the Gauss-Lobatto quadrature rule.
        w (ndarray): Weights of the Gauss-Lobatto quadrature rule.
        tau_j (float): Time step size \tau_j.
        t (float): Current time level t_j.
        func_f (callable): Right-hand side function f(x, t_j).
        sol_vector_s (ndarray): Coefficient vector from the previous time step U^{j-1}.
        matrix_L (ndarray): Matrix representation of operator L.
        matrix_M (ndarray): Matrix representation of operator M.
        a (float): Left endpoint of the spatial domain.
        b (float): Right endpoint of the spatial domain.
        le_n_g (ndarray): Evaluations of integrated Legendre polynomials at quadrature points x.
        dim_V (int): Dimension of the ansatz space.

    Returns:
        ndarray: Coefficients of the computed approximate solution U^j.
    """
    
    # The reference interval [-1,1]
    ref_array = np.array([-1,1])
    

    # --------------------------------------------------------------------------------------------------------------
    # ASSOCIATED BUTCHER TABLEAU
    # --------------------------------------------------------------------------------------------------------------

    r"""The vectors of the Butcher tableau:

        c | A
        ------
          | b^T

    The exact values can be found in [1]. Note that the last row of A coincides with b^T.
    """

    # The vector c 
    c = np.array([(4-np.sqrt(6))/10, (4+np.sqrt(6))/10, 1])
    
    # The matrix A 
    A = np.array([[(88-7*np.sqrt(6))/360, (296-169*np.sqrt(6))/1800, (-2+3*np.sqrt(6))/225], 
                  [(296+169*np.sqrt(6))/1800, (88+7*np.sqrt(6))/360, (-2-3*np.sqrt(6))/225],
                  [(16-np.sqrt(6))/36, (16+np.sqrt(6))/36, 1/9]])


    # --------------------------------------------------------------------------------------------------------------
    # COMPUTATION OF THE APPROXIMATE SOLUTION U^j
    # --------------------------------------------------------------------------------------------------------------
        
    # Assemble the coefficient matrix 
    coef_matrix_above = np.hstack((matrix_L / tau_j + A[0,0] * matrix_M, A[0,1] * matrix_M, A[0,2] * matrix_M))
    coef_matrix_middle = np.hstack((A[1,0] * matrix_M, matrix_L/tau_j + A[1,1] * matrix_M, A[1,2] * matrix_M))
    coef_matrix_under = np.hstack((A[2,0] * matrix_M, A[2,1] * matrix_M, matrix_L/tau_j + A[2,2] * matrix_M))
    
    coef_matrix = np.vstack((coef_matrix_above,coef_matrix_middle))
    coef_matrix = np.vstack((coef_matrix,coef_matrix_under))
    
    # Assemble the right hand side            
    f_v0 = func_f(int_mapp(x, ref_array, np.array([a, b])), t + (c[0]-1)*tau_j)
    f_v1 = func_f(int_mapp(x, ref_array, np.array([a, b])), t + (c[1]-1)*tau_j)
    f_v2 = func_f(int_mapp(x, ref_array, np.array([a, b])), t + (c[2]-1)*tau_j)

    f_v0 = f_v0 * w
    f_v1 = f_v1 * w
    f_v2 = f_v2 * w
    
    b0 = (b-a)/2*np.dot(le_n_g, A[0,0] * f_v0 + A[0,1] * f_v1 + A[0,2] * f_v2) + np.dot(matrix_L / tau_j, sol_vector_s)
    b1 = (b-a)/2*np.dot(le_n_g, A[1,0] * f_v0 + A[1,1] * f_v1 + A[1,2] * f_v2) + np.dot(matrix_L / tau_j, sol_vector_s)
    b2 = (b-a)/2*np.dot(le_n_g, A[2,0] * f_v0 + A[2,1] * f_v1 + A[2,2] * f_v2) + np.dot(matrix_L / tau_j, sol_vector_s)
    
    result_vector = np.concatenate((b0, b1, b2)) 
    
    return(solve(coef_matrix, result_vector)[2*dim_V:])


# --------------------------------------------------------------------------------------------------------------
# BIBLIOGRAPHY
# --------------------------------------------------------------------------------------------------------------

# [1] K. Strehmel, R. Weiner, and H. Podhaisky; 
#     Numerik gewöhnlicher Differentialgleichungen: nichtsteife, steife und differential-algebraische Gleichungen; 
#     Studium, Springer Spektrum, Wiesbaden, 2.; 2012.