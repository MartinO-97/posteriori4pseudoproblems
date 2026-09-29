import numpy as np
from typing import Callable
from ...mesh_tools.interval_mapp import interval_mapp as int_mapp

def dg_three_spectral(x : np.ndarray, 
                      w : np.ndarray, 
                      tau_j : float, 
                      t : float, 
                      func_f: Callable[[np.ndarray, float], np.ndarray], 
                      sol_vector_s : np.ndarray, 
                      matrix_L : np.ndarray, 
                      matrix_M : np.ndarray, 
                      a : float, 
                      b : float, 
                      le_n_g : np.ndarray,
                      dim_V: int) -> np.ndarray: 
    
    r"""
    dG(3)-method for a spectral Galerkin discretization in a spatial domain [a,b].

    This function computes an approximation U^j of the solution u(t_j)
    to a (pseudo-)parabolic equation of the form

    L \partial_t u(t) + M u(t) = f(t),

    using the dG(3) method.

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
        dim_V (int): Dimension of the ansatz space.

    Returns:
        np.ndarray: Coefficients of the computed approximate solution U^j.
    """

    # the vector c
    cB = np.array([0.0885879595127039474, 0.40946686444073471088, 0.7876594617608470559, 1.0])
        
    # the matrices C_s, V_s and A from Strehmel/Weiner/Podhaisky 8.1.3
    C_s = np.transpose(np.reshape(np.concatenate([cB, cB**2/2, cB**3/3, cB**4/4]), (4,4)))
    V_s = np.transpose(np.reshape(np.concatenate([np.ones_like(cB), cB, cB**2, cB**3]), (4,4)))
    A = np.matmul(C_s, np.linalg.inv(V_s))
        
    
    ''' construction of the coefficient matrix ''' 
    coef_matrix_oben = np.hstack((matrix_L/tau_j+A[0,0]*matrix_M, A[0,1]*matrix_M, A[0,2]*matrix_M, A[0,3]*matrix_M))
    coef_matrix_oben_h = np.hstack((A[1,0]*matrix_M, matrix_L/tau_j+A[1,1]*matrix_M, A[1,2]*matrix_M, A[1,3]*matrix_M))
    coef_matrix_unten_h = np.hstack((A[2,0]*matrix_M, A[2,1]*matrix_M, matrix_L/tau_j+A[2,2]*matrix_M, A[2,3]*matrix_M))
    coef_matrix_unten = np.hstack((A[3,0]*matrix_M, A[3,1]*matrix_M, A[3,2]*matrix_M, matrix_L/tau_j+A[3,3]*matrix_M))
    
    coef_matrix = np.vstack((coef_matrix_oben,coef_matrix_oben_h))
    coef_matrix = np.vstack((coef_matrix,coef_matrix_unten_h))
    coef_matrix = np.vstack((coef_matrix,coef_matrix_unten))
    
    ''' construction of the right hand side '''         
    f_v0 = func_f(int_mapp(x, np.array([-1,1]), np.array([a, b])), t+(cB[0]-1)*tau_j)
    f_v1 = func_f(int_mapp(x, np.array([-1,1]), np.array([a, b])), t+(cB[1]-1)*tau_j)
    f_v2 = func_f(int_mapp(x, np.array([-1,1]), np.array([a, b])), t+(cB[2]-1)*tau_j)
    f_v3 = func_f(int_mapp(x, np.array([-1,1]), np.array([a, b])), t+(cB[3]-1)*tau_j)

    f_v0 = f_v0 * w
    f_v1 = f_v1 * w
    f_v2 = f_v2 * w
    f_v3 = f_v3 * w
    
    b0 = (b-a)/2*np.dot(le_n_g, A[0,0]*f_v0 + A[0,1]*f_v1 + A[0,2]*f_v2 + A[0,3]*f_v3) + np.dot(matrix_L, sol_vector_s)/tau_j
    b1 = (b-a)/2*np.dot(le_n_g, A[1,0]*f_v0 + A[1,1]*f_v1 + A[1,2]*f_v2 + A[1,3]*f_v3) + np.dot(matrix_L, sol_vector_s)/tau_j
    b2 = (b-a)/2*np.dot(le_n_g, A[2,0]*f_v0 + A[2,1]*f_v1 + A[2,2]*f_v2 + A[2,3]*f_v3) + np.dot(matrix_L, sol_vector_s)/tau_j
    b3 = (b-a)/2*np.dot(le_n_g, A[3,0]*f_v0 + A[3,1]*f_v1 + A[3,2]*f_v2 + A[3,3]*f_v3) + np.dot(matrix_L, sol_vector_s)/tau_j
    
    result_vector = np.concatenate((b0, b1, b2, b3))
    
    sol_vector = np.linalg.solve(coef_matrix, result_vector)[3*dim_V:]   
    
    return(sol_vector)
