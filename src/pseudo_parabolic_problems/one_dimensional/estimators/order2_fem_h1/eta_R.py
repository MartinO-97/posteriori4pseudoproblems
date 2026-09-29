import numpy as np

from numpy import ndarray
from typing import Callable
from ..pseudo_elliptic_h1_1d import pseudo_elliptic_h1_1d as elliptic_est
from ...approximate_solutions import fem_solution_gauss_lobatto as fem_sol 
from ...norms.h1_norm_function_fem import h1_norm_function_fem as h1_norm
from ....common.delta_t import delta_t
from ....common.eta_S_1 import eta_S_1

def eta_R(sol_vector: ndarray,
          sol_vector_s: ndarray,  
          psi_vector: ndarray,
          psi_vector_s: ndarray,
          func_a: Callable[[ndarray], ndarray],
          func_c: Callable[[ndarray], ndarray],
          func_f: Callable[[ndarray, float], ndarray],
          C_a: float, 
          c_a: float, 
          c_c: float,
          t_j: float,
          t_j_s: float,
          tau_j: float,
          T: float,
          psi_r_v: ndarray,
          psi_l_v: ndarray, 
          le_n_g: ndarray,
          le_p_g: ndarray,
          le_p_g_x: ndarray,
          r: int,
          x: ndarray, 
          w: ndarray,
          h: ndarray, 
          Delta: ndarray,
          N: int, 
          elliptic_estimator: bool,
          sol_vector_elliptic: ndarray,
          sol_vector_elliptic_s: ndarray,
          r_elliptic: int, 
          le_n_g_elliptic: ndarray,
          le_p_g_elliptic: ndarray) -> float:
    
    r"""
    The component \eta_R of the estimator \eta_{H^1_0} at time t_j:

    \eta^j_{\Psi} = \sigma_j \tau_j ||\psi^j_h + \delta_t u^j_h||_{1,\Omega}
         
    Args:
        sol_vector(ndarray): Coefficients of u^j_h.
        sol_vector(ndarray): Coefficients of u^{j-1}_h.
        psi_vector(ndarray): Coefficients of \psi^j_h.
        psi_vector_s(ndarray): Coefficients of \psi^{j-1}_h.
        func_f (Callable[[ndarray, float], ndarray]): The function f(\cdot, \cdot): \Omega \times [0,T].
        func_a (Callable[[ndarray], ndarray]): The function a.
        func_c (Callable[[ndarray], ndarray]): The function c.
        C_a (float): Boundedness constant of the bilinear form a.
        c_a (float): Coercivity constant of the bilinear form a.
        c_c (float): 'Coercivity' constant of the bilinear form c.
        t_j (float): The time t_j.
        t_j_s (float): The time t_{j-1}.
        tau_j (float): The time step size t_j-t_{j-1}.
        T (float): The final time T.
        psi_r_v (ndarray): The shape functions \psi_R, evaluated at the Gauss-Lobatto nodes.
        psi_l_v (ndarray): The shape functions \psi_L, evaluated at the Gauss-Lobatto nodes.
        le_n_g (ndarray): The integrated Legendre Polynomials N_l, l=1,...,r, evaluated at the Gauss-Lobatto nodes.
        le_p_g(ndarray): The Legendre Polynomials P_l, l=1,...,r, evaluated at the Gauss-Lobatto nodes.
        le_p_g_x (ndarray): The first derivative of the Legendre Polynomials P'_l, l=1,...,r, evaluated at the Gauss-Lobatto nodes.
        r (int): r+1 is the maximum degree of the considered integrated Legendre polynomials. 
        x (ndarray): The nodes of the GL formula.
        w (ndarray): The weights of the GL formula.
        h (ndarray): The spatial step sizes.
        Delta (ndarray): The mesh points of the spatial mesh of [a,b]. 
        N (int): Number of spatial subintervals.
        elliptic_estimator (bool): Shall an elliptic estimator be utilized or shall the following approximation be used:
                                    || u(t_j) - u^j_h||_{1,\Omega} \leq || v^j_h - u^j_h||_{1,\Omega} + || u(t_j) - v^j_h||
                                   where v^j_h is an approximation of higher order.
        sol_vector_elliptic (ndarray): Coefficients of v^j_h.
        sol_vector_elliptic_s (ndarray): Coefficients of v^{j-1}_h.
        r_elliptic (int): r_elliptic+1 is the maximum degree of the considered integrated Legendre polynomials for the approximation of higher order.     
        le_n_g_elliptic (ndarray): The integrated Legendre Polynomials N_l, l=1,...,r_elliptic, evaluated at the Gauss-Lobatto nodes.
        le_p_g_elliptic (ndarray): The Legendre Polynomials P_l, l=1,...,r_elliptic, evaluated at the Gauss-Lobatto nodes.  
    
    Returns:
        float: The component eta_{init}.

    """    
    
    # The value \sigma_j
    sigma_j = np.max(np.array([eta_S_1(C_a, c_a, c_c, T-t_j), eta_S_1(C_a, c_a, c_c, T-t_j_s)]))

    if elliptic_estimator:
        
        # The functions (u^j_h + u^{j-1}_h)/2, (\psi^j_h + \psi^{j-1}_h)/2 and (f^j + f^{j-1})/2
        sol_vector_h = (sol_vector + sol_vector_s)/2
        psi_vector_h = (psi_vector + psi_vector_s)/2
        func_f_h = lambda y: (func_f(y, t_j) + func_f(y, t_j_s))/2

        # The functions \delta_t u^j_h, \delta_t \psi^j_h and \delta_t f^j
        delta_sol_vector = delta_t(sol_vector, sol_vector_s, tau_j)
        delta_psi_vector = delta_t(psi_vector, psi_vector_s, tau_j)
        delta_func_f = lambda y: delta_t(func_f(y, t_j), func_f(y, t_j_s), tau_j)

        # The value \eta^j_{ell, 1/2}
        eta_ell_h = elliptic_est(sol_vector_h, sol_vector_h, psi_vector_h, func_a, func_c, func_f_h, x, w, h, Delta, c_a, psi_r_v, psi_l_v, le_n_g, le_p_g_x, r, N)
        
        # The value \eta^j_{ell, \delta_t}
        eta_ell_delta = elliptic_est(delta_sol_vector, delta_sol_vector, delta_psi_vector, func_a, func_c, delta_func_f, x, w, h, Delta, c_a, psi_r_v, psi_l_v, le_n_g, le_p_g_x, r, N)

    else:

        # The functions (u^j_h + u^{j-1}_h)/2 and (v^j_h + v^{j-1}_h)/2 
        sol_vector_h = (sol_vector + sol_vector_s)/2
        sol_vector_elliptic_h = (sol_vector_elliptic + sol_vector_elliptic_s)/2

        # The functions \delta_t u^j_h and \delta_t v^j_h 
        delta_sol_vector = delta_t(sol_vector, sol_vector_s, tau_j)
        delta_sol_vector_elliptic = delta_t(sol_vector_elliptic, sol_vector_elliptic_s, tau_j)

        # Evaluate (u^j_h + u^{j-1}_h)/2 and its derivative
        func_fem_sol_h = fem_sol.approximate_solution_intervals(sol_vector_h, psi_r_v, psi_l_v, le_n_g, r, N)
        func_der_fem_sol_h = fem_sol.derivative_approximate_solution_intervals(sol_vector_h, le_p_g, np.shape(x)[0], r, N, h)

        # Evaluate (v^j_h + v^{j-1}_h)/2 and its derivative
        func_fem_sol_elliptic_h = fem_sol.approximate_solution_intervals(sol_vector_elliptic_h, psi_r_v, psi_l_v, le_n_g_elliptic, r_elliptic, N)
        func_der_fem_sol_elliptic_h = fem_sol.derivative_approximate_solution_intervals(sol_vector_elliptic_h, le_p_g_elliptic, np.shape(x)[0], r_elliptic, N, h)

        # Evaluate \delta_t u^j_h and its derivative
        func_fem_sol_delta = fem_sol.approximate_solution_intervals(delta_sol_vector, psi_r_v, psi_l_v, le_n_g, r, N)
        func_der_fem_sol_delta = fem_sol.derivative_approximate_solution_intervals(delta_sol_vector, le_p_g, np.shape(x)[0], r, N, h)

        # Evaluate \delta_t v^j_h and its derivative
        func_fem_sol_elliptic_delta = fem_sol.approximate_solution_intervals(delta_sol_vector_elliptic, psi_r_v, psi_l_v, le_n_g_elliptic, r_elliptic, N)
        func_der_fem_sol_elliptic_delta = fem_sol.derivative_approximate_solution_intervals(delta_sol_vector_elliptic, le_p_g_elliptic, np.shape(x)[0], r_elliptic, N, h)

        # Compute the H^1-norm of '(u^j_h + u^{j-1}_h)/2 - (v^j_h + v^{j-1}_h)/2' and '\delta_t u^j_h - \delta_t v^j_h '
        eta_ell_h = h1_norm(func_fem_sol_h, func_fem_sol_elliptic_h, func_der_fem_sol_h, func_der_fem_sol_elliptic_h, x, w, h, Delta)
        eta_ell_delta = h1_norm(func_fem_sol_delta, func_fem_sol_elliptic_delta, func_der_fem_sol_delta, func_der_fem_sol_elliptic_delta, x, w, h, Delta)

    # The value \eta^j_{ell}
    eta_ell = sigma_j * (tau_j * eta_ell_h + tau_j**2/2*eta_ell_delta)

    return(eta_ell)