import numpy as np

from numpy import ndarray
from typing import Callable
from ..mesh_tools.interval_mapp import interval_mapp as int_mapp
from ..approximate_solutions import fem_solution_gauss_lobatto as fem_solution

def pseudo_elliptic_h1_1d(sol_vector: ndarray, 
                         phi_vector: ndarray,
                         psi_vector: ndarray,
                         func_a: Callable[[ndarray], ndarray],
                         func_c: Callable[[ndarray], ndarray],
                         func_g: Callable[[ndarray], ndarray], 
                         x: ndarray, 
                         w: ndarray, 
                         h: ndarray, 
                         Delta: ndarray,
                         c_a: float,
                         psi_r_v: ndarray,
                         psi_l_v: ndarray,
                         le_n_g: ndarray | None,
                         le_p_g_x: ndarray | None,
                         r: int, 
                         N: int) -> float:
    
    r"""
    The estimator \eta_{ell} for the 'pseudo'-elliptic problem:

    L y = (L+M) \phi_h - g - L\psi_h,   with Ly = -y'' + ay and My = -y'' + cy

    where \psi_h, \phi_h \in - V^0_h and g \in L^2(a,b). 

    The estimator \eta_{ell} is given by 

    \eta_{ell} = (\sum_{i=1}^N h_i ||-2*\phi_h'' + (a+c)\phi_h - g + \psi_h'' - a\psi_h + y_h'' - ay_h'')^{1/2}/(\pi * c_a).   
    
    In he follwoing we use the abbreviation: 

    G = -2*\phi_h'' + (a+c)\phi_h - g + \psi_h'' - a\psi_h + y_h'' - ay_h''.
        
    Parameters
    ----------
    sol_vector: ndarray 
        Coefficients associated with the FEM solution y_h.

    phi_vector: ndarray 
        Coefficients associated with the function \phi_h.
    
    psi_vector: ndarray 
        Coefficients associated with the function \psi_h.
    
    func_a: Callable[[ndarray], ndarray] 
        The function a.
    
    func_c: Callable[[ndarray], ndarray] 
        The function c.
    
    func_g: Callable[[ndarray], ndarray]
        The function g.
    
    x: ndarray 
        The nodes of the GL formula.
    
    w: ndarray 
        The weights of the GL formula.
    
    h: ndarray 
        The spatial step sizes.
    
    Delta: ndarray 
        The mesh points of the spatial mesh of [a,b]. 
    
    c_a: float 
        The Coercivity constant of the bilinear form a(,), associated with the operator L.
    
    psi_r_v: ndarray 
        The shape functions \psi_R, evaluated at the Gauss-Lobatto nodes.
    
    psi_l_v: ndarray 
        The shape functions \psi_L, evaluated at the Gauss-Lobatto nodes.
    
    le_n_g: ndarray 
        The integrated Legendre Polynomials N_l, l=1,...,r, evaluated at the Gauss-Lobatto nodes.
    
    le_p_g_x: ndarray 
        The first derivative of the Legendre Polynomials P'_l, l=1,...,r, evaluated at the Gauss-Lobatto nodes.
    
    r: int 
        r+1 is the maximum degree of the considered integrated Legendre polynomials. 
    
    N: int 
        Number of spatial subintervals.                       
        
    Returns
    -------
    :float 
        \eta_{ell}
        
    """
    
    # The array [-1,1]
    ref_array = np.array([-1,1])
    
    # The functions a(\chi_i^{-1}), c(\chi_i^{-1}) and g(\chi_i^{-1}) evaluated at the Gauss-Lobatto nodes for each i=1,...,N.
    a_values = func_a(int_mapp(x, ref_array, np.array([Delta[:-1], Delta[1:]]).T))
    c_values = func_c(int_mapp(x, ref_array, np.array([Delta[:-1], Delta[1:]]).T))
    g_values = func_g(int_mapp(x, ref_array, np.array([Delta[:-1], Delta[1:]]).T))

    # u_h(\chi_i^{-1}), \phi_h(\chi_i^{-1}) and \psi_h(\chi_i^{-1}) evaluated at the Gauss-Lobatto nodes, i=1,...,N.
    y_h_values = fem_solution.approximate_solution_intervals(sol_vector, psi_r_v, psi_l_v, le_n_g, r, N)
    phi_h_values = fem_solution.approximate_solution_intervals(phi_vector, psi_r_v, psi_l_v, le_n_g, r, N)
    psi_h_values = fem_solution.approximate_solution_intervals(psi_vector, psi_r_v, psi_l_v, le_n_g, r, N)

    # u''_h(\chi_i^{-1}), \phi''_h(\chi_i^{-1}) and \psi''_h(\chi_i^{-1}) evaluated at the Gauss-Lobatto nodes, i=1,...,N.
    second_der_y_h_values = fem_solution.second_derivative_approximate_solution_intervals(sol_vector, le_p_g_x, np.shape(x)[0], r, N, h)
    second_der_phi_h_values = fem_solution.second_derivative_approximate_solution_intervals(phi_vector, le_p_g_x, np.shape(x)[0], r, N, h)
    second_der_psi_h_values = fem_solution.second_derivative_approximate_solution_intervals(psi_vector, le_p_g_x, np.shape(x)[0], r, N, h)

    # The function G(\chi_i^{-1}), evaluated at the Gauss-Lobatto nodes, i=1,...,N.
    G_values = -2*second_der_phi_h_values + (a_values + c_values) * phi_h_values - g_values + second_der_psi_h_values - a_values * psi_h_values \
               + second_der_y_h_values - a_values * y_h_values
    
    # Approximation of \int_{-1}^1 (G(\chi_i^{-1}(x)))^2 dx, i=1,...,N
    err_value = np.dot(G_values**2, w)

    # Approximation of h_i/2 \int_{-1}^1 (G(\chi_i^{-1}(x)))^2 dx, i=1,...,N,
    err_value = h/2 * err_value

    # Approximation of (\sum_{i=1}^N h_i ||-2*\phi_h'' + (a+c)\phi_h - g + \psi_h'' - a\psi_h + y_h'' - ay_h'')^{1/2}
    err_value = np.dot(h**2, err_value)

    # Approximation of (\sum_{i=1}^N h_i ||-2*\phi_h'' + (a+c)\phi_h - g + \psi_h'' - a\psi_h + y_h'' - ay_h'')^{1/2}/(\pi c_a)
    err_value = np.sqrt(err_value)/(np.pi * c_a)
    
    return(err_value)