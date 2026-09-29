import numpy as np
import scipy.sparse as sp

from numpy import ndarray
from ..mesh_tools.interval_mapp import interval_mapp as int_mapp
from typing import Callable

def elliptic_projection_fem(N: int, 
                            r: int, 
                            Delta: ndarray, 
                            h: ndarray, 
                            x: ndarray, 
                            w: ndarray, 
                            matrix_L: sp.sparray, 
                            psi_r_v: ndarray, 
                            psi_l_v: ndarray, 
                            func_u: Callable[[ndarray], ndarray], 
                            func_der_u: Callable[[ndarray], ndarray], 
                            func_a: Callable[[ndarray], ndarray], 
                            le_n_g: ndarray, 
                            le_p_g: ndarray) -> ndarray:
    
    r"""
    The elliptic projection of a given function u into the FEM space V^0_h \subset H^1_0(a,b).

    Given an elliptic, bounded and coercive operator L: H^1_0(a,b) \to H^{-1}(a,b), which is given by
    
    Lu = -u'' + au
    
    the projection 

    a(u_h, v_h) = a(u, v_h)     for all v_h \in V^0_h,

    is computed. Here, a() denotes the bilinear form associated with the operator L.         
    
    Args:
        N (int): Number of spatial subintervals. 
        r (int): r+1 is the maximum degree of the considered integrated Legendre polynomials; if 'r==0', no integrated Legendre polynomials are considered. 
        Delta (ndarray): The spatial mesh.
        h (ndarray): The spatial step sizes.
        x (ndarray): Nodes of the Gauss-Lobatto formula.
        w (ndarray): Weights of the Gauss-Lobatto formula.
        matritx_L (sp.sparray): Mass matrix with respect to V^0_h.
        psi_r_v (ndarray): The shape function \psi_R evaluated at x. 
        psi_l_v (ndarray): The shape function \psi_L evaluated at x.
        func_u (Callable[[ndarray], ndarray]): The function u \in H^1_0(\Omega).
        der_func_u (Callable[[ndarray], ndarray]): The derivative of the function u.
        func_a (Callable[[ndarray], ndarray]): The function a.
        le_n_g (ndarray): The integrated Legendre polynomials N_1,...,N_r evaluated at x; empty array if 'r=0'.
        le_p_g (ndarray): The Legendre polynomials P_1,...,P_r evaluated at x; empty array if 'r=0'.
                            
    
    Returns:
        ndarray: The coefficients of the the elliptic projection u_h.
        
    """
    
    # The reference array [-1,1]
    ref_array = np.array([-1,1])
    
    r"""Evaluate u(\chi_i^{-1}), u'(\chi_i^{-1}) and a(\chi_i^{-1}) in the Gauss-Lobatto nodes x, where 
        \chi_i:[x_{i-1}, x_i] \to [-1,1] denote affine-linear transformations, i=1,...,N. """
    u_value = func_u(int_mapp(x, ref_array, np.array([Delta[:-1], Delta[1:]]).T))
    der_u_value = func_der_u(int_mapp(x, ref_array, np.array([Delta[:-1], Delta[1:]]).T))
    a_value = func_a(int_mapp(x, ref_array, np.array([Delta[:-1], Delta[1:]]).T))
    
    # ----------------------------------------------------------------------------------------------------
    # COMPUTE a(u, \phi_i), WHERE \phi_i DENOTE THE BASSIS FUNCTIONS OF V^0_h, i=1,...,dim(V^0_h)
    # ----------------------------------------------------------------------------------------------------
    
    # Initialize result_vector subjected to a(u,\phi_i), and sol_vector, that stores the coefficients of u_h
    result_vector = np.zeros((r+1)*N-1)
    sol_vector = np.zeros((r+1)*N+1)
    
    # BASIS FUNCTIONS ASSOCIATED WITH THE SHAPE FUNCTIONS \psi_R AND \psi_L
    l = np.linspace(1,N-1,N-1, True, dtype = int)

    # (au, \phi_i)
    result_vector[(r+1)*l-1] = h[:-1]/2 * np.dot(u_value[:-1] * a_value[:-1] * psi_r_v, w) \
                                + h[1:]/2 * np.dot(u_value[1:] * a_value[1:] * psi_l_v, w)
    # (u', \phi'_i)
    result_vector[(r+1)*l-1] += 1/2 * np.dot(der_u_value[:-1], w) - 1/2 * np.dot(der_u_value[1:], w) 
    
    # BASIS FUNCTIONS ASSOCIATED WITH THE INTEGRATED LEGENDRE POLYNOMIALS, N_1,...,N_r
    l = np.linspace(0,N-1,N, True, dtype = int)
    for j in range(0,r):
        
        # (au, \phi_i)
        result_vector[(r+1)*l+j] = h/2 * np.dot(le_n_g[j] * u_value * a_value, w)
        
        # (u', \phi'_i)
        result_vector[(r+1)*l+j] += np.dot(le_p_g[j] * der_u_value, w)
    
    sol_vector[1:-1] = sp.linalg.spsolve(matrix_L, result_vector)
    
    return(sol_vector) 