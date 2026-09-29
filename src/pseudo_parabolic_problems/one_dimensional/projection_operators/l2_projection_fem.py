import numpy as np


from numpy import ndarray
from scipy.sparse import csr_array, csc_array
from typing import Callable
from scipy.sparse.linalg import spsolve
from ..mesh_tools.interval_mapp import interval_mapp as int_mapp

def l2_projection_fem(N: int, 
                      r: int, 
                      Delta: ndarray, 
                      h: ndarray, 
                      x: ndarray, 
                      w: ndarray, 
                      mass: csr_array | csc_array, 
                      psi_r_v: ndarray, 
                      psi_l_v: ndarray, 
                      func_u: Callable[[ndarray], ndarray], 
                      le_n_g: ndarray) -> ndarray:
    
    r"""
    Computation of the discrete L^2 projection of a function u \in H^1_0(a,b) into the FEM space V^0_h \subspace H^1_0(a,b).

    Given a function u \in H^1_0(a,b), the discrete L^2 projection

    (u_h, v_h) = (u, v_h)   for all v_h \in V^0_h

    is computed.       

    Args: 
        N (int): Number of spatial subintervals. 
        r (int): r+1 is the maximum degree of the considered integrated Legendre polynomials; if 'r==0', no integrated Legendre polynomials are considered. 
        Delta (ndarray): The spatial mesh.
        h (ndarray): The spatial step sizes.
        x (ndarray): Nodes of the Gauss-Lobatto formula.
        w (ndarray): Weights of the Gauss-Lobatto formula.
        mass (sp.sparray): Mass matrix with respect to V^0_h.
        psi_r_v (ndarray): The shape function \psi_R evaluated at x. 
        psi_l_v (ndarray): The shape function \psi_L evaluated at x.
        func_u (Callable[[ndarray], ndarray]): The function u \in H^1_0(\Omega).
        le_n_g (ndarray): The integrated Legendre polynomials N_1,...,N_r evaluated at x; empty if 'r=0'.
                            
    
    Returns:
        ndarray: The coefficients of the the L^2 projection u_h over (a,b).   
    """

    # The reference array [-1,1]
    ref_array = np.array([-1, 1])
    
    # Initialize the array, that stores the coefficients
    sol_vector = np.zeros((r+1)*N+1)

    r""" Evaluate u(\chi_i^{-1}) at the Gauss-Lobatto nodes x for each i=1,...,N. \chi_i:[x_{i-1}, x_i] \to [-1,1] denote 
         affine-linear transformations, i=1,...,N."""
    u_value = func_u(int_mapp(x, ref_array, np.array([Delta[:-1], Delta[1:]]).T))
    
    # COMPUTE (u, \phi_i), WHERE \phi_i DENOTE THE BASSIS FUNCTIONS OF V^0_h, i=1,...,dim(V^0_h)
    result_vector = np.zeros((r+1)*N-1)
    
    # Basis functions associated with the shape functions \psi_R and \psi_L
    l = np.linspace(1,N-1,N-1, True, dtype = int)
    result_vector[(r+1)*l-1] = h[:-1]/2 * np.dot(u_value[:-1] * psi_r_v,w) + h[1:]/2 * np.dot(u_value[1:] * psi_l_v,w)
    
    # Basis functions associated with the integrated Legendre polynomials N_1,...,N_r
    l = np.linspace(0,N-1,N, True, dtype = int)
    for j in range(0,r):
        
        result_vector[(r+1)*l+j] = h/2 * np.dot(u_value * le_n_g[j], w)
    
    sol_vector[1:-1] = spsolve(mass, result_vector) # type: ignore
    
    return(sol_vector)    