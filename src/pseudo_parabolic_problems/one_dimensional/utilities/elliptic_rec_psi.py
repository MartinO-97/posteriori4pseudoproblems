import numpy as np
import scipy.sparse as sp

from numpy import ndarray
from scipy.sparse import csr_array, csc_array
from ..mesh_tools.interval_mapp import interval_mapp as int_mapp
from typing import Callable

def elliptic_rec_psi(N: int,  
                     r: int, 
                     func_f: Callable[[ndarray], ndarray], 
                     Delta: ndarray, 
                     h: ndarray, 
                     x: ndarray, 
                     w: ndarray, 
                     sol_vector: ndarray, 
                     matrix_L: csr_array | csc_array, 
                     matrix_M: csr_array | csc_array, 
                     psi_r_v: ndarray, 
                     psi_l_v: ndarray, 
                     le_n_g: ndarray) -> ndarray:
    
    r"""
    Computation of the function \psi^j_h for ellitpic reconstruction.

    \psi^j_h is defined by 

    a_h(\psi^j_h, v_h) = c_h(u^j_h, v_h) - (f^j, v_h)   for all v_h \in V^0_h. 

    Args:
        N (int): Number of space intervals.
        r (int): Polynomial degree parameter for the integrated Legendre polynomials.
        func_f (Callable[[ndarray], ndarray]): Right-hand side function f(x, t_j).
        Delta (ndarray): Mesh points in space.
        h (ndarray): spatial step size.
        x (ndarray): Gauss-Lobatto nodes.
        w (ndarray): Gauss-Lobatto weights.
        func_f (callable): Right-hand side function f(x, t_j).
        sol_vector (ndarray): Coefficient vector subjected to u^j_h.
        matrix_L (csr_array | csc_array): Sparse matrix representing the operator L.
        matrix_M (csr_array | csc_array): Sparse matrix representing the operator M.
        psi_r_v (ndarray): 'Right' shape function \psi_R evaluated at quadrature nodes x.
        psi_l_v (ndarray): 'Left' shape function \psi_L evaluated at quadrature nodes x.
        le_n_g (ndarray | None): Integrated Legendre polynomials evaluated at quadrature points x.    
    """

    # The reference array 
    ref_array = np.array([-1,1])

    # determination of the result_vector
    f_j = func_f(int_mapp(x, ref_array, np.array([Delta[:-1], Delta[1:]]).T)) 
    
    l = np.linspace(1,N-1,N-1, True, dtype = int)
    result_vector = np.zeros((r+1)*N-1)
    
    # (f,\phi_i)
    result_vector[(r+1)*l-1] = h[:-1]/2*np.dot(f_j[:-1] * psi_r_v, w) + h[1:]/2*np.dot(f_j[1:] * psi_l_v, w)
    
    # (f, \phi_{i,j})    
    l = np.linspace(0,N-1,N, True, dtype = int)
    for mu in range(0,r):  
        result_vector[(r+1)*l+mu] = h/2*np.dot(f_j * le_n_g[mu], w)

    psi_vector = np.zeros((r+1)*N+1)
    psi_vector[1:-1] = sp.linalg.spsolve(matrix_L, matrix_M.dot(sol_vector[1:-1]) - result_vector)    # \psi^j_h # type: ignore

    return(psi_vector)    