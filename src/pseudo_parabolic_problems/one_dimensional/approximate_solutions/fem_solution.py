import numpy as np
from ..basis_functions import legendre_polynomials as lp

def fem_solution(s_v: np.ndarray, 
                 N: int, 
                 mesh_le: np.ndarray, 
                 ref: int, 
                 r: int) -> np.ndarray:
    
    r"""The approxiatem solution u_h, computed by a FEM.

    Given the shape functions \psi_L and \psi_R as well as the integrated
    Legendre polynomials N_1, ..., N_r, the u_h is given by

    u_h(x) = \sum_{i=1}^N ( \alpha_i \psi_R(\chi_i(x)) + \sum_{l=1}^r \alpha_{il}N_l(\chi_i(x)) 
                                + \alpha_{i-1}\psi_L(\chi_{i-1}(x)))

    where N is the number of spatial mesh intervals, x_0,...,x_N the nodes of the spatial mesh 
    and \chi_{i}:[x_{i-1}, x_i] \to [-1,1], i=1,...,N, affine-linear transformations.                            
        
    Args:
        s_v (np.ndarray): The coefficients of u_h.
        N (int): The number of spatial subintervals.
        mesh_le (np.ndarray): The refined mesh Delte_refined mapped to [-1,1].
        ref (int): The number of points between two nodes in Delta.
        r (int): r+1 the highest degree of the utilized itegrated Legendre polynomials.
        a (float): Starting point of the spatial interval.
        b (float): End point of the spatial interval.
    
    Returns:
        np.ndarray: The approximate solution evaluated at the points of the refined mesh Delta_refined. 
        
    """
    
    z_sol = np.zeros((ref+1)*N+1)

    for l in range(0,ref+1):
        z_sol[l:-ref-1+l:ref+1] = ((ref+1-l)*s_v[:-1-r:r+1] + l*s_v[1+r::r+1])/(ref+1)
    
    z_sol[-1] = s_v[-1]
    
    
    for l in range(1,r+1):
        y = np.concatenate((np.repeat(s_v[l:-1-r+l:r+1], ref+1), np.array([1])))
        z_sol = z_sol + y * lp.integrated_legendre(l,mesh_le)
    
    return(z_sol)    