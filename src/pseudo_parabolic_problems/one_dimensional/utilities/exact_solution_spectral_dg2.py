from typing import Callable
from numpy import ndarray
from ..time_discretization_methods.spectral.dg_two_spectral import dg_two_spectral
from ..projection_operators.l2_projection_spectral import l2_projection_spectral as l2_projection

def exact_solution_spectral_dg2(M: int, 
                                Tau: ndarray, 
                                x: ndarray, 
                                w: ndarray, 
                                func_u0: Callable[[ndarray], ndarray], 
                                func_f: Callable[[ndarray, float], ndarray], 
                                mass: ndarray,
                                matrix_L: ndarray, 
                                matrix_M: ndarray, 
                                dim_V: int, 
                                a: int, 
                                b: int, 
                                le_n_g: ndarray) -> ndarray:
    
    
    """
    Computation of a reference solution, using a dG(2)/spectral Galerink method. 
        
    Args:
        M (int): Number of time intervals.
        T (int): Final time.
        x (ndarray): Nodes of the Gauss Lobatto formula.
        w (ndarray): Weights of the Gauss Lobatto formula.
        func_u0 (Callable[[ndarray], ndarray]): The initial condition u_0.
        func_f (Callable[[ndarray, float], ndarray]): The function f.
        mass (ndarray): The mass matrix.
        matrix_L (ndarray): Coefficient matrix associated with the operator L.
        matrix_M (ndarray): Coefficient matrix associated with the operator M.
        dim_V (int): Dimension of the ansatz space.
        a (int): Startinf point of the space interval.
        b (int): End point of the space interval.
        le_n_g (ndarray): The integrated Legendre polynomials N_1,...,N_{dim_V} evaluated at x.
        
    Returns:
        ndarray: The coefficients of the reference solution at the final time T.
    """
    
    # The L^2 projection of the initial condition into the finite element space V^0_h
    sol_vector = l2_projection(func_u0, a, b, mass, x, w, le_n_g)
    
    for j in range(1,M+1):
        
        t = j*Tau[j-1]   # the current time
            
        # determination of the coefficient array
        sol_vector = dg_two_spectral(x, w, Tau[j-1], t, func_f, sol_vector, matrix_L, matrix_M, a, b, le_n_g, dim_V)
    
    return(sol_vector)