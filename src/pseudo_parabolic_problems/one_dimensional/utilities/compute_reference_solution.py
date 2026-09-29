import numpy as np

from numpy import ndarray
from typing import Callable
from ..approximate_solutions import spectral_solution as sol
from ..matrix_assembly.assemble_coeff_matrices_spectral import assemble_coeff_matrices_spectral 
from ..time_discretization_methods.spectral.dg_two_spectral import dg_two_spectral as dg2
from ..time_discretization_methods.spectral.dg_three_spectral import dg_three_spectral as dg3
from ...common import temporal_mesh_generation as time_mesh
from ..projection_operators import l2_projection_spectral as l2_spectral

def compute_reference_solution(x: ndarray, 
                               w: ndarray, 
                               func_a: Callable[[ndarray], ndarray], 
                               func_c: Callable[[ndarray], ndarray],
                               func_u0: Callable[[ndarray], ndarray], 
                               func_f: Callable[[ndarray, float], ndarray],
                               det_a: bool, 
                               dim_V: int, 
                               a: float, 
                               b: float, 
                               eps_a: float,
                               eps_c: float,
                               le_n_g: ndarray,
                               dg2_or_dg3: str, 
                               T: float,
                               pseudo_or_normal: str) -> tuple[Callable[[ndarray], ndarray], 
                                                               Callable[[ndarray], ndarray], 
                                                               ndarray, ndarray, ndarray]:

    r"""
    Compute a reference solution U: [a,b] \times {t_0,t_1,...,t_M} \to \RR as an 'exact" solution.

    The reference solution is computed by either using the dG(2)- or dG(3)-method
    in time and an spectral Galerkin method in space.

    Parameters
    ----------
    x: ndarray 
        Nodes of the GL formula.
    
    w: ndarray 
        Weights of the GL formula.
    
    func_a: Callable 
        The function a.
    
    func_c: Callable 
        The function c.
    
    func_u0: Callable 
        The initial condition u0.
    
    det_a: bool 
        Shall the matrix, associated with au, be assembled: 'True' for yes, 'False' for no.
        
    dim_V: int 
        Dimension of the Ansatz space.
    a: float 
        Starting point of the interval [a,b].
    
    b: float 
        End point of the interval [a,b].

    eps_a: float
        Perturbation parameter for the operator L.

    eps_c: float
        Perturbation parameter for the operator M.
    
    le_n_g: ndarray 
        Integrated Legendre polynomials evaluated at x.
    
    dg2_or_dg3: str 
        Shall the dG(2)-method ('dg2') or dG(3)-method ('dg3') be used.
    
    T: float 
        The final time T.
    
    pseudo_or_normal: str 
        Do we consider a pseudo parabolic equation ('pseudo') or a normal parabolic equation ('normal').

    Returns
    -------    
    func_u: Callable[[ndarray], ndarray] 
        The approximation of u(T).
    
    func_der_u: Callable[[ndarray], ndarray] 
        The approximation of u'(T).
    
    z_reference: ndarray 
        Points, to evaluate U(t_j), j=0,...,M.
    
    result_reference: ndarray 
        U(t_j) evaluated at 'z_reference', j=0,...,M.   
    
    omega_t: ndarray 
        Mesh in time.                             
    """
    
    # Validate that 'dg2_or_dg3' is either 'dg2' or 'dg3'
    if not (dg2_or_dg3 == 'dg2' or dg2_or_dg3 == 'dg3'):
        raise ValueError('\'dg2_or_dg3\' must be either \'dg2\' or \'dg3\'!')
    
    # Validate that 'pseudo_or_normal' is either 'pseudo' or 'normal'
    if not (pseudo_or_normal == 'pseudo' or pseudo_or_normal == 'normal'):
        raise ValueError('\'pseudo_or_normal\' must be either \'pseudo\' or \'normal\'!')
    
    # Assembly for the spectral Galerkin method
    mass, reaction_a, reaction_c, stiff, matrix_L, matrix_M \
        = assemble_coeff_matrices_spectral(x, w, func_a, func_c, det_a, dim_V, a, b, eps_a, eps_c, le_n_g)

    # Set 'matrix_L = mass' if a 'normal' parabolic equation is considered
    if pseudo_or_normal == 'normal': 
        matrix_L = np.copy(mass)

    # Number of time intervals
    M = 128

    # Points to evaluate the reference solution and array to save the evaluation
    z_reference = np.linspace(a,b,2*M)
    result_reference = np.zeros((M+1, np.size(z_reference)))

    # The L^2 projection of the initial condition into the ansatz space
    sol_vector = l2_spectral.l2_projection_spectral(func_u0, a, b, mass, x, w, le_n_g)

    # Evaluate u0 for the graphical illustration
    result_reference[0] = func_u0(z_reference)

    # temporal mesh 
    omega_t, Tau = time_mesh.equidistant_mesh(T, M)

    for j in range(1,M+1):

        # the dG(2)-method
        if dg2_or_dg3 == 'dg2':

            sol_vector = dg2(x, w, Tau[j-1], omega_t[j], func_f, sol_vector, matrix_L, matrix_M, a, b, le_n_g, dim_V)

        # the dG(2)-method
        if dg2_or_dg3 == 'dg3':

            sol_vector = dg3(x, w, Tau[j-1], omega_t[j], func_f, sol_vector, matrix_L, matrix_M, a, b, le_n_g, dim_V)    
    
        # Approximate u(t_j)
        result_reference[j] = sol.approximate_solution(z_reference, sol_vector, a, b, dim_V)

    # Define the reference solution to approximate u(T) and its first derivative
    func_u = lambda z: sol.approximate_solution(z, sol_vector, a, b, dim_V)
    func_der_u = lambda z: sol.derivative_approximate_solution(z, sol_vector, a, b, dim_V)  

    return(func_u, func_der_u, z_reference, result_reference, omega_t)  




        

     
    


