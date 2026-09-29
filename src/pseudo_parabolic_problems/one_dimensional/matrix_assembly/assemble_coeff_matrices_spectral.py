import numpy as np

from numpy import ndarray
from typing import Callable, Tuple
from ..mesh_tools.interval_mapp import interval_mapp

def assemble_coeff_matrices_spectral(x : ndarray, 
                                     w : ndarray, 
                                     func_a : Callable[[ndarray], ndarray], 
                                     func_c : Callable[[ndarray], ndarray], 
                                     det_a : bool, 
                                     dim_V : int, 
                                     a : float, 
                                     b : float,
                                     eps_a : float, 
                                     eps_c: float, 
                                     le_n_g : ndarray) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]:
    
    r"""
    This functions assembles the matrices, associated with the operators 
            
    Id u,  -\Delta u, au, cu
    
    where a,c :[a,b] \to \RR, \Delta is the Laplacian and Id is the identity operator, for a spectral Galerkin method. 
    If only a parabolic problem, is considered, then the matrix, associated with au, doesn't need to be assembled (det_a = False).
    We use Gauss-Lobatto (GL) as quadrature formula.   
     
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
        
    det_a: bool 
        Shall the matrix, associated with au, be assembled: 'True' for yes and 'False' for no.
        
    dim_V: int 
        Dimension of the Ansatz space.
    
    a: float 
        Starting point of the interval [a,b].
    
    b: float 
        End point of the interval [a,b].

    eps_a: float
        Pertubation parameter for the operator L.

    eps_c: float
        Pertubation parameter for the operator M.    

    le_n_g: ndarray 
        Integrated Legendre polynomials evaluated at x.                
        
    Returns
    -------
    mass: ndarray 
        Mass matrix
    
    reaction_a: ndarray
        Matrix associated the the reaction term au; empty array if det_a = False
    
    reaction_c: ndarray 
        Matrix associated with the reaction term cu
    
    stiff: ndarray 
        Stiffness matrix 
    
    matrix_L: ndarray 
        Matrix associated to the operator L; empty array if det_a = False
    
    matrix_M: ndarray 
        Matrix associated to the operator M
    """

    # The integers 1,...,dim_V
    m_i = np.linspace(1, dim_V, dim_V, True)  
    
    # Mass matrix
    mass_off_diag = -(b-a)/((2*m_i[:-2]+1)*(2*m_i[:-2]+3)*(2*m_i[:-2]+5))   # \int_a^b \phi_i \phi_{i+2}
    mass_diag = (b-a)/(2*m_i+1)**2*(1/(2*m_i+3)+1/(2*m_i-1))                # \int_a^b \phi_i^2
    mass = np.diag(mass_off_diag, -2)+np.diag(mass_diag, 0)+np.diag(mass_off_diag, 2)
    
    # Stiffness matrix
    stiff = 4/(b-a)*np.diag(1/(2*m_i+1))
    
    # The matrix reaction_a, associated with the reaction term au
    if det_a: 
        a_value = func_a(interval_mapp(x, np.array([-1,1]), np.array([a,b])))           # a(\chi^{-1}(x)), \chi:[a,b] -> [-1,1]
        reaction_a = (b-a)/2*le_n_g.dot(((a_value*w)*le_n_g).T)
    else:
        reaction_a = np.array([])    

    # The matrix reaction_c, associated with the reaction term cu
    c_value = func_c(interval_mapp(x, np.array([-1,1]), np.array([a,b])))     # c(\chi^{-1}(x)), \chi:[a,b] -> [-1,1]              
    reaction_c = (b-a)/2*le_n_g.dot(((c_value*w)*le_n_g).T)

    # The matrices, subjected to the operators L and M
    if det_a: 
        matrix_L = eps_a*stiff + reaction_a  
    else:
        matrix_L = np.array([])

    matrix_M = eps_c * stiff + reaction_c    

    return(mass, reaction_a, reaction_c, stiff, matrix_L, matrix_M)