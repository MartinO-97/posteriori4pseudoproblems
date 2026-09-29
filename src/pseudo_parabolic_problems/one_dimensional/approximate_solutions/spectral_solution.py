import numpy as np

from numpy import ndarray
from typing import Callable
from ..basis_functions import legendre_polynomials as lp
from ..mesh_tools.interval_mapp import interval_mapp as int_mapp


# ----------------------------------------------------------------------------------------------------
# THE APPROXIMATE SOLUTION
# ----------------------------------------------------------------------------------------------------

def approximate_solution(x: ndarray, 
                         s_v: ndarray, 
                         a: float, 
                         b: float, 
                         dim_V: int) -> ndarray:
    
    r"""
    The approximate solution in one-dimension, computed by using a spectral Galerkin method.

    The approximate solution is given by

    U(x) = \sum_{l=1}^{dim_V} \alpha_l \phi_l(x),   x \in [a,b],

    where \phi_l:[a,b] \to \RR is given by

    \phi_l(x) = N_l(\chi(x)),   l=1,...,dim_V.

    Here, N_l denote the integrated Legendre polynomials, l=1,...,dim_V, and \chi:[a,b] \to [-1,1]
    an affine-linear transformation.        

    Args:
        x (ndarray): Point(s), where the approximate solution U shall be evaluated at. 
        s_v (ndarray): Array, which stores the coefficients \alpha_1,...,\alpha_{dim_V}
        a (float): Starting point of the space interval [a,b].
        b (float): End point of the space interval [a,b].
        dim_V (int): Dimension of the ansatz space. 
        
    Returns:
        ndarray: The approximate solution, evaluated at x. 
    """    
    
    def phi(l: int,
            y: ndarray) -> ndarray:

        r"""The basis function

            \phi_l(y) = N_l(\chi(y)), l=1,...,dim_V-1,

        where N_l denotes the integrated Legendre polynomials and \chi:[a,b] \to [-1,1]
        an affine linear transformation.

        Args:
            l (int): Number of basis function.
            y (ndarray): Point(s), where \phi_l shall be evaluated at.

        Returns:
            ndarray: \phi_l, evaluated at y.
        """

        return(lp.integrated_legendre(l, int_mapp(y, np.array([a,b]), np.array([-1,1]))))

    return(_spectral_sum(x,s_v, a, b, dim_V, phi))


# ----------------------------------------------------------------------------------------------------
# THE DERIVATIVE OF THE APPROXIMATE SOLUTION
# ----------------------------------------------------------------------------------------------------

def derivative_approximate_solution(x: ndarray, 
                                    s_v: ndarray, 
                                    a: float, 
                                    b: float, 
                                    dim_V: int):
    
    r"""The derivative of the approximate solution in one dimension, 
    computed by using a spectral Galerkin method.

    The derivative of the approximate solution is given by

        U'(x) = \sum_{l=1}^{dim_V} \alpha_l \phi'_l(x),   x \in [a,b],

    where \phi_l:[a,b] \to \RR is given by

        \phi_l(x) = N_l(\chi(x)),   l=1,...,dim_V.

    Here, N_l denote the integrated Legendre polynomials, l=1,...,dim_V, and \chi:[a,b] \to [-1,1]
    an affine-linear transformation. So, we have
    
        \phi'_l(x) = \chi'(x) P_l(\chi(x)),   l=1,...,dim_V.

    Note, that 
    
        \chi'(x) = 2/(b-a),     x \in [a,b].

    Args:
        x (ndarray): Point(s), where the approximate solution U shall be evaluated at. 
        s_v (ndarray): Array, which stores the coefficients \alpha_1,...,\alpha_{dim_V}
        a (float): Starting point of the space interval [a,b].
        b (float): End point of the space interval [a,b].
        dim_V (int): Dimension of the ansatz space. 
        
    Returns:
        ndarray: The derivative of the approximate solution, evaluated at x. 
    """    
    
    def der_phi(l: int,
                y: ndarray) -> ndarray:

        r"""The derivative of the basis function

            \phi_l(y) = N_l(\chi(y)), l=1,...,dim_V-1,

        where N_l denotes the integrated Legendre polynomials and \chi:[a,b] \to [-1,1]
        an affine linear transformation. So, we have
    
            \phi'_l(x) = \chi'(x) P_l(\chi(x)),   l=1,...,dim_V.

        Args:
            l (int): Number of basis function.
            y (ndarray): Point(s), where \phi_l shall be evaluated at.

        Returns:
            ndarray: \phi'_l, evaluated at y.
        """

        return(2/(b-a)*lp.legendre(l, int_mapp(y, np.array([a,b]), np.array([-1,1]))))


    return(_spectral_sum(x,s_v, a, b, dim_V, der_phi)) 


# ----------------------------------------------------------------------------------------------------
# AUXILIARY FUNCTION TO EVALUATE THE APPROXIAMTE SOLUTION AND ITS DERIVATIVE
# ----------------------------------------------------------------------------------------------------

def _spectral_sum(x: ndarray, 
                  s_v: ndarray, 
                  a: float, 
                  b: float, 
                  dim_V: int, 
                  basis_function: Callable[[int, ndarray], ndarray]) -> ndarray:
    
    r"""Auxiliary function, to evaluate functions of type

        v(x) = \sum_{l=1}^{dim_V} \alpha_l \phi_l(x),   \quad x\in [a,b]

    where \phi_l, l=1,...,dim_V, are some given basis functions.     

    Args:
        x (ndarray): Point(s), where the the function v shall be evaluated at. 
        s_v (ndarray): Array, which stores the coefficients \alpha_1,...,\alpha_{dim_V}
        a (float): Starting point of the space interval [a,b].
        b (float): End point of the space interval [a,b].
        dim_V (int): Dimension of the ansatz space.
        basis_function (callable): Basis function(s). 
        
    Returns:
        ndarray: The function v, evaluated at x. 
    """

    z = np.zeros(np.shape(x))
    for k in range(1, dim_V+1):
        z += s_v[k-1] * basis_function(k,x)
    return(z) 