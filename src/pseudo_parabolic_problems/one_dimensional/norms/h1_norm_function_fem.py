import numpy as np

from numpy import ndarray
from typing import Callable
from ..mesh_tools.interval_mapp import interval_mapp as int_mapp

def h1_norm_function_fem(func_u: Callable[[ndarray], ndarray] | ndarray, 
                         func_v: Callable[[ndarray], ndarray] | ndarray, 
                         der_func_u: Callable[[ndarray], ndarray] | ndarray, 
                         der_func_v: Callable[[ndarray], ndarray] | ndarray, 
                         x: ndarray, 
                         w: ndarray, 
                         h: ndarray, 
                         Delta: ndarray) -> float:
    
    r"""
    The H^1(a,b) norm error between two given functions u and v with respect to a FEM:

    ||u-v||_{1,(a,b)} = \sqrt{\int_a^b (u(x)-v(x))^2 dx + \int_a^b (u'(x)-v'(x))^2 dx}.

    If u and/or v are elements of the FEM space V^0_h \subset H^1_0(a,b), func_u and func_v are ndarrays, that stores 
    the evaluation of u(\chi_i^{-1}) and/or v(\chi_i^{-1}) at the Gauss-Lobatto (GL) nodes x, where \chi_i:[x_{i-1},x_i] \to [-1,1],
    i=1,...,N, denote affine-linear transforamtions from the spatial subintervals to [-1,1]. The same holds for der_func_u and der_func_v. 
    
    x_0,...,x_N denote the nodes of the spatial mesh. x_0,...,x_n are not related to the GL nodes x.    
    
    Args:
        func_u Callable[[ndarray], ndarray] | ndarray -> The function u; or the function u(\chi_i^{-1}) evaluated at x for each i=1,...,N.
        func_v Callable[[ndarray], ndarray] | ndarray -> The function v; or the function v(\chi_i^{-1}) evaluated at x for each i=1,...,N.
        der_func_u Callable[[ndarray], ndarray] | ndarray -> The function u'; or the function u'(\chi_i^{-1}) evaluated at x for each i=1,...,N.
        der_func_v Callable[[ndarray], ndarray] | ndarray -> The function v'; or the function v'(\chi_i^{-1}) evaluated at x for each i=1,...,N.
        x (ndarray): The nodes of the GL formula.
        w (ndarray): The weights of the GL formula.
        h (ndarray): The spatial step sizes.
        Delta (ndarray): The mesh points of the spatial mesh of [a,b].              
        
    Returns:
        float: ||u-v||_{1,(a,b)} = \sqrt{\int_a^b (u(x)-v(x))^2 dx + \int_a^b (u'(x)-v'(x))^2 dx}.
        
    """
    
    # The array [-1,1]
    ref_array = np.array([-1,1])

    # The functions u(\chi_i^{-1}) and v(\chi_i^{-1}) evaluated at the Gauss-Lobatto nodes for each i=1,...,N.
    if callable(func_u):
        u_v = func_u(int_mapp(x, ref_array, np.array([Delta[:-1], Delta[1:]]).T))
    else:
        u_v = np.copy(func_u)

    if callable(func_v):
        v_v = func_v(int_mapp(x, ref_array, np.array([Delta[:-1], Delta[1:]]).T))
    else:
        v_v = np.copy(func_v)

    # The functions u'(\chi_i^{-1}) and v'(\chi_i^{-1}) evaluated at the Gauss-Lobatto nodes for each i=1,...,N.
    if callable(der_func_u):
        der_u_v = der_func_u(int_mapp(x, ref_array, np.array([Delta[:-1], Delta[1:]]).T))
    else:
        der_u_v = np.copy(der_func_u)

    if callable(der_func_v):
        der_v_v = der_func_v(int_mapp(x, ref_array, np.array([Delta[:-1], Delta[1:]]).T))
    else:
        der_v_v = np.copy(der_func_v)

    # Approximation of \int_{-1}^1 ((u-v)(\chi_i^{-1}(x)))^2 + \int_{-1}^1 ((u'-v')(\chi_i^{-1}(x)))^2 dx, i=1,...,N
    h1_v = np.dot((u_v-v_v)**2, w) + np.dot((der_u_v-der_v_v)**2, w)

    # Approximation of \sum_{i=1}^N (h_i/2 \int_{-1}^1 ((u-v)(\chi_i^{-1}(x)))^2 dx + h_i/2 \int_{-1}^1 ((u'-v')(\chi_i^{-1}(x)))^2 dx), i=1,...,N,
    h1_v = np.dot(h/2, h1_v)

    # Approximation of \sqrt{\sum_{i=1}^N (h_i/2 \int_{-1}^1 ((u-v)(\chi_i^{-1}(x)))^2 dx + h_i/2 \int_{-1}^1 ((u'-v')(\chi_i^{-1}(x)))^2 dx)}, i=1,...,N,
    h1_v = np.sqrt(h1_v)

    return(h1_v)