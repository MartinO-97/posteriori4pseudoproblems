import numpy as np

from numpy import ndarray
from typing import Callable
from ..mesh_tools.interval_mapp import interval_mapp as int_mapp

def l2_norm_function_fem(func_u: Callable[[ndarray], ndarray] | ndarray, 
                         func_v: Callable[[ndarray], ndarray] | ndarray, 
                         x: ndarray, 
                         w: ndarray, 
                         h: ndarray, 
                         Delta: ndarray) -> float:
    
    r"""
    The L^2(a,b) norm error between two given functions u and v with respect to a FEM:

    ||u-v||_{0,(a,b)} = \sqrt{\int_a^b (u(x)-v(x))^2 dx} = \sqrt{\sum_{i=1}^N \int_{x_{i-1}}^{x_i} (u(x)-v(x))^2 dx} .

    If u and/or v are elements of the FEM space V^0_h \subset H^1_0(a,b), func_u and func_v are ndarrays, that stores 
    the evaluation of u(\chi_i^{-1}) and/or v(\chi_i^{-1}) at the Gauss-Lobatto (GL) nodes x, where \chi_i:[x_{i-1},x_i] \to [-1,1],
    i=1,...,N, denote affine-linear transforamtions from the spatial subintervals to [-1,1]. Here, 
    x_0,...,x_N denote the nodes of the spatial mesh. x_0,...,x_n are not related to the GL nodes x.    
    
    Args:
        func_u Callable[[ndarray], ndarray] | ndarray -> The function u; or the function u(\chi_i^{-1}) evaluated at x for each i=1,...,N.
        func_v Callable[[ndarray], ndarray] | ndarray -> The function v; or the function v(\chi_i^{-1}) evaluated at x for each i=1,...,N.
        x (ndarray): The nodes of the GL formula.
        w (ndarray): The weights of the GL formula.
        h (ndarray): The spatial step sizes.
        Delta (ndarray): The mesh points of the spatial mesh of [a,b].              
        
    Returns:
        float: ||u-v||_{0,(a,b)} = \sqrt{\int_a^b (u(x)-v(x))^2 dx}.
        
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

    # Approximation of \int_{-1}^1 ((u-v)(\chi_i^{-1}(x)))^2 dx, i=1,...,N
    l2_v = np.dot((u_v-v_v)**2, w)

    # Approximation of \sum_{i=1}^N h_i/2 \int_{-1}^1 ((u-v)(\chi_i^{-1}(x)))^2 dx, i=1,...,N,
    l2_v = np.dot(h/2, l2_v)
    
    # Approximation of \sqrt{\sum_{i=1}^N h_i/2 \int_{-1}^1 ((u-v)(\chi_i^{-1}(x)))^2 dx}, i=1,...,N,
    l2_v = np.sqrt(l2_v)
    
    return(l2_v)