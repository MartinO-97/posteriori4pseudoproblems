import numpy as np

from numpy import ndarray
from typing import Callable
from ..discretization_dataclasses import SpatialDiscParameters
from ..quadrature import Quadrature
from .l2_norm_fem import values_at_quadrature_nodes

def h1_norm_function_fem(func_u: Callable[[ndarray], ndarray] | ndarray,
                         func_v: Callable[[ndarray], ndarray] | ndarray,
                         der_func_u: Callable[[ndarray], ndarray] | ndarray,
                         der_func_v: Callable[[ndarray], ndarray] | ndarray,
                         spatial: SpatialDiscParameters,
                         quadrature: Quadrature) -> float:

    r"""
    The H^1(a,b) norm error between two given functions u and v with respect to a FEM:

    ||u-v||_{1,(a,b)} = \sqrt{\int_a^b (u(x)-v(x))^2 dx + \int_a^b (u'(x)-v'(x))^2 dx}.

    If u and/or v are elements of the FEM space V^0_h \subset H^1_0(a,b), func_u and func_v are ndarrays, that store
    the evaluation of u(\chi_i^{-1}) and/or v(\chi_i^{-1}) at the Gauss-Lobatto (GL) nodes x, where \chi_i:[x_{i-1},x_i] \to [-1,1],
    i=1,...,N, denote affine-linear transformations from the spatial subintervals to [-1,1]. The same holds for der_func_u and der_func_v.

    x_0,...,x_N denote the nodes of the spatial mesh. x_0,...,x_N are not related to the GL nodes x.

    Args:
        func_u (Callable[[ndarray], ndarray] | ndarray): The function u; or the function u(\chi_i^{-1})
            evaluated at x for each i=1,...,N.
        func_v (Callable[[ndarray], ndarray] | ndarray): The function v; or the function v(\chi_i^{-1})
            evaluated at x for each i=1,...,N.
        der_func_u (Callable[[ndarray], ndarray] | ndarray): The function u'; or the function u'(\chi_i^{-1})
            evaluated at x for each i=1,...,N.
        der_func_v (Callable[[ndarray], ndarray] | ndarray): The function v'; or the function v'(\chi_i^{-1})
            evaluated at x for each i=1,...,N.
        spatial (SpatialDiscParameters): Spatial discretization parameters;
            provides the spatial mesh `Delta` and the spatial step size `h`.
        quadrature (Quadrature): Quadrature rule; provides the Gauss-Lobatto
            nodes and weights used for numerical integration.

    Returns:
        float: ||u-v||_{1,(a,b)} = \sqrt{\int_a^b (u(x)-v(x))^2 dx + \int_a^b (u'(x)-v'(x))^2 dx}.
    """

    h = spatial.h
    assert h is not None, "spatial must provide h for a FEM discretization."

    _, w = quadrature.nodes_and_weights

    # The functions u(\chi_i^{-1}), v(\chi_i^{-1}), u'(\chi_i^{-1}) and v'(\chi_i^{-1}) evaluated at the
    # Gauss-Lobatto nodes for each i=1,...,N.
    u_v = values_at_quadrature_nodes(func_u, spatial, quadrature)
    v_v = values_at_quadrature_nodes(func_v, spatial, quadrature)
    der_u_v = values_at_quadrature_nodes(der_func_u, spatial, quadrature)
    der_v_v = values_at_quadrature_nodes(der_func_v, spatial, quadrature)

    # Approximation of \int_{-1}^1 ((u-v)(\chi_i^{-1}(x)))^2 dx + \int_{-1}^1 ((u'-v')(\chi_i^{-1}(x)))^2 dx, i=1,...,N
    h1_v = np.dot((u_v-v_v)**2, w) + np.dot((der_u_v-der_v_v)**2, w)

    # Approximation of \sqrt{\sum_{i=1}^N h_i/2 (\int_{-1}^1 ((u-v)(\chi_i^{-1}(x)))^2 dx + \int_{-1}^1 ((u'-v')(\chi_i^{-1}(x)))^2 dx)}
    return float(np.sqrt(np.dot(h/2, h1_v)))
