import numpy as np

from typing import Callable
from numpy import ndarray
from scipy.sparse import csr_array
from scipy.sparse.linalg import spsolve
from ..interval_transformation import interval_transformation as int_mapp
from ..pseudo_parabolic_pde_class import PseudoParabolicPDE
from ..discretization_dataclasses import SpatialDiscParameters
from ..quadrature import Quadrature
from ..legendre_galerkin import PkLegendreFEM

def elliptic_projection_fem(func_u: Callable[[ndarray], ndarray],
                            func_der_u: Callable[[ndarray], ndarray],
                            pde: PseudoParabolicPDE,
                            spatial: SpatialDiscParameters,
                            quadrature: Quadrature,
                            ref_functions: PkLegendreFEM,
                            matrix_L: csr_array) -> ndarray:

    r"""
    The elliptic projection of a given function u into the FEM space V^0_h \subset H^1_0(a,b).

    Given an elliptic, bounded and coercive operator L: H^1_0(a,b) \to H^{-1}(a,b), which is given by

    Lu = -u'' + au

    the projection

    a(u_h, v_h) = a(u, v_h)     for all v_h \in V^0_h,

    is computed. Here, a() denotes the bilinear form associated with the operator L.

    Args:
        func_u (Callable[[ndarray], ndarray]): The function u \in H^1_0(a,b).
        func_der_u (Callable[[ndarray], ndarray]): The derivative u' of the function u.
        pde (PseudoParabolicPDE): The pseudo-parabolic PDE, providing the function a.
        spatial (SpatialDiscParameters): Spatial discretization parameters;
            provides the number of spatial subintervals `N`, the polynomial
            degree `k` of the P_k-FEM, the spatial mesh `Delta` and the
            spatial step size `h`.
        quadrature (Quadrature): Quadrature rule; provides the Gauss-Lobatto
            nodes and weights used for numerical integration.
        ref_functions (PkLegendreFEM): Shape functions \psi_L, \psi_R, integrated
            Legendre polynomials N_1,...,N_r and Legendre polynomials P_1,...,P_r,
            evaluated at the quadrature nodes.
        matrix_L (csr_array): Matrix subjected to the operator L with respect to V^0_h,
            e.g. as returned by `assemble_coeff_matrices_fem_1d`.

    Returns:
        ndarray: Coefficients of the elliptic projection u_h, of length (r+1)*N+1.
    """

    N = spatial.N
    Delta = spatial.Delta
    h = spatial.h

    assert N is not None and spatial.k is not None and Delta is not None and h is not None, \
        "spatial must provide N, k, Delta and h for a FEM discretization."

    r = spatial.k - 1

    x, w = quadrature.nodes_and_weights
    psi_l_v, psi_r_v, le_n_g, le_p_g = ref_functions.evaluated_reference_functions

    r"""Evaluate u(\chi_i^{-1}), u'(\chi_i^{-1}) and a(\chi_i^{-1}) in the Gauss-Lobatto nodes x, where
        \chi_i:[x_{i-1}, x_i] \to [-1,1] denote affine-linear transformations, i=1,...,N. """
    x_v = int_mapp(x, np.array([-1,1]), np.array([Delta[:-1], Delta[1:]]).T)
    u_value = func_u(x_v)
    der_u_value = func_der_u(x_v)
    a_value = pde.func_a(x_v)

    # ----------------------------------------------------------------------------------------------------
    # COMPUTE a(u, \phi_i), WHERE \phi_i DENOTE THE BASIS FUNCTIONS OF V^0_h, i=1,...,dim(V^0_h)
    # ----------------------------------------------------------------------------------------------------

    # Initialize result_vector subjected to a(u,\phi_i), and sol_vector, that stores the coefficients of u_h
    result_vector = np.zeros((r+1)*N-1)
    sol_vector = np.zeros((r+1)*N+1)

    # BASIS FUNCTIONS ASSOCIATED WITH THE SHAPE FUNCTIONS \psi_R AND \psi_L
    l = np.linspace(1,N-1,N-1, True, dtype = int)

    # (au, \phi_i)
    result_vector[(r+1)*l-1] = h[:-1]/2 * np.dot(u_value[:-1] * a_value[:-1] * psi_r_v, w) \
                                + h[1:]/2 * np.dot(u_value[1:] * a_value[1:] * psi_l_v, w)
    # (u', \phi'_i)
    result_vector[(r+1)*l-1] += 1/2 * np.dot(der_u_value[:-1], w) - 1/2 * np.dot(der_u_value[1:], w)

    # BASIS FUNCTIONS ASSOCIATED WITH THE INTEGRATED LEGENDRE POLYNOMIALS, N_1,...,N_r
    if r > 0:
        assert le_n_g is not None and le_p_g is not None, \
            "ref_functions has no (integrated) Legendre polynomials for r > 0."

        l = np.linspace(0,N-1,N, True, dtype = int)
        for j in range(0,r):

            # (au, \phi_i)
            result_vector[(r+1)*l+j] = h/2 * np.dot(le_n_g[j] * u_value * a_value, w)

            # (u', \phi'_i)
            result_vector[(r+1)*l+j] += np.dot(le_p_g[j] * der_u_value, w)

    sol_vector[1:-1] = np.asarray(spsolve(matrix_L, result_vector))

    return sol_vector
