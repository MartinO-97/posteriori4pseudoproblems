import numpy as np

from numpy import ndarray
from ..discretization_dataclasses import SpatialDiscParameters
from .pk_legendre_fem import PkLegendreFEM

def evaluate_pk_solution_at_quadrature_nodes(coeffs: ndarray,
                                             spatial: SpatialDiscParameters,
                                             ref_functions: PkLegendreFEM) -> ndarray:

    r""" Evaluates a P_k-FEM solution u_h(\chi_i^{-1}) at the quadrature nodes
    x for each i=1,...,N, where \chi_i:[x_{i-1},x_i] \to [-1,1] denote the
    affine-linear transformations from the spatial subintervals to [-1,1].

    On the mesh interval [x_{i-1}, x_i], i=1,...,N, the FEM solution reads

    u_h(\chi_i^{-1}(x)) = a_{i-1} \psi_L(x) + \sum_{l=1}^{r} b_{i,l} N_l(x) + a_i \psi_R(x),

    so only the reference functions \psi_L, \psi_R, N_1,...,N_r evaluated at
    the quadrature nodes are needed.

    Args:
        coeffs (ndarray): Coefficients of the FEM solution, a vector of
            length (r+1)*N+1, laid out interval by interval as
            [a_0, b_{1,1},...,b_{1,r}, a_1, b_{2,1},...,b_{2,r}, a_2, ..., a_N].
        spatial (SpatialDiscParameters): Spatial discretization parameters;
            provides the number of spatial subintervals `N` and the
            polynomial degree `k` of the P_k-FEM.
        ref_functions (PkLegendreFEM): Shape functions \psi_L, \psi_R and
            integrated Legendre polynomials N_1,...,N_r, evaluated at the
            quadrature nodes.

    Returns:
        ndarray: u_h(\chi_i^{-1}) evaluated at the quadrature nodes, of shape
            (N, number of quadrature nodes).
    """

    N = spatial.N
    assert N is not None and spatial.k is not None, "spatial must provide N and k for a FEM discretization."
    r = spatial.k - 1

    psi_l_v, psi_r_v, le_n_g, _ = ref_functions.evaluated_reference_functions
    assert psi_l_v is not None and psi_r_v is not None, \
        "ref_functions has not been evaluated at the quadrature nodes."

    # Consideration of the shape functions \psi_R and \psi_L
    l = np.arange(0, N + 1)
    u_h_v = coeffs[l[1:] * (r + 1), None] * psi_r_v + coeffs[l[:-1] * (r + 1), None] * psi_l_v

    # Consideration of the integrated Legendre polynomials
    if r > 0:
        assert le_n_g is not None, "ref_functions has no integrated Legendre polynomials for r > 0."
        l = np.arange(0, N)
        for mu in range(1, r + 1):
            u_h_v = u_h_v + coeffs[l * (r + 1) + mu, None] * le_n_g[mu - 1]

    return u_h_v


def evaluate_pk_derivative_at_quadrature_nodes(coeffs: ndarray,
                                               spatial: SpatialDiscParameters,
                                               ref_functions: PkLegendreFEM) -> ndarray:

    r""" Evaluates the first derivative u'_h(\chi_i^{-1}) of a P_k-FEM
    solution at the quadrature nodes x for each i=1,...,N, where
    \chi_i:[x_{i-1},x_i] \to [-1,1] denote the affine-linear transformations
    from the spatial subintervals to [-1,1].

    On the mesh interval [x_{i-1}, x_i], i=1,...,N, with h_i = x_i - x_{i-1},
    the derivative of the FEM solution reads

    u'_h(\chi_i^{-1}(x)) = (a_i - a_{i-1})/h_i + \sum_{l=1}^{r} 2/h_i b_{i,l} P_l(x),

    since N'_l = P_l, so only the Legendre polynomials P_1,...,P_r evaluated
    at the quadrature nodes are needed.

    Args:
        coeffs (ndarray): Coefficients of the FEM solution, laid out as for
            `evaluate_pk_solution_at_quadrature_nodes`.
        spatial (SpatialDiscParameters): Spatial discretization parameters;
            provides the number of spatial subintervals `N`, the polynomial
            degree `k` of the P_k-FEM and the spatial step size `h`.
        ref_functions (PkLegendreFEM): Shape functions \psi_L, \psi_R and
            Legendre polynomials P_1,...,P_r, evaluated at the quadrature
            nodes.

    Returns:
        ndarray: u'_h(\chi_i^{-1}) evaluated at the quadrature nodes, of shape
            (N, number of quadrature nodes).
    """

    N = spatial.N
    h = spatial.h
    assert N is not None and spatial.k is not None and h is not None, \
        "spatial must provide N, k and h for a FEM discretization."
    r = spatial.k - 1

    psi_l_v, _, _, le_p_g = ref_functions.evaluated_reference_functions
    assert psi_l_v is not None, "ref_functions has not been evaluated at the quadrature nodes."

    # Consideration of the shape functions \psi_R and \psi_L
    l = np.arange(0, N + 1)
    der_u_h_v = ((coeffs[l[1:] * (r + 1)] - coeffs[l[:-1] * (r + 1)]) / h)[:, None] * np.ones((N, psi_l_v.shape[0]))

    # Consideration of the integrated Legendre polynomials
    if r > 0:
        assert le_p_g is not None, "ref_functions has no Legendre polynomials for r > 0."
        l = np.arange(0, N)
        for mu in range(1, r + 1):
            der_u_h_v = der_u_h_v + (2 / h * coeffs[l * (r + 1) + mu])[:, None] * le_p_g[mu - 1]

    return der_u_h_v
