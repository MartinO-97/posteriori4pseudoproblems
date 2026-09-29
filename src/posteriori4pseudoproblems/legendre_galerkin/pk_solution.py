import numpy as np

from numpy import ndarray
from ..discretization_dataclasses import SpatialDiscParameters
from .legendre_fem_reference_functions import psi_l, psi_r, integrated_legendre
from .pk_legendre_fem import PkLegendreFEM

def evaluate_pk_solution(coeffs: ndarray,
                         elements: ndarray,
                         bary: ndarray,
                         spatial: SpatialDiscParameters) -> ndarray:

    r""" Evaluates a P_k-FEM solution, based on the integrated Legendre
    polynomials, at points given by their barycentric coordinates.

    On the mesh interval [x_{i-1}, x_i], i=1,...,N, the FEM solution reads

    u_h(x) = a_{i-1} \psi_L(\xi_i(x)) + \sum_{l=1}^{r} b_{i,l} N_l(\xi_i(x)) + a_i \psi_R(\xi_i(x)),

    where \psi_L, \psi_R are the 'left'/'right' hat functions, N_l the
    l-th integrated Legendre polynomial, r the maximum degree of the
    integrated Legendre polynomials used (r=0 corresponds to a P_1-FEM)
    and \xi_i the affine map of [x_{i-1}, x_i] onto the reference interval
    [-1,1].

    A point x lying in [x_{i-1}, x_i] is given by its barycentric
    coordinates (\lambda_0, \lambda_1), \lambda_0 + \lambda_1 = 1, such
    that x = \lambda_0 x_{i-1} + \lambda_1 x_i; equivalently
    \xi_i(x) = \lambda_1 - \lambda_0.

    Args:
        coeffs (ndarray): Coefficients of the FEM solution, as e.g.
            returned by `backward_euler_fem_1d` or `bdf_fem_1d`; a vector
            of length (r+1)*N+1, laid out interval by interval as
            [a_0, b_{1,1},...,b_{1,r}, a_1, b_{2,1},...,b_{2,r}, a_2, ..., a_N].
        elements (ndarray): Interval number i for every evaluation point,
            such that the point lies in [x_{i-1}, x_i], i=1,...,N.
        bary (ndarray): Barycentric coordinates (\lambda_0, \lambda_1) of
            every evaluation point with respect to its interval, given as
            an array of shape (2, n_points).
        spatial (SpatialDiscParameters): Spatial discretization parameters;
            provides the polynomial degree `k` of the P_k-FEM.

    Returns:
        ndarray: The FEM solution u_h evaluated at the given points.
    """

    assert spatial.k is not None, "spatial must provide k for a FEM discretization."
    r = spatial.k - 1

    elements = np.asarray(elements)
    lambda_0, lambda_1 = bary

    xi = lambda_1 - lambda_0

    base = (elements - 1) * (r + 1)
    result = coeffs[base] * psi_l(xi) + coeffs[base + r + 1] * psi_r(xi)

    for l in range(1, r + 1):
        result = result + coeffs[base + l] * integrated_legendre(l, xi)

    return result


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
