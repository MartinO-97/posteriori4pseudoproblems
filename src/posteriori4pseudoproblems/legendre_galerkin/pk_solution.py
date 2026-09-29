import numpy as np

from numpy import ndarray
from ..discretization_dataclasses import SpatialDiscParameters
from .legendre_fem_reference_functions import psi_l, psi_r, integrated_legendre

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
