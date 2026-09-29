import numpy as np

from numpy import ndarray
from ..interval_transformation import interval_transformation as int_mapp
from ..pseudo_parabolic_pde_class import PseudoParabolicPDE
from .legendre_fem_reference_functions import integrated_legendre, legendre

def evaluate_spectral_solution(coeffs : ndarray,
                               x : ndarray,
                               pde : PseudoParabolicPDE) -> ndarray:

    r""" Evaluates a spectral Galerkin solution

    U(x) = \sum_i coeffs[i] N_{i+1}(\chi(x)),

    where N_1,...,N_{dim_V} are the integrated Legendre polynomials used as
    ansatz functions and \chi maps the spatial domain of `pde` onto the
    reference interval [-1,1], at physical points x.

    Args:
        coeffs (ndarray): Coefficients of the spectral Galerkin solution,
            e.g. the `sol_vector` returned by `dg_two_spectral`.
        x (ndarray): Physical points, lying in the spatial domain of `pde`,
            where the solution shall be evaluated at.
        pde (PseudoParabolicPDE): The pseudo-parabolic PDE, providing the spatial domain
            [a,b].

    Returns:
        ndarray: The solution U evaluated at x.
    """

    a, b = pde.spatial_interval
    xi = int_mapp(x, np.array([a, b]), np.array([-1, 1]))

    result = np.zeros_like(xi, dtype=float)
    for i, c in enumerate(coeffs):
        result += c * integrated_legendre(i + 1, xi)

    return result


def evaluate_spectral_derivative(coeffs : ndarray,
                                 x : ndarray,
                                 pde : PseudoParabolicPDE) -> ndarray:

    r""" Evaluates the first derivative of a spectral Galerkin solution

    U'(x) = \chi'(x) \sum_i coeffs[i] P_{i+1}(\chi(x)),    \chi'(x) = 2/(b-a),

    where P_1,...,P_{dim_V} are the Legendre polynomials, i.e. the
    derivatives of the integrated Legendre polynomials N_1,...,N_{dim_V},
    and \chi maps the spatial domain [a,b] of `pde` onto the reference
    interval [-1,1], at physical points x.

    Args:
        coeffs (ndarray): Coefficients of the spectral Galerkin solution,
            e.g. the `sol_vector` returned by `dg_two_spectral`.
        x (ndarray): Physical points, lying in the spatial domain of `pde`,
            where the derivative shall be evaluated at.
        pde (PseudoParabolicPDE): The pseudo-parabolic PDE, providing the spatial domain
            [a,b].

    Returns:
        ndarray: The derivative U' evaluated at x.
    """

    a, b = pde.spatial_interval
    xi = int_mapp(x, np.array([a, b]), np.array([-1, 1]))

    result = np.zeros_like(xi, dtype=float)
    for i, c in enumerate(coeffs):
        result += c * legendre(i + 1, xi)

    return 2/(b-a) * result
