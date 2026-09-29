import numpy as np

from typing import Callable
from numpy import ndarray
from ..interval_transformation import interval_transformation as int_mapp
from ..pseudo_parabolic_pde_class import PseudoParabolicPDE
from ..quadrature import Quadrature
from ..legendre_galerkin import SpectralLegendre

def project_onto_spectral_basis(func : Callable[[ndarray], ndarray],
                                pde : PseudoParabolicPDE,
                                quadrature : Quadrature,
                                ref_functions : SpectralLegendre,
                                mass : ndarray) -> ndarray:

    r""" L2-projects `func` onto the spectral Galerkin ansatz space spanned
    by the integrated Legendre polynomials N_1,...,N_{dim_V}, by solving

    mass @ coeffs = rhs,   rhs_i = \int_a^b func(x) N_i(x) dx.

    Args:
        func (Callable[[ndarray], ndarray]): The function to project, e.g.
            the initial condition u_0.
        pde (PseudoParabolicPDE): The pseudo-parabolic PDE, providing the spatial domain
            [a,b].
        quadrature (Quadrature): Quadrature rule; provides the Gauss-Lobatto
            nodes and weights used for numerical integration.
        ref_functions (SpectralLegendre): Integrated Legendre polynomials,
            evaluated at the quadrature nodes.
        mass (ndarray): Mass matrix of the spectral Galerkin ansatz space.

    Returns:
        ndarray: Coefficients of the L2 projection of `func`.
    """

    a, b = pde.spatial_interval

    x, w = quadrature.nodes_and_weights
    le_n_g, _ = ref_functions.evaluated_reference_functions
    assert le_n_g is not None, "ref_functions has not been evaluated at the quadrature nodes."

    x_v = int_mapp(x, np.array([-1, 1]), np.array([a, b]))
    rhs = (b - a) / 2 * le_n_g.dot(func(x_v) * w)

    return np.linalg.solve(mass, rhs)
