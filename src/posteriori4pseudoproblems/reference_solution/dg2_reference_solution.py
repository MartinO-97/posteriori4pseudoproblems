import numpy as np

from numpy import ndarray
from ..pseudo_parabolic_pde_class import PseudoParabolicPDE
from ..quadrature import Quadrature
from ..legendre_galerkin import SpectralLegendre
from ..discretization_dataclasses import TemporalDiscParameters
from ..projection_operators import project_onto_spectral_basis
from ..time_discretization_methods import SpectralTimeSteppingParameters, dg_two_spectral

_M_REF = 128


def compute_reference_solution(pde: PseudoParabolicPDE,
                               quadrature: Quadrature,
                               ref_functions: SpectralLegendre,
                               mass: ndarray,
                               matrix_L: ndarray,
                               matrix_M: ndarray,
                               T: float) -> ndarray:

    r""" Reference solution u(.,T), represented by spectral coefficients,
    computed with a dG(2) time discretization on a uniform temporal mesh
    with M_REF = 128 subintervals.

    Args:
        pde (PseudoParabolicPDE): The pseudo-parabolic PDE, providing the
            initial condition u0.
        quadrature (Quadrature): Quadrature rule; provides the nodes and
            weights used for numerical integration.
        ref_functions (SpectralLegendre): Reference functions of the
            spectral ansatz space, evaluated at the quadrature nodes.
        mass (ndarray): Mass matrix, used for the L^2 projection of the
            initial condition, e.g. as returned by
            `assemble_coeff_matrices_spectral`.
        matrix_L (ndarray): Matrix representing the operator L, e.g. as
            returned by `assemble_coeff_matrices_spectral`.
        matrix_M (ndarray): Matrix representing the operator M, e.g. as
            returned by `assemble_coeff_matrices_spectral`.
        T (float): The final time.

    Returns:
        ndarray: The reference solution's spectral coefficients at t=T.
    """

    c0 = project_onto_spectral_basis(pde.func_u0, pde, quadrature, ref_functions, mass)

    temporal_disc_data = TemporalDiscParameters(disc_type="dg2", M=_M_REF,
                                                temporal_mesh=np.linspace(0.0, T, _M_REF + 1))
    stepping_params = SpectralTimeSteppingParameters(disc_type="dg2", num_prev=1,
                                                     matrix_L=matrix_L, matrix_M=matrix_M)
    stepping_params.update(c0)

    sol_vector = c0
    for j in range(1, _M_REF + 1):
        sol_vector = dg_two_spectral(pde, temporal_disc_data, stepping_params, quadrature, ref_functions, j)

    return sol_vector
