import numpy as np

from numpy import ndarray
from ..pseudo_parabolic_pde_class import PseudoParabolicPDE
from ..quadrature import Quadrature
from ..legendre_galerkin import SpectralLegendre
from ..discretization_dataclasses import TemporalDiscParameters
from ..projection_operators import project_onto_spectral_basis
from ..time_discretization_methods import SpectralTimeSteppingParameters, dg_two_spectral, dg_three_spectral

_M_REF = {"dg2": 512, "dg3": 128}


def compute_reference_solution(pde: PseudoParabolicPDE,
                               quadrature: Quadrature,
                               ref_functions: SpectralLegendre,
                               mass: ndarray,
                               matrix_L: ndarray,
                               dg2_or_dg3: str,
                               T: float) -> ndarray:

    r""" Reference solution u(.,T), represented by spectral coefficients,
    computed with a dG(2) or dG(3) time discretization (`dg2_or_dg3`, one
    of "dg2", "dg3") on a uniform temporal mesh with M_REF subintervals --
    M_REF = 512 for dG(2), M_REF = 128 for dG(3).

    Args:
        pde (PseudoParabolicPDE): The pseudo-parabolic PDE, providing the initial
            condition u0.
        quadrature (Quadrature): Quadrature rule; provides the nodes and
            weights used for numerical integration.
        ref_functions (SpectralLegendre): Reference functions of the
            spectral ansatz space, evaluated at the quadrature nodes.
        mass (ndarray): Mass matrix, e.g. as returned by
            `assemble_coeff_matrices_spectral`.
        matrix_L (ndarray): Matrix representing the operator L, e.g. as
            returned by `assemble_coeff_matrices_spectral`.
        dg2_or_dg3 (str): Which dG method to compute the reference
            solution with: "dg2" or "dg3".
        T (float): The final time.

    Returns:
        ndarray: The reference solution's spectral coefficients at t=T.

    Raises:
        ValueError: If `dg2_or_dg3` is neither "dg2" nor "dg3".
    """

    if dg2_or_dg3 == "dg2":
        step = dg_two_spectral
    elif dg2_or_dg3 == "dg3":
        step = dg_three_spectral
    else:
        raise ValueError(f'dg2_or_dg3 must be "dg2" or "dg3", got {dg2_or_dg3!r}.')

    M_ref = _M_REF[dg2_or_dg3]

    c0 = project_onto_spectral_basis(pde.func_u0, pde, quadrature, ref_functions, mass)

    temporal_disc_data = TemporalDiscParameters(disc_type=dg2_or_dg3, M=M_ref,
                                                temporal_mesh=np.linspace(0.0, T, M_ref + 1))
    stepping_params = SpectralTimeSteppingParameters(disc_type=dg2_or_dg3, num_prev=1, mass=mass, matrix_L=matrix_L)
    stepping_params.update(c0)

    sol_vector = c0
    for j in range(1, M_ref + 1):
        sol_vector = step(pde, temporal_disc_data, stepping_params, quadrature, ref_functions, j)

    return sol_vector
