import numpy as np

from ..pseudo_parabolic_pde_class import PseudoParabolicPDE
from ..discretization_dataclasses import TemporalDiscParameters, SpatialDiscParameters
from ..quadrature import Quadrature
from ..legendre_galerkin import (PkLegendreFEM, evaluate_pk_solution_at_quadrature_nodes,
                                 evaluate_pk_derivative_at_quadrature_nodes)
from ..time_discretization_methods import FemTimeSteppingParameters
from ..norms import h1_norm_function_fem
from ._semigroup_bounds import _sigma_j


def _compute_eta_delta_psi(pde: PseudoParabolicPDE,
                           temporal_disc_data: TemporalDiscParameters,
                           spatial_disc_data: SpatialDiscParameters,
                           quadrature: Quadrature,
                           ref_functions: PkLegendreFEM,
                           fem_time_stepping_parameters: FemTimeSteppingParameters,
                           norm_used: str,
                           j: int) -> float:

    r""" The component \eta_{\delta\psi} on the subinterval I_j = [t_{j-1}, t_j]:

        \eta^j_{\delta\psi} = \sigma_j \chi_j ||\delta_t \psi^j_h||_{1,\Omega}       ("h1"),
        \eta^j_{\delta\psi} = C_a \sigma_j \chi_j ||\delta_t \psi^j_h||_{1,\Omega}   ("l2"),

    with \chi_j = min{\tau_j^2/4, C_c/c_a \tau_j^3/12}.

    Args:
        pde (PseudoParabolicPDE): The pseudo-parabolic PDE, providing the
            final time T and the generic constants.
        temporal_disc_data (TemporalDiscParameters): Temporal discretization
            parameters, providing the temporal mesh t_0,...,t_M.
        spatial_disc_data (SpatialDiscParameters): Spatial discretization
            parameters; provides N, k and h.
        quadrature (Quadrature): Quadrature rule; provides the nodes and
            weights used for numerical integration.
        ref_functions (PkLegendreFEM): Reference functions of the P_k-FEM,
            evaluated at the quadrature nodes.
        fem_time_stepping_parameters (FemTimeSteppingParameters): State of
            the FEM time-stepping scheme; provides \psi^{j-1}_h, \psi^j_h as
            the last two entries of `prev_psi`.
        norm_used (str): The norm the estimator bounds the error in: "l2"
            or "h1".
        j (int): Index of the time level t_j.

    Returns:
        float: The contribution of I_j to the component eta_delta_psi.
    """

    temporal_mesh = temporal_disc_data.temporal_mesh
    prev_psi = fem_time_stepping_parameters.prev_psi
    assert temporal_mesh is not None, "temporal_disc_data must provide temporal_mesh."
    assert prev_psi is not None and prev_psi.shape[0] >= 2, \
        "fem_time_stepping_parameters must provide psi^{j-1}_h and psi^j_h."

    C_a, c_a, C_c, _, _, _, _, _ = pde.generic_constants

    t_jm1, t_j = temporal_mesh[[j-1, j]]
    tau_j = t_j - t_jm1

    # The value \chi_j
    chi_j = min(tau_j**2/4, C_c/c_a * tau_j**3/12)

    # \delta_t \psi^j_h and its spatial derivative
    coeffs = (prev_psi[-1] - prev_psi[-2])/tau_j
    delta_psi_v = evaluate_pk_solution_at_quadrature_nodes(coeffs, spatial_disc_data, ref_functions)
    der_delta_psi_v = evaluate_pk_derivative_at_quadrature_nodes(coeffs, spatial_disc_data, ref_functions)

    h1_value = h1_norm_function_fem(np.zeros_like(delta_psi_v), delta_psi_v, np.zeros_like(der_delta_psi_v),
                                    der_delta_psi_v, spatial_disc_data, quadrature)

    value = _sigma_j(pde, t_j, t_jm1) * chi_j * h1_value

    if norm_used == "l2":
        value *= C_a

    return value
