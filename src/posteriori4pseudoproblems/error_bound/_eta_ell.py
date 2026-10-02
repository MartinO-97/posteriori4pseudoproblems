import numpy as np

from ..pseudo_parabolic_pde_class import PseudoParabolicPDE
from ..discretization_dataclasses import TemporalDiscParameters, SpatialDiscParameters
from ..quadrature import Quadrature
from ..legendre_galerkin import (PkLegendreFEM, evaluate_pk_solution_at_quadrature_nodes,
                                 evaluate_pk_derivative_at_quadrature_nodes)
from ..time_discretization_methods import FemTimeSteppingParameters
from ..norms import h1_norm_function_fem
from ._semigroup_bounds import _sigma_j, _mu_j


def _compute_eta_ell(pde: PseudoParabolicPDE,
                   temporal_disc_data: TemporalDiscParameters,
                   spatial_disc_data: SpatialDiscParameters,
                   higher_order_spatial_disc_data: SpatialDiscParameters,
                   quadrature: Quadrature,
                   ref_functions: PkLegendreFEM,
                   higher_order_ref_functions: PkLegendreFEM,
                   fem_time_stepping_parameters: FemTimeSteppingParameters,
                   higher_order_fem_time_stepping_parameters: FemTimeSteppingParameters,
                   norm_used: str,
                   j: int) -> float:

    r""" The component \eta_\ell on the subinterval I_j = [t_{j-1}, t_j]:

        \eta^j_R = \sigma_j (\tau_j \eta^j_{ell,1/2} + \tau_j^2/2 \eta^j_{ell,\delta_t})                                  ("h1"),
        \eta^j_R = C_a C_I |||L^{-1}|||_{0,2} \mu_j max_i h_i (\tau_j \eta^j_{ell,1/2} + \tau_j^2/2 \eta^j_{ell,\delta_t})   ("l2"),

    where the elliptic errors are approximated by means of a higher-order
    approximation v_h:

        \eta^j_{ell,1/2}    = ||(u^j_h + u^{j-1}_h)/2 - (v^j_h + v^{j-1}_h)/2||_{1,\Omega},
        \eta^j_{ell,\delta_t} = ||\delta_t u^j_h - \delta_t v^j_h||_{1,\Omega}.

    Args:
        pde (PseudoParabolicPDE): The pseudo-parabolic PDE, providing the
            final time T and the generic constants.
        temporal_disc_data (TemporalDiscParameters): Temporal discretization
            parameters, providing the temporal mesh t_0,...,t_M.
        spatial_disc_data (SpatialDiscParameters): Spatial discretization
            parameters for the actual approximation u_h.
        higher_order_spatial_disc_data (SpatialDiscParameters): Spatial
            discretization parameters for the higher-order approximation
            v_h; same N, Delta and h as `spatial_disc_data`, only the
            polynomial degree k differs.
        quadrature (Quadrature): Quadrature rule; provides the nodes and
            weights used for numerical integration.
        ref_functions (PkLegendreFEM): Reference functions of the P_k-FEM of
            u_h, evaluated at the quadrature nodes.
        higher_order_ref_functions (PkLegendreFEM): Reference functions of
            the P_k-FEM of v_h, evaluated at the quadrature nodes.
        fem_time_stepping_parameters (FemTimeSteppingParameters): State of
            the FEM time-stepping scheme of u_h; provides u^{j-1}_h, u^j_h
            as the last two entries of `prev_solutions`.
        higher_order_fem_time_stepping_parameters (FemTimeSteppingParameters):
            State of the FEM time-stepping scheme of v_h; provides v^{j-1}_h,
            v^j_h as the last two entries of `prev_solutions`.
        norm_used (str): The norm the estimator bounds the error in: "l2"
            or "h1".
        j (int): Index of the time level t_j.

    Returns:
        float: The contribution of I_j to the component eta_ell.
    """

    temporal_mesh = temporal_disc_data.temporal_mesh
    h = spatial_disc_data.h
    u = fem_time_stepping_parameters.prev_solutions
    v = higher_order_fem_time_stepping_parameters.prev_solutions
    assert temporal_mesh is not None, "temporal_disc_data must provide temporal_mesh."
    assert h is not None, "spatial_disc_data must provide h."
    assert u is not None and u.shape[0] >= 2, \
        "fem_time_stepping_parameters must provide u^{j-1}_h and u^j_h."
    assert v is not None and v.shape[0] >= 2, \
        "higher_order_fem_time_stepping_parameters must provide v^{j-1}_h and v^j_h."

    C_a, _, _, _, _, _, C_I, L_inverse = pde.generic_constants

    t_jm1, t_j = temporal_mesh[[j-1, j]]
    tau_j = t_j - t_jm1

    def h1_distance(u_coeffs: np.ndarray, v_coeffs: np.ndarray) -> float:
        u_v = evaluate_pk_solution_at_quadrature_nodes(u_coeffs, spatial_disc_data, ref_functions)
        der_u_v = evaluate_pk_derivative_at_quadrature_nodes(u_coeffs, spatial_disc_data, ref_functions)
        v_v = evaluate_pk_solution_at_quadrature_nodes(v_coeffs, higher_order_spatial_disc_data,
                                                       higher_order_ref_functions)
        der_v_v = evaluate_pk_derivative_at_quadrature_nodes(v_coeffs, higher_order_spatial_disc_data,
                                                             higher_order_ref_functions)
        return h1_norm_function_fem(u_v, v_v, der_u_v, der_v_v, spatial_disc_data, quadrature)

    # ||(u^j_h + u^{j-1}_h)/2 - (v^j_h + v^{j-1}_h)/2||_{1,\Omega}
    eta_ell_h = h1_distance((u[-1] + u[-2])/2, (v[-1] + v[-2])/2)

    # ||\delta_t u^j_h - \delta_t v^j_h||_{1,\Omega}
    eta_ell_delta = h1_distance((u[-1] - u[-2])/tau_j, (v[-1] - v[-2])/tau_j)

    elliptic_part = tau_j * eta_ell_h + tau_j**2/2 * eta_ell_delta

    if norm_used == "l2":
        return C_a * C_I * L_inverse * _mu_j(pde, t_j, t_jm1) * np.max(h) * elliptic_part

    return _sigma_j(pde, t_j, t_jm1) * elliptic_part
