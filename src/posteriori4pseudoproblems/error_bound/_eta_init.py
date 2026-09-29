import numpy as np

from ..interval_transformation import interval_transformation as int_mapp
from ..pseudo_parabolic_pde_class import PseudoParabolicPDE
from ..discretization_dataclasses import SpatialDiscParameters
from ..quadrature import Quadrature
from ..legendre_galerkin import (PkLegendreFEM, evaluate_pk_solution_at_quadrature_nodes,
                                 evaluate_pk_derivative_at_quadrature_nodes)
from ..time_discretization_methods import FemTimeSteppingParameters
from ..norms import h1_norm_function_fem
from ._semigroup_bounds import _eta_S_1, _eta_S_star_2


def _compute_eta_init(pde: PseudoParabolicPDE,
                      spatial_disc_data: SpatialDiscParameters,
                      quadrature: Quadrature,
                      ref_functions: PkLegendreFEM,
                      fem_time_stepping_parameters: FemTimeSteppingParameters,
                      norm_used: str) -> float:

    r""" The component \eta_{init}, capturing the error of the initial
    approximation u^0_h of u_0:

        \eta_{init} = \eta_{S,1}(T) ||u_0 - u^0_h||_{1,\Omega}                                  ("h1"),
        \eta_{init} = C_a C_I |||L^{-1}|||_{0,2} max_i h_i \eta_{S_*,2}(T) ||u_0 - u^0_h||_{1,\Omega}   ("l2").

    Args:
        pde (PseudoParabolicPDE): The pseudo-parabolic PDE, providing u_0, u'_0,
            the final time T and the generic constants.
        spatial_disc_data (SpatialDiscParameters): Spatial discretization
            parameters; provides N, k, Delta and h.
        quadrature (Quadrature): Quadrature rule; provides the nodes and
            weights used for numerical integration.
        ref_functions (PkLegendreFEM): Reference functions of the P_k-FEM,
            evaluated at the quadrature nodes.
        fem_time_stepping_parameters (FemTimeSteppingParameters): State of
            the FEM time-stepping scheme; its latest approximation
            `prev_solutions[-1]` must be u^0_h.
        norm_used (str): The norm the estimator bounds the error in: "l2"
            or "h1".

    Returns:
        float: The component eta_init.
    """

    Delta = spatial_disc_data.Delta
    h = spatial_disc_data.h
    assert Delta is not None and h is not None, "spatial_disc_data must provide Delta and h."
    assert fem_time_stepping_parameters.prev_solutions is not None, \
        "fem_time_stepping_parameters has no initial approximation u^0_h."

    C_a, _, _, _, _, _, C_I, L_inverse = pde.generic_constants
    T = pde.final_time

    # u_0 and u'_0 evaluated at the quadrature nodes of every spatial subinterval
    x, _ = quadrature.nodes_and_weights
    x_v = int_mapp(x, np.array([-1,1]), np.array([Delta[:-1], Delta[1:]]).T)
    u0_v = pde.func_u0(x_v)
    der_u0_v = pde.func_der_u0(x_v)
    assert der_u0_v is not None, "pde must provide the derivative u'_0 of the initial condition."

    # u^0_h and its derivative evaluated at the quadrature nodes
    sol_vector = fem_time_stepping_parameters.prev_solutions[-1]
    u0_h_v = evaluate_pk_solution_at_quadrature_nodes(sol_vector, spatial_disc_data, ref_functions)
    der_u0_h_v = evaluate_pk_derivative_at_quadrature_nodes(sol_vector, spatial_disc_data, ref_functions)

    h1_error = h1_norm_function_fem(u0_v, u0_h_v, der_u0_v, der_u0_h_v, spatial_disc_data, quadrature)

    if norm_used == "l2":
        return C_a * C_I * L_inverse * np.max(h) * _eta_S_star_2(pde, T) * h1_error

    return _eta_S_1(pde, T) * h1_error
