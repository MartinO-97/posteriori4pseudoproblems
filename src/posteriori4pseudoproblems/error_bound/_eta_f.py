import numpy as np

from ..interval_transformation import interval_transformation as int_mapp
from ..pseudo_parabolic_pde_class import PseudoParabolicPDE
from ..discretization_dataclasses import TemporalDiscParameters, SpatialDiscParameters
from ..quadrature import Quadrature
from ..utilities import compute_divided_differences, use_modified_horner
from ..norms import l2_norm_function_fem
from ._semigroup_bounds import _sigma_j


def _compute_eta_f(pde: PseudoParabolicPDE,
                   temporal_disc_data: TemporalDiscParameters,
                   spatial_disc_data: SpatialDiscParameters,
                   quadrature: Quadrature,
                   j: int) -> float:

    r""" The component \eta_f on the subinterval I_j = [t_{j-1}, t_j]:

        \eta^j_f = \sigma_j/c_a \int_{t_{j-1}}^{t_j} || (f - \tilde{f})(s) ||_{0,\Omega} ds,

    where \tilde{f} denotes the linear interpolant of f at t_{j-1} and t_j.
    It is the same for the L^2 and the H^1 estimator.

    The temporal integral is approximated by the Simpson rule, i.e. Gauss-Lobatto
    with 3 nodes. Since f and \tilde{f} coincide at t_{j-1} and t_j, only the
    interior node, the midpoint of I_j, has to be considered.

    Args:
        pde (PseudoParabolicPDE): The pseudo-parabolic PDE, providing the
            source term F, the final time T and the generic constants.
        temporal_disc_data (TemporalDiscParameters): Temporal discretization
            parameters, providing the temporal mesh t_0,...,t_M.
        spatial_disc_data (SpatialDiscParameters): Spatial discretization
            parameters; provides Delta and h.
        quadrature (Quadrature): Quadrature rule; provides the nodes and
            weights used for the spatial integration.
        j (int): Index of the time level t_j.

    Returns:
        float: The contribution of I_j to the component eta_f.
    """

    temporal_mesh = temporal_disc_data.temporal_mesh
    Delta = spatial_disc_data.Delta
    assert temporal_mesh is not None, "temporal_disc_data must provide temporal_mesh."
    assert Delta is not None, "spatial_disc_data must provide Delta."

    _, c_a, _, _, _, _, _, _ = pde.generic_constants

    t_jm1, t_j = temporal_mesh[[j-1, j]]
    tau_j = t_j - t_jm1
    t_values = np.array([t_jm1, t_j])

    # The reference array
    ref_array = np.array([-1,1])

    # The interior Gauss-Lobatto node and its weight to approximate the temporal integral
    gl_nodes = np.array([0])
    gl_weights = np.array([4/3])

    # The interior Gauss-Lobatto node mapped to [t_{j-1}, t_j]
    gl_nodes_mapped = int_mapp(gl_nodes, ref_array, t_values)

    # The spatial quadrature nodes of every spatial subinterval
    x, _ = quadrature.nodes_and_weights
    x_v = int_mapp(x, ref_array, np.array([Delta[:-1], Delta[1:]]).T)

    def evaluate_F(t: float) -> np.ndarray:
        xt_points = np.column_stack((x_v.ravel(), np.full(x_v.size, t)))
        return pde.func_F(xt_points).reshape(x_v.shape)

    # f(t_{j-1}) and f(t_j) evaluated at the spatial quadrature nodes
    f_values = np.stack([evaluate_F(t_jm1), evaluate_F(t_j)], axis=0)

    # The divided differences f[t_j] and f[t_{j-1}, t_j] of the linear interpolant \tilde{f}
    divided_diff = compute_divided_differences(t_values, f_values)

    # \tilde{f} evaluated at the interior Gauss-Lobatto node
    tilde_f_values = use_modified_horner(gl_nodes_mapped, t_values, divided_diff)

    # || (f-\tilde{f})(gl_1) ||_{0,\Omega}
    l2_norm_gl_1 = l2_norm_function_fem(evaluate_F(gl_nodes_mapped[0]), tilde_f_values[0],
                                        spatial_disc_data, quadrature)

    # \int_{t_{j-1}}^{t_j} || (f - \tilde{f})(s) ||_{0,\Omega} ds
    l2_integral = tau_j/2 * gl_weights[0] * l2_norm_gl_1

    return _sigma_j(pde, t_j, t_jm1)/c_a * l2_integral
