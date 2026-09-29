import numpy as np

from numpy import ndarray
from ...interval_transformation import interval_transformation as int_mapp
from ...pseudo_parabolic_pde_class import PseudoParabolicPDE
from ...discretization_dataclasses import TemporalDiscParameters
from ...quadrature import Quadrature
from ...legendre_galerkin import SpectralLegendre
from .spectral_time_stepping_parameters import SpectralTimeSteppingParameters
from numpy.linalg import solve

def dg_two_spectral(pde : PseudoParabolicPDE,
                    temporal : TemporalDiscParameters,
                    stepping_params : SpectralTimeSteppingParameters,
                    quadrature : Quadrature,
                    ref_functions : SpectralLegendre,
                    j : int) -> ndarray:

    r"""dG(2)-method for a spectral Galerkin discretization in a spatial domain [a,b].

    This function computes an approximation U^j of the solution u(t_j)
    to a pseudo-parabolic equation of the form

    L \partial_t u(t) + M u(t) = f(t),

    using the dG(2) method.

    Args:
        pde (PseudoParabolicPDE): The pseudo-parabolic PDE, providing the
            spatial domain [a,b] and the right-hand side F.
        temporal (TemporalDiscParameters): Temporal discretization parameters,
            providing the temporal mesh t_0,...,t_M.
        stepping_params (SpectralTimeSteppingParameters): State of the spectral
            time-stepping scheme; provides the matrices L and M
            and the previous approximation U^{j-1}. Updated in place with
            the newly computed approximation U^j.
        quadrature (Quadrature): Quadrature rule; provides the nodes and
            weights used for numerical integration.
        ref_functions (SpectralLegendre): Integrated Legendre polynomials,
            evaluated at the quadrature nodes.
        j (int): Index of the time level t_j to advance to; t_{j-1} is the
            previous time level.

    Returns:
        ndarray: Coefficients of the computed approximate solution U^j.
    """

    a, b = pde.spatial_interval
    dim_V = stepping_params.matrix_L.shape[0]

    assert temporal.temporal_mesh is not None, "temporal must provide temporal_mesh to advance the solution."
    t = temporal.temporal_mesh[j]
    tau_j = t - temporal.temporal_mesh[j-1]

    x, w = quadrature.nodes_and_weights
    le_n_g, _ = ref_functions.evaluated_reference_functions
    assert le_n_g is not None, "ref_functions has not been evaluated at the quadrature nodes."

    assert stepping_params.prev_solutions is not None, "stepping_params has no previous approximation U^{j-1} to advance from."
    sol_vector_s = stepping_params.prev_solutions[-1]

    # The reference interval [-1,1]
    ref_array = np.array([-1,1])


    # --------------------------------------------------------------------------------------------------------------
    # ASSOCIATED BUTCHER TABLEAU
    # --------------------------------------------------------------------------------------------------------------

    r"""The vectors of the Butcher tableau:

        c | A
        ------
          | b^T

    The exact values can be found in [1]. Note that the last row of A coincides with b^T.
    """

    # The vector c
    c = np.array([(4-np.sqrt(6))/10, (4+np.sqrt(6))/10, 1])

    # The matrix A
    A = np.array([[(88-7*np.sqrt(6))/360, (296-169*np.sqrt(6))/1800, (-2+3*np.sqrt(6))/225],
                  [(296+169*np.sqrt(6))/1800, (88+7*np.sqrt(6))/360, (-2-3*np.sqrt(6))/225],
                  [(16-np.sqrt(6))/36, (16+np.sqrt(6))/36, 1/9]])


    # --------------------------------------------------------------------------------------------------------------
    # COMPUTATION OF THE APPROXIMATE SOLUTION U^j
    # --------------------------------------------------------------------------------------------------------------

    # Assemble the coefficient matrix
    matrix_L = stepping_params.matrix_L
    matrix_M = stepping_params.matrix_M

    coef_matrix_above = np.hstack((matrix_L / tau_j + A[0,0] * matrix_M, A[0,1] * matrix_M, A[0,2] * matrix_M))
    coef_matrix_middle = np.hstack((A[1,0] * matrix_M, matrix_L/tau_j + A[1,1] * matrix_M, A[1,2] * matrix_M))
    coef_matrix_under = np.hstack((A[2,0] * matrix_M, A[2,1] * matrix_M, matrix_L/tau_j + A[2,2] * matrix_M))

    coef_matrix = np.vstack((coef_matrix_above,coef_matrix_middle))
    coef_matrix = np.vstack((coef_matrix,coef_matrix_under))

    # Assemble the right hand side
    x_v = int_mapp(x, ref_array, np.array([a, b]))
    xt_points_0 = np.column_stack((x_v, np.full(x_v.shape[0], t + (c[0]-1)*tau_j)))
    xt_points_1 = np.column_stack((x_v, np.full(x_v.shape[0], t + (c[1]-1)*tau_j)))
    xt_points_2 = np.column_stack((x_v, np.full(x_v.shape[0], t + (c[2]-1)*tau_j)))

    f_v0 = pde.func_F(xt_points_0)
    f_v1 = pde.func_F(xt_points_1)
    f_v2 = pde.func_F(xt_points_2)

    f_v0 = f_v0 * w
    f_v1 = f_v1 * w
    f_v2 = f_v2 * w

    b0 = (b-a)/2*np.dot(le_n_g, A[0,0] * f_v0 + A[0,1] * f_v1 + A[0,2] * f_v2) + np.dot(matrix_L / tau_j, sol_vector_s)
    b1 = (b-a)/2*np.dot(le_n_g, A[1,0] * f_v0 + A[1,1] * f_v1 + A[1,2] * f_v2) + np.dot(matrix_L / tau_j, sol_vector_s)
    b2 = (b-a)/2*np.dot(le_n_g, A[2,0] * f_v0 + A[2,1] * f_v1 + A[2,2] * f_v2) + np.dot(matrix_L / tau_j, sol_vector_s)

    result_vector = np.concatenate((b0, b1, b2))

    sol_vector = solve(coef_matrix, result_vector)[2*dim_V:]

    stepping_params.update(sol_vector)

    return sol_vector


# --------------------------------------------------------------------------------------------------------------
# BIBLIOGRAPHY
# --------------------------------------------------------------------------------------------------------------

# [1] K. Strehmel, R. Weiner, and H. Podhaisky;
#     Numerik gewöhnlicher Differentialgleichungen: nichtsteife, steife und differential-algebraische Gleichungen;
#     Studium, Springer Spektrum, Wiesbaden, 2.; 2012.
