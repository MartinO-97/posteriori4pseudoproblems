from typing import Callable

import numpy as np
from numpy import ndarray

from ..discretization_dataclasses import SpatialDiscParameters
from ..legendre_galerkin import evaluate_pk_solution


def compute_max_norm(sol_vector: ndarray,
                     spatial_disc_data: SpatialDiscParameters,
                     func: Callable[[ndarray], ndarray] | None = None) -> float:

    r""" Maximum-norm of a P_k-FEM solution, or of its error against a
    reference function, approximated at spatial_disc_data.points_per_interval
    equidistant points on every spatial mesh interval.

    Args:
        sol_vector (ndarray): Coefficients of the FEM solution, as e.g.
            returned by `backward_euler_fem_1d` or `bdf_fem_1d`.
        spatial_disc_data (SpatialDiscParameters): Spatial discretization
            parameters; provides N, Delta, k and points_per_interval for
            the FEM discretization.
        func (Callable[[ndarray], ndarray] | None): If given, evaluated at
            the same points as the FEM solution and the max-norm error
            between the two is returned instead of the plain max-norm of
            the FEM solution. Takes the evaluation points x (shape
            (n_points,)) and returns the function values at these points
            (same shape).

    Returns:
        float: The maximum norm of the FEM solution, or, if `func` is
            given, of its error against `func`.
    """

    N = spatial_disc_data.N
    Delta = spatial_disc_data.Delta
    points_per_interval = spatial_disc_data.points_per_interval
    assert N is not None and Delta is not None and points_per_interval is not None

    lambda_1_local = np.linspace(0.0, 1.0, points_per_interval)
    lambda_0_local = 1.0 - lambda_1_local

    elements = np.repeat(np.arange(1, N + 1), points_per_interval)
    lambda_1 = np.tile(lambda_1_local, N)
    lambda_0 = np.tile(lambda_0_local, N)
    bary = np.vstack((lambda_0, lambda_1))

    u_h = evaluate_pk_solution(sol_vector, elements, bary, spatial_disc_data)

    if func is None:
        return float(np.max(np.abs(u_h)))

    x_eval = lambda_0 * Delta[elements - 1] + lambda_1 * Delta[elements]
    return float(np.max(np.abs(u_h - func(x_eval))))
