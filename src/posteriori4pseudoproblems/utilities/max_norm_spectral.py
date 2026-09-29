from typing import Callable

import numpy as np
from numpy import ndarray

from ..discretization_dataclasses import SpatialDiscParameters
from ..parabolicPDEclass import ParabolicPDE
from ..legendre_galerkin import evaluate_spectral_solution


def compute_max_norm_spectral(sol_vector: ndarray,
                              spatial_disc_data: SpatialDiscParameters,
                              pde: ParabolicPDE,
                              func: Callable[[ndarray], ndarray] | None = None) -> float:

    r""" Maximum-norm of a spectral Galerkin solution, or of its error
    against a reference function, approximated at
    spatial_disc_data.points_per_interval equidistant points across the
    whole spatial domain -- the spectral counterpart of `compute_max_norm`
    (there, one FEM mesh interval at a time; here, spectral has no mesh,
    so all points lie in the single global domain `pde.spatial_interval`).

    Args:
        sol_vector (ndarray): Coefficients of the spectral Galerkin
            solution, as e.g. returned by `backward_euler_spectral`,
            `bdf_spectral`, `dg_two_spectral` or `dg_three_spectral`.
        spatial_disc_data (SpatialDiscParameters): Spatial discretization
            parameters; provides `points_per_interval`, here read as the
            total number of (equidistant) points the solution is
            evaluated at.
        pde (ParabolicPDE): The parabolic PDE, providing the spatial
            domain `spatial_interval` that `evaluate_spectral_solution`
            needs to map physical points onto the reference interval.
        func (Callable[[ndarray], ndarray] | None): If given, evaluated at
            the same points as the spectral solution and the max-norm
            error between the two is returned instead of the plain
            max-norm of the spectral solution. Takes the evaluation
            points x (shape (n_points,)) and returns the function values
            at these points (same shape).

    Returns:
        float: The maximum norm of the spectral solution, or, if `func`
            is given, of its error against `func`.
    """

    points_per_interval = spatial_disc_data.points_per_interval
    assert points_per_interval is not None, "spatial_disc_data must provide points_per_interval."

    a, b = pde.spatial_interval
    x_eval = np.linspace(a, b, points_per_interval)

    u_h = evaluate_spectral_solution(sol_vector, x_eval, pde)

    if func is None:
        return float(np.max(np.abs(u_h)))

    return float(np.max(np.abs(u_h - func(x_eval))))
