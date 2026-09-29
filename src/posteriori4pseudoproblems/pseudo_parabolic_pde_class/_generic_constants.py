import numpy as np

from numpy import ndarray
from typing import Callable


def compute_generic_constants(a: Callable[[ndarray], ndarray],
                              c: Callable[[ndarray], ndarray],
                              spatial_interval: tuple[float, float],
                              n_points: int = 5000) -> tuple[float, float, float, float,
                                                             float, float, float, float]:

    r""" Computes the boundedness and coercivity constants of the bilinear forms
    associated with the operators L and M as well as some further constants for
    the computation of \eta_{L^2} and \eta_{H^1_0}.

    Given the two operators

        Lu = -u'' + au  and  Mu = -u'' + cu,    with a > 0,

    the boundedness and coercivity constants of the bilinear forms associated
    with L and M are

        C_a = max(1, max_x a(x)),   c_a = min(1, min_x a(x)),
        C_c = max(1, max_x c(x)),   c_c = min(1, min_x c(x)).

    Note that the 'coercivity' constant c_c might be zero or negative. max_x and
    min_x are approximated on `n_points` equidistant points on the closure of
    the spatial domain. Furthermore,

        M_{2,*} = 1,  omega_{2,*} = 1,  C_I = 1/2 sqrt(1/2) (b-a+1),  ||| L^{-1} |||_{0,2} = 1.

    Args:
        a (Callable[[ndarray], ndarray]): The function a.
        c (Callable[[ndarray], ndarray]): The function c.
        spatial_interval (tuple[float, float]): The closure of the spatial domain.
        n_points (int): Number of equidistant points, on the closure of the
            spatial domain, used to approximate the extrema of a and c. Defaults
            to 5000.

    Returns:
        tuple[float, float, float, float, float, float, float, float]:
            (C_a, c_a, C_c, c_c, M_2_star, omega_2_star, C_I, L_inverse).
    """

    # The spatial interval [a,b]
    x_a, x_b = spatial_interval

    # Initialization of some auxiliary array
    z = np.linspace(x_a, x_b, n_points)

    # the constants C_a and c_a
    a_values = a(z)
    C_a = np.max(np.array([1, np.max(a_values)]))
    c_a = np.min(np.array([1, np.min(a_values)]))

    # the constants C_c and c_c
    c_values = c(z)
    C_c = np.max(np.array([1, np.max(c_values)]))
    c_c = np.min(np.array([1, np.min(c_values)]))

    # the constants M_{2,*} and omega_{2,*}
    M_2_star = 1
    omega_2_star = 1

    # the constant C_I
    C_I = 1/2*np.sqrt(1/2)*(x_b-x_a+1)

    # the constant ||| L^{-1} |||_{0,2}
    L_inverse = 1

    return C_a, c_a, C_c, c_c, M_2_star, omega_2_star, C_I, L_inverse
