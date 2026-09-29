from dataclasses import dataclass
from numpy import ndarray

from ..time_stepping_parameters import TimeSteppingParameters

@dataclass
class SpectralTimeSteppingParameters(TimeSteppingParameters):

    r""" State of a spectral Galerkin time-stepping scheme for

    L \partial_t u(t) + M u(t) = f(t).

    Unlike the FEM path, the spectral time-stepping scripts solve the
    coefficient system directly at every time step, so no factorization is
    cached here.

    Args:
        matrix_L (ndarray): Matrix representing the operator L.
        matrix_M (ndarray): Matrix representing the operator M.
    """

    matrix_L: ndarray
    matrix_M: ndarray

    def get_lu(self, alpha: float) -> ndarray:

        r""" Assembles and returns the coefficient matrix `matrix_L * alpha +
        matrix_M`. No factorization is cached: the spectral time-stepping
        scripts solve the dense coefficient system directly at every time
        step.

        Args:
            alpha (float): Coefficient multiplying `matrix_L` in the
                coefficient matrix.

        Returns:
            ndarray: The coefficient matrix `matrix_L * alpha + matrix_M`.
        """

        return self.matrix_L * alpha + self.matrix_M
