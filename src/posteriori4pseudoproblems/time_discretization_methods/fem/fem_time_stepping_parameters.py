from dataclasses import dataclass, field
from scipy.sparse import csr_array, csc_array
from scipy.sparse.linalg import SuperLU, splu

from ..time_stepping_parameters import TimeSteppingParameters

@dataclass
class FemTimeSteppingParameters(TimeSteppingParameters):

    r""" State of a FEM-Galerkin time-stepping scheme for

    L \partial_t u(t) + M u(t) = f(t).

    In addition to the matrices representing L and M, this caches the LU
    factorization of the coefficient matrix `matrix_L * alpha + matrix_M`.
    The factorization is reused across time steps as long as the leading
    coefficient `alpha` (e.g. `1/tau_j` for the Backward Euler method, or the
    leading BDF coefficient `bdf[0]`) is unchanged, and recomputed
    automatically otherwise.

    Args:
        matrix_L (csr_array): Sparse matrix representing the operator L.
        matrix_M (csr_array): Sparse matrix representing the operator M.
    """

    matrix_L: csr_array
    matrix_M: csr_array
    lu_coef: SuperLU | None = None
    _alpha: float | None = field(init=False, default=None, repr=False)

    def get_lu(self, alpha: float) -> SuperLU:

        r""" Returns the LU factorization of `matrix_L * alpha + matrix_M`,
        (re)computing and caching it only if `alpha` differs from the value
        used for the currently cached factorization.

        Args:
            alpha (float): Coefficient multiplying `matrix_L` in the
                coefficient matrix.

        Returns:
            SuperLU: LU factorization of `matrix_L * alpha + matrix_M`.
        """

        if self.lu_coef is None or alpha != self._alpha:
            self.lu_coef = splu(csc_array(self.matrix_L * alpha + self.matrix_M))
            self._alpha = alpha

        return self.lu_coef
