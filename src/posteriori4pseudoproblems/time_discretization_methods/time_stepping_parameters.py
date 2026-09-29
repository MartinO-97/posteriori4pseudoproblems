import numpy as np

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from numpy import ndarray
from scipy.sparse.linalg import SuperLU

@dataclass
class TimeSteppingParameters(ABC):

    r""" Base state of a time-stepping scheme for a pseudo-parabolic equation

    L \partial_t u(t) + M u(t) = f(t).

    Abstract; only its subclasses (`FemTimeSteppingParameters`,
    `SpectralTimeSteppingParameters`) are ever instantiated. Stores the
    temporal discretization method used to advance the solution in time,
    together with the previous approximations it relies on to compute the
    next one. One-step methods (``"backward_euler"``, ``"dg2"``) need only
    the latest approximation u^{j-1}, i.e. `num_prev = 1`, while a k-step
    BDF method needs the `k` most recent approximations u^{j-1},...,u^{j-k},
    i.e. `num_prev = k`. The FEM and spectral Galerkin methods each add the
    matrices representing L and M, together with whatever caching they
    require, in their own subclass.

    `prev_solutions` and `prev_psi` each hold one more entry than
    `num_prev`, i.e. u^{j-k},...,u^j and \psi^{j-k},...,\psi^j respectively:
    before u^j (and \psi^j) is computed, their last entry is u^{j-1}
    (\psi^{j-1}), the k previous approximations `num_prev = k` alone would
    provide; once u^j (\psi^j) has been computed, `update` (`update_psi`)
    registers it as the new last entry, so that the estimator, which
    additionally needs the value at the current step j itself, has
    u^{j-k},...,u^j (\psi^{j-k},...,\psi^j) available together.

    Args:
        disc_type (str): The temporal discretization method, e.g.
            ``"backward_euler"``, ``"bdf2"``, ``"dg2"``.
        num_prev (int): Number of previous approximations the method relies
            on to compute the next one.
    """

    disc_type: str
    num_prev: int
    prev_solutions: ndarray | None = field(init=False, default=None, repr=False)
    prev_psi: ndarray | None = field(init=False, default=None, repr=False)

    @abstractmethod
    def get_lu(self, alpha: float) -> SuperLU | ndarray:

        r""" Returns the object used to solve the coefficient system
        `matrix_L * alpha + matrix_M` against a right-hand side.

        FEM subclasses cache and reuse an LU factorization across time steps
        as long as `alpha` is unchanged. The spectral subclass has no such
        cache: it assembles and returns the dense coefficient matrix afresh,
        since it solves the system directly at every step.

        Args:
            alpha (float): Coefficient multiplying `matrix_L` in the
                coefficient matrix.
        """

        ...

    def update(self, new_solution: ndarray) -> None:

        r""" Registers a newly computed approximation u^j as the latest
        entry of `prev_solutions`, evicting the oldest one once `num_prev+1`
        approximations are stored.

        Args:
            new_solution (ndarray): Coefficient vector of the approximation
                u^j.
        """

        if self.prev_solutions is None:
            self.prev_solutions = new_solution[None, :]
        elif self.prev_solutions.shape[0] < self.num_prev + 1:
            self.prev_solutions = np.vstack((self.prev_solutions, new_solution))
        else:
            self.prev_solutions = np.vstack((self.prev_solutions[1:], new_solution))

    def update_psi(self, new_psi: ndarray) -> None:

        r""" Registers a newly computed \psi^j as the latest entry of
        `prev_psi`, evicting the oldest one once `num_prev+1`
        approximations are stored.

        Args:
            new_psi (ndarray): Coefficient vector of \psi^j.
        """

        if self.prev_psi is None:
            self.prev_psi = new_psi[None, :]
        elif self.prev_psi.shape[0] < self.num_prev + 1:
            self.prev_psi = np.vstack((self.prev_psi, new_psi))
        else:
            self.prev_psi = np.vstack((self.prev_psi[1:], new_psi))
