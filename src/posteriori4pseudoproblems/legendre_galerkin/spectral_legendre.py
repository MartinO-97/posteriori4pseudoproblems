import numpy as np

from numpy import ndarray
from ..quadrature import Quadrature
from .legendre_fem_reference_functions import integrated_legendre, legendre

class SpectralLegendre():

    r""" Class that provides data for a spectral Galerkin method based on
    the integrated Legendre polynomials."""

    def __init__(self,
                 dim_V: int,
                 spatial_interval: tuple[float, float]) -> None:

        r""" Initialization of data associated with a spectral Galerkin method.

        Args:
            dim_V (int): Dimension of the ansatz space, i.e. the number of
                integrated Legendre polynomials of degree 1,...,dim_V used
                as basis functions.
            spatial_interval (tuple[float, float]): Spatial interval of the PDE considered."""

        self._dim_V: int = dim_V
        self._spatial_interval: tuple[float, float] = spatial_interval
        self._le_n_g: ndarray | None = None
        self._le_p_g: ndarray | None = None

    def evaluate_reference_function_for_quadrature(self,
                                                   quadrature: Quadrature) -> None:

        r""" Evaluates the integrated Legendre polynomials N_1,...,N_{dim_V}
        and the Legendre polynomials P_1,...,P_{dim_V} at the quadrature nodes
        of the employed quadrature rule; they are returned by
        `evaluated_reference_functions`.

        Args:
            quadrature (Quadrature): Quadrature rule; provides the nodes.
        """

        q_nodes, _ = quadrature.nodes_and_weights

        self._le_n_g = np.zeros((self._dim_V, q_nodes.shape[0]))
        self._le_p_g = np.zeros((self._dim_V, q_nodes.shape[0]))

        for l in range(0, self._dim_V):
            self._le_n_g[l] = integrated_legendre(l+1, q_nodes)
            self._le_p_g[l] = legendre(l+1, q_nodes)

    @property
    def evaluated_reference_functions(self) -> tuple[ndarray | None,
                                                     ndarray | None]:

        r""" Returns the reference functions evaluated at the quadrature nodes

        Returns:
            - tuple[ndarray | None, ndarray | None]:
                Tuple consisting of:
                    - **ndarray | None**: Integrated Legendre polynomials evaluated at the
                                          quadrature nodes
                    - **ndarray | None**: Legendre polynomials evaluated at the quarature nodes"""

        return self._le_n_g, self._le_p_g
