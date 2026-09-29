from numpy import ndarray
from ..quadrature import Quadrature
from .legendre_fem_reference_functions import *

class PkLegendreFEM():

    r""" Class that provides data for a P_k-FEM based on the integrated 
    Legendre polynomials."""

    def __init__(self,
                 k: int, 
                 spatial_interval: tuple[float, float]) -> None:

        r""" Initialization of data associated with a P_k-FEM.
        
        Args:
            k (int): Polynomial degree
            spatial_interval (tuple[float, float]): Spatial interval of the PDE considered."""
        
        self._k: int = k
        self._spatial_interval: tuple[float, float] = spatial_interval
        self._psi_l_v: ndarray | None = None
        self._psi_r_v: ndarray | None = None
        self._le_n_g: ndarray | None = None
        self._le_p_g: ndarray | None = None

    def evaluate_reference_function_for_quadrature(self,
                                                   quadrature: Quadrature) -> None:

        r""" Evaluates the reference functions \psi_L, \psi_R, the integrated
        Legendre polynomials N_1,...,N_{k-1} and the Legendre polynomials
        P_1,...,P_{k-1} at the quadrature nodes of the employed quadrature
        rule; they are returned by `evaluated_reference_functions`.

        Args:
            quadrature (Quadrature): Quadrature rule; provides the nodes.
        """

        q_nodes, _ = quadrature.nodes_and_weights
        self._psi_l_v = psi_l(q_nodes)
        self._psi_r_v = psi_r(q_nodes)

        if self._k > 1:

            self._le_n_g = np.zeros((self._k-1, q_nodes.shape[0]))
            self._le_p_g = np.zeros((self._k-1, q_nodes.shape[0]))

            for l in range(0,self._k-1):
                self._le_n_g[l] = integrated_legendre(l+1, q_nodes)
                self._le_p_g[l] = legendre(l+1, q_nodes)

    @property
    def evaluated_reference_functions(self) -> tuple[ndarray | None,
                                                     ndarray | None,
                                                     ndarray | None,
                                                     ndarray | None]:

        r""" Returns the reference functions evaluated at the quadrature nodes
        
        Returns:
            - tuple[ndarray | None, ndarray | None, ndarray | None, ndarray | None]:
                Tuple consisting of:
                    - **ndarray | None**: \psi_L evaluated at the quadrature nodes
                    - **ndarray | None**: \psi_R evaluated at the quadrature nodes
                    - **ndarray | None**: Integrated Legendre polynomials evaluated at the 
                                          quadrature nodes
                    - **ndarray | None**: Legendre polynomials evaluated at the quarature nodes"""

        return self._psi_l_v, self._psi_r_v, self._le_n_g, self._le_p_g
