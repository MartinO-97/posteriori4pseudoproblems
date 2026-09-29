from dataclasses import dataclass
from numpy import ndarray

@dataclass
class SpatialDiscParameters:

    r""" Parameters for the spatial discretization of a parabolic PDE.

    Bundles the configuration shared by the Galerkin Finite Element method
    (``disc_type = "fem"``) and the spectral Galerkin method
    (``disc_type = "spectral"``). Only the fields relevant to the chosen
    ``disc_type`` need to be set: ``N``, ``k``, ``Delta`` and ``h`` for the
    FEM, ``dim_V`` for the spectral method.

    Args:
        disc_type (str): The spatial discretization method: ``"fem"`` or ``"spectral"``.
        N (int | None): FEM only. Number of spatial subintervals. Defaults to ``None``.
        k (int | None): FEM only. Polynomial degree of the P_k-FEM, i.e. the
            number of integrated Legendre polynomials used is ``k-1``;
            ``k=1`` corresponds to a P_1-FEM. Defaults to ``None``.
        Delta (ndarray | None): FEM only. Spatial mesh (a=x_0 < x_1 < ... < x_N = b).
            Defaults to ``None``.
        h (ndarray | None): FEM only. Diameter x_i - x_{i-1} of every interval
            [x_{i-1},x_i], i=1,...,N. Defaults to ``None``.
        dim_V (int | None): Spectral method only. Dimension of the Ansatz space.
            Defaults to ``None``.
    """

    disc_type: str
    N: int | None = None
    k: int | None = None
    Delta: ndarray | None = None
    h: ndarray | None = None
    dim_V: int | None = None
