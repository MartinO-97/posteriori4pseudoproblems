from dataclasses import dataclass
from numpy import ndarray

@dataclass
class TemporalDiscParameters:

    r""" Parameters for the temporal discretization of a parabolic PDE.

    Args:
        disc_type (str): The temporal discretization method.
        M (int | None): The number of temporal subinterval. Defaults to ``None``.
        temporal_mesh (ndarray | None): The temporal mesh. Defaults to ``None``.
        m (int | None): Polynomial degree of the temporal reconstruction.
            Defaults to ``None``.
    """

    disc_type: str
    M: int | None = None
    temporal_mesh: ndarray | None = None
    m: int | None = None
