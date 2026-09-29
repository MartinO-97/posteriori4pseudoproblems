from dataclasses import dataclass

from .spatial_disc_parameters import SpatialDiscParameters
from .temporal_disc_parameters import TemporalDiscParameters

@dataclass
class DiscParameters:

    r""" Parameters for the discretization of a parabolic PDE.

    Bundles the spatial and temporal discretization parameters.

    Args:
        spatial (SpatialDiscParameters): The spatial discretization parameters.
        temporal (TemporalDiscParameters): The temporal discretization parameters.
    """

    spatial: SpatialDiscParameters
    temporal: TemporalDiscParameters
