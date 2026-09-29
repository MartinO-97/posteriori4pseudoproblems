import numpy as np

from numpy import ndarray

def interval_transformation(z : ndarray,
                            int_start : tuple[float, float] | ndarray,
                            int_end : list[tuple[float, float]] | tuple[float, float] | ndarray) -> ndarray:
    
    r"""
    Function that map points from a given interval onto another interval or a set of
    intervals.

    This function linearly maps the array `z`, whose values lie in `int_start`,
    into the interval(s) defined by `int_end`. It supports both 1D and
    multi-interval mappings (for example, in finite element meshes).

    Args:
        z (ndarray): Array of points within `int_start` to be mapped.
        int_start (tuple[float, float] | ndarray): Interval where `z` lies.
        int_end (list[tuple[float, float]] | tuple[float, float] | ndarray): Target interval(s):
            - Shape (2,) for a single interval
            - Shape (m, 2) for multiple intervals [[a1, b1], [a2, b2], ...].

    Returns:
        ndarray: `z` mapped onto the intervals defined by `int_end`.

    Example:
        >>> import numpy as np
        >>> z = np.array([-1, 0, 1])
        >>> int_start = (-1,1)
        >>> int_end = (2, 4)
        >>> interval_transformation(z, int_start, int_end)
        array([2., 3., 4.])
    """

    if not isinstance(int_end, ndarray):
        if isinstance(int_end, list) and len(int_end) == 1:
            int_end_array = np.array(int_end[0])
        else:    
            int_end_array = np.array(int_end)
    else:
        int_end_array = np.copy(int_end)

    int_start_array = np.array(int_start)

    # Case 1: Both int_start and int_end are 1D intervals
    if int_start_array.ndim == 1 and int_end_array.ndim == 1:
        psi = lambda x: int_end_array[0] + (x - int_start_array[0]) * (
              (int_end_array[1] - int_end_array[0]) / (int_start_array[1] - int_start_array[0])
              )

    # Case 2: int_start is 1D and int_end is 2D (mapping to multiple intervals)
    elif int_start_array.ndim == 1 and int_end_array.ndim == 2:
        psi = lambda x: int_end_array[:, 0].reshape((-1, 1)) + (x - int_start_array[0]) * (
              (int_end_array[:, -1] - int_end_array[:, 0]) / (int_start_array[-1] - int_start_array[0])
              ).reshape((-1, 1))

    else:
        raise ValueError(
            "Unsupported input dimensions: "
            f"int_start.ndim={int_start_array.ndim}, int_end.ndim={int_end_array.ndim}"
        )

    return psi(z)