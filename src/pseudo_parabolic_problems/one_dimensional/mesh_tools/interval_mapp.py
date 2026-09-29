from numpy import ndarray

def interval_mapp(z: ndarray, 
                  int_start: ndarray, 
                  int_end : ndarray) -> ndarray:
    
    """
    Map points from one interval or set of intervals to another.

    This function linearly maps the array `z`, whose values lie in `int_start`,
    into the interval(s) defined by `int_end`. It supports both 1D and
    multi-interval mappings (for example, in finite element meshes).

    Args:
        z (ndarray): Array of points within `int_start` to be mapped.
        int_start (ndarray): Interval where `z` lies. 
        int_end (ndarray): Target interval(s).
            - Shape (2,) for a single interval 
            - Shape (m, 2) for multiple intervals [[a1, b1], [a2, b2], ...].

    Returns:
        ndarray: Array of `z` mapped from `int_start` to `int_end`.

    Raises:
        ValueError: If input dimensionalities are inconsistent.

    Example:
        >>> import numpy as np
        >>> z = np.array([-1, 0, 1])
        >>> int_start = np.array([-1, 1])
        >>> int_end = np.array([2, 4])
        >>> interval_mapp(z, int_start, int_end)
        array([2., 3., 4.])
    """

    # Case 1: Both int_start and int_end are 1D intervals
    if int_start.ndim == 1 and int_end.ndim == 1:
        psi = lambda x: int_end[0] + (x - int_start[0]) * (
              (int_end[1] - int_end[0]) / (int_start[1] - int_start[0])
              )

    # Case 2: int_start is 1D and int_end is 2D (mapping to multiple intervals)
    elif int_start.ndim == 1 and int_end.ndim == 2:
        m = int_end.shape[0] - 1
        psi = lambda x: int_end[:, 0].reshape((m + 1, 1)) + (x - int_start[0]) * (
              (int_end[:, -1] - int_end[:, 0]) / (int_start[-1] - int_start[0])
              ).reshape((m + 1, 1))

    else:
        raise ValueError(
            "Unsupported input dimensions: "
            f"int_start.ndim={int_start.ndim}, int_end.ndim={int_end.ndim}"
        )

    return psi(z)

