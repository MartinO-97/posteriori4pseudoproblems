import numpy.linalg as lg

from numpy import ndarray, inf

def max_norm(x: ndarray) -> float:

    r"""
    Computation of the maximum norm of a given ndarray.

    Parameters
    ----------
    x: ndarray
        The array x, whose maximum shall be computed.

    Returns
    -------
    : float
        The maximum norm of x.
    """

    return float(lg.norm(x, ord=inf))