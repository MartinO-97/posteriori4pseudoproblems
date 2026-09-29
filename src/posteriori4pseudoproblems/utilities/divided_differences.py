import numpy as np

def compute_divided_differences(
        t_values: np.ndarray, 
        v_values: np.ndarray) -> np.ndarray:

    r"""
    Computation of the divided differences:

    v[t_{j-k},...,t_j] = (v[t_{j-k+1},...,t_j] - v[t_{j-k},...,t_{j-1}])/(t_j - t_{j-k}),

    v[t_j] = v^j.

    Since we are only need Newton interpolation -- and not the extended Hermite interpolation --
    we don't need to consider possible deriviatives interpolation, like

        p'(t_j) = v'(t_j).

    Parameters
    ----------
    t_values: ndarray 
        The values t_{j-k},...,t_j (in this order). In particular, t_{j-k} is stored at index 0.
    
    v_values: ndarray 
        The function v, evaluated at t_{j-k},...,t_j (in this order). Axis 0 must be associated with t_l, l=j-k,...,j.

    Returns
    -------
    :ndarray 
        The divided differences v[t_j], v[t_{j-1}, t_j],..., v[t_{j-k},...,t_j] (in this order). In particular, v[t_j] is stored 
        at index 0.                
    
    """

    t_local = np.copy(t_values[::-1])
    v_local = np.copy(v_values[::-1])

    Results = np.zeros(np.shape(v_values))

    # v[t_j]
    Results[0] = np.copy(v_local[0])

    # v[t_{j-l},...,v[t_j]]

    for l in range(1, np.shape(v_values)[0]):

        divisor = np.reshape(t_local[l:] - t_local[:-l], (np.size(t_local[l:] - t_local[:-l]),) + (1,)*(np.ndim(v_local)-1))
        v_local = (v_local[1:] - v_local[:-1])/divisor 

        Results[l] = np.copy(v_local[0])

    return(Results)    