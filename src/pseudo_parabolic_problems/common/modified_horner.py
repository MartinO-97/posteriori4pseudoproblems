import numpy as np
from numpy import ndarray

def modified_horner(alpha_values: ndarray, 
                    t_values: ndarray, 
                    divided_diff: ndarray) -> ndarray:

    r"""
    The modified Horner schema to evaluate an interpolation polynomial p,
    given as Newton interpolation polynomial:

               |    t_j    |        t_{j-1}        |  ...  |          t_{j-k}
        --------------------------------------------------------------------------------
        ...    |    a_j    |        a_{j-1}        |  ...  |          a_{j-k}
        \alpha |    ...    |  c_j(\alpha-t_{j-1})  |  ...  |  c_{j-k+1}(\alpha-t_{j-k})   
        --------------------------------------------------------------------------------
        ...    |    c_j    |        c_{j-1}        |  ...  |          c_{j-k}
        
    where a_l = v[t_l,...,t_j], l=j-k,...,j. 
        
    Parameters
    ----------
    alpha_values: ndarray 
        Values, where p shall be evaluated at.
    
    t_values: ndarray 
        Interpolation points t_{j-k},...,t_j, where t_{j-k} has to be at index 0.
    
    divided_diff: ndarray 
        The divided differences v[t_j], v[t_{j-1}, t_j], ..., v[t_{j-k},...,t_j], where v[t_j] has to be stored at index 0. 
        Axis 0 must be associated with t_l, l=j-k,...,j. 
    
    Returns
    -------
    :ndarray 
        p evaluated at alpha_values.
    """

    divided_diff_local = np.copy(divided_diff[::-1])
    t_local = np.copy(t_values)

    # c_j
    Result = np.tile(divided_diff_local[0], ((np.size(alpha_values),) + (1,)*(np.ndim(divided_diff)-1)))

    for l in range(1, np.shape(divided_diff_local)[0]):
        
        # reshape alpha_values - t_l
        value = np.reshape(alpha_values - t_local[l], ((np.size(alpha_values),) + (1,)*(np.ndim(divided_diff)-1)))

        # c_l = a_l + (alpha_values - t_l) * (c_{l+1})
        Result = divided_diff_local[l] + Result * value
    return(Result)


if __name__ == '__main__':

    alpha = np.array([0,1,3,2])
    t = np.array([0,1,2,3])
    divided = np.array([8, 4, 1, 1/6])

    print(modified_horner(alpha, t, divided))


