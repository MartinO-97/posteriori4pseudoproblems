import numpy as np
from numpy import ndarray

def bdf_coefficients(t_array: ndarray, 
                     k: int) -> ndarray:
    
    r"""
    The coefficients of u^j,...,u^{j-k} for the BDF operator.

    The BDF operator to approximate the first derivative of a given function u
    is given by

    D^1_{t,k}u(t_j) = \beta_0 u(t_j) + \beta_1 u(t_{j-1}) + ... + \beta_k u(t_{j-k}),
                    = \sum_{l=1}^k (\product_{i=1}^{l-1} \sum_{\mu=0}^{i-1} \tau_{j-\mu}) \delta_t^l u(t_j)

    where t_j,...,t_{j-k} are time steps, \tau_{j-\mu} = t_{j-\mu}-t_{j-\mu-1} and \delta_t^l u(t_j) the divided differences:
    
    \delta_t^l u(t_j) = (\delta_t^{l-1}u(t_j) - \delta_t^{l-1}u(t_{j-1})/(t_j - t_{j-l}),   \delta_t^0 u(t_j) = u(t_j).

    The present function computes the coefficients \beta_0, ..., \beta_k for given t_j,...,t_{j-k}.
        
    Paramters
    ---------
    t_array: ndarray 
        The time steps t_{j-k},...,t_j. t_{j-k} has to be stored at index 0 and t_j at index k-1.

    k: int 
        Order of the BDF method.
             
    Returns
    -------
    bdf: ndarray 
        Coefficients \beta_0,...,\beta_k, in the following order, where \beta_0 is associated with u(t_j) and stored at index 0. 
    """     
    
    # Initialize the array, that stores the coefficients \beta_0,...,\beta_k
    bdf = np.zeros(np.shape(t_array))                  
    
    # Initialize the array, that stores the sums (\product_{i=1}^{l-1} \sum_{\mu=0}^{i-1} \tau_{j-\mu}), l=1,...,k.
    vector_tau = np.zeros((np.shape(t_array)[0]-1, k))
    
    # Compute \tau_{j-k+1},...,\tau_j
    vector_tau[0,:] = t_array[1:] - t_array[:-1]
    
    # Compute the coefficients of the first column of the 'Steigungsschema', i.e.
    # 1/\tau_{j-l} and -1/\tau_{j-l}, l=0,...,k-1
    delta = np.vstack((1/vector_tau[0,:], -1/vector_tau[0,:]))
    
    tau = 1
    
    # Add the coefficients, coming from \delta_t^1 u(t_j)
    bdf[:2] = np.copy(delta[:,k-1]*tau)
    
    
    for l in range(2,k+1):
        
        # Compute the sum (\sum_{\mu=0}^{i-1} \tau_{j-\mu}), i=1,...,l-1
        vector_tau[l-1,l-1:] = vector_tau[0,:-l+1] + vector_tau[l-2,l-1:]       # determination of the needed sums of tau_{n-k+1}, ..., \tau_{n}
        
        delta_help = np.copy(delta)
        delta = np.zeros((l+1, k-(l-1)))
        
        # the lth column of the 'Steigungsschema'

        # the coefficient of u^j  
        delta[0,:] = np.copy(delta_help[0,1:])/ vector_tau[l-1,l-1:]    

        # the coefficient of u^{j-l}   
        delta[-1,:] = np.copy(-delta_help[-1,:-1])/ vector_tau[l-1,l-1:]

        # the coefficients of u^{j-l+1}, ..., u^{j-1}
        delta[1:-1,:] = (delta_help[1:,:0:-1] - delta_help[:-1,-2::-1]) / vector_tau[l-1,l-1:]
        
        # the coefficients of u^{n-l}, u^{n-l+1}, ..., u^{n-1}, u^n for the BDF operator
        bdf[:l+1] = bdf[:l+1] + np.prod(vector_tau[l-2::-1,-1], 0) * delta[:,-1]
                       
    return bdf  