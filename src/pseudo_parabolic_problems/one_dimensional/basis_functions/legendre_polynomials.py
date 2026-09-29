import numpy as np

from numpy import ndarray
# ----------------------------------------------------------------------------------------------------
# THE INTEGRATED LEGENDRE POLYNOMIALS
# ----------------------------------------------------------------------------------------------------

def integrated_legendre(n: int,
                        x: ndarray) -> ndarray: 
    
    r"""
    Evaluation of the n-th integrated Legendre polynomial at x \in [-1,1]. 
    
    The integrated Legendre polynomial N_n is given by:
        
    N_n = (P_{n+1}-P_{n-1})/(2*n+1),    n=1,...,    N_0 = x+1
    
    Args:
        n (int): Number of the integrated Legendre polynomial, which shall be evaluated at x; must be non-negative.
        x (ndarray): Value, where the n-th integrated Legendre polynomial shall be evaluated at. 

    Returns:
        ndarray: N_n, evaluated at x.

    Raises: 
        ValueError: If n is negative or if x \notin [-1,1].     
    """
    
    if n < 0 or np.any(x) < -1 or np.any(x) > 1:
        raise ValueError('legendre_polynomials: n must be greater than zero and x must be in [-1,1]')

    # N_0(x)
    elif n == 0:
        integrated_legendre_evaluated = x+1             
    
    # N_n(x), n>1
    else:
        P_vs = np.ones_like(x)   
        P_s  = x
        for i in range(2,n+1):
            P_b = (2*i-1)/i * x * P_s - (i-1)/i * P_vs
            P_vs = np.copy(P_s)
            P_s = np.copy(P_b)  
        
        P_b = (2*n+1)/(n+1) * x * P_s - n/(n+1) * P_vs

        integrated_legendre_evaluated = (P_b - P_vs)/(2*n+1)      
    
    return(integrated_legendre_evaluated)


# ----------------------------------------------------------------------------------------------------
# THE LEGENDRE POLYNOMIALS
# ----------------------------------------------------------------------------------------------------

def legendre(n: int,
             x: ndarray) -> ndarray:
    
    r"""
    Evaluation of the n-th Legendre polynomial at x. 
    
    The Legendre polynomial P_n is given by the three term recursion:
        
    P_n = (2n-1)/n \mu_1 * P_{n-1} - (n-1)/n P_{n-2},   n=2,...,   P_0 = \mu_0,    P_1 = \mu_1.
    
    The Legendre polynomials form a sequence of orthognal polynomials on [-1,1] with respect to
    
    \rho(x) = 1.
    
    Args:
        n (int): Number of the Legendre polynomial, which shall be evaluated at x; must be non-negative.
        x (ndarray): Value, where the n-th Legendre polynomial shall be evaluated at. 

    Returns:
        ndarray: P_n, evaluated at x.

    Raises: 
        ValueError: If n is negative or if x\notin [-1,1].     
    """

    if n < 0 or np.any(x) < -1 or np.any(x) > 1:
        raise ValueError('legendre_polynomials: n must be greater than zero and x must be in [-1,1]')

    # P_0(x)
    elif n == 0:
        legendre_evaluated = np.ones_like(x)    
    
    # P_1(x)
    elif n == 1:
        legendre_evaluated = x       
    
    # P_n(x), n>1
    else:
        P_vs = np.full_like(x,1)
        P_s  = x
        for i in range(2,n+1):
            P_b = (2*i-1)/i * x * P_s - (i-1)/i * P_vs
            P_vs = np.copy(P_s)
            P_s = np.copy(P_b)  
        legendre_evaluated = P_b      
    
    return(legendre_evaluated)


# ----------------------------------------------------------------------------------------------------
# THE DERIVATIVE OF THE LEGENDRE POLYNOMIALS
# ----------------------------------------------------------------------------------------------------
  
def derivative_legendre(n: int,
                        x: ndarray) -> ndarray:

    r"""
    Evaluation of the derivative of the n-th Legendre polynomial at x. 
    
    The derivative of the n-th Legendre polynomial P'_n is given by the three term recursion:
        
    P'_n = (2n-1)/(n-1) \mu_1 * P'_{n-1} - n/(n-1) P'_{n-2},   n=2,...,   P'_0 = 0,    P'_1 = \mu_0.
    
    Args:
        n (int): Number of the Legendre polynomial, which shall be evaluated at x; must be non-negative.
        x (ndarray): Value, where the n-th Legendre polynomial shall be evaluated at. 

    Returns:
        ndarray: P'_n, evaluated at x.

    Raises: 
        ValueError: If n is negative or if x\notin [-1,1].     
    """

    if n < 0 or np.any(x) < -1 or np.any(x) > 1:
        raise ValueError('legendre_polynomials: n must be greater than zero and x must be in [-1,1]')

    # P'_0(x)
    elif n == 0:
        derivative_legendre_evaluated = np.zeros_like(x)      
    
    # P'_1(x)
    elif n == 1:
        derivative_legendre_evaluated = np.ones_like(x)       
    
    # P'_n(x), n>1
    else:
        der_P_vs = np.zeros_like(x)
        der_P_s  = np.ones_like(x)
        for i in range(2,n+1):
            der_P_b = (2*i-1)/(i-1) * x * der_P_s - i/(i-1) * der_P_vs
            der_P_vs = np.copy(der_P_s)
            der_P_s = np.copy(der_P_b)  
        derivative_legendre_evaluated = der_P_b      
    
    return(derivative_legendre_evaluated) 