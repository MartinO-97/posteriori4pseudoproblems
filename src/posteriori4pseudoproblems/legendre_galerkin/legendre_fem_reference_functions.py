r""" Refrence functions for finite elements based on hat functions and integrated legendre polynomials.
Provided are:
    - The 'left' hat function \psi_L on [-1,1]
    - The 'right' hat function \psi_R on [-1,1]
    - The integrated Legendre polynomials
    - The Legendre polynomials
"""

import numpy as np
from numpy import ndarray

# ----------------------------------------------------------------------------------------------------
# THE 'LEFT' HAT FUNCTION \psi_L
# ----------------------------------------------------------------------------------------------------

def psi_l(x: ndarray) -> ndarray:

    r""" The hat function 
    
    \psi_L(x) = (1-x)/2
    
    Args:
        x (ndarray): Values where \psi_L shall be evaluated at.

    Returns:
        ndarray: \psi_L evaluated at the input array x.
    """

    return (1-x)/2

# ----------------------------------------------------------------------------------------------------
# THE 'RIGHT' HAT FUNCTION \psi_R
# ----------------------------------------------------------------------------------------------------

def psi_r(x: ndarray) -> ndarray:

    r""" The hat function 
    
    \psi_R(x) = (1+x)/2
    
    Args:
        x (ndarray): Values where \psi_R shall be evaluated at.

    Returns:
        ndarray: \psi_R evaluated at the input array x.
    """

    return (1+x)/2

# ----------------------------------------------------------------------------------------------------
# THE LEGENDRE POLYNOMIALS
# ----------------------------------------------------------------------------------------------------

def legendre(n: int,
             x: ndarray) -> ndarray:
    
    r"""Evaluation of the n-th Legendre polynomial at x. 
    
    The Legendre polynomial P_n is given by the three term recursion:
        
        P_n = (2n-1)/n \mu_1 * P_{n-1} - (n-1)/n P_{n-2},   n=2,...,   P_0 = \mu_0,    P_1 = \mu_1.
    
    The Legendre polynomials form a sequence of orthognal polynomials on [-1,1] with respect to
    
        \rho(x) = 1.
    
    Args:
        n (int): Number of the Legendre polynomial, which shall be evaluated at x; must be non-negative.
        x (ndarray): Value, where the n-th Legendre polynomial shall be evaluated at. 

    Returns:
        ndarray: The n-th Legendre polynomial P_n evaluated at x.    
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
        P_vs = np.ones_like(x)
        P_s  = x
        for i in range(2,n+1):
            P_b = (2*i-1)/i * x * P_s - (i-1)/i * P_vs
            P_vs = np.copy(P_s)
            P_s = np.copy(P_b)  
        legendre_evaluated = P_b      
    
    return legendre_evaluated

# ----------------------------------------------------------------------------------------------------
# THE INTEGRATED LEGENDRE POLYNOMIALS
# ----------------------------------------------------------------------------------------------------

def integrated_legendre(n: int,
                        x: ndarray) -> np.ndarray: 
    
    r"""Evaluation of the n-th integrated Legendre polynomial at x \in [-1,1]. 
    
    The integrated Legendre polynomial N_n is given by:
        
        N_n = (P_{n+1}-P_{n-1})/(2*n+1),    n=1,...,    N_0 = x+1
    
    Args:
        n (int): Number of the integrated Legendre polynomial, which shall be evaluated at x; must be non-negative.
        x (ndarray): Value, where the n-th integrated Legendre polynomial shall be evaluated at. 

    Returns:
        np.ndarray: The n-th integrated Legendre polynomial N_n evaluated at x.    
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
    
    return integrated_legendre_evaluated


# ----------------------------------------------------------------------------------------------------
# THE DERIVATIVE OF THE LEGENDRE POLYNOMIALS
# ----------------------------------------------------------------------------------------------------
  
def derivative_legendre(n: int,
                        x: np.ndarray) -> np.ndarray:

    r"""Evaluation of the derivative of the n-th Legendre polynomial at x. 
    
    The derivative of the n-th Legendre polynomial P'_n is given by the three term recursion:
        
        P'_n = (2n-1)/(n-1) \mu_1 * P'_{n-1} - n/(n-1) P'_{n-2},   n=2,...,   P'_0 = 0,    P'_1 = \mu_0.
    
    Args:
        n (int): Number of the Legendre polynomial, which shall be evaluated at x; must be non-negative.
        x (np.ndarray): Value, where the n-th Legendre polynomial shall be evaluated at. 

    Returns:
        np.ndarray: P'_n, evaluated at x.

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
    
    return derivative_legendre_evaluated 