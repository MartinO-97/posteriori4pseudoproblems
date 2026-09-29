import numpy as np

from numpy import ndarray
from numpy.linalg import norm

def gauss_lobatto_quadrature(n: int) -> tuple[ndarray, ndarray]: 

    r"""
    Compute the 'n' nodes and weights of the Gauss-Lobatto quadrature formula, 
    which is exact for polynoimals of degree 2*n-3.

    The Gauss-Lobatto quadrature formula approximates the integral

    \int_{-1}^1 f(x) dx,
    
    where the nodes x_1,...x_n are the n roots of the (n-1)-th integrated Legendre
    polynomial N_{n-1}. Furthermore, the nodes satisfy

    -1 = x_1 < x_2 < ... < x_{n-1} < x_n = 1.

    The nodes are determined by a Newton iteration applied to N_{n-1}(x), where
    N_{n-1} describes the (n-1)-th integrated Legendre polynomial.

    The weights are given by

    w_l = 2/(n*(n-1)) * 1/P'_{n-1}**2,  l = 1,...,n,

    where P_{n-1} denotes the (n-1)-th Legendre polynomial.     

    Parameters
    ----------
    n: int 
        Number of nodes and weights (n>=2).

    Returns
    -------
    x: ndarray 
        Nodes of the Gauss-Lobatto formula.
    
    w: ndarray 
        Weights of the Gauss-Lobatto formula. 

    Raises:
        ValueError: If 'n<2'.             
    """
    
    if n<2:
        raise ValueError('gauss_lobatto_quadrature: n must be greater or equal 2.')

    else:     
        # Extrema of the (n-1)-th Chebyshev polynomial of first kind T_{n-1}
        x = np.cos(np.arange(0,n)*np.pi/(n-1))[::-1]

        # Initialize x_{i-1}
        x_old = 0*x+2

        stop = 0

        # Newton step
        while norm(x-x_old,ord=np.inf) > 10**(-13):
            
            stop += 1

            # x_{i-1}
            x_old = np.copy(x) 
            
            # Initialize P_0(x_old) and P_1(x_old) 
            P_vs = 0*x+1  
            P_s = np.copy(x)

            # Compute P_{n-1}(x_old)  
            for k in range(2,n):
                P_b = (2*k-1)/k * x * P_s - (k-1)/k * P_vs      # P_k
                P_vs = np.copy(P_s)                             # P_{k-2}
                P_s = np.copy(P_b)                              # P_{k-1}

            # Compute P_n
            P_b = (2*n-1)/n * x * P_s - (n-1)/n * P_vs

            # Compute N_{n-1}(x_old); then P_b corresponds to P_n, P_s to P_{n-1} and P_vs to P_{n-2}
            N = (P_b - P_vs)/(2*n-1)
            
            # Update x
            x = x_old - N/P_s

        # Set manually x[0] = -1 and x[-1] = 1 
        x[np.array([0,-1])] = np.array([-1,1])

        # weights
        w = 2/(n**2-n) * 1/P_s**2
          
    return(x,w)    