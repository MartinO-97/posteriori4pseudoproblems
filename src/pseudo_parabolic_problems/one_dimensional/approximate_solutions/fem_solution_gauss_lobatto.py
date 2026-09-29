import numpy as np

from numpy import ndarray

def approximate_solution_intervals(sol_v: ndarray,
                                   psi_r_v: ndarray,
                                   psi_l_v: ndarray,
                                   le_n_g: ndarray | None,
                                   r: int,
                                   N: int) -> ndarray:

    r"""
    The FEM solution u_h(\chi^{-1}_i(x)) evaluated at the Gauss-Lobatto nodes for each i=1,...,N.

    In the FEM analysis, we are interested in the computation of

    ||u-u_h||_{L^2(a,b)}    and    ||u-u_h||_{H^1(a,b)},

    where ||\cdot||_{L^2(a,b)} and ||\cdot||_{H^1(a,b)} denote the L^2 and H^1
    norm over (a,b). u denotes the exact solution of the given pde and u_h
    its approximation, computed by a FEM. 

    'approximate_solution_intervals' evaluates u_h(\chi^{-1}_i(x)) at the Gauss-Lobatto
    nodes for each i=1,...,N, where \chi_i:[x_{i-1},x_i] \to [-1,1] denote affine-linear
    transformations. 
    
    Args:
        sol_v (ndarray): The coefficients of the approximate solution u_h.
        psi_r_v (ndarray): The shape functions \psi_R, evaluated at the Gauss-Lobatto nodes.
        psi_l_v (ndarray): The shape functions \psi_L, evaluated at the Gauss-Lobatto nodes.
        le_n_g (ndarray): The integrated Legendre Polynomials N_l, l=1,...,r, evaluated at the Gauss-Lobatto nodes.
        r (int): r+1 is the maximum degree of the considered integrated Legendre polynomials. 
        N (int): Number of spatial subintervals.

    Returns:
        ndarray: u_h(\chi_i^{-1}) evaluated at the Gauss-Lobatto nodes, for each i=1,...,N.    
    """

    # Consideration of the shape functions \psi_R and \psi_L
    l = np.arange(0,N+1)
    u_h_v = sol_v[l[1:]*(r+1), None] * psi_r_v + sol_v[l[:-1]*(r+1), None] * psi_l_v

    # Consideration of the integrated Legendre polynomials
    l = np.arange(0,N)
    for k in range(1,r+1):
        u_h_v += sol_v[l*(r+1)+k, None]*le_n_g[k-1]     #type: ignore

    return(u_h_v)    


def derivative_approximate_solution_intervals(sol_v: ndarray,
                                              le_p_g: ndarray | None,
                                              number_gauss_lobatto: int,
                                              r: int,
                                              N: int, 
                                              h: ndarray) -> ndarray:

    r"""
    The u'_h(\chi^{-1}_i(x)) evaluated at the Gauss-Lobatto nodes for each i=1,...,N.

    In the FEM analysis, we are interested in the computation of

    ||u-u_h||_{L^2(a,b)}    and    ||u-u_h||_{H^1(a,b)},

    where ||\cdot||_{L^2(a,b)} and ||\cdot||_{H^1(a,b)} denote the L^2 and H^1
    norm over (a,b). u denotes the exact solution of the given pde and u_h
    its approximation, computed by a FEM. 

    'derivative_approximate_solution_intervals' evaluates u'_h(\chi^{-1}_i(x)) at the Gauss-Lobatto
    nodes for each i=1,...,N, where \chi_i:[x_{i-1},x_i] \to [-1,1] denote affine-linear
    transformations. 
    
    Args:
        sol_v (ndarray): The coefficients of the approximate solution u_h.
        le_p_g (ndarray): The Legendre Polynomials P_l, l=1,...,r, evaluated at the Gauss-Lobatto nodes.
        number_gauss_lobatto (int): Number of Gauss-Lobatto points.
        r (int): r+1 is the maximum degree of the considered integrated Legendre polynomials. 
        N (int): Number of spatial subintervals.
        h (ndarray): Spatial step sizes.

    Returns:
        ndarray: u'_h(\chi_i^{-1}) evaluated at the Gauss-Lobatto nodes, for each i=1,...,N.    
    """

    # Consideration of the shape functions \psi_R and \psi_L
    l = np.arange(0,N+1)
    der_u_h_v = ((sol_v[l[1:]*(r+1)] * 1/h)[:,None] - (sol_v[l[:-1]*(r+1)] * 1/h)[:,None]) * np.ones((N, number_gauss_lobatto))   #type: ignore

    # Consideration of the integrated Legendre polynomials
    l = np.arange(0,N)
    for k in range(1,r+1):
        der_u_h_v += (2/h*sol_v[l*(r+1)+k])[:,None] * le_p_g[k-1]   # type: ignore

    return(der_u_h_v)    


def second_derivative_approximate_solution_intervals(sol_v: ndarray,
                                                     le_p_g_x: ndarray | None,
                                                     number_gauss_lobatto: int,
                                                     r: int,
                                                     N: int, 
                                                     h: ndarray) -> ndarray:

    r"""
    The u''_h(\chi^{-1}_i(x)) evaluated at the Gauss-Lobatto nodes for each i=1,...,N.

    u''_h(\chi^{-1}_i(x)) is needed for the elliptic residual based a posteriori error boound.

    'second_derivative_approximate_solution_intervals' evaluates u''_h(\chi^{-1}_i(x)) at the Gauss-Lobatto
    nodes for each i=1,...,N, where \chi_i:[x_{i-1},x_i] \to [-1,1] denote affine-linear
    transformations. 

    Note that for r=0 we have u''_h = 0.
    
    Args:
        sol_v (ndarray): The coefficients of the approximate solution u_h.
        le_p_g_x (ndarray): The first derivative of Legendre Polynomials P'_l, l=1,...,r, evaluated at the Gauss-Lobatto nodes.
        number_gauss_lobatto: int,
        r (int): r+1 is the maximum degree of the considered integrated Legendre polynomials. 
        N (int): Number of spatial subintervals.
        h (ndarray): Spatial step sizes.

    Returns:
        ndarray: u''_h(\chi_i^{-1}) evaluated at the Gauss-Lobatto nodes, for each i=1,...,N.    
    """

    # Initialization of u''_h
    second_der_u_h_v = np.zeros((N, number_gauss_lobatto))

    # Consideration of the integrated Legendre polynomials
    l = np.arange(0,N)
    for k in range(1,r+1):
        second_der_u_h_v += (4/h**2*sol_v[l*(r+1)+k])[:,None] * le_p_g_x[k-1]   # type: ignore

    return(second_der_u_h_v)    