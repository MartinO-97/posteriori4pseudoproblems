import numpy as np

from numpy import ndarray
from typing import Callable
from scipy.sparse import csr_array, csc_array
from scipy.sparse.linalg import SuperLU, splu
from ...mesh_tools.interval_mapp import interval_mapp as int_mapp
from ....common.bdf_coefficients import bdf_coefficients

def bdf_fem_1d(k: int, 
               sol_vector_s: ndarray, 
               t_array: ndarray,
               Delta: ndarray, 
               h: ndarray, 
               x: ndarray, 
               w: ndarray, 
               matrix_L: csr_array, 
               matrix_M: csr_array, 
               lu_coeff: SuperLU,
               func_f: Callable[[ndarray, float], ndarray], 
               psi_r_v: ndarray, 
               psi_l_v: ndarray, 
               le_n_g: ndarray, 
               r:int, 
               N: int) -> tuple[ndarray, SuperLU]:
    
    r"""
    BDF method for a FEM discretization in a spatial domain [a,b].

    This function computes an approximation u^j_h of the solution u(t_j)
    to a (pseudo-)parabolic equation of the form

    L \partial_t u(t) + M u(t) = f(t),

    using the BDF time discretization:

    L D^1_{t,k}u^j_h + M u^j_h = f(t_j),

    where u^{j-1}_h approximates u(t_{j-1}) and 
    D^1_{t,k}u^j_h denotes the BDF operator of order k.
    
    Parameters
    ----------
    k: int 
        Order of the BDF method   
    
    sol_vector_s: ndarray 
        Array, that stores the coefficients of u^{j-1}_h,...,u^{j-k}_h, 
        where the l-th row is associated to u^{j-l-1}_h.  
    
    t_array: ndarray 
        Arrray, that stores t_j,...,t_{j-k},t_{j-k-1} where the index 0 is associated to t_{j-k-1};
        if only t_j,...,t_{j-k} are passed, the LU factorization of the coefficient matrix is computed.
    
    Delta: ndarray 
        Mesh points in space.
    
    h: ndarray 
        spatial step size. 
    
    x: ndarray 
        Gauss-Lobatto nodes.
    
    w: ndarray 
        Gauss-Lobatto weights.    
    
    matrix_L: csr_array
        Sparse matrix representing the operator L.
    
    matrix_M: csr_array 
        Sparse matrix representing the operator M. 
    
    lu_coeff: SuperLU 
        LU factorization of the coefficient matrix.
    
    func_f: Callable[[ndarray, float], ndarray] 
        Right-hand side function f(x, t). 
    
    psi_r_v: ndarray 
        'Right' shape function \psi_R evaluated at quadrature nodes x.
    
    psi_l_v: ndarray 
        'Left' shape function \psi_L evaluated at quadrature nodes x.
    
    le_n_g: ndarray 
        Integrated Legendre polynomials evaluated at quadrature points x; empty array if 'r=0'.    
    
    r: int 
        r+1 is the maximum degree of the utilized integrated Legendre polynomials.             
    
    N: int 
        Number of space intervals.

    Returns
    -------
    sol_vector: ndarray 
        Coefficients of the approximate solution u_h^j.
    
    lu_coeff: SuperLU 
        LU factorization of the coefficient matrix.

    Raises
    ------
    TypeError 
        If matrix_L or matrix_M are not scipy.sparse sparse arrays.       
    """

    # Validate sparse matrices
    if not isinstance(matrix_L, csr_array) or not isinstance(matrix_M, csr_array):
        raise TypeError("matrix_L and matrix_M must be scipy.csr_array sparse arrays.")

    # the array [-1,1]
    ref_array = np.array([-1,1])

    # determination of the result_vector
    f_j = func_f(int_mapp(x, ref_array, np.array([Delta[:-1], Delta[1:]]).T), t_array[-1]) 
    
    l = np.linspace(1,N-1,N-1, True, dtype = int)
    result_vector = np.zeros((r+1)*N-1)
    sol_vector = np.zeros((r+1)*N+1)
    
    # (f,\phi_i)
    result_vector[(r+1)*l-1] = h[:-1]/2*(f_j[:-1] * psi_r_v).dot(w) + h[1:]/2*(f_j[1:]*psi_l_v).dot(w)
    
    # (f, \phi_{i,j})    
    l = np.linspace(0,N-1,N, True, dtype = int)
    for mu in range(0,r):  
        result_vector[(r+1)*l+mu] = h/2*np.dot(f_j * le_n_g[mu], w)

    # Compute the bdf coefficients    
    if (k+2) == np.shape(t_array)[0]:
        bdf = bdf_coefficients(t_array[1:], k) 
    else:
        bdf = bdf_coefficients(t_array, k)     

    # Check if the LU factorization of the coefficient matrix has to be computed.
    # This has to be done, if either \tau_j, ..., \tau_{j-k+1} have changed, compared
    # to the previous time step, or if j-k-1 < 0
    if not ((k+2) == np.shape(t_array)[0] and np.array_equal(t_array[2:]-t_array[1:-1], t_array[1:-1] - t_array[0:-2])):         

        # Compute the coefficient matrix matrix_L * bdf[0] + matrix_M
        coef_matrix = csc_array(matrix_L*bdf[0] + matrix_M)

        # Compute LU factorization of the coefficient matrix
        lu_coeff = splu(coef_matrix)
    
    # Compute \sum_{l=1}^k u^{j-1}_h * bdf[l]
    previous = np.zeros(np.shape(sol_vector_s)[1])
    for l in range(1,k+1):
        previous[1:-1] += bdf[l] * sol_vector_s[-l, 1:-1]

    # (f, \phi_i) - matrix_L * \sum_{l=1}^k u^{j-1}_h * bdf[l]
    result_vector =  result_vector - matrix_L.dot(previous[1:-1])

    # the coefficients for u^j_h
    sol_vector = np.zeros((r+1)*N+1)
    sol_vector[1:-1] = np.asarray(lu_coeff.solve(result_vector)) 

    return(sol_vector, lu_coeff)   