import numpy as np

from numpy import ndarray
from typing import Callable 
from scipy.sparse import csr_array, csc_array
from scipy.sparse.linalg import SuperLU, splu
from ...mesh_tools.interval_mapp import interval_mapp as int_mapp

def backward_euler_fem_1d(N : int, 
                          x : ndarray, 
                          w : ndarray, 
                          tau_j : float,
                          tau_j_s: float | None,  
                          t : float, 
                          func_f: Callable[[ndarray, float], ndarray], 
                          sol_vector_s : ndarray, 
                          matrix_L : csr_array, 
                          matrix_M : csr_array, 
                          lu_coef: SuperLU,
                          Delta : ndarray, 
                          h : ndarray, 
                          psi_r_v : ndarray, 
                          psi_l_v : ndarray, 
                          r : int, 
                          le_n_g : ndarray) -> tuple[ndarray, SuperLU]:
    
    r"""
    Backward Euler method for a FEM-Galerkin discretization in a spatial domain [a,b].

    This function computes an approximation u^j_h of the solution u(t_j)
    to a (pseudo-)parabolic equation of the form

    L \partial_t u(t) + M u(t) = f(t),

    using the implicit Backward Euler time discretization:

    L (U^j - U^{j-1}) / \tau_j + M U^j = f(t_j),

    where u^{j-1}_h approximates u(t_{j-1}).
    
    Parameters
    ----------
    N: int 
        Number of space intervals.
    
    x: ndarray
        Gauss-Lobatto nodes.
    
    w: ndarray
        Gauss-Lobatto weights.
    
    tau_j: float 
        Time step size.
    
    tau_j_s: float | None 
        Previous time step size; 'None' if 'j=1' or if the LU factorization has to be definitely 
        computed again.
    
    t: float 
        Current time level.
    
    func_f: Callable[[ndarray, float], ndarray] 
        Right-hand side function f(x, t_j).
    
    sol_vector_s: ndarray 
        Coefficient vector from the previous time step.
    
    matrix_L: csr_array 
        Sparse matrix representing the operator L.
    
    matrix_M: csr_array 
        Sparse matrix representing the operator M.
    
    lu_coef: SuperLU 
        LU factorization of the coefficient matrix.
    
    Delta: ndarray 
        Mesh points in space.
    
    h: ndarray
        spatial step size.
    
    psi_r_v: ndarray 
        'Right' shape function \psi_R evaluated at quadrature nodes x.
    
    psi_l_v: ndarray 
        'Left' shape function \psi_L evaluated at quadrature nodes x.
    
    r: int 
        Polynomial degree parameter for the integrated Legendre polynomials.
    
    le_n_g: ndarray
        Integrated Legendre polynomials evaluated at quadrature points x; empty if 'r=0'.

    Returns
    -------
    sol_vector: ndarray 
        Coefficients of the approximate solution u_h^j.

    lu_coef: SuperLU 
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

    # Check if tau_j changed. If not, a new computation of the LU factorization of the coefficient matrix
    # is not required

    if tau_j != tau_j_s or tau_j_s == None:
            
        # the coefficient matrix for the Euler-Method    
        coef_matrix = matrix_L * 1/tau_j + matrix_M
        coef_matrix = csc_array(coef_matrix)

        lu_coef = splu(coef_matrix)

    # determination of the result_vector
    f_j = func_f(int_mapp(x, ref_array, np.array([Delta[:-1], Delta[1:]]).T), t) 
    
    l = np.linspace(1,N-1,N-1, True, dtype = int)
    result_vector = np.zeros((r+1)*N-1)
    sol_vector = np.zeros((r+1)*N+1)
    
    # (f,\phi_i)
    result_vector[(r+1)*l-1] = h[:-1]/2*(f_j[:-1] * psi_r_v).dot(w) + h[1:]/2*(f_j[1:]*psi_l_v).dot(w)
    
    # (f, \phi_{i,j})    
    l = np.linspace(0,N-1,N, True, dtype = int)
    for mu in range(0,r):  
        result_vector[(r+1)*l+mu] = h/2*np.dot(f_j * le_n_g[mu], w) 
    
    # f + L u^{j-1}_h / tau_j
    result_vector = result_vector + matrix_L.dot(sol_vector_s[1:-1])/tau_j
    
    # determination of u^j_h
    sol_vector[1:-1] = np.asarray(lu_coef.solve(result_vector))    # type: ignore
    
    return(sol_vector, lu_coef)