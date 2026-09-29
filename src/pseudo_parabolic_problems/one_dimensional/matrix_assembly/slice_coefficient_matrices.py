import numpy as np
import scipy.sparse as sp

def slice_coefficient_matrices(r: int,
                               r_ref: int,
                               N: int,
                               mass: sp.csr_array,
                               reaction_a: sp.csr_array,
                               reaction_c: sp.csr_array,
                               stiff: sp.csr_array,
                               matrix_L: sp.csr_array,
                               matrix_M: sp.csr_array) -> tuple[sp.csr_array, sp.csr_array, sp.csr_array, 
                                                                sp.csr_array, sp.csr_array, sp.csr_array, 
                                                                sp.linalg.SuperLU]:

    r"""
    Slicing of given coefficient matrices for one dimenisonal problems.
    
    If instead of an elliptic estimator a solution of higher order is computed, 
    it is more efficient to slice the the corresponding matrices. Furthermore, 
    an initialization of the LU factorization of the corresponding coefficient 
    matrix is returned.
    
    Args:
        r (int): Maximum grade of the integrated Legendre polynomials; if r=0 a P_1-FEM is considered
        r (int): Maximum grade of the integrated Legendre polynomials for the solution of higher order.
        N (int): Number of spatial subintervals.
        mass (sp.csr_array): Mass matrix.
        reaction_a (sp.csr_array): Matrix associated to the reaction term au; None if det_a = False.
        reaction_c (sp.csr_array): Matrix associated with the reaction term cu.
        stiff (sp.csr_array): Stiffness matrix, associated with -\Delta u.
        matrix_L (sp.csr_array): Matrix subjected to the operator L; None if det_a = False.
        matrix_M (sp.csr_array): Matrix subjected to the operator M.

    Returns:
        tuple[sp.csr_array, sp.csr_array, sp.csr_array, sp.csr_array, sp.csr_array, sp.csr_array, sp.linalg.SuperLU]:
            - mass_sliced (sp.csr_array): Mass matrix.
            - reaction_a_sliced (sp.csr_array): Matrix associated to the reaction term au.
            - reaction_c_sliced (sp.csr_array): Matrix associated with the reaction term cu.
            - stiff_sliced (sp.csr_array): Stiffness matrix, associated with -\Delta u.
            - matrix_L_sliced (sp.csr_array): Matrix subjected to the operator L.
            - matrix_M_sliced (sp.csr_array): Matrix subjected to the operator M.
            - lu_coef_init (sp.linalg.SuperLU): Initialization of the LU factorization of the corresponding coefficient matrix.
    """
    
    # DETERMINATION OF THE INDEX ARRAY

    # Initialization of the index array
    index = np.zeros((r+1)*N-1)

    # Indices for the o and x entries (see 'assemble_coeff_matrices_fem_1d.py')
    l_o_x = np.arange(1,N)
    index[(r+1)*l_o_x - 1] = (r_ref+1)*l_o_x-1

    # Indices for the * entries (see 'assemble_coeff_matrices_fem_1d.py')
    l_star = np.arange(0,N)
    b = np.arange(0,r)  
    index_help = (b+np.reshape(l_star, (N,1))*(r+2)).flatten()

    index[np.setdiff1d(np.arange(0, (r+1)*N-1), (r+1)*l_o_x - 1)] = index_help

    # SLICING THE MATRICES
    mass_sliced = mass[index, :].tocsc()[:, index].tocsr()
    reaction_a_sliced = reaction_a[index, :].tocsc()[:, index].tocsr()
    reaction_c_sliced = reaction_c[index, :].tocsc()[:, index].tocsr()
    stiff_sliced = stiff[index, :].tocsc()[:, index].tocsr()
    matrix_L_sliced = matrix_L[index, :].tocsc()[:, index].tocsr()
    matrix_M_sliced = matrix_M[index, :].tocsc()[:, index].tocsr()

    # INITIALIZATION OF THE LU FACTORIZATION
    lu_coef_init = sp.linalg.splu(sp.csc_array(np.eye(1)))

    return(mass_sliced, reaction_a_sliced, reaction_c_sliced, stiff_sliced, matrix_L_sliced, matrix_M_sliced, lu_coef_init)
