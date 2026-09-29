import numpy as np

from scipy.sparse import csr_array
from ..discretization_dataclasses import SpatialDiscParameters

def slice_coefficient_matrices(spatial: SpatialDiscParameters,
                               spatial_ref: SpatialDiscParameters,
                               mass: csr_array,
                               matrix_L: csr_array,
                               matrix_M: csr_array) -> tuple[csr_array, csr_array, csr_array]:

    r"""
    Slicing of given coefficient matrices for one dimenisonal problems.

    If instead of an elliptic estimator a solution of higher order is computed,
    it is more efficient to slice the the corresponding matrices. Furthermore,
    an initialization of the LU factorization of the corresponding coefficient
    matrix is returned.

    Args:
        spatial (SpatialDiscParameters): Spatial discretization parameters;
            provides the number of spatial subintervals `N` and the
            polynomial degree `k` of the P_k-FEM (k=1 corresponds to a
            P_1-FEM).
        spatial_ref (SpatialDiscParameters): Spatial discretization
            parameters for the FEM solution of higher order; provides the
            polynomial degree `k` of the P_k-FEM used for that solution.
        mass (csr_array): Mass matrix, assembled for `spatial_ref`.
        matrix_L (csr_array): Matrix subjected to the operator L, assembled
            for `spatial_ref`.
        matrix_M (csr_array): Matrix subjected to the operator M, assembled
            for `spatial_ref`.

    Returns:
        tuple[csr_array, csr_array, csr_array]: Tuple, consisting of
            - **csr_array**: Mass matrix.
            - **csr_array**: Matrix subjected to the operator L.
            - **csr_array**: Matrix subjected to the operator M.
    """

    N = spatial.N

    assert N is not None and spatial.k is not None, "spatial must provide N and k for a FEM discretization."
    assert spatial_ref.k is not None, "spatial_ref must provide k for a FEM discretization."

    r = spatial.k - 1
    r_ref = spatial_ref.k - 1

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
    matrix_L_sliced = matrix_L[index, :].tocsc()[:, index].tocsr()
    matrix_M_sliced = matrix_M[index, :].tocsc()[:, index].tocsr()

    return mass_sliced, matrix_L_sliced, matrix_M_sliced
