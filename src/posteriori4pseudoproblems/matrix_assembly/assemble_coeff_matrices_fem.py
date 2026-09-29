import numpy as np

from typing import Tuple
from numpy import ndarray
from scipy.sparse import csr_array, coo_array
from ..interval_transformation import interval_transformation as int_mapp
from ..pseudo_parabolic_pde_class import PseudoParabolicPDE
from ..discretization_dataclasses import SpatialDiscParameters
from ..quadrature import Quadrature
from ..legendre_galerkin import PkLegendreFEM

def assemble_coeff_matrices_fem_1d(pde : PseudoParabolicPDE,
                                   spatial : SpatialDiscParameters,
                                   quadrature : Quadrature,
                                   ref_functions : PkLegendreFEM) -> tuple[csr_array, csr_array, csr_array]:

    r"""
    This functions assembles the matrices, associated with the operators Id u, -\Delta u, au, cu,
    where a, c :[a,b] \to \RR and \Delta is the Laplacian, for a FEM-Galerkin method.

    As shape functionswe use

    \psi_L(x) = (1-x)/2, \psi_R(x)=(1+x)/2

    and the integrated Legendre polynomials N_i, i=1,...,r, where is a non-negative integer.

    If r=0, a standard P_1-FEM is used.

    Gauss-Lobatto (GL) quadrature is used for numerical integration.

    When assembling our matrices, we have to consider three constilations. Let \phi_i denote the basis functions, subjected to \psi_L and
    \psi_R, and let \phi_{ij} denote the basis functions, subjected to the integrated Legendre polynomials. If
    we denote by k() the bilinear form, associated with the considered operator, we have the three cases:

    - k(\phi_i, \phi_k), i,k=1,...,N-1       (referred to as 'x')
    - k(\phi_i, \ph_{ij}), j=1,...,r         (referred to as 'o')
    - k(\phi_{ik}, \ph_{ij}), j,k=1,...,r    (referred to as '*')

    Args:
        pde (PseudoParabolicPDE): The pseudo-parabolic PDE, providing the functions a and c.
        spatial (SpatialDiscParameters): Spatial discretization parameters;
            provides the number of
            spatial subintervals `N`, the polynomial degree `k` of the
            P_k-FEM (k=1 corresponds to a P_1-FEM), the spatial mesh
            `Delta` and the spatial step size `h`.
        quadrature (Quadrature): Quadrature rule; provides the Gauss-Lobatto
            nodes and weights used for numerical integration.
        ref_functions (PkLegendreFEM): Shape functions \psi_L, \psi_R and
            integrated Legendre polynomials, evaluated at the quadrature
            nodes.

    Returns:
        tuple[csr_array, csr_array, csr_array]: Tuple, consisting of
            - **csr_array**: Mass matrix.
            - **csr_array**: Matrix subjected to the operator L.
            - **csr_array**: Matrix subjected to the operator M.
    """

    def build_block(data : ndarray, rows : ndarray, cols : ndarray, shape : Tuple[int, int]) -> csr_array:

        r"""Internal support function, to assemble the sparse matrix blocks.

        This function constructs a sparse matrix block in CSR format from
        the given data and index arrays. It is used to initialize
        the x-, o-, and *-blocks of the global matrices.

        Args:
            data (ndarray): Nonzero entries of the sparse block.
            rows (ndarray): Row indices corresponding to 'data'.
            cols (ndarray): Column indices corresponding to 'data'.
            shape (Tuple[int, int]): Shape of the resulting csr_array

        Returns:
            csr_array: Data, stored in a csr_array at [rows, cols].
        """

        return coo_array((data, (rows.astype(int), cols.astype(int))), shape).tocsr()

    N = spatial.N
    k = spatial.k
    Delta = spatial.Delta
    h = spatial.h

    assert N is not None and k is not None and Delta is not None and h is not None, \
        "spatial must provide N, k, Delta and h for a FEM discretization."

    r = k - 1

    x, w = quadrature.nodes_and_weights
    psi_l_v, psi_r_v, le_n_g, _ = ref_functions.evaluated_reference_functions

    a_v = pde.func_a(int_mapp(x, np.array([-1,1]), np.array([Delta[:-1], Delta[1:]]).T))    # the function a evaluated at x
    c_v = pde.func_c(int_mapp(x, np.array([-1,1]), np.array([Delta[:-1], Delta[1:]]).T))    # the function c evaluated at x

    # shape of all matrices
    shape_matrices = ((r+1)*N-1, (r+1)*N-1)

    # ----------------------------------------------------------------------------------------------------
    # x BLOCKS
    # ----------------------------------------------------------------------------------------------------

    # rows and cols indices
    rows_diag = cols_diag = (r+1)*np.linspace(1,N-1,N-1, dtype=np.int64)-1
    rows_off_above = (r+1)*np.linspace(1,N-2,N-2, dtype=np.int64)-1
    cols_off_above = (r+1)*np.linspace(1,N-2,N-2, dtype=np.int64)+r
    rows_off_under = cols_off_above
    cols_off_under = rows_off_above

    # diagonal elements
    # mass
    data_x_diags = h[:-1]/3 + h[1:]/3
    mass_x_diags = build_block(data_x_diags, rows_diag, cols_diag, shape_matrices)

    # stiffness
    data_x_diags = 1/h[:-1] + 1/h[1:]
    stiff_x_diags = build_block(data_x_diags, rows_diag, cols_diag, shape_matrices)

    # reaction_a -> subjected to au
    data_x_diags = h[:-1]/2 * np.dot(a_v[:-1] * psi_r_v**2, w) + h[1:]/2 * np.dot(a_v[1:] * psi_l_v**2, w)
    reaction_a_x_diags = build_block(data_x_diags, rows_diag, cols_diag, shape_matrices)

    # reaction_c -> subjected to cu
    data_x_diags = h[:-1]/2 * np.dot(c_v[:-1] * psi_r_v**2, w) + h[1:]/2 * np.dot(c_v[1:] * psi_l_v**2, w)
    reaction_c_x_diags = build_block(data_x_diags, rows_diag, cols_diag, shape_matrices)

    # off-diagonals
    # mass
    data_x_off_diags = h[1:-1]/6
    mass_x_off_diags_above = build_block(data_x_off_diags, rows_off_above, cols_off_above, shape_matrices)
    mass_x_off_diags_under = build_block(data_x_off_diags, rows_off_under, cols_off_under, shape_matrices)

    # stiffness
    data_x_off_diags = -1/h[1:-1]
    stiff_x_off_diags_above = build_block(data_x_off_diags, rows_off_above, cols_off_above, shape_matrices)
    stiff_x_off_diags_under = build_block(data_x_off_diags, rows_off_under, cols_off_under, shape_matrices)

    # reaction_a -> subjected to au
    data_x_off_diags = h[1:-1]/2*np.dot(a_v[1:-1]*psi_r_v*psi_l_v, w)
    reaction_a_x_off_diags_above = build_block(data_x_off_diags, rows_off_above, cols_off_above, shape_matrices)
    reaction_a_x_off_diags_under = build_block(data_x_off_diags, rows_off_under, cols_off_under, shape_matrices)

    # reaction_c -> subjected to cu
    data_x_off_diags = h[1:-1]/2*np.dot(c_v[1:-1]*psi_r_v*psi_l_v, w)
    reaction_c_x_off_diags_above = build_block(data_x_off_diags, rows_off_above, cols_off_above, shape_matrices)
    reaction_c_x_off_diags_under = build_block(data_x_off_diags, rows_off_under, cols_off_under, shape_matrices)

    # assemble x blocks
    mass_x = mass_x_diags + mass_x_off_diags_above + mass_x_off_diags_under
    stiff_x = stiff_x_diags + stiff_x_off_diags_above + stiff_x_off_diags_under
    reaction_a_x = reaction_a_x_diags + reaction_a_x_off_diags_above + reaction_a_x_off_diags_under
    reaction_c_x = reaction_c_x_diags + reaction_c_x_off_diags_above + reaction_c_x_off_diags_under

    # ----------------------------------------------------------------------------------------------------
    # o BLOCKS
    # ----------------------------------------------------------------------------------------------------
    r"""We distinguish two types of o-block elements:
        k(\phi_{i-1}, \phi_{ij})  and  k(\phi_i, \phi_{ij}),   for j = 1, ..., r.

    We first consider k(\phi_i, \phi_{ij}), referred to as 'r' (right) and 'u' (under):
        x o o x
        o * * o r
        o * * o r
        x o o x
          u u

    Afterwards we will consider k(\phi_{i-1}, \phi_{ij}), referred to as 'l' (left) and 'a' (above):
            a a
          x o o x
        l o * * o
        l o * * o
          x o o x

    Notes:
        - We only have o blocks, if r>0.
        - We don't have to consider the stiffness matrix, since \phi_i' is in \RR and j>0. Therefore
          (\phi_i', \phi'_{ij}) = 1/2 \int_{-1}^1 P_j(x) dx = 0, where P_j denotes the j-th Legendre polynomial.
    """

    if r>0:
        assert le_n_g is not None, "ref_functions has no integrated Legendre polynomials for r > 0."

        # row and column indices
        rows_r = cols_u = np.tile(np.arange(r), N-1) + np.repeat(np.arange(N-1), r) * (r + 1)
        rows_u = cols_r = np.repeat(-1 + (r+1)*np.linspace(1,N-1,N-1, dtype=np.int64), r)
        rows_l = cols_a = np.tile(np.arange(r), N-1) + np.repeat(np.arange(1,N), r) * (r + 1)
        rows_a = cols_l = np.repeat(-1 + (r+1)*np.linspace(1,N-1,N-1, dtype=np.int64), r)

        # r and u
        # mass
        data_o_ru = (np.tile(h[:-1, None]/2, (1,r)) * np.dot(psi_r_v * le_n_g, w)).flatten()
        mass_o_r = build_block(data_o_ru, rows_r, cols_r, shape_matrices)
        mass_o_u = build_block(data_o_ru, rows_u, cols_u, shape_matrices)

        # reaction_a
        data_o_ru = np.repeat(h[:-1]/2, r) * np.dot(np.tile(a_v[:-1, None, :], (1,r,1)) * psi_r_v * le_n_g, w).flatten()
        reaction_a_o_r = build_block(data_o_ru, rows_r, cols_r, shape_matrices)
        reaction_a_o_u = build_block(data_o_ru, rows_u, cols_u, shape_matrices)

        # reaction_c
        data_o_ru = np.repeat(h[:-1]/2, r) * np.dot(np.tile(c_v[:-1, None, :], (1,r,1)) * psi_r_v * le_n_g, w).flatten()
        reaction_c_o_r = build_block(data_o_ru, rows_r, cols_r, shape_matrices)
        reaction_c_o_u = build_block(data_o_ru, rows_u, cols_u, shape_matrices)

        # l and a
        # mass
        data_o_la = (np.tile(h[1:, None]/2, (1,r)) * np.dot(psi_l_v * le_n_g, w)).flatten()
        mass_o_l = build_block(data_o_la, rows_l, cols_l, shape_matrices)
        mass_o_a = build_block(data_o_la, rows_a, cols_a, shape_matrices)

        # reaction_a
        data_o_la = np.repeat(h[1:]/2, r) * np.dot(np.tile(a_v[1:, None, :], (1,r,1)) * psi_l_v * le_n_g, w).flatten()
        reaction_a_o_l = build_block(data_o_la, rows_l, cols_l, shape_matrices)
        reaction_a_o_a = build_block(data_o_la, rows_a, cols_a, shape_matrices)

        # reaction_c
        data_o_la = np.repeat(h[1:]/2, r) * np.dot(np.tile(c_v[1:, None, :], (1,r,1)) * psi_l_v * le_n_g, w).flatten()
        reaction_c_o_l = build_block(data_o_la, rows_l, cols_l, shape_matrices)
        reaction_c_o_a = build_block(data_o_la, rows_a, cols_a, shape_matrices)

        # assemble o blocks
        mass_o = mass_o_r + mass_o_u + mass_o_l + mass_o_a
        reaction_a_o = reaction_a_o_r + reaction_a_o_u + reaction_a_o_l + reaction_a_o_a
        reaction_c_o = reaction_c_o_r + reaction_c_o_u + reaction_c_o_l + reaction_c_o_a


    # ----------------------------------------------------------------------------------------------------
    # * BLOCKS
    # ----------------------------------------------------------------------------------------------------
    """Notes:
        - We only have * blocks, if r>0.
    """

    if r > 0:
        assert le_n_g is not None, "ref_functions has no integrated Legendre polynomials for r > 0."

        # row and column indices, except for the stiffness matrix
        rows_star = np.repeat(np.tile(np.arange(r), N) + np.repeat((r + 1) * np.arange(N), r), r)
        cols_star = np.tile(np.arange(r), r * N) + np.repeat((r + 1) * np.repeat(np.arange(N), r), r)

        # mass
        data_star = (np.tile(h[:, None, None], (1,r,r))/2 * np.dot(le_n_g, (w*le_n_g).T)).flatten()   # type: ignore
        mass_star = build_block(data_star, rows_star, cols_star, shape_matrices)

        # reaction_a
        data_star = (h[:,None, None]/2*np.dot(a_v[:, None, :]*le_n_g, (w*le_n_g).T)).flatten()   # type: ignore
        reaction_a_star = build_block(data_star, rows_star, cols_star, shape_matrices)

        # reaction_c
        data_star = (h[:,None, None]/2*np.dot(c_v[:, None, :]*le_n_g, (w*le_n_g).T)).flatten()   # type: ignore
        reaction_c_star = build_block(data_star, rows_star, cols_star, shape_matrices)

        # row and column indices for the stiffness matrix
        """Due to the orthogonality property of the Legendre polynomials, only the diagonal
        * entries are non zero"""
        rows_star = cols_star = np.tile(np.arange(r), N) + np.repeat((r + 1) * np.arange(N), r)

        # stiffness
        data_star = np.outer(2/h, 2/(2*np.linspace(1,r,r)+1)).flatten()
        stiff_star = build_block(data_star, rows_star, cols_star, shape_matrices)

    # ----------------------------------------------------------------------------------------------------
    # ASSEMBLE mass, stiff, reaction_a and reaction_c
    # ----------------------------------------------------------------------------------------------------

    if r==0:
        mass = mass_x
        stiff = stiff_x
        reaction_a = reaction_a_x
        reaction_c = reaction_c_x
    else:
        mass = mass_x + mass_o + mass_star
        stiff = stiff_x + stiff_star
        reaction_a = reaction_a_x + reaction_a_o + reaction_a_star
        reaction_c = reaction_c_x + reaction_c_o + reaction_c_star

    # The matrices, subjected to the operators L and M
    matrix_L = stiff + reaction_a
    matrix_M = stiff + reaction_c

    return mass, matrix_L, matrix_M
