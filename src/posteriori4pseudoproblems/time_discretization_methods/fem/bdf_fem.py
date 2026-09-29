import numpy as np

from numpy import ndarray
from scipy.sparse import csr_array
from ...interval_transformation import interval_transformation as int_mapp
from ...pseudo_parabolic_pde_class import PseudoParabolicPDE
from ...discretization_dataclasses import SpatialDiscParameters, TemporalDiscParameters
from ...quadrature import Quadrature
from ...legendre_galerkin import PkLegendreFEM
from ..bdf_coefficients import bdf_coefficients
from .fem_time_stepping_parameters import FemTimeSteppingParameters

def bdf_fem_1d(pde : PseudoParabolicPDE,
               spatial : SpatialDiscParameters,
               temporal : TemporalDiscParameters,
               stepping_params : FemTimeSteppingParameters,
               quadrature : Quadrature,
               ref_functions : PkLegendreFEM,
               j : int) -> ndarray:

    r"""
    BDF method for a FEM discretization in a spatial domain [a,b].

    This function computes an approximation u^j_h of the solution u(t_j)
    to a pseudo-parabolic equation of the form

    L \partial_t u(t) + M u(t) = f(t),

    using the BDF time discretization:

    matrix_L D^1_{t,k}u^j_h + matrix_M u^j_h = f(t_j),

    where u^{j-1}_h approximates u(t_{j-1}) and
    D^1_{t,k}u^j_h denotes the BDF operator of order k = `stepping_params.num_prev`.

    Args:
        pde (PseudoParabolicPDE): The pseudo-parabolic PDE, providing the
            spatial domain [a,b] and the right-hand side F.
        spatial (SpatialDiscParameters): Spatial discretization parameters;
            provides the number of spatial subintervals `N`, the polynomial
            degree `k` of the P_k-FEM, the spatial mesh `Delta` and the
            spatial step size `h`.
        temporal (TemporalDiscParameters): Temporal discretization parameters,
            providing the temporal mesh t_0,...,t_M.
        stepping_params (FemTimeSteppingParameters): State of the FEM time-stepping
            scheme; provides the order `bdf_order = num_prev` of the BDF
            method, the matrices L and M, the cached LU
            factorization and the `bdf_order` previous approximations
            u^{j-1}_h,...,u^{j-k}_h, where k denotes `bdf_order`. Updated
            in place with the newly computed approximation u^j_h.
        quadrature (Quadrature): Quadrature rule; provides the Gauss-Lobatto
            nodes and weights used for numerical integration.
        ref_functions (PkLegendreFEM): Shape functions \psi_L, \psi_R and
            integrated Legendre polynomials, evaluated at the quadrature
            nodes.
        j (int): Index of the time level t_j to advance to; t_{j-1},...,t_{j-k}
            are the `bdf_order` previous time levels the BDF operator
            relies on.

    Returns:
        ndarray: Coefficients of the approximate solution u_h^j.

    Raises:
        TypeError: If the matrices L or M stored in `stepping_params` are not
            scipy.sparse csr_array sparse arrays.
    """

    # Validate sparse matrices
    if not isinstance(stepping_params.matrix_L, csr_array) or not isinstance(stepping_params.matrix_M, csr_array):
        raise TypeError("stepping_params.matrix_L and stepping_params.matrix_M must be scipy.csr_array sparse arrays.")

    N = spatial.N
    Delta = spatial.Delta
    h = spatial.h

    assert N is not None and spatial.k is not None and Delta is not None and h is not None, \
        "spatial must provide N, k, Delta and h for a FEM discretization."
    assert stepping_params.prev_solutions is not None, "stepping_params has no previous approximations u^{j-1}_h,...,u^{j-k}_h to advance from."

    r = spatial.k - 1

    bdf_order = stepping_params.num_prev

    assert temporal.temporal_mesh is not None, "temporal must provide temporal_mesh to advance the solution."

    # t_{j-k},...,t_j
    t_array = temporal.temporal_mesh[j-bdf_order:j+1]

    x, w = quadrature.nodes_and_weights
    psi_l_v, psi_r_v, le_n_g, _ = ref_functions.evaluated_reference_functions

    # the array [-1,1]
    ref_array = np.array([-1,1])

    # determination of the result_vector
    # x_v has shape (N, len(x)): one row of quadrature points per spatial interval.
    x_v = int_mapp(x, ref_array, np.array([Delta[:-1], Delta[1:]]).T)
    xt_points = np.column_stack((x_v.ravel(), np.full(x_v.size, t_array[-1])))
    f_j = pde.func_F(xt_points).reshape(x_v.shape)

    l = np.linspace(1,N-1,N-1, True, dtype = int)
    result_vector = np.zeros((r+1)*N-1)
    sol_vector = np.zeros((r+1)*N+1)

    # (f,\phi_i)
    result_vector[(r+1)*l-1] = h[:-1]/2*(f_j[:-1] * psi_r_v).dot(w) + h[1:]/2*(f_j[1:]*psi_l_v).dot(w)

    # (f, \phi_{i,j})
    if r > 0:
        assert le_n_g is not None, "ref_functions has no integrated Legendre polynomials for r > 0."
        l = np.linspace(0,N-1,N, True, dtype = int)
        for mu in range(0,r):
            result_vector[(r+1)*l+mu] = h/2*np.dot(f_j * le_n_g[mu], w)

    # Compute the bdf coefficients
    bdf = bdf_coefficients(t_array, bdf_order)

    # LU factorization of the coefficient matrix matrix_L * bdf[0] + matrix_M, cached in stepping_params
    # and recomputed only if the leading BDF coefficient bdf[0] changed since the previous time step.
    lu_coeff = stepping_params.get_lu(bdf[0])

    # Compute \sum_{l=1}^k u^{j-1}_h * bdf[l]
    previous = np.zeros(np.shape(stepping_params.prev_solutions)[1])
    for l in range(1,bdf_order+1):
        previous[1:-1] += bdf[l] * stepping_params.prev_solutions[-l, 1:-1]

    # (f, \phi_i) - matrix_L * \sum_{l=1}^k u^{j-1}_h * bdf[l]
    result_vector =  result_vector - stepping_params.matrix_L.dot(previous[1:-1])

    # the coefficients for u^j_h
    sol_vector = np.zeros((r+1)*N+1)
    sol_vector[1:-1] = np.asarray(lu_coeff.solve(result_vector))

    stepping_params.update(sol_vector)

    return sol_vector
