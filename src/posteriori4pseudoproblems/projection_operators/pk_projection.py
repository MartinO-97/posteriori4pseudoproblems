import numpy as np

from typing import Callable
from numpy import ndarray
from scipy.sparse import csr_array
from scipy.sparse.linalg import spsolve
from ..interval_transformation import interval_transformation as int_mapp
from ..discretization_dataclasses import SpatialDiscParameters
from ..quadrature import Quadrature
from ..legendre_galerkin import PkLegendreFEM

def project_onto_pk_basis(func: Callable[[ndarray], ndarray],
                          spatial: SpatialDiscParameters,
                          quadrature: Quadrature,
                          ref_functions: PkLegendreFEM,
                          mass: csr_array) -> ndarray:

    r""" L2-projects `func` onto the P_k-FEM ansatz space spanned by the
    'left'/'right' hat functions \psi_L, \psi_R and the integrated
    Legendre polynomials N_1,...,N_r, r=k-1, by solving

    mass @ dof = rhs,   rhs_i = \int_a^b func(x) \phi_i(x) dx,

    for every FEM basis function \phi_i, and embedding the resulting
    interior degrees of freedom `dof` into the coefficient vector

    [a_0, b_{1,1},...,b_{1,r}, a_1, b_{2,1},...,b_{2,r}, a_2, ..., a_N],

    with homogeneous Dirichlet boundary values a_0 = a_N = 0, as expected
    by `evaluate_pk_solution`.

    Args:
        func (Callable[[ndarray], ndarray]): The function to project, e.g.
            the initial condition u_0.
        spatial (SpatialDiscParameters): Spatial discretization parameters;
            provides the number of spatial subintervals `N`, the polynomial
            degree `k` of the P_k-FEM, the spatial mesh `Delta` and the
            spatial step size `h`.
        quadrature (Quadrature): Quadrature rule; provides the Gauss-Lobatto
            nodes and weights used for numerical integration.
        ref_functions (PkLegendreFEM): Shape functions \psi_L, \psi_R and
            integrated Legendre polynomials, evaluated at the quadrature
            nodes.
        mass (csr_array): Mass matrix of the P_k-FEM ansatz space, e.g. as
            returned by `assemble_coeff_matrices_fem_1d`.

    Returns:
        ndarray: Coefficients of the L2 projection of `func`, of length
            (r+1)*N+1.
    """

    N = spatial.N
    Delta = spatial.Delta
    h = spatial.h

    assert N is not None and spatial.k is not None and Delta is not None and h is not None, \
        "spatial must provide N, k, Delta and h for a FEM discretization."

    r = spatial.k - 1

    x, w = quadrature.nodes_and_weights
    psi_l_v, psi_r_v, le_n_g, _ = ref_functions.evaluated_reference_functions

    x_v = int_mapp(x, np.array([-1, 1]), np.array([Delta[:-1], Delta[1:]]).T)
    f_v = func(x_v)

    # (func, \phi_i), i=1,...,N-1
    l = np.linspace(1, N - 1, N - 1, True, dtype=int)
    rhs = np.zeros((r + 1) * N - 1)
    rhs[(r + 1) * l - 1] = h[:-1] / 2 * (f_v[:-1] * psi_r_v).dot(w) + h[1:] / 2 * (f_v[1:] * psi_l_v).dot(w)

    # (func, \phi_{i,mu}), i=1,...,N, mu=1,...,r
    if r > 0:
        assert le_n_g is not None, "ref_functions has no integrated Legendre polynomials for r > 0."
        l = np.linspace(0, N - 1, N, True, dtype=int)
        for mu in range(0, r):
            rhs[(r + 1) * l + mu] = h / 2 * np.dot(f_v * le_n_g[mu], w)

    dof = spsolve(mass, rhs)

    sol_vector = np.zeros((r + 1) * N + 1)
    sol_vector[1:-1] = dof

    return sol_vector
