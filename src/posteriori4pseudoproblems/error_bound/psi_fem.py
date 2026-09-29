import numpy as np

from numpy import ndarray
from scipy.sparse import csr_array
from scipy.sparse.linalg import spsolve
from ..interval_transformation import interval_transformation as int_mapp
from ..pseudo_parabolic_pde_class import PseudoParabolicPDE
from ..discretization_dataclasses import SpatialDiscParameters
from ..quadrature import Quadrature
from ..legendre_galerkin import PkLegendreFEM
from ..time_discretization_methods import FemTimeSteppingParameters


def _assemble_load_vector(pde: PseudoParabolicPDE,
                          spatial: SpatialDiscParameters,
                          quadrature: Quadrature,
                          ref_functions: PkLegendreFEM,
                          t: float) -> ndarray:

    r""" The FEM load vector (f(t), \phi_i)_h, i.e. the discrete L^2 inner
    product of the source term f(t) with every interior basis function
    \phi_i of V^0_h. """

    N = spatial.N
    k = spatial.k
    Delta = spatial.Delta
    h = spatial.h
    assert N is not None and k is not None and Delta is not None and h is not None, \
        "spatial must provide N, k, Delta and h for a FEM discretization."

    r = k - 1

    x, w = quadrature.nodes_and_weights
    psi_l_v, psi_r_v, le_n_g, _ = ref_functions.evaluated_reference_functions
    ref_array = np.array([-1, 1])

    x_v = int_mapp(x, ref_array, np.array([Delta[:-1], Delta[1:]]).T)
    xt_points = np.column_stack((x_v.ravel(), np.full(x_v.size, t)))
    f_v = pde.func_F(xt_points).reshape(x_v.shape)

    load_vector = np.zeros((r + 1) * N - 1)

    # (f, \phi_i)
    l = np.linspace(1, N - 1, N - 1, True, dtype=int)
    load_vector[(r + 1) * l - 1] = h[:-1] / 2 * (f_v[:-1] * psi_r_v).dot(w) + h[1:] / 2 * (f_v[1:] * psi_l_v).dot(w)

    # (f, \phi_{i,mu})
    if r > 0:
        assert le_n_g is not None, "ref_functions has no integrated Legendre polynomials for r > 0."
        l = np.linspace(0, N - 1, N, True, dtype=int)
        for mu in range(r):
            load_vector[(r + 1) * l + mu] = h / 2 * np.dot(f_v * le_n_g[mu], w)

    return load_vector


def compute_psi(pde: PseudoParabolicPDE,
                spatial: SpatialDiscParameters,
                quadrature: Quadrature,
                ref_functions: PkLegendreFEM,
                stepping_params: FemTimeSteppingParameters,
                sol_vector: ndarray,
                t: float) -> ndarray:

    r""" Computes the function \psi^j_h of the elliptic reconstruction,
    defined by

    a_h(\psi^j_h, v_h) = c_h(u^j_h, v_h) - (f^j, v_h)   for all v_h \in V^0_h,

    where a_h(,) and c_h(,) are the bilinear forms induced by the operators L
    and M, associated with `stepping_params.matrix_L` and
    `stepping_params.matrix_M`. In matrix form, restricted to the interior
    degrees of freedom of V^0_h,

    matrix_L \psi^j = matrix_M u^j_h - f^j,

    where f^j is the FEM load vector (f(t_j), \phi_i)_h.

    Args:
        pde (PseudoParabolicPDE): The pseudo-parabolic PDE, providing the
            source term F.
        spatial (SpatialDiscParameters): Spatial discretization parameters;
            provides the number of spatial subintervals `N`, the polynomial
            degree `k` of the P_k-FEM, the spatial mesh `Delta` and the
            spatial step size `h`.
        quadrature (Quadrature): Quadrature rule; provides the Gauss-Lobatto
            nodes and weights used for numerical integration.
        ref_functions (PkLegendreFEM): Shape functions \psi_L, \psi_R and
            integrated Legendre polynomials, evaluated at the quadrature
            nodes.
        stepping_params (FemTimeSteppingParameters): State of the FEM
            time-stepping scheme; provides the matrices L and M.
        sol_vector (ndarray): Coefficients of the FEM solution u^j_h, as
            e.g. returned by `bdf_fem_1d`.
        t (float): The time t_j at which u^j_h approximates the solution.

    Returns:
        ndarray: Coefficients of \psi^j_h, laid out like `sol_vector` (with
            the two boundary degrees of freedom set to zero).

    Raises:
        TypeError: If the matrices L or M stored in `stepping_params` are
            not scipy.sparse csr_array sparse arrays.
    """

    if not isinstance(stepping_params.matrix_L, csr_array) or not isinstance(stepping_params.matrix_M, csr_array):
        raise TypeError("stepping_params.matrix_L and stepping_params.matrix_M must be scipy.csr_array sparse arrays.")

    N = spatial.N
    k = spatial.k
    assert N is not None and k is not None, "spatial must provide N and k for a FEM discretization."
    r = k - 1

    load_vector = _assemble_load_vector(pde, spatial, quadrature, ref_functions, t)
    rhs = stepping_params.matrix_M.dot(sol_vector[1:-1]) - load_vector

    psi_vector = np.zeros((r + 1) * N + 1)
    psi_vector[1:-1] = np.asarray(spsolve(stepping_params.matrix_L, rhs))

    return psi_vector
