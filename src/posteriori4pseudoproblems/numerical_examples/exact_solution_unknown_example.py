import numpy as np

from scipy.sparse import csr_array
from posteriori4pseudoproblems.error_bound import ErrorBoundEvaluationFEM, RESULTS_DIR, compute_psi
from posteriori4pseudoproblems.pseudo_parabolic_pde_class import PseudoParabolicPDE
from posteriori4pseudoproblems.discretization_dataclasses import SpatialDiscParameters, TemporalDiscParameters
from posteriori4pseudoproblems.quadrature import GaussLobatto
from posteriori4pseudoproblems.legendre_galerkin import (PkLegendreFEM, SpectralLegendre, evaluate_spectral_solution,
                                                         evaluate_spectral_derivative,
                                                         evaluate_pk_solution_at_quadrature_nodes,
                                                         evaluate_pk_derivative_at_quadrature_nodes)
from posteriori4pseudoproblems.projection_operators import project_onto_pk_basis, elliptic_projection_fem
from posteriori4pseudoproblems.matrix_assembly import (assemble_coeff_matrices_fem_1d, assemble_coeff_matrices_spectral,
                                                       slice_coefficient_matrices)
from posteriori4pseudoproblems.time_discretization_methods import FemTimeSteppingParameters, bdf_fem_1d
from posteriori4pseudoproblems.reference_solution import compute_reference_solution
from posteriori4pseudoproblems.norms import l2_norm_function_fem, h1_norm_function_fem


r""" The numerical example of the paper: the error and the a posteriori
error estimator, with respect to the L^2 or the H^1 norm, of a
BDF-2-in-time, P_k-FEM-in-space discretization of the pseudo-parabolic
problem

    -u_{xxt} + (5x+6) u_t - u_{xx} - e^{-x} u = e^{2t} + \cos(\pi(x+t)^2)   on (-1,1) x (0,2],

i.e. L u_t + M u = F with Lu = -u'' + au, Mu = -u'' + cu, a(x) = 5x+6 and
c(x) = -e^{-x}, initial condition u_0(x) = \sin(\pi x) and homogeneous
Dirichlet boundary conditions, for M = N = 2^p, p = 6,...,13.

Since the exact solution is unknown, the error of u^M_h is measured against
a reference solution at T, computed by a dG(2)-in-time, spectral-Galerkin-
in-space discretization (ansatz-space dimension DIM_V).

For the L^2 estimator, P_1 elements are employed (k=1), and the Galerkin
projection of u_0 with respect to L serves as u^0_h; for the H^1 estimator,
P_2 elements are employed (k=2), and the L^2 projection of u_0 serves as
u^0_h. The starting value u^1_h is computed by the backward Euler method.
The higher-order approximations v_h, needed for the eta_ell component of the
estimator, are computed the same way, but with P_{k+1} elements; the
coefficient matrices are assembled for the P_{k+1}-FEM only, and those of
the P_k-FEM are obtained by slicing them. All spatial integrals are
approximated by the Gauss-Lobatto formula with N_QUAD_FEM = 4 nodes, which
is exact for polynomials of degree 5.
"""

SPATIAL_INTERVAL = (-1.0, 1.0)
T_FINAL = 2.0
P_VALUES = range(6, 14)
DIM_V = 30
N_QUAD_SPECTRAL = 32
N_QUAD_FEM = 4
RESULTS_PATH = RESULTS_DIR / "exact_solution_unknown_example"

# The polynomial degree k of the P_k-FEM employed for each norm
K_VALUES = {"l2": 1, "h1": 2}


def source(xt_points: np.ndarray) -> np.ndarray:
    r""" The source function F(x,t) = e^{2t} + \cos(\pi(x+t)^2).

    Args:
        xt_points (np.ndarray): Points in the space-time domain, a
            two-dimensional array whose first column is associated with x
            and whose last column is associated with t.

    Returns:
        np.ndarray: F evaluated at `xt_points`.
    """

    x = xt_points[:, 0]
    t = xt_points[:, -1]
    return np.exp(2*t) + np.cos(np.pi*(x+t)**2)


def initial_condition(x: np.ndarray) -> np.ndarray:
    r""" The initial condition u_0(x) = \sin(\pi x).

    Args:
        x (np.ndarray): Points in the spatial domain.

    Returns:
        np.ndarray: u_0 evaluated at `x`.
    """

    return np.sin(np.pi*x)


def derivative_initial_condition(x: np.ndarray) -> np.ndarray:
    r""" The derivative u'_0(x) = \pi \cos(\pi x) of the initial condition.

    Args:
        x (np.ndarray): Points in the spatial domain.

    Returns:
        np.ndarray: u'_0 evaluated at `x`.
    """

    return np.pi*np.cos(np.pi*x)


def build_pde() -> PseudoParabolicPDE:
    r""" Builds the pseudo-parabolic PDE of the example and computes its
    generic constants.

    Returns:
        PseudoParabolicPDE: The PDE with a(x) = 5x+6, c(x) = -e^{-x}, the
            source function `source`, the initial condition
            `initial_condition` and homogeneous Dirichlet boundary conditions.
    """

    pde = PseudoParabolicPDE(
        F=source,
        u0=initial_condition,
        Psi=lambda xt: np.zeros(xt.shape[0]),
        a=lambda x: 5*x + 6,
        c=lambda x: -np.exp(-x),
        T=T_FINAL,
        spatial_interval=SPATIAL_INTERVAL,
        der_u0=derivative_initial_condition,
    )
    # C_a, c_a, C_c, c_c, M_2_star, omega_2_star, C_I, L_inverse are needed by the estimator.
    pde.compute_generic_constants()
    return pde


def _build_spectral_setup(dim_V: int) -> tuple[SpatialDiscParameters, GaussLobatto, SpectralLegendre]:
    r""" Builds the data of the spectral Galerkin method used for the
    reference solution.

    Args:
        dim_V (int): Dimension of the ansatz space.

    Returns:
        tuple[SpatialDiscParameters, GaussLobatto, SpectralLegendre]: The
            spatial discretization parameters, the Gauss-Lobatto quadrature
            rule with N_QUAD_SPECTRAL nodes and the reference functions,
            evaluated at its nodes.
    """

    quadrature = GaussLobatto(N_QUAD_SPECTRAL)
    ref_functions = SpectralLegendre(dim_V=dim_V, spatial_interval=SPATIAL_INTERVAL)
    ref_functions.evaluate_reference_function_for_quadrature(quadrature)
    spatial_disc_data = SpatialDiscParameters(disc_type="spectral", dim_V=dim_V)

    return spatial_disc_data, quadrature, ref_functions


def _build_fem_setup(k: int, N: int, quadrature: GaussLobatto) -> tuple[SpatialDiscParameters, PkLegendreFEM]:
    r""" Builds the data of a P_k-FEM on an equidistant mesh of the spatial
    domain.

    Args:
        k (int): Polynomial degree of the P_k-FEM.
        N (int): Number of spatial subintervals.
        quadrature (GaussLobatto): Quadrature rule, at whose nodes the
            reference functions are evaluated.

    Returns:
        tuple[SpatialDiscParameters, PkLegendreFEM]: The spatial
            discretization parameters and the reference functions of the
            P_k-FEM, evaluated at the quadrature nodes.
    """

    a, b = SPATIAL_INTERVAL
    Delta = np.linspace(a, b, N + 1)
    h = np.diff(Delta)

    spatial_disc_data = SpatialDiscParameters(disc_type="fem", N=N, k=k, Delta=Delta, h=h)
    ref_functions = PkLegendreFEM(k=k, spatial_interval=SPATIAL_INTERVAL)
    ref_functions.evaluate_reference_function_for_quadrature(quadrature)

    return spatial_disc_data, ref_functions


def _initial_value(pde: PseudoParabolicPDE,
                   norm_used: str,
                   spatial_disc_data: SpatialDiscParameters,
                   quadrature: GaussLobatto,
                   ref_functions: PkLegendreFEM,
                   mass: csr_array,
                   matrix_L: csr_array) -> np.ndarray:
    r""" The initial approximation u^0_h: the Galerkin projection of u_0
    with respect to L for the L^2 estimator, the L^2 projection of u_0 for
    the H^1 estimator.

    Args:
        pde (PseudoParabolicPDE): The pseudo-parabolic PDE, providing u_0 and
            the function a.
        norm_used (str): The norm of the estimator: "l2" or "h1".
        spatial_disc_data (SpatialDiscParameters): Spatial discretization
            parameters of the P_k-FEM.
        quadrature (GaussLobatto): Quadrature rule; provides the nodes and
            weights used for numerical integration.
        ref_functions (PkLegendreFEM): Reference functions of the P_k-FEM,
            evaluated at the quadrature nodes.
        mass (csr_array): Mass matrix of the P_k-FEM.
        matrix_L (csr_array): Matrix subjected to the operator L of the
            P_k-FEM.

    Returns:
        np.ndarray: Coefficients of u^0_h.
    """

    if norm_used == "l2":
        return elliptic_projection_fem(initial_condition, derivative_initial_condition, pde, spatial_disc_data,
                                       quadrature, ref_functions, matrix_L)

    return project_onto_pk_basis(pde.func_u0, spatial_disc_data, quadrature, ref_functions, mass)


def _backward_euler_value(pde: PseudoParabolicPDE,
                          spatial_disc_data: SpatialDiscParameters,
                          temporal_disc_data: TemporalDiscParameters,
                          quadrature: GaussLobatto,
                          ref_functions: PkLegendreFEM,
                          stepping_params: FemTimeSteppingParameters) -> np.ndarray:
    r""" The starting value u^1_h (or v^1_h), obtained from u^0_h (v^0_h)
    by a single backward Euler step, i.e. the BDF-1 method, on [t_0, t_1].

    Args:
        pde (PseudoParabolicPDE): The pseudo-parabolic PDE, providing the
            source term F.
        spatial_disc_data (SpatialDiscParameters): Spatial discretization
            parameters of the P_k-FEM.
        temporal_disc_data (TemporalDiscParameters): Temporal discretization
            parameters, providing the temporal mesh t_0,...,t_M.
        quadrature (GaussLobatto): Quadrature rule; provides the nodes and
            weights used for numerical integration.
        ref_functions (PkLegendreFEM): Reference functions of the P_k-FEM,
            evaluated at the quadrature nodes.
        stepping_params (FemTimeSteppingParameters): State of the BDF-2
            time-stepping scheme; provides the matrices L and M and, as its
            latest approximation, u^0_h (v^0_h). It is not modified.

    Returns:
        np.ndarray: Coefficients of u^1_h (v^1_h).
    """

    assert stepping_params.prev_solutions is not None
    euler_params = FemTimeSteppingParameters(disc_type="backward_euler", num_prev=1,
                                             matrix_L=stepping_params.matrix_L, matrix_M=stepping_params.matrix_M)
    euler_params.update(stepping_params.prev_solutions[-1])

    return bdf_fem_1d(pde, spatial_disc_data, temporal_disc_data, euler_params, quadrature, ref_functions, 1)


def run_bdf2_fem(pde: PseudoParabolicPDE, norm_used: str, M: int, coeffs_ref: np.ndarray,
                 evaluation: ErrorBoundEvaluationFEM) -> float:
    r""" Runs the BDF-2, P_k-FEM discretization (alongside the P_{k+1}-FEM
    higher-order approximation v_h) on a mesh with N=M spatial subintervals,
    updates `evaluation`'s estimator components along the way, and returns
    the error of u^M_h against the reference solution at T_FINAL in the
    norm `norm_used`.

    Args:
        pde (PseudoParabolicPDE): The pseudo-parabolic PDE of the example.
        norm_used (str): The norm of the error and the estimator: "l2" or
            "h1"; determines k via K_VALUES.
        M (int): Number of temporal subintervals, which equals the number
            of spatial subintervals N.
        coeffs_ref (np.ndarray): Spectral coefficients of the reference
            solution at T_FINAL.
        evaluation (ErrorBoundEvaluationFEM): Evaluation of the estimator,
            whose components are updated in place.

    Returns:
        float: The error ||u(T) - u^M_h|| in the norm `norm_used`.
    """

    k = K_VALUES[norm_used]
    quadrature = GaussLobatto(N_QUAD_FEM)
    spatial, ref_functions = _build_fem_setup(k, M, quadrature)
    spatial_hat, ref_functions_hat = _build_fem_setup(k + 1, M, quadrature)

    # The coefficient matrices of the P_{k+1}-FEM, and those of the P_k-FEM obtained by slicing them
    mass_hat, matrix_L_hat, matrix_M_hat = assemble_coeff_matrices_fem_1d(pde, spatial_hat, quadrature,
                                                                          ref_functions_hat)
    mass, matrix_L, matrix_M = slice_coefficient_matrices(spatial, spatial_hat, mass_hat, matrix_L_hat,
                                                          matrix_M_hat)

    temporal_mesh = np.linspace(0.0, T_FINAL, M + 1)
    temporal_disc_data = TemporalDiscParameters(disc_type="bdf2", M=M, temporal_mesh=temporal_mesh)

    fem_params = FemTimeSteppingParameters(disc_type="bdf2", num_prev=2, matrix_L=matrix_L, matrix_M=matrix_M)
    fem_params_hat = FemTimeSteppingParameters(disc_type="bdf2", num_prev=2,
                                               matrix_L=matrix_L_hat, matrix_M=matrix_M_hat)

    # u^0_h, v^0_h
    u_0 = _initial_value(pde, norm_used, spatial, quadrature, ref_functions, mass, matrix_L)
    v_0 = _initial_value(pde, norm_used, spatial_hat, quadrature, ref_functions_hat, mass_hat, matrix_L_hat)
    fem_params.update(u_0)
    fem_params_hat.update(v_0)
    fem_params.update_psi(compute_psi(pde, spatial, quadrature, ref_functions, fem_params, u_0, temporal_mesh[0]))

    evaluation.update_eta_init(pde, spatial, quadrature, ref_functions, fem_params)

    sol_vector = u_0
    for j in range(1, M + 1):

        # u^1_h, v^1_h by the backward Euler method; u^j_h, v^j_h, j>1, by the BDF-2 method
        if j == 1:
            sol_vector = _backward_euler_value(pde, spatial, temporal_disc_data, quadrature, ref_functions,
                                               fem_params)
            fem_params.update(sol_vector)
            fem_params_hat.update(_backward_euler_value(pde, spatial_hat, temporal_disc_data, quadrature,
                                                        ref_functions_hat, fem_params_hat))
        else:
            sol_vector = bdf_fem_1d(pde, spatial, temporal_disc_data, fem_params, quadrature, ref_functions, j)
            bdf_fem_1d(pde, spatial_hat, temporal_disc_data, fem_params_hat, quadrature, ref_functions_hat, j)

        psi_j = compute_psi(pde, spatial, quadrature, ref_functions, fem_params, sol_vector, temporal_mesh[j])
        fem_params.update_psi(psi_j)

        evaluation.update_components(pde, temporal_disc_data, spatial, spatial_hat, quadrature, ref_functions,
                                     ref_functions_hat, fem_params, fem_params_hat, j)

    # The error ||u(T) - u^M_h|| against the reference solution
    u_h_v = evaluate_pk_solution_at_quadrature_nodes(sol_vector, spatial, ref_functions)
    u_ref = lambda x: evaluate_spectral_solution(coeffs_ref, x, pde)

    if norm_used == "l2":
        return l2_norm_function_fem(u_ref, u_h_v, spatial, quadrature)

    der_u_h_v = evaluate_pk_derivative_at_quadrature_nodes(sol_vector, spatial, ref_functions)
    der_u_ref = lambda x: evaluate_spectral_derivative(coeffs_ref, x, pde)

    return h1_norm_function_fem(u_ref, u_h_v, der_u_ref, der_u_h_v, spatial, quadrature)


def main() -> None:
    r""" Runs the example for the L^2 and the H^1 estimator and writes the
    results as LaTeX tables to RESULTS_PATH. """

    pde = build_pde()

    print(f"Computing reference solution (spectral dG(2), dim_V={DIM_V})...")
    spectral_spatial, spectral_quadrature, spectral_ref_functions = _build_spectral_setup(DIM_V)
    spectral_mass, spectral_matrix_L, spectral_matrix_M = assemble_coeff_matrices_spectral(
        pde, spectral_spatial, spectral_quadrature, spectral_ref_functions)
    coeffs_ref = compute_reference_solution(pde, spectral_quadrature, spectral_ref_functions, spectral_mass,
                                            spectral_matrix_L, spectral_matrix_M, T=T_FINAL)

    for norm_used, norm_name in (("l2", "L^2"), ("h1", "H^1")):
        k = K_VALUES[norm_used]
        print(f"Running BDF-2 (P_{k}-FEM, N=M) with the {norm_name} estimator...")

        evaluation = ErrorBoundEvaluationFEM(norm_used=norm_used)
        for p in P_VALUES:
            M = 2**p
            error = run_bdf2_fem(pde, norm_used, M, coeffs_ref, evaluation)
            evaluation.update(M=M, N=M, error=error)
            print(f"  M = N = {M}: error = {error:.3e}")

        filename = f"bdf2_{norm_used}.txt"
        evaluation.write_to_file(
            filename, str(RESULTS_PATH),
            caption=f"BDF-2, P_{k}-FEM: results of the {norm_name} estimator",
            label=f"tab:exact-solution-unknown-example-{norm_used}",
            components_caption=f"BDF-2, P_{k}-FEM: components of the {norm_name} estimator",
            components_label=f"tab:exact-solution-unknown-example-{norm_used}-components",
        )
        print(f"  results written to {RESULTS_PATH / filename}")


if __name__ == "__main__":
    main()
