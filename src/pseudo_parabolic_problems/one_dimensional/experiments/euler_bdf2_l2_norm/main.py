# Absolute imports
import numpy as np
import pandas as pd

# Relative imports
from .config import *
from ..functions import *
from ...mesh_tools import spatial_mesh_generation as spatial_mesh
from ...matrix_assembly import assemble_coeff_matrices_fem as fem_assembly
from ...approximate_solutions import fem_solution_gauss_lobatto as fem_sol
from ...estimators.order2_fem_l2 import eta_init, eta_f, eta_Psi, eta_delta_psi, eta_R
from ...norms.l2_norm_function_fem import l2_norm_function_fem as l2_norm
from ...projection_operators import elliptic_projection_fem as elliptic_projection
from ...time_discretization_methods.fem import backward_euler_fem as euler_fem, bdf_fem as bdf
from ...quadrature.gauss_lobatto import gauss_lobatto_quadrature
from ...basis_functions.legendre_evaluation import legendre_evaluation
from ...basis_functions.p1_shape_functions_evaluation import p1_shape_functions_evaluation_1d as psi_evaluate
from ...utilities.elliptic_rec_psi import elliptic_rec_psi as det_psi
from ...utilities.show_solution import show_solution
from ...utilities.compute_reference_solution import compute_reference_solution as reference_solution
from ...matrix_assembly.slice_coefficient_matrices import slice_coefficient_matrices 
from ....common import temporal_mesh_generation as time_mesh

def bdf2_l2_norm():

    r"""
    A BDF-2/FEM discretization of a pseudo parabolic partial differential equation:
    
        L \pt_t u + M u = F,        in (a,b) \times (0,T]
                 u(x,t) = 0,        for (x,t) \in {a,b} \times [0,T]
                 u(x,0) = u_0(x)    for x \in [a,b]

    An approximate solution is computed by employing a BDF-2/FEM 
    or backward Euler/FEM discretization. 
    To compute the 'exact' solution, a reference solution is determined by using
    a dG(2)-method in time and a spectral Galerkin method in space is used. 
    """

    # ----------------------------------------------------------------------------------------------------
    # THE GENERIC CONSTANTS C_a, c_a, C_c, c_c, M_2_star, omega_2_star, C_I AND L_inverse
    # ----------------------------------------------------------------------------------------------------
    C_a, c_a, C_c, c_c, M_2_star, omega_2_star, C_I, L_inverse = determine_generic_constants_l2(a,b)

    # ----------------------------------------------------------------------------------------------------
    # INITIALIZATION OF UTILITIES FOR THE GAUSS-LOBATTO QUADRATURE RULE
    # ----------------------------------------------------------------------------------------------------

    # Number of nodes x and weights w for the Gauss-Lobatto quadrature 
    number_gl_nodes_weights_spectral = np.max([np.array([30, dim_V+2])])
    number_gl_nodes_weights_fem = 3

    # Computation of the nodes and weights
    x_spectral, w_spectral = gauss_lobatto_quadrature(number_gl_nodes_weights_spectral)
    x_fem, w_fem = gauss_lobatto_quadrature(number_gl_nodes_weights_fem)

    # Evaluation of the integrated Legendre, the Legendre and the first derivates of the Legendre
    # polynomials for the spectral Galerkin method at x
    le_n_g_spectral, le_p_g_spectral, le_p_g_x_spectral = legendre_evaluation(x_spectral, dim_V)

    # Evaluation of the integrated Legendre, the Legendre and the first derivates of the Legendre
    # polynomials for the FEM method at x
    if not elliptic_estimator:
        le_n_g_elliptic, le_p_g_elliptic, le_p_g_x_elliptic = legendre_evaluation(x_fem, r_elliptic)
        le_n_g, le_p_g, le_p_g_x = le_n_g_elliptic[:r], le_p_g_elliptic[:r], le_p_g_x_elliptic[:r]
    else:
        le_n_g_elliptic, le_p_g_elliptic, le_p_g_x_elliptic = np.array([]), np.array([]), np.array([])
        le_n_g, le_p_g, le_p_g_x = legendre_evaluation(x_fem, r)

    # Evaluation of the shape functions \psi_L and \psi_r for the FEM method at x
    psi_l_v, psi_r_v = psi_evaluate(x_fem)

    # ----------------------------------------------------------------------------------------------------
    # ASSEMBLY OF THE COEFFICIENT MATRICES FOR THE SPECTRAL GALERKIN METHOD
    # ----------------------------------------------------------------------------------------------------

    func_u, func_der_u, z_reference, result_reference, omega_t_reference = \
        reference_solution(x_spectral, w_spectral, func_a, func_c, func_u0, func_f, True, dim_V, a, b, 
                           eps_a, eps_c, le_n_g_spectral, 'dg2', T, 'pseudo')

    # ----------------------------------------------------------------------------------------------------
    # INITIALIZATION OF ARRAYS, THAT STORE THE RESULTS
    # ----------------------------------------------------------------------------------------------------

    # Number of subintervals in time
    result_M = np.zeros(p_end-p_start + 1)

    # Number of spatial subintervals  
    result_N = np.zeros(p_end-p_start + 1)

    # Error ||(u-u_h)(T)||_{1,(a,b)} and its experimental convergence order
    result_err = np.zeros([p_end-p_start + 1, 2])

    # estimator and its experimental convergence order
    result_est = np.zeros([p_end-p_start + 1, 2])

    # Experimental efficiency of the estimator
    result_eff = np.zeros(p_end-p_start + 1)

    # The components \eta_{init}, \eta_f, \eta_{ell}, \eta_{\Psi}, \eta_{\delta_t psi} 
    # and its experimental convergence orders
    result_init = np.zeros([p_end-p_start + 1, 2])
    result_f = np.zeros([p_end-p_start + 1, 2])
    result_ell = np.zeros([p_end-p_start + 1, 2])
    result_Psi = np.zeros([p_end-p_start + 1, 2])
    result_delta_psi = np.zeros([p_end-p_start + 1, 2])
    

    # ----------------------------------------------------------------------------------------------------
    # APPROXIMATE SOLUTION
    # ----------------------------------------------------------------------------------------------------
    r"""sol_vector is associated with the time step t_j, whereas sol_vector_s is associated with t_{j-1}.
        The coefficients for the reference solution is stored on sol_vector_ref, since the the coefficients
        at t_{j-1} are only needed to compute the approximation in t_j."""

    for p in range(p_start, p_end+1):

        # ----------------------------------------------------------------------------------------------------
        # INITIALIZATION OF THE TEMPORAL AND SPATIAL MESHES AND THE POINTS FOR EVALUATION OF THE REFERENCE SOLUTION
        # ----------------------------------------------------------------------------------------------------

        # number of intervals in time and space
        if backward_euler:    
            if p == p_start: 
                N = N_reference = int(2**p)
            else:
                N_reference *= np.sqrt(2) 
                N = int(np.floor(N_reference))
        else:
            N = int(2**p)
        
        if np.mod(N, 2) !=0:
            N += 1
        M = int(2**p)

        result_M[p-p_start] = M
        result_N[p-p_start] = N

        print(M,N)
        # temporal mesh 
        omega_t, Tau = time_mesh.equidistant_mesh(T, M)

        # spatial mesh
        Delta, h = spatial_mesh.equidistant_mesh(a, b, N)

        # ----------------------------------------------------------------------------------------------------
        # INITIALIZATION OF THE ARRAY, THAT STORES THE COEFFICIENTS OF u^l_h and \psi^l_h, l=j-k,...,j.
        # ----------------------------------------------------------------------------------------------------
        sol_vector = np.zeros((3,(r+1)*N+1))
        psi_vector = np.zeros((3,(r+1)*N+1))
        sol_vector_elliptic = np.array([[0],[0]])

        if not elliptic_estimator:
            sol_vector_elliptic = np.zeros((3,(r_elliptic+1)*N+1))

        # ----------------------------------------------------------------------------------------------------
        # ASSEMBLY OF THE COEFFICIENT MATRICES FOR THE FEM 
        # ----------------------------------------------------------------------------------------------------
        
        if not elliptic_estimator:
            mass_elliptic, reaction_a_elliptic, reaction_c_elliptic, stiff_elliptic, matrix_L_elliptic, \
            matrix_M_elliptic, lu_coeff_elliptic \
                = fem_assembly.assemble_coeff_matrices_fem_1d(N, x_fem, w_fem, func_a, func_c, True, Delta, h, 
                                                              psi_r_v, psi_l_v, r_elliptic, le_n_g_elliptic)
            
            mass, reaction_a, reaction_c, stiff, matrix_L, matrix_M, lu_coeff \
                = slice_coefficient_matrices(r, r_elliptic, N, mass_elliptic, reaction_a_elliptic, reaction_c_elliptic,
                                             stiff_elliptic, matrix_L_elliptic, matrix_M_elliptic)

        else:
            mass, reaction_a, reaction_c, stiff, matrix_L, matrix_M, lu_coeff \
                = fem_assembly.assemble_coeff_matrices_fem_1d(N, x_fem, w_fem, func_a, func_c, True, Delta, h, psi_r_v, psi_l_v, 
                                                              r, le_n_g) 

        # ----------------------------------------------------------------------------------------------------
        # COMPUTATION OF THE L^2 PROJECTION OF THE INTIAL CONDITION AND OF \eta_{init}
        # ----------------------------------------------------------------------------------------------------
        
        # FEM approximate solution 
        sol_vector[-2] =  elliptic_projection.elliptic_projection_fem(N, r, Delta, h, x_fem, w_fem, matrix_L, psi_r_v, 
                                                                      psi_l_v, func_u0, func_der_u0, func_a, 
                                                                      le_n_g, le_p_g)

        if not elliptic_estimator:
            sol_vector_elliptic[-2] \
                = elliptic_projection.elliptic_projection_fem(N, r_elliptic, Delta, h, x_fem, w_fem, matrix_L_elliptic, 
                                                              psi_r_v, psi_l_v, func_u0, func_der_u0, func_a, 
                                                              le_n_g_elliptic, le_p_g_elliptic)

        # eta_{init}
        result_init[p-p_start, 0] = eta_init.eta_init(sol_vector[-2], func_u0, func_der_u0, C_a, 
                                                      C_I, L_inverse, M_2_star, omega_2_star, T,
                                                      psi_r_v, psi_l_v, le_n_g, le_p_g, r, x_fem, w_fem, h, Delta, N)

        # \psi^0_h
        psi_vector[-2] = det_psi(N, r, lambda y: func_f(y, 0), Delta, h, x_fem, w_fem, sol_vector[-2], matrix_L,    # type: ignore
                                 matrix_M, psi_r_v, psi_l_v, le_n_g)


        # ----------------------------------------------------------------------------------------------------
        # ITERATION OVER THE TIME STEPS t_1,...,t_M 
        # ----------------------------------------------------------------------------------------------------

        for j in range(1, M+1):

            # ----------------------------------------------------------------------------------------------------
            # COMPUTATION OF THE REFERENCE SOLUTION, APPROXIMATE SOLUTION AND COMPONENTS OF THE ESTIMATOR
            # ----------------------------------------------------------------------------------------------------

            # Approximate solution for j=1 -> backward Euler
            if j == 1 or backward_euler:
                sol_vector[-1], lu_coeff = euler_fem.backward_euler_fem_1d(N, x_fem, w_fem, Tau[j-1], None, omega_t[j], func_f, sol_vector[-2], 
                                                                 matrix_L, matrix_M, lu_coeff, Delta, h, psi_r_v, psi_l_v, r, le_n_g) 
                if not elliptic_estimator:
                    sol_vector_elliptic[-1], lu_coeff_elliptic \
                        = euler_fem.backward_euler_fem_1d(N, x_fem, w_fem, Tau[j-1], None, omega_t[j], func_f, sol_vector_elliptic[-2],
                                                          matrix_L_elliptic, matrix_M_elliptic, lu_coeff_elliptic, Delta, h, 
                                                          psi_r_v, psi_l_v, r_elliptic, le_n_g_elliptic)

            # Approximate solution for j>1 -> BDF-2 method:
            else:
                
                # Compute previous time steps. If 'j==2', we have only t_0, t_1 and t_2. Then the LU factorization of the 
                # coefficient matrix is definitely recomputed. 

                if j==2: 
                    omega_t_previous = omega_t[j-2:j+1]
                else:
                    omega_t_previous = omega_t[j-3:j+1]    

                sol_vector[-1], lu_coeff = bdf.bdf_fem_1d(2, sol_vector[:-1], omega_t_previous, Delta, h, x_fem, w_fem, matrix_L, matrix_M, 
                                                          lu_coeff, func_f, psi_r_v, psi_l_v, le_n_g, r, N)
                
                if not elliptic_estimator:
                    sol_vector_elliptic[-1], lu_coeff_elliptic \
                        = bdf.bdf_fem_1d(2, sol_vector_elliptic[:-1], omega_t_previous, Delta, h, x_fem, w_fem, matrix_L_elliptic, matrix_M_elliptic,
                                           lu_coeff_elliptic, func_f, psi_r_v, psi_l_v, le_n_g_elliptic, r_elliptic, N)

            # \psi^j_h
            psi_vector[-1] = det_psi(N, r, lambda y: func_f(y, omega_t[j]), Delta, h, x_fem, w_fem, sol_vector[-1], matrix_L,   
                                     matrix_M, psi_r_v, psi_l_v, le_n_g)
            
            # computation of the components of the estimator
            result_f[p-p_start, 0] += eta_f.eta_f(func_f, omega_t[j], omega_t[j-1], Tau[j-1], C_a, c_a, c_c, T, x_fem, w_fem, h, Delta)
            
            result_Psi[p-p_start,0] += eta_Psi.eta_Psi(sol_vector[-1], sol_vector[-2], psi_vector[-1], psi_vector[-2], C_a, c_a, c_c, omega_t[j], 
                                                       omega_t[j-1], Tau[j-1], T, psi_r_v, psi_l_v, le_n_g, le_p_g, r, x_fem, w_fem, h, Delta, N)
            
            result_delta_psi[p-p_start,0] += \
                eta_delta_psi.eta_delta_psi(psi_vector[-1], psi_vector[-2], C_a, c_a, C_c, c_c, omega_t[j], omega_t[j-1], Tau[j-1], T, psi_r_v,
                                            psi_l_v, le_n_g, le_p_g, r, x_fem, w_fem, h, Delta, N)
            
            result_ell[p-p_start, 0] += \
                eta_R.eta_R(sol_vector[-1], sol_vector[-2], psi_vector[-1], psi_vector[-2], func_a, func_c, func_f, C_a,
                                c_a, C_I, L_inverse, M_2_star, omega_2_star, omega_t[j], omega_t[j-1], Tau[j-1], T, psi_r_v,
                                psi_l_v, le_n_g, le_p_g, le_p_g_x, r, w_fem, w_fem, h, Delta, N, elliptic_estimator, 
                                sol_vector_elliptic[-1], sol_vector_elliptic[-2], r_elliptic, le_n_g_elliptic, le_p_g_elliptic)

            # preparation for the next time step
            sol_vector[:-1] = np.copy(sol_vector[1:])
            psi_vector[:-1] = np.copy(psi_vector[1:])
            if not elliptic_estimator:
                sol_vector_elliptic[:-1] = np.copy(sol_vector_elliptic[1:])

        # ----------------------------------------------------------------------------------------------------
        # COMPUTATION OF THE ERROR, THE ESTIMATOR AND THE EXPERIMENTAL CONVERGENCE ORDERS
        # ----------------------------------------------------------------------------------------------------

        # Define the fem solution
        func_fem_sol = fem_sol.approximate_solution_intervals(sol_vector[-1], psi_r_v, psi_l_v, le_n_g, r, N)

        # The H^1-norm error
        result_err[p-p_start,0] = l2_norm(func_u, func_fem_sol, x_fem, w_fem, h, Delta)

        # The estimator
        result_est[p-p_start,0] = result_init[p-p_start, 0] + result_f[p-p_start,0] + result_Psi[p-p_start,0] \
                                  + result_delta_psi[p-p_start,0] + result_ell[p-p_start,0]
        
    # The efficiency
    result_eff[:] = result_err[:,0]/result_est[:,0]

    # The experimental order of convergence
    result_err[1:,1] = np.log2(result_err[:-1,0]/result_err[1:,0])
    result_est[1:,1] = np.log2(result_est[:-1,0]/result_est[1:,0])
    result_init[1:,1] = np.log2(result_init[:-1,0]/result_init[1:,0])
    result_f[1:,1] = np.log2(result_f[:-1,0]/result_f[1:,0])
    result_Psi[1:,1] = np.log2(result_Psi[:-1,0]/result_Psi[1:,0])
    result_delta_psi[1:,1] = np.log2(result_delta_psi[:-1,0]/result_delta_psi[1:,0])
    result_ell[1:,1] = np.log2(result_ell[:-1,0]/result_ell[1:,0])

    # ----------------------------------------------------------------------------------------------------
    # SAVE THE RESULTS
    # ----------------------------------------------------------------------------------------------------
    # Store as .txt file
    output_file_err = path_results + data_file_name_results + '.txt'  
    output_file_components = path_results + data_file_name_results + '_components.txt'

    with open(output_file_err, "w") as f_out, open(output_file_components, 'w') as f_comp_out:
        f_out.write(r"\(M\) & \(N\) & \(\norm_{u(T)-u^M_h}_{1,\Omega\) & \(p_N\) & \(\eta_{H^1_0}\) & \(1/\mathrm{eff}_N\) \\" + "\n")
        f_out.write(r"\hline" + "\n")
        f_comp_out.write(r"\(M\) & \(\eta_{\mathrm{init}}\) & \(\eta_f\) & \(\eta_{\mathrm{ell}}\) & \(\eta_\Psi\) & \(\eta_{\delta \psi}\) \\ " + "\n")
        f_comp_out.write(r"\hline" + "\n")
        for j in range(np.size(result_M)):
            f_out.write(f"{int(result_M[j])} & {int(result_N[j])} & {result_err[j,0]:.3e} & {result_err[j,1]:.2f} & {result_est[j,0]:.3e} & {(1/result_eff[j]):.2f} ")
            f_out.write(r"\\" + "\n")
            f_comp_out.write(f"{int(result_M[j])} & {int(result_N[j])} & {result_init[j,0]:.3e} & {result_f[j,0]:.3e} & {result_ell[j,0]:.3e} & {result_Psi[j,0]:.3e} & ")
            f_comp_out.write(f"{result_delta_psi[j,0]:.3e} \\" + "\n")

    # Save the value M, at which the reference solution was computed, and the starting and ending point a and b of the interval [a,b]
    with open(path_results + txt_file_name_reference_solution, 'w') as f:
        f.write(f"M = {int(result_M[-1])}\n")
        f.write(f"a = {a}\n")
        f.write(f"b = {b}\n")

    # Generate the graphic that shows the reference solution  
    show_solution(z_reference, omega_t_reference, result_reference, True, path_results)

if __name__ == '__main__':

    bdf2_l2_norm()