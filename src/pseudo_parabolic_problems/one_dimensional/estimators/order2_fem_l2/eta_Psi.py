import numpy as np

from numpy import ndarray
from ...norms.h1_norm_function_fem import h1_norm_function_fem as h1_norm
from ...approximate_solutions import fem_solution_gauss_lobatto as fem_solution
from ....common.delta_t import delta_t
from ....common.eta_S_1 import eta_S_1

def eta_Psi(sol_vector: ndarray, 
            sol_vector_s: ndarray,
            psi_vector: ndarray,
            psi_vector_s: ndarray,
            C_a: float, 
            c_a: float, 
            c_c: float,
            t_j: float,
            t_j_s: float,
            tau_j: float,
            T: float,
            psi_r_v: ndarray,
            psi_l_v: ndarray, 
            le_n_g: ndarray | None,
            le_p_g: ndarray | None,
            r: int,
            x: ndarray, w: ndarray,
            h: ndarray, 
            Delta: ndarray,
            N: int) -> float:
    
    r"""The component \eta_{Psi} of the estimator \eta_{H^1_0}  at time t_j.
         
    Args:
        sol_vector(ndarray): Coefficients of u^j_h.
        sol_vector_s(ndarray): Coefficients of u^{j-1}_h.
        psi_vector(ndarray): Coefficients of \psi^j_h.
        psi_vector_s(ndarray): Coefficients of \psi^j_h.
        C_a (float): Boundedness constant of the bilinear form a.
        c_a (float): Coercivity constant of the bilinear form a.
        c_c (float): 'Coercivity' constant of the bilinear form c.
        t_j (float): The time t_j.
        t_j_s (float): The time t_{j-1}.
        tau_j (float): The time step size t_j-t_{j-1}.
        T (float): The final time T.
        psi_r_v (ndarray): The shape functions \psi_R, evaluated at the Gauss-Lobatto nodes.
        psi_l_v (ndarray): The shape functions \psi_L, evaluated at the Gauss-Lobatto nodes.
        le_n_g (ndarray): The integrated Legendre Polynomials N_l, l=1,...,r, evaluated at the Gauss-Lobatto nodes.
        le_p_g (ndarray): The Legendre Polynomials P_l, l=1,...,r, evaluated at the Gauss-Lobatto nodes.
        r (int): r+1 is the maximum degree of the considered integrated Legendre polynomials. 
        x (ndarray): The nodes of the GL formula.
        w (ndarray): The weights of the GL formula.
        h (ndarray): The spatial step sizes.
        Delta (ndarray): The mesh points of the spatial mesh of [a,b]. 
        N (int): Number of spatial subintervals.         
    
    Returns:
        float: The component eta_{init}.

    """    
    
    # The value \sigma_j
    sigma_j = np.max(np.array([eta_S_1(C_a, c_a, c_c, T-t_j), eta_S_1(C_a, c_a, c_c, T-t_j_s)]))

    # \psi^j_h + \delta u^j_h and its spatial derivative
    func_Psi = \
        fem_solution.approximate_solution_intervals((psi_vector + psi_vector_s)/2 + delta_t(sol_vector, sol_vector_s, tau_j), psi_r_v, psi_l_v, le_n_g, r, N)
    func_der_Psi = \
        fem_solution.derivative_approximate_solution_intervals((psi_vector + psi_vector_s)/2 + delta_t(sol_vector, sol_vector_s, tau_j), le_p_g, np.shape(x)[0], r, N, h)

    value = C_a * sigma_j * tau_j * h1_norm(np.zeros(np.shape(func_Psi)), func_Psi, np.zeros(np.shape(func_der_Psi)), func_der_Psi, x, w, h, Delta)

    return(value)