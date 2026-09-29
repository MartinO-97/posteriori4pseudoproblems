import numpy as np
from typing import Callable

from ...norms.l2_norm_function_fem import l2_norm_function_fem as l2_norm
from ...mesh_tools import interval_mapp
from ....common import divided_differences, modified_horner
from ....common.eta_S_1 import eta_S_1

def eta_f(func_f: Callable[[np.ndarray, float], np.ndarray],
          t_j: float,
          t_j_s: float, 
          tau_j: float,
          C_a: float, 
          c_a: float, 
          c_c: float,
          T: float,
          x: np.ndarray, 
          w: np.ndarray,
          h: np.ndarray, 
          Delta: np.ndarray) -> float:
    
    r"""
    The component \eta_f of the estimator \eta_{H^1_0}
    
    To approximate the temporal integral:

    \int_{t_{j-1}}^{t_j} || (f - \tilde{f})(s) ||_{0,\Omega} ds

    we employ the Simpson-Rule. Note that since the boundary Gauss-Lobatto nodes -1 and 1 conicides
    with \chi^{-1}(t_{j-1}) and \chi^{-1}(t_j), where \chi:[t_{j-1}, t_j] \to [-1,1], we only have 
    to consider the interior point gl_1=0.   

    Args:
        sol_vector(np.ndarray): Coefficients of u^j_h.
        func_u0(Callable[[np.ndarray], np.ndarray] | None): The initial condition 'u_0'.
        func_der_u0(Callable[[np.ndarray], np.ndarray] | None): The first derivative of initial condition 'u_0'.
        C_a (float): Boundedness constant of the bilinear form a.
        c_a (float): Coercivity constant of the bilinear form a.
        c_c (float): 'Coercivity' constant of the bilinear form c.
        T (float): The final time T.
        x (np.ndarray): The nodes of the GL formula.
        w (np.ndarray): The weights of the GL formula.
        h (np.ndarray): The spatial step sizes.
        Delta (np.ndarray): The mesh points of the spatial mesh of [a,b].  
    
    Returns:
        float: The component eta_f.

    """    

    # The reference array
    ref_array = np.array([-1,1])

    # The interior Gauss-Lobatto points gl_1, gl_2 and gl_3 as well as the weights associated weights to approximate tha 
    # temporal integral
    gl_nodes = np.array([0])
    gl_weights = np.array([4/3])

    # The Gauss-Lobatto points gl_1, gl_2 and gl_3 mapped to [t_{j-1}, t_j]
    gl_nodes_mapped = interval_mapp.interval_mapp(gl_nodes, ref_array, np.array([t_j_s, t_j]))

    # Evaluate f(\chi_i^{-1}, t_j) and f(\chi_i^{-1}, t_{j-1}) at the 'spatial' Gauss-Lobatto nodes
    f_values_j = func_f(interval_mapp.interval_mapp(x, ref_array, np.array([Delta[:-1], Delta[1:]]).T), t_j)
    f_values_j_s = func_f(interval_mapp.interval_mapp(x, ref_array, np.array([Delta[:-1], Delta[1:]]).T), t_j_s)

    # Compute the divided differences for the interpolation polynomial \tilde{f}, which interpolates f at t_j and t_{j-1}.
    # Note that we need the divided differences f[t_j] and f[t_{j-1}, t_j]
    divided_diff = divided_differences.divided_differences(np.array([t_j_s, t_j]), np.array([f_values_j_s, f_values_j]))

    # Evaluate \tilde{f} at the temporal Gauss-Lobatto nodes
    tilde_f_values = modified_horner.modified_horner(gl_nodes_mapped, np.array([t_j_s, t_j]), divided_diff)

    # Approximate || (f-\tilde{f})(gl_l) ||_{0,\Omega} l=1,2,3
    l2_norm_gl_1 = l2_norm(lambda mu: func_f(mu, gl_nodes_mapped[0]), tilde_f_values[0], x, w, h, Delta)
    #l2_norm_gl_2 = l2_norm(lambda mu: func_f(mu, gl_nodes_mapped[1]), tilde_f_values[1], x, w, h, Delta)
    #l2_norm_gl_3 = l2_norm(lambda mu: func_f(mu, gl_nodes_mapped[2]), tilde_f_values[2], x, w, h, Delta)

    # Approximate \int_{t_{j-1}}^{t_j} || (f - \tilde{f})(s) ||_{0,\Omega} ds
    l2_integral = tau_j/2*gl_weights[0]*l2_norm_gl_1

    # The value \sigma_j
    sigma_j = np.max(np.array([eta_S_1(C_a, c_a, c_c, T-t_j), eta_S_1(C_a, c_a, c_c, T-t_j_s)]))

    value = sigma_j/c_a * l2_integral

    return(value)