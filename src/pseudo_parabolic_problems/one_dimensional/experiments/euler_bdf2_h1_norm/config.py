from ..functions import *
import argparse

# ----------------------------------------------------------------------------------------------------
# CONFIGURATION FILE FOR main.py
# ----------------------------------------------------------------------------------------------------

r"""
This module defines all configuration parameters for the numerical experiments
related to the pseudo-parabolic problem solved with the BDF-2/FEM in 1D.
"""

# ----------------------------------------------------------------------------------------------------
# GENERAL SETTINGS
# ----------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------
# PARSE ARGUMENTS
# ----------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--elliptic", action="store_true", help="Use Elliptic estimator. If this " \
                    "argument is not given, a reference solution of higher order is computed. " \
                    "If it is passed, the elliptic estimator is used.")
parser.add_argument("--euler", action="store_true", help="If '--euler' is given, the time " \
                    "discretization is only based on the backward euler method.")
parse_args = parser.parse_args()

# Spatial interval [a, b].
a: float = -1.0
b: float = 1.0

# Range of spatial refinement levels.
# The mesh is defined as N = 2^p, for p in [p_start, ..., p_end].
p_start: int = 3
p_end: int = 13

# The final time T.
T: float = 2.0

# Type of time discretization:
#   1 -> equidistant time mesh
time_steps: int = 1

# Shall an elliptic estimator be used or shall an approximation of higher order be computed
# 'True' -> Elliptic estimator
# 'False' -> Approximation of higher order
elliptic_estimator: bool = parse_args.elliptic

# Shall only the backward Euler method be employed
backward_euler: bool = parse_args.euler

# Polynomial degree r of the integrated Legendre basis functions.
# If r == 0, no integrated Legendre polynomials will be used.
# r_elliptic is used to compute an approximation of higher order for the elliptic estimator
if backward_euler:
    r: int = 0
else:
    r: int = 1
r_elliptic: int = r+1

# Dimension of the ansatz space for the spectral dG2 method.
dim_V: int = 30

# Perturbation parameters
eps_a: float = 1.0
eps_c: float = 1.0

# ----------------------------------------------------------------------------------------------------
# PATHS TO STORE THE RESULTS   
# ----------------------------------------------------------------------------------------------------

path_results = './data/one_dimensional/'

if elliptic_estimator:
    if backward_euler:
        path_results += 'with_elliptic_estimator/euler_h1_pseudo/'
    else:
        path_results += 'with_elliptic_estimator/bdf2_h1_pseudo/'
else:
    if backward_euler:
        path_results += 'no_elliptic_estimator/euler_h1_pseudo/'
    else:
        path_results += 'no_elliptic_estimator/bdf2_h1_pseudo/'  
data_file_name_results = 'err_est_results'

# Name of the .txt file, which stores the number of temporal intervals for the reference solution
txt_file_name_reference_solution = 'coefficients_reference_sol_info.txt'

