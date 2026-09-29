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
elliptic_estimator: bool = True

# Polynomial degree r of the integrated Legendre basis functions.
# If r == 0, no integrated Legendre polynomials will be used.
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

path_results += 'with_elliptic_estimator/cn_h1_pseudo/'
data_file_name_results = 'err_est_results'

# Name of the .txt file, which stores the number of temporal intervals for the reference solution
txt_file_name_reference_solution = 'coefficients_reference_sol_info.txt'

