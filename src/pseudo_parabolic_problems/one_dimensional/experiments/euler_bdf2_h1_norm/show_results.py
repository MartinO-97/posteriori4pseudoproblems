import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .config import path_results
from ...approximate_solutions.spectral_solution import approximate_solution as ref_sol

def show_results():

    r"""Creation of the tables, which only need to be copy pasted into the tex file. 
    
    Furthermore .png files are created and safed, which shows the course of the error and estimator.
    """

    # load the results for the estimator
    results = pd.read_csv(path_results + 'err_est_results.csv')
    
    # load the coeffcients of the reference solution
    coefficients = pd.read_csv(path_results + 'coefficients_reference_sol.csv').to_numpy().flatten()
    
    with open(path_results + 'coefficients_reference_sol_info.txt', 'r') as f:
        for line in f:
            key, value = line.split('=')
            key = key.strip()
            value = value.strip()

            if key == 'M':
                M = int(value)   
            elif key == 'a':
                a = float(value)
            elif key == 'b':
                b = float(value)
    
    # Evaluate the reference soltuion
    z = np.linspace(a,b, 2*np.size(coefficients))
    ref_sol_evaluated = ref_sol(z, coefficients, a, b, int(np.size(coefficients)))

    # Create graphic, that shows the evaluated reference solution and safe at 'path_result + 'solution.png''
    plt.plot(z, ref_sol_evaluated, 'k', label=r'$u(\cdot, T)$')
    plt.legend(loc='best')
    

if __name__ == '__main__':

    show_results()    