import numpy as np

def roots_dg_three():

    ''' the needed polynomial and its derivative to construct the dG3 method '''
        
    '''def p3(x):
        return(35*x**3-45*x**2+15*x-1)
    
    def p3_x(x):
        return(105*x**2-90*x+15)
    '''
    
    ''' the Butscher-Tableau, cB is the vector c, bB the vector b and AB the matrix A'''
    ''' # starting values for the Newton-Method
    cB = array([0.089, 0.409, 0.788])
    
    # Newton-Method to determine the roots of p3
    if roots_dg3 == True: 
        cB_old = 2
        while norm(cB-cB_old,ord=inf) > (2.5*10**(-19)):
            cB_old = copy(cB)
            cB = cB_old-p3(cB_old)/p3_x(cB_old)
        
        cB = append(cB, array([1])) 
    
    else:
        cB = array([0.0885879595127039474, 0.40946686444073471088, 0.7876594617608470559, 1.0])  '''  