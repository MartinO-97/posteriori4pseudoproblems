from numpy import ndarray

def psi_R(x : ndarray) -> ndarray:
    
    r"""
    'Right' shape function on [-1,1]: \psi_R = (x+1)/2.
    
    Args:
        x (ndarray): value(s), where \psi_R shall be evaluated at
        
    Returns:
        ndarray: \psi_R, evaluated at x     
    """   
    
    return((x+1)/2)


def psi_L(x : ndarray) -> ndarray:
    
    r"""
    'Left' shape function on [-1,1]: \psi_L = (1-x)/2
    
    Args:
        x (ndarray): value(s), where \psi_L shall be evaluated at
        
    Returns:
        ndarray: \psi_L, evaluated at x     
    """  

    return((1-x)/2)