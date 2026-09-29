import numpy as np

from numpy import ndarray

def spatial_mesh_refinement_1d(Delta: ndarray, 
                               ref: int, 
                               prepare_legendre: bool) -> tuple[ndarray, ndarray]:
    
    r"""
    This function refines a given spatial mesh \Delta of the interval [a,b] of N+1 points, given by
        
    \Delta: a = x_0 < x_1 < ... < x_{N-1} < x_N = b. 

    Each subinterval [x_{i-1}, x_i], i = 1,...,N, is is refined to ref+1 smaller subintevals with ref+2 
    equidistant points, i.e.

    x_{i-1} = \xi_{i,0} < \xi_{i,1} < ... < \xi_{i,ref} < \xi_{i,ref+1} = x_i, i=1,...,N.  

    The resulting refined mesh \Delta_{ref} consists of N*(ref+1)+1 points. 
    
    If 'prepare_legendre' is 'True', each local mesh interval {\xi_{i,k}}_{k=0}^{ref+1}, i = 1,...,N, is also mapped to the reference 
    interval [-1,1].

    Note that for subsequent computations it is irrelevant, if the nodes {x_i}_{i=0}^N are mapped to -1 or 1.      
        
    Parameters
    ----------
    Delta: ndarray 
        Mesh of [a,b].
    
    ref: int 
        Number of points between two nodes x_{i-1} and x_i; must be greater than zero. 
    
    prepare_legendre: bool 
        If 'True', return local meshes mapped to [-1,1].

    Returns
    -------
    Delta_refined: ndarray 
        The refined mesh Delta. 
    
    Delta_refined_Legendre: ndarray
        [x_{i-1}, x_i], i = 1,...,N, mapped to [-1,1]; empty array, if 'prepare_legendre=False'. 

    Raises:
        ValueError: If 'ref' is not greater than zero.        
    """    

    # Check, if ref > 0
    if ref <= 0:
        raise ValueError('spatial_mesh_refinement_1d: ref must be > 0.')

    # Number of spatial subintervals
    N = np.shape(Delta)[0]-1

    # Initialization of Delta_refined
    Delta_refined = np.zeros((ref+1)*N+1)
    Delta_refined[np.arange(0,N+1)*(ref+1)] = np.copy(Delta)
    
    # Refinement of \Delta
    for l in range(1,ref+1):
        Delta_refined[l:-ref-1+l:ref+1] = ((ref+1-l)*Delta[:-1] + l*Delta[1:])/(ref+1)
    
    # Preparation of evaluating the spline approximation with Legendre polynomials
    if prepare_legendre:

        # Initialization of Delta_refined_Legendre
        Delta_refined_Legendre = np.ones((ref+1)*N+1)

        # Affine transformation of the local meshintervals to [-1,1]
        for l in range(1,ref+1):
            Delta_refined_Legendre[l:-ref-1+l:ref+1] = (2*Delta_refined[l:-ref-1+l:ref+1] - (Delta[:-1] + Delta[1:]))/(Delta[1:]-Delta[:-1])

    else:
        Delta_refined_Legendre = ndarray([])

    return(Delta_refined, Delta_refined_Legendre)