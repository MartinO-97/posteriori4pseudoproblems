import numpy as np

# The function f
def func_f(x: np.ndarray,
           t: float) -> np.ndarray:

    r"""The function f:

        f(x,t) = e^{2t} + sin(x\pi)
    
    Args:
        x (np.ndarray): The spatial variable.
        t (float): The time variable.

    Returns:
        np.ndarray: f evaluated at (x,t).    
    """

    return(np.exp(2*t)+ np.cos(np.pi*(x+t)**2))


# The function a
def func_a(x: np.ndarray) -> np.ndarray:

    r"""The function a:

        a(x) = x^2+1
    
    Args:
        x (np.ndarray): The spatial variable.

    Returns:
        np.ndarray: c evaluated at x.    
    """

    return(5*x+6)


# The function c
def func_c(x: np.ndarray) -> np.ndarray:

    r"""The function c:

        c(x) = e^{x}
    
    Args:
        x (np.ndarray): The spatial variable.

    Returns:
        np.ndarray: c evaluated at x.    
    """

    return(-np.exp(-x))


# The initial condition u_0
def func_u0(x: np.ndarray) -> np.ndarray:

    r"""The initial condition u_0:

        u_0(x) = sin(x \pi)
    
    Args:
        x (np.ndarray): The spatial variable.

    Returns:
        np.ndarray: u_0 evaluated at x.    
    """

    return(np.sin(x* np.pi))


# The first derivative of the intial condition u_0
def func_der_u0(x: np.ndarray) -> np.ndarray:

    r"""The first derivative of the initial condition u'_0:

        u'_0(x) = \pi cos(x \pi)
    
    Args:
        x (np.ndarray): The spatial variable.

    Returns:
        np.ndarray: u'_0 evaluated at x.    
    """

    return(np.pi * np.cos(x* np.pi))


# Whether we have an exact solution
def func_exact_sol() -> bool:

    r"""Returns, whether we have an exact solution.

    Returns:
        bool: 'True' if we have an exact solution; 'False' else.
    """

    return(False)


# Generic constants
def determine_generic_constants_h1(a: float,
                                   b: float) -> tuple[float, float, float, float]:

    r"""
    Computation of the boundedness and coercivity constants of the bilinear forms a and c
    for the computation of \eta_{H^1_0}

    Given the two operators

    Lu = -u'' + au  and  Mu = -u'' + cu,    with a>0,

    we compute the boundedness and coercivity constants of the bilinear forms a and c, that
    are associated with L and M, respectively. Note that for c the 'coercivity' constant
    might be zero or negative. The functions u,a and c are assumed to be defined on the
    interval [a,b].

        Parameters
    ----------
    a: float 
        Starting point of [a,b].
    
    b: float 
        End point of [a,b].

    Returns
    -------
    C_a: float 
        Boundedness constant of a.
    
    c_a: float 
        Coercivity constant of a.
    
    C_c: float 
        Boundedness constant of c.
    
    c_c: float 
        'Coercivity' constant of c.
    """

    # Initialization of some auxiliary array
    z = np.linspace(a,b,5000)

    # the constant C_a
    a_max = np.max(func_a(z))
    C_a = np.max(np.array([1, a_max]))

    # the constant c_a
    a_min = np.min(func_a(z))
    c_a = np.min(np.array([1, a_min]))

    # the constant C_c
    c_max = np.max(func_c(z))
    C_c = np.max(np.array([1, c_max]))

    # the constant c_a
    c_min = np.min(func_c(z))
    c_c = np.min(np.array([1, c_min]))

    return(C_a, c_a, C_c, c_c)


def determine_generic_constants_l2(a: float,
                                   b: float) -> tuple[float, float, float, float, float, float, float, float]:

    r"""
    Computation of the boundedness and coercivity constants of the bilinear forms a and c
    as well as some further constants for the computation of \eta_{L^2}
    
    Given the two operators

    Lu = -u'' + au  and  Mu = -u'' + cu,    with a>0,

    we compute the boundedness and coercivity constants of the bilinear forms a and c, that
    are associated with L and M, respectively. Note that for c the 'coercivity' constant
    might be zero or negative. The functions u,a and c are assumed to be defined on the
    interval [a,b].

    Parameters
    ----------
    a: float 
        Starting point of [a,b].
    
    b: float 
        End point of [a,b].

    Returns
    -------
    C_a: float 
        Boundedness constant of a.
    
    c_a: float 
        Coercivity constant of a.
    
    C_c: float 
        Boundedness constant of c.
    
    c_c: float 
        'Coercivity' constant of c.
    
    M_2_star: float 
        The constant M_{2,*}.
    
    omega_2_star: float 
        The constant omega_{2,*}.
    
    C_I: float 
        The constant C_I.
    
    L_inverse: float
        The constant ||| L^{-1} |||_{0,2}.
    """

    # Initialization of some auxiliary array
    z = np.linspace(a,b,5000)

    # the constant C_a
    a_max = np.max(func_a(z))
    C_a = np.max(np.array([1, a_max]))

    # the constant c_a
    a_min = np.min(func_a(z))
    c_a = np.min(np.array([1, a_min]))

    # the constant C_c
    c_max = np.max(func_c(z))
    C_c = np.max(np.array([1, c_max]))

    # the constant c_a
    c_min = np.min(func_c(z))
    c_c = np.min(np.array([1, c_min]))

    # the constant M_{2,*}
    M_2_star = 1

    # the constant omega_{2,*}
    omega_2_star = 1

    # the constant C_I
    C_I = 1/2*np.sqrt(1/2)*(b-a+1)

    # the constant ||| L^{-1} |||_{0,2}
    L_inverse = 1

    return(C_a, c_a, C_c, c_c, M_2_star, omega_2_star, C_I, L_inverse)