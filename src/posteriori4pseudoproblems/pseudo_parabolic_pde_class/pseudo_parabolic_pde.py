from numpy import ndarray
from typing import Callable

from ._generic_constants import compute_generic_constants as _compute_generic_constants

class PseudoParabolicPDE:

    r""" A pseudo-parabolic PDE

        L u_t + M u = F    with Lu = -\Delta u + au, a > 0, and Mu = -\Delta u + cu,

    on \Omega x (0,T]. Bundles the problem data -- the source function, the
    initial and boundary conditions, the functions a and c, the final time
    and the spatial domain -- together with the generic constants needed by
    the a posteriori error estimator, which are computed via
    `compute_generic_constants`.
    """

    def __init__(self,
                 F: Callable[[ndarray], ndarray],
                 u0: Callable[[ndarray], ndarray],
                 Psi: Callable[[ndarray], ndarray],
                 a: Callable[[ndarray], ndarray],
                 c: Callable[[ndarray], ndarray],
                 T: float,
                 spatial_interval: tuple[float, float],
                 u: Callable[[ndarray], ndarray] | None = None,
                 der_u0: Callable[[ndarray], ndarray] | None = None) -> None:

        r""" Initialization of a pseudo-parabolic PDE:
            Lu_t + Mu = -\Delta u_t + au_t - \Delta u + cu = F on \Omega x (0,T]
            u(x,0) = u_0(x) on \overline{\Omega}
            u(x,t) = Psi(x) for (x,t) \in \pt \Omega x [0,T],
        where Lu = -\Delta u + au with a > 0 and Mu = -\Delta u + cu.

        Args:
            F (Callable[[ndarray], ndarray]): The source function F.
            u0 (Callable[[ndarray], ndarray]): The initial condition u0.
            Psi (Callable[[ndarray], ndarray]): The boundary condition Psi.
            a (Callable[[ndarray], ndarray]): The function a.
            c (Callable[[ndarray], ndarray]): The function c.
            T (float): The final time T.
            spatial_interval (tuple[float, float]): The spatial domain.
            u (Callable[[ndarray], ndarray] | None): The exact solution. Defaults to ``None``.
            der_u0 (Callable[[ndarray], ndarray] | None): The first derivative u0' of the
                initial condition. Defaults to ``None``.
        """

        if len(spatial_interval) != 2:
            raise ValueError(f"""spatial_interval must consist of two values but
                             got {len(spatial_interval)}!""")
        self._F = F
        self._u0 = u0
        self._der_u0 = der_u0
        self._Psi = Psi
        self._u = u
        self._T = T
        self._spatial_interval = spatial_interval
        self._a = a
        self._c = c
        self._C_a: float | None = None
        self._c_a: float | None = None
        self._C_c: float | None = None
        self._c_c: float | None = None
        self._M_2_star: float | None = None
        self._omega_2_star: float | None = None
        self._C_I: float | None = None
        self._L_inverse: float | None = None

    def func_a(self,
               x_points: ndarray) -> ndarray:

        """ The function a.

        Args:
            x_points (ndarray): The points in the spatial domain where
                a shall be evaluated at. Must be two-dimensional.

        Returns:
            ndarray: The function ``a`` evaluated at x_points.
        """

        return self._a(x_points)

    def func_c(self,
               x_points: ndarray) -> ndarray:

        """ The function c.

        Args:
            x_points (ndarray): The points in the spatial domain where
                c shall be evaluated at. Must be two-dimensional.

        Returns:
            ndarray: The function ``c`` evaluated at x_points.
        """

        return self._c(x_points)

    def compute_generic_constants(self,
                                  n_points: int = 5000) -> None:

        r""" Computes and stores the boundedness and coercivity constants of the
        bilinear forms associated with the operators L and M as well as some
        further constants for the computation of \eta_{L^2} and \eta_{H^1_0}:

            C_a = max(1, max_x a(x)),   c_a = min(1, min_x a(x)),
            C_c = max(1, max_x c(x)),   c_c = min(1, min_x c(x)),
            M_{2,*} = 1,  omega_{2,*} = 1,  C_I = 1/2 sqrt(1/2) (b-a+1),  ||| L^{-1} |||_{0,2} = 1.

        Note that the 'coercivity' constant c_c might be zero or negative. max_x
        and min_x are approximated on `n_points` equidistant points on the
        closure of the spatial domain.

        Args:
            n_points (int): Number of equidistant points, on the closure of
                the spatial domain, used to approximate the extrema of a and c.
                Defaults to 5000.
        """

        (self._C_a, self._c_a, self._C_c, self._c_c, self._M_2_star,
         self._omega_2_star, self._C_I, self._L_inverse) = \
            _compute_generic_constants(self._a, self._c, self._spatial_interval, n_points)

    def func_F(self,
               xt_points: ndarray) -> ndarray:

        """ The source function of the PDE

        Args:
            xt_points (ndarray): The points in the space-time
                domain where F shall be evaluated at. Must be a
                two-dimensional array, where the last column is associated
                with the time variable.

        Returns:
            ndarray: The source function F evaluated at
                xt_points
        """

        return self._F(xt_points)

    def func_u0(self,
                x_points: ndarray) -> ndarray:

        """ The initial condition.

        Args:
            x_points (ndarray): The points in closure of the
                spatial domain where u0 shall be evaluated at. Must be
                a two-dimensional array.

        Returns:
            ndarray: The initial condition evaluated at
                x_points.
        """

        return self._u0(x_points)

    def func_der_u0(self,
                    x_points: ndarray) -> ndarray | None:

        """ The first derivative of the initial condition.

        Args:
            x_points (ndarray): The points in closure of the
                spatial domain where u0' shall be evaluated at. Must be
                a two-dimensional array.

        Returns:
            ndarray | None: The first derivative of the initial condition
                evaluated at x_points. If it is not known, ``None`` is
                returned.
        """

        if self._der_u0 is None:
            return None
        else:
            return self._der_u0(x_points)

    def func_Psi(self,
                 xt_points: ndarray) -> ndarray:

        """ The boundary condition.

        Args:
            xt_points (ndarray): The points in the space-time
                domain where Psi shall be evaluated at. Must be a
                two-dimensional array, where the last column
                must be associated with the time variable.

        Returns:
            ndarray: The boundary condition evaluated at
                xt_points.
        """

        return self._Psi(xt_points)

    def func_u(self,
               xt_points: ndarray) -> ndarray | None:

        """ The exact solution of the PDE.
        Args:
            xt_points (ndarray): The points in the space-time
                domain where u shall be evaluated at. Must be two-dimensional
                array, where the last column
                must be associated with the time variable.

        Returns:
            ndarray | None: The exact solution evaluated at
                xt_points. If no exact solution is known, ``None``
                is returned.
        """
        if self._u is None:
            return None
        else:
            return self._u(xt_points)

    @property
    def exact_solution(self) -> bool:

        """ Query if we have an exact solution.

        Returns:
            bool: Returns ``True`` if we have an exact solution, otherwise
                ``False``.
        """
        if self._u is None:
            return False
        else:
            return True

    @property
    def spatial_interval(self) -> tuple[float, float]:

        r""" Returns the spatial interval as list of floats

        Returns:
            tuple[float, float]: The spatial interval.

        """

        return self._spatial_interval

    @property
    def final_time(self) -> float:

        r""" Returns the final time.

        Returns:
            float: The final time
        """

        return self._T

    @property
    def generic_constants(self) -> tuple[float, float, float, float, float, float, float, float]:

        r""" Returns the constants C_a, c_a, C_c, c_c, M_2_star, omega_2_star,
        C_I, L_inverse, where C_a and c_a are the boundedness and coercivity
        constants of a, C_c and c_c are the boundedness and 'coercivity'
        constants of c, M_2_star and omega_2_star are the constants M_{2,*}
        and omega_{2,*}, C_I is the constant C_I and L_inverse is the constant
        ||| L^{-1} |||_{0,2}.

        Returns:
            tuple[float, float, float, float, float, float, float, float]:
                (C_a, c_a, C_c, c_c, M_2_star, omega_2_star, C_I, L_inverse).
        """

        assert (self._C_a is not None and self._c_a is not None
                and self._C_c is not None and self._c_c is not None
                and self._M_2_star is not None and self._omega_2_star is not None
                and self._C_I is not None and self._L_inverse is not None), \
            "compute_generic_constants() has not been called yet."

        return (self._C_a, self._c_a, self._C_c, self._c_c, self._M_2_star,
                self._omega_2_star, self._C_I, self._L_inverse)
