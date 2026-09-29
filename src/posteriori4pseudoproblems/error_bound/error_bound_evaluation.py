import numpy as np

from dataclasses import dataclass, field
from numpy import ndarray

from ._eta_init import _compute_eta_init
from ._eta_f import _compute_eta_f
from ._eta_Psi import _compute_eta_Psi
from ._eta_delta_psi import _compute_eta_delta_psi
from ._eta_R import _compute_eta_R
from ._latex_table import _write_to_file
from ..pseudo_parabolic_pde_class import PseudoParabolicPDE
from ..discretization_dataclasses import TemporalDiscParameters, SpatialDiscParameters
from ..quadrature import Quadrature
from ..legendre_galerkin import PkLegendreFEM
from ..time_discretization_methods import FemTimeSteppingParameters

_NORMS = ("l2", "h1")


def _append(history: ndarray | None, value: float) -> ndarray:

    r""" Appends `value` to `history`, creating it if it does not exist yet. """

    if history is None:
        return np.array([value])

    return np.append(history, value)


@dataclass
class ErrorBoundEvaluationFEM:

    r""" Results of an a posteriori error bound evaluation for a FEM
    discretization of a pseudo-parabolic PDE, with respect to the L^2 or
    the H^1 norm.

    Bundles, for a sequence of meshes registered one at a time via `update`,
    the error ||u(T) - u^M_h|| in the norm `norm_used`, the associated
    convergence order, the value of the a posteriori error estimator

        \eta = \eta_{init} + \eta_f + \eta_R + \eta_\Psi + \eta_{\delta\psi}

    and its efficiency, and provides `write_to_file` to store them as a
    LaTeX table.

    Per run, `update_eta_init` is called once for the initial approximation
    u^0_h, and `update_components` once for every time step j=1,...,M.
    Before each of these calls, the caller registers u^j_h (and v^j_h for
    the higher-order approximation) via `FemTimeSteppingParameters.update`
    and \psi^j_h, computed via `compute_psi`, via
    `FemTimeSteppingParameters.update_psi`.

    Args:
        norm_used (str): The norm the estimator bounds the error in: "l2"
            or "h1".

    Raises:
        ValueError: If `norm_used` is neither "l2" nor "h1".
    """

    norm_used: str
    history_M: ndarray | None = field(init=False, default=None, repr=False)
    history_N: ndarray | None = field(init=False, default=None, repr=False)
    history_error: ndarray | None = field(init=False, default=None, repr=False)
    history_estimator: ndarray | None = field(init=False, default=None, repr=False)
    history_order: ndarray | None = field(init=False, default=None, repr=False)
    history_efficiency: ndarray | None = field(init=False, default=None, repr=False)
    history_eta_init: ndarray | None = field(init=False, default=None, repr=False)
    history_eta_f: ndarray | None = field(init=False, default=None, repr=False)
    history_eta_R: ndarray | None = field(init=False, default=None, repr=False)
    history_eta_Psi: ndarray | None = field(init=False, default=None, repr=False)
    history_eta_delta_psi: ndarray | None = field(init=False, default=None, repr=False)
    eta_init: float = field(init=False, default=0.0)
    eta_f: float = field(init=False, default=0.0)
    eta_R: float = field(init=False, default=0.0)
    eta_Psi: float = field(init=False, default=0.0)
    eta_delta_psi: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:

        r""" Validates `norm_used`. """

        if self.norm_used not in _NORMS:
            raise ValueError(f'norm_used must be "l2" or "h1", got {self.norm_used!r}.')

    def update_eta_init(self,
                        pde: PseudoParabolicPDE,
                        spatial_disc_data: SpatialDiscParameters,
                        quadrature: Quadrature,
                        ref_functions: PkLegendreFEM,
                        fem_time_stepping_parameters: FemTimeSteppingParameters) -> None:

        r""" Adds the eta_init component, computed via
        `_eta_init._compute_eta_init`, to the running total `self.eta_init`.

        Must be called right after registering the initial approximation
        u^0_h in `fem_time_stepping_parameters`, and before the time
        marching procedure begins: it reads off u^0_h as
        `fem_time_stepping_parameters.prev_solutions[-1]`.

        Args:
            pde (PseudoParabolicPDE): The pseudo-parabolic PDE, providing
                u_0, u'_0, the final time T and the generic constants.
            spatial_disc_data (SpatialDiscParameters): Spatial
                discretization parameters for the approximation u_h.
            quadrature (Quadrature): Quadrature rule; provides the nodes
                and weights used for numerical integration.
            ref_functions (PkLegendreFEM): Reference functions of the
                P_k-FEM, evaluated at the quadrature nodes.
            fem_time_stepping_parameters (FemTimeSteppingParameters): State
                of the FEM time-stepping scheme; provides u^0_h.
        """

        self.eta_init += _compute_eta_init(pde, spatial_disc_data, quadrature, ref_functions,
                                           fem_time_stepping_parameters, self.norm_used)

    def update_components(self,
                          pde: PseudoParabolicPDE,
                          temporal_disc_data: TemporalDiscParameters,
                          spatial_disc_data: SpatialDiscParameters,
                          higher_order_spatial_disc_data: SpatialDiscParameters,
                          quadrature: Quadrature,
                          ref_functions: PkLegendreFEM,
                          higher_order_ref_functions: PkLegendreFEM,
                          fem_time_stepping_parameters: FemTimeSteppingParameters,
                          higher_order_fem_time_stepping_parameters: FemTimeSteppingParameters,
                          j: int) -> None:

        r""" Adds the contributions of subinterval I_j = [t_{j-1}, t_j] to
        eta_f, eta_R, eta_Psi and eta_delta_psi to their running totals,
        computed via `_eta_f._compute_eta_f`, `_eta_R._compute_eta_R`,
        `_eta_Psi._compute_eta_Psi` and `_eta_delta_psi._compute_eta_delta_psi`
        respectively.

        Args:
            pde (PseudoParabolicPDE): The pseudo-parabolic PDE, providing
                the source term F, the final time T and the generic constants.
            temporal_disc_data (TemporalDiscParameters): Temporal
                discretization parameters, providing the temporal mesh
                t_0,...,t_M.
            spatial_disc_data (SpatialDiscParameters): Spatial
                discretization parameters for the approximation u_h.
            higher_order_spatial_disc_data (SpatialDiscParameters): Spatial
                discretization parameters for the higher-order approximation
                v_h; same N, Delta and h as `spatial_disc_data`, only the
                polynomial degree k differs.
            quadrature (Quadrature): Quadrature rule; provides the nodes
                and weights used for numerical integration.
            ref_functions (PkLegendreFEM): Reference functions of the
                P_k-FEM of u_h, evaluated at the quadrature nodes.
            higher_order_ref_functions (PkLegendreFEM): Reference functions
                of the P_k-FEM of v_h, evaluated at the quadrature nodes.
            fem_time_stepping_parameters (FemTimeSteppingParameters): State
                of the FEM time-stepping scheme of u_h; provides
                u^{j-1}_h, u^j_h and \psi^{j-1}_h, \psi^j_h.
            higher_order_fem_time_stepping_parameters (FemTimeSteppingParameters):
                State of the FEM time-stepping scheme of v_h; provides
                v^{j-1}_h, v^j_h.
            j (int): Index of the current time level t_j.
        """

        self.eta_f += _compute_eta_f(pde, temporal_disc_data, spatial_disc_data, quadrature, j)

        self.eta_R += _compute_eta_R(pde, temporal_disc_data, spatial_disc_data, higher_order_spatial_disc_data,
                                     quadrature, ref_functions, higher_order_ref_functions,
                                     fem_time_stepping_parameters, higher_order_fem_time_stepping_parameters,
                                     self.norm_used, j)

        self.eta_Psi += _compute_eta_Psi(pde, temporal_disc_data, spatial_disc_data, quadrature, ref_functions,
                                         fem_time_stepping_parameters, self.norm_used, j)

        self.eta_delta_psi += _compute_eta_delta_psi(pde, temporal_disc_data, spatial_disc_data, quadrature,
                                                     ref_functions, fem_time_stepping_parameters,
                                                     self.norm_used, j)

    def update(self, M: int, N: int, error: float) -> None:

        r""" Registers the results of a single (M,N) run as a new row: the
        estimator \eta = eta_init + eta_f + eta_R + eta_Psi + eta_delta_psi
        is computed from the current running totals of the components and,
        together with M, N, error, the convergence order (log2(previous
        error / error), 0 for the first row) and the efficiency (error /
        estimator), appended to the histories; the components themselves
        are appended to their histories as well. The running totals are
        then reset to 0.0, so the next run's calls to
        `update_eta_init`/`update_components` start from a clean state.

        Args:
            M (int): Number of temporal subintervals.
            N (int): Number of spatial subintervals.
            error (float): The error ||u(T)-u^M_h|| in the norm `norm_used`.
        """

        estimator = self.eta_init + self.eta_f + self.eta_R + self.eta_Psi + self.eta_delta_psi
        order = 0.0 if self.history_error is None else float(np.log2(self.history_error[-1] / error))

        self.history_M = _append(self.history_M, M)
        self.history_N = _append(self.history_N, N)
        self.history_error = _append(self.history_error, error)
        self.history_estimator = _append(self.history_estimator, estimator)
        self.history_order = _append(self.history_order, order)
        self.history_efficiency = _append(self.history_efficiency, error / estimator)
        self.history_eta_init = _append(self.history_eta_init, self.eta_init)
        self.history_eta_f = _append(self.history_eta_f, self.eta_f)
        self.history_eta_R = _append(self.history_eta_R, self.eta_R)
        self.history_eta_Psi = _append(self.history_eta_Psi, self.eta_Psi)
        self.history_eta_delta_psi = _append(self.history_eta_delta_psi, self.eta_delta_psi)

        self.eta_init = 0.0
        self.eta_f = 0.0
        self.eta_R = 0.0
        self.eta_Psi = 0.0
        self.eta_delta_psi = 0.0

    def write_to_file(self, filename: str, path: str, caption: str = "", label: str = "",
                      components_caption: str = "", components_label: str = "") -> None:

        r""" Writes the results as two LaTeX tables to `path/filename`: M,
        N, the error, its convergence order, the estimator and its
        efficiency, followed by a second table, in the same style, of the
        estimator's components eta_init, eta_f, eta_R, eta_Psi, eta_delta_psi.

        Args:
            filename (str): Name of the .txt file the tables are written to.
            path (str): Directory the file is written to; must be located
                inside the project's `results` folder.
            caption (str): Caption of the error/estimator table. Defaults
                to an empty string.
            label (str): Label of the error/estimator table. Defaults to
                an empty string.
            components_caption (str): Caption of the components table.
                Defaults to an empty string.
            components_label (str): Label of the components table.
                Defaults to an empty string.

        Raises:
            ValueError: If `path` is not located inside the project's
                `results` folder.
        """

        assert self.history_M is not None and self.history_N is not None and self.history_error is not None \
            and self.history_estimator is not None and self.history_order is not None \
            and self.history_efficiency is not None and self.history_eta_init is not None \
            and self.history_eta_f is not None and self.history_eta_R is not None \
            and self.history_eta_Psi is not None and self.history_eta_delta_psi is not None, \
            "no results have been registered via update() yet."

        _write_to_file(self.norm_used, self.history_M, self.history_N, self.history_error, self.history_order,
                       self.history_estimator, self.history_efficiency, self.history_eta_init,
                       self.history_eta_f, self.history_eta_R, self.history_eta_Psi, self.history_eta_delta_psi,
                       filename, path, caption, label, components_caption, components_label)
