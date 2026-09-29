import numpy as np

from typing import Tuple
from numpy import ndarray
from ..interval_transformation import interval_transformation as interval_mapp
from ..pseudo_parabolic_pde_class import PseudoParabolicPDE
from ..discretization_dataclasses import SpatialDiscParameters
from ..quadrature import Quadrature
from ..legendre_galerkin import SpectralLegendre

def assemble_coeff_matrices_spectral(pde : PseudoParabolicPDE,
                                     spatial : SpatialDiscParameters,
                                     quadrature : Quadrature,
                                     ref_functions : SpectralLegendre) -> Tuple[ndarray, ndarray, ndarray]:

    r"""
    This functions assembles the matrices, associated with the operators

    Id u,  -\Delta u, au, cu

    where a, c :[a,b] \to \RR and \Delta is the Laplacian, for a spectral Galerkin method.
    We use Gauss-Lobatto (GL) as quadrature formula.

    Args:
        pde (PseudoParabolicPDE): The pseudo-parabolic PDE, providing the spatial domain
            [a,b] and the functions a and c.
        spatial (SpatialDiscParameters): Spatial discretization parameters;
            provides the dimension `dim_V` of the ansatz space.
        quadrature (Quadrature): Quadrature rule; provides the Gauss-Lobatto
            nodes and weights used for numerical integration.
        ref_functions (SpectralLegendre): Integrated Legendre polynomials,
            evaluated at the quadrature nodes.

    Returns:
        tuple[ndarray, ndarray, ndarray]: Tuple, consisting of
            - **ndarray**: Mass matrix.
            - **ndarray**: Matrix subjected to the operator L.
            - **ndarray**: Matrix subjected to the operator M.
    """

    a, b = pde.spatial_interval
    dim_V = spatial.dim_V

    assert dim_V is not None, "spatial must provide dim_V for a spectral discretization."

    x, w = quadrature.nodes_and_weights
    le_n_g, _ = ref_functions.evaluated_reference_functions
    assert le_n_g is not None, "ref_functions has not been evaluated at the quadrature nodes."

    # The integers 1,...,dim_V
    m_i = np.linspace(1, dim_V, dim_V, True)

    # Mass matrix
    mass_off_diag = -(b-a)/((2*m_i[:-2]+1)*(2*m_i[:-2]+3)*(2*m_i[:-2]+5))   # \int_a^b \phi_i \phi_{i+2}
    mass_diag = (b-a)/(2*m_i+1)**2*(1/(2*m_i+3)+1/(2*m_i-1))                # \int_a^b \phi_i^2
    mass = np.diag(mass_off_diag, -2)+np.diag(mass_diag, 0)+np.diag(mass_off_diag, 2)

    # Stiffness matrix
    stiff = 4/(b-a)*np.diag(1/(2*m_i+1))

    # The matrix reaction_a, associated with the reaction term au
    a_value = pde.func_a(interval_mapp(x, np.array([-1,1]), np.array([a,b])))           # a(\chi^{-1}(x)), \chi:[a,b] -> [-1,1]
    reaction_a = (b-a)/2*le_n_g.dot(((a_value*w)*le_n_g).T)

    # The matrix reaction_c, associated with the reaction term cu
    c_value = pde.func_c(interval_mapp(x, np.array([-1,1]), np.array([a,b])))           # c(\chi^{-1}(x)), \chi:[a,b] -> [-1,1]
    reaction_c = (b-a)/2*le_n_g.dot(((c_value*w)*le_n_g).T)

    # The matrices, subjected to the operators L and M
    matrix_L = stiff + reaction_a
    matrix_M = stiff + reaction_c

    return mass, matrix_L, matrix_M
