import numpy as np
import matplotlib.pyplot as plt

from numpy import ndarray
from typing import Callable

def graphical_illustrations(func_v: Callable[[ndarray], ndarray]    
                                | Callable[[ndarray, float | ndarray], ndarray],
                            path: str,
                            one_or_two_d: int,
                            coords: ndarray,
                            exact_sol: bool):

    r"""
    A graphical illustration of a given function v.

    The matplotlib figure will be saved in the given path.

    Parameters
    ----------
    func_v: Callable[[ndarray], ndarray] | Callable[[ndarray, float], ndarray]
        The function v, which shall be illustrated graphically.

    path: str
        Path where the graphic shall be stored.

    one_or_two_d: int
        Is the spatial domain one-dimensional (1) or two-dimensional
        (2)?
        
    coords: ndarray
        Coordinates of the given mesh. If a function in a one-dimensional
        spatial domain shall be illustrated, coords must describe a mesh
        in the spatial-time domain.
    
    elements3: ndarray
        Vertices of the triangles. 'None', if the spatial domain is 
        one-dimensional. 
    """

    # Verify that one_or_two_d is in [1,2]
    if one_or_two_d not in [1,2]:
        raise ValueError(f"\'one_or_two_d\' must be in [1,2], but {one_or_two_d} was given!")

    # ----------------------------------------------------------------------------------------------------
    # ONE-DIMENSIONAL CASE
    # ----------------------------------------------------------------------------------------------------
    if one_or_two_d == 1:
        
        # EVALUATE v in the coordinates
        func_v_coords = func_v(coords[:, 0], coords[:,1])   # type: ignore

        # SURFACE PLOT
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_trisurf(coords[:,0], coords[:,1], func_v_coords, cmap='plasma', antialiased=False)

        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$t$")

        if exact_sol:
            ax.set_zlabel(r"$u(x,t)$")
        else:
            ax.set_zlabel(r"$u_h(x,t)$")

        fig.savefig(path + "sol.png")

    # ----------------------------------------------------------------------------------------------------
    # TWO-DIMENSIONAL CASE
    # ----------------------------------------------------------------------------------------------------        
    elif one_or_two_d == 2:
        
        # EVALUATE v in the coordinates
        func_v_coords = func_v(coords)   # type: ignore

        # PSEUDOCOLOR PLOT OF THE SOLUTION
        fig_tripcolor, ax_triplcolor = plt.subplots()
        
        contour_plot = ax_triplcolor.tripcolor(coords[:,0], coords[:,1], 
                                            func_v_coords, cmap='plasma')     #type: ignore

        fig_tripcolor.colorbar(contour_plot, ax=ax_triplcolor, label=r"Value of $u$")
        #ax_triplcolor.triplot(coords[:,0], coords[:, 1], elements3, color='k', linewidth=0.3) # type: ignore

        # PLOT SOLUTION IN A 3D SYSTEM
        fig_sol = plt.figure()
        ax_sol = fig_sol.add_subplot(projection='3d')

        #ax_sol.plot_trisurf(coords[:, 0], coords[:, 1], elements3, func_v_coords, 
        #                    cmap='plasma', edgecolor='k', linewidth=0.3)
        ax_sol.plot_trisurf(coords[:,0], coords[:,1], func_v_coords, cmap='plasma', antialiased=False)
        ax_sol.set_xlabel(r"$x_1$")
        ax_sol.set_ylabel(r"$x_2$")
        ax_sol.set_zlabel(r"$u(x_1, x_2)$")

        # SAVE FIGURES
        fig_tripcolor.savefig(path + "contour.png", dpi=300)
        fig_sol.savefig(path + "sol.png", dpi=300)