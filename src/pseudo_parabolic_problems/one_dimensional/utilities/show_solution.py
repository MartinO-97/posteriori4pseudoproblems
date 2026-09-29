import numpy as np
import matplotlib.pyplot as plt

from numpy import ndarray

def show_solution(x: ndarray,
                  y: ndarray,
                  results: ndarray,
                  y_time: bool, 
                  path: str):

    r"""
    Create a graphic, that shows the solution of a given timedependent PDE.
    
    Args:
        x (ndarray): x coordinate values (x_1,...,x_n).
        y (ndarray): y coordinate values (y_1,...,y_m) (time steps for problems in one spatial domain).
        results (ndarray): The solution u evaluated at {(x_i,y_j) | i=1,...,n and j=1,...,m}.
        y_time (bool): 'True' if y describes the time variable; else 'False'.
        path (str): Path, where the graphic shall be saved.
    """

    # generate the figure and axis    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    
    # generating the grid
    X, Y = np.meshgrid(x, y)
    
    # Plot the surface.
    surf = ax.plot_surface(X, Y, results, cmap='plasma',        # type: ignore
                           linewidth=0, antialiased=False)
    
    # set axis ticks
    z_max = np.amax(results) 
    z_min = np.amin(results)
    ax.set_xticks([x[0], x[0]+(x[-1]-x[0])/3, x[0]+2*(x[-1]-x[0])/3, x[-1]])
    ax.set_yticks([y[0], y[0]+(y[-1]-y[0])/3, y[0]+2*(y[-1]-y[0])/3, y[-1]])
    ax.set_zticks([z_min, z_min + (z_max-z_min)/3, z_min + 2*(z_max-z_min)/3, z_max])   # type: ignore
    
    # set labels
    ax.set_xlabel(r'$x$')
    if y_time:
        ax.set_ylabel(r'$t$')
    else:
        ax.set_ylabel(r'$y$')    
    ax.zaxis.set_rotate_label(False)    # type: ignore
    if y_time:
        ax.set_zlabel(r'$ u(t,x)$', rotation = 0, labelpad = 20)    # type: ignore
    else:
        ax.set_zlabel(r'$ u(x,y)$', rotation = 0, labelpad = 20)    # type: ignore    
    
    # Set face color
    #ax.set_facecolor('lightgrey')
    
    #ax.legend(loc='best') 

    # set title
    #ax.set_title(r'the solution $u$ computed by using a spectral Galerkin method and dG$(2)$')
    
    # safe the figure
    fig.savefig(path + 'solution.png', format="png", dpi=300)
    #fig.savefig(path + 'solution.eps', format="eps", dpi=300)
    #plt.close()