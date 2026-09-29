import numpy as np

from abc import ABC, abstractmethod
from typing import Callable
from numpy import ndarray

class Quadrature(ABC):

    def __init__(self, 
                 number_nodes: int) -> None:
        
        r""" Initialization of nodes and weights for a 
        quadrature rule with ``number_nodes``
        nodes and weights.
        
        Args:
            number_nodes (int): Number of nodes and weights of
            device (str): Device where tensors are stored
        """

        self._number_nodes = number_nodes
        self.nodes: ndarray
        self.weights: ndarray
        self.compute_nodes_and_weights()

    @property
    def number_nodes(self) -> int:
        
        """ Function that returns the number of nodes and weights.
        
        Returns:
            int: The number of nodes and weights
        """

        return self._number_nodes
    
    @property
    def nodes_and_weights(self) -> tuple[ndarray,
                                         ndarray]:
        
        """ Function that returns the nodes and weights of 
        the quadrature rule.
        
        Returns:
            tuple[ndarray, ndarray]: Tuple, consiting of
                - **ndarray**: Nodes of the quadrature formula.
                - **ndarray**: Weights of the quadrature formula.
        """

        return self.nodes, self.weights

    @abstractmethod
    def compute_nodes_and_weights(self) -> None:

        r""" Compute nodes and weights for the quadrature
        rule with ``n=self._number_nodes`` nodes and weights.  
        """

        
    @staticmethod
    def interval_transformation(interval: tuple[float, float], 
                                x: ndarray) -> ndarray:
        
        r""" Transformation of values in [-1,1] to a given interval.
        
        Args:
            interval (list[float]): Interval where values ``x`` shall be mapped to.
            x (ndarray): Values that shall be mapped onto the given interval.

        Returns:
            ndarray: Transformed values of ``x``.
        """

        alpha = interval[0]
        beta = interval[1]
        
        return alpha + (beta-alpha)/2*(x+1)

    @abstractmethod
    def approximate_integral(self,
                             func_y: Callable[[ndarray], ndarray],
                             interval: tuple[float, float]) -> float:
        
        r""" Approximation of the integral 
        
        \int_{\alpha}^\beta y(x) dx
        
        by employing a quadrature rule. 

        Args:
            func_y (Callable[[ndarray], ndarray]): The integrand ``y``.
            interval (tuple[float, float]): The interval [\alpha, \beta] over which 
                shall be integrated

        Returns:
            float: Approximation of the \int_{\alpha}^\beta y(x) dx.
        """

        mapped_nodes = self.interval_transformation(interval, self.nodes)
        alpha = interval[0]
        beta = interval[1]
        integral = np.dot(self.weights.T, func_y(mapped_nodes))

        return (beta-alpha)/2 * integral
    
    @abstractmethod
    def approximate_integral_from_values(self,
                             func_y_eval: ndarray,
                             interval: tuple[float, float]) -> float:
        
        r""" Approximation of the integral 
        
        \int_{\alpha}^\beta y(x) dx
        
        by employing a quadrature rule. The function y is 
        already evaluated at the nodes of the quadrature rule 

        Args:
            func_y_eval (ndarray): The integrand ``y`` evaluated at the nodes
                of the Gauss-Lobatto quadrature rule.
            interval (tuple[float, float]): The interval [\alpha, \beta] over which 
                shall be integrated

        Returns:
            float: Approximation of the \int_{\alpha}^\beta y(x) dx.
        """

        alpha = interval[0]
        beta = interval[1]
        integral = np.dot(self.weights.T, func_y_eval)

        return (beta-alpha)/2 * integral