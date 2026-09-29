import numpy as np

from numpy import ndarray

# \omega_{l,j}
def omega(t_j: float, 
          t_j_s: float,
          s: ndarray, 
          l: int) -> ndarray:

        r"""
        The basis polynomial \omega_{l,j}.            
            
        Parameters
        ----------
        t_j: float 
            The time t_j.
            
        t_j_s: float 
            The time t_{j-1}.
        
        s: ndarray 
            Point, where \omega_{l,j} shall be evaluated at.
        
        l: int 
            l+2 is the degree of \omega_{l,j}.
        
        Returns
        -------
        :ndarray 
            \omega_{l,j} evaluated in s.
        """     
        
        # t_{j-1/2}
        t_j_h = (t_j + t_j_s)/2

        if l < -1:
            value_omega = np.zeros_like(s)
        elif l == -1:
            value_omega = s-t_j      
        else:
            value_omega = (s-t_j)*(s-t_j_h)**l*(s-t_j_s)
            
        return(value_omega)    


# \omega'_{l,j}
def der_omega(t_j: float, 
              t_j_s: float,
              s: ndarray, 
              l: int) -> ndarray:

        r"""
        The basis polynomial \omega'_{l,j}.            
            
        Parameters
        ----------
        t_j: float 
            The time t_j.
        
        t_j_s: float 
            The time t_{j-1}.
        
        s: ndarray 
            Point, where \omega'_{l,j} shall be evaluated at.
        
        l: int 
            l+1 is the degree of \omega'_{l,j}.
        
        Returns
        -------
        :ndarray 
            \omega'_{l,j} evaluated in s.    
        """     
        
        # t_{j-1/2}
        t_j_h = (t_j + t_j_s)/2

        if l <= -2:
            value_omega = np.zeros_like(s)
        elif l == -1:
            value_omega = np.ones_like(s)
        elif l == 0:
            value_omega = 2*(s-t_j_h)      
        else:
            value_omega = 2*(s-t_j_h)**(l+1) + l*(s-t_j)*(s-t_j_h)**(l-1)*(s-t_j_s)
            
        return(value_omega)   


# \int_{I_j} |\omega'_{l,j}(s)| ds
def get_integral_der_omega(tau_j: float,
                           nu: int) -> float:

        r"""
        Auxiliary function, to compute:
            
        \int_{t_{j-1}}^{t_j} |\omega'_{nu,j}(s)| ds

        Note that in the used Python version, we have 0^0=1 !!
            
        Parameters
        ----------
        tau_j: float
            The time step size t_j - t_{j-1}.
        
        nu: int 
            nu+1 is the degree of \omega'_{nu,j}.

        Returns
        -------
        :float 
            \int_{t_{j-1}}^{t_j} |\omega'_{nu,j}(s)| ds.
        """

        if nu < -1:
            result = 0
        elif nu == -1:
            result = tau_j
        elif nu == 0:
            result = tau_j**2/2
        else:
            result = tau_j**(2+nu)/2**nu * 2 /(nu+2) * (nu/(2+nu))**(nu/2)           

        return(result) 