import numpy as np
from scipy.special import lambertw
from scipy.interpolate import interp1d
from constants import *

class NFWHalo:
	def __init__(self, m=110., c=None):
		""" Class that defines an NFW halo and its parameters. All radii in units of kpc.

		:param m: Halo mass m200 in units of 1e10*M_s.
		:param c: Halo concentration. If concentration c is not given, determine c from the 
				  default mass-concentration relation 
		"""

		self.m = m
		if c is None:
			self.c = mean_concentration(self.m)
		else:
			self.c = c

		# NFW parameters. Get all by using standard definitions of 
		# scale (NFW) and virial (= 200) quantities!
		self.delta_c = delta_c_pf/3*self.c**3/nfw_func(self.c) # Characteristic overdensity
		self.rho_s = self.delta_c*rho_c  # Scale density
		self.r_vir = (self.m/(delta_c_pf*rho_c)/(4*np.pi/3))**(1/3)  # Virial radius
		self.r_s = self.r_vir/self.c  # Scale radius

		# Scale mass. This is the prefactor that appears in the integral of (rho_NFW dV)
		# up to some arbitrary radius.
		self.m_s = 4*np.pi*self.rho_s*self.r_s**3   

		# Scale luminosity. This is the prefactor that appears in the integral of (rho_NFW^2 dV)
		# up to some arbitrary radius.
		self.l_s = 4*np.pi/3.*self.rho_s**2*self.r_s**3

	def radius(self, m):
		""" Inversion of m(r). See eq. A1/A2 of 1509.02175. This is only used
			to get the NFW truncation radius for integrating up to -- so not
			so important, since the integral rho^2 is pretty insensitive to
			the actual truncation radius outside r_s.
		"""
		return -(1./lambertw(-np.exp(-(1. + m/self.m_s))) + 1.).real*self.r_s

	def mass(self, r):
		""" Cumulative mass profile, i.e. the integral of (rho_NFW dV) up to radius r. 
		"""
		return self.m_s*nfw_func(r/self.r_s)

	def luminosity(self, r):
		""" Luminosity profile, i.e. the integral of (rho_NFW^2 dV) up to radius r.
			This is quite insensitive to truncation outside r_s.
		"""
		return (1.-(1.+r/self.r_s)**-3)*self.l_s #

	def density(self, r):
		""" NFW density profile
		"""
		x = r/self.r_s
		return self.rho_s/(x*(1+x)**2)

def mean_concentration(m200, model='Prada'):
    """ Mean concentration at z=0. 

        :param m200: Halo mass in units of 1e10*M_s/h 
        :param model: Halo concentration. 'Ludlow', 'Prada' or 'MaccioW1'.
    """

    if model == 'Prada':
        """ According to Sanchez-Conde&Prada14, with scatter of 0.14dex.
        """
        x=np.log(m200*1e10)
        pars=[37.5153, -1.5093, 1.636e-2, 3.66e-4, -2.89237e-5, 5.32e-7][::-1]
        return np.polyval(pars, x)

    if model == 'MaccioW1':
        """ Maccio08 WMAP1, C200, z=0, relaxed; sigma(log10)=0.11
        """
        return 8.26*(m200/1e2)**-0.104 

    if model == 'Ludlow':
        """ Ludlow14, z=0
        """
        return ludlow_spline(np.log10(m200)) 

    print("Unknown model!")
    raise
    
def nfw_func(x):
    """ Function that appears in the integral of (rho_NFW dV). Note that is is 
        logarithmically divergent!
    """
    return np.log(1+x)-x/(1+x)