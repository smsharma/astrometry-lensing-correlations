## TODO: 
# - Add precompute for other profiles
# - Add more concentration-mass, spatial distribution, mass functions etc
# - Can incorporate distance-dependent c?
# - Create separate class for distributions
# - Judicious choise of interpolation parameters

import sys
sys.path.append("../Simulations/")

import numpy as np
from scipy.special import erf, jn, jv, kn
from scipy.interpolate import interp2d
import mpmath as mp
from tqdm import *

from units import *

class PowerSpectra:
    def __init__(self, precompute_NFW):
        if self.precompute_NFW:
            self.procompute_NFW_MjdivM0()

    def F(self, x):
        """ Helper function for NFW deflection, from astro-ph/0102341
        """
        if x > 1:
            return mp.atan(mp.sqrt(x**2-1))/(mp.sqrt(x**2 - 1))
        elif x == 1:
            return 1
        elif x < 1:
            return mp.atanh(mp.sqrt(1-x**2))/(mp.sqrt(1-x**2))
        
    def Ft(self, x, tau):
        """ Helper function for truncated NFW deflection
        """
        return tau**2/(tau**2 + 1)**2*((tau**2 + 1 + 2*(x**2 - 1))*self.F(x) + tau*mp.pi + (tau**2 - 1)*mp.log(tau) + mp.sqrt(tau**2 + x**2)*(-mp.pi + (tau**2 - 1)/tau*self.L(x, tau)))

    def L(self, x, tau):
        """ Helper function for truncated NFW deflection
        """
        return mp.log(x/(mp.sqrt(tau**2 + x**2) + tau))

    def Fb(self, x):
        """ Helper function for Burkert deflection
        """
        if x > 1:
            return mp.log(x/2.) + mp.pi/4.*(mp.sqrt(x**2 + 1) - 1) + mp.sqrt(x**2 + 1)/2*mp.acoth(mp.sqrt(x**2 + 1)) - 0.5*mp.sqrt(x**2 - 1)*mp.atan(mp.sqrt(x**2 - 1))
        elif x == 1:
            return -mp.log(2.) - mp.pi/4. + 1/(2*mp.sqrt(2))*(mp.pi + mp.log(3 + 2*mp.sqrt(2)))
        elif x < 1:
            return mp.log(x/2.) + mp.pi/4.*(mp.sqrt(x**2 + 1) - 1) + mp.sqrt(x**2 + 1)/2*mp.acoth(mp.sqrt(x**2 + 1)) + 0.5*mp.sqrt(1 - x**2)*mp.atanh(mp.sqrt(1 - x**2))

    ################################################################################

    def MGauss(self, theta, M0, beta0):
        """ Enclosed mass in cylinder, Gaussian profile
        """
        return M0*(1-mp.exp(-theta**2/(2*beta0**2)))

    def MPlumm(self, theta, M0, beta0):
        """ Enclosed mass in cylinder, Plummer profile
        """
        return M0*theta**2/(theta**2 + beta0**2)

    ################################################################################

    def MNFWdivM0(self, x):
        """ Enclosed mass in cylinder, NFW profile
        """
        return (mp.log(x/2) + self.F(x))


    def MtNFWdivM0(self, x, tau=15.):
        """ Enclosed mass in cylinder, NFW profile
        """
        return self.Ft(x, tau)

    def MBurkdivM0(self, x):
        """ Enclosed mass in cylinder, NFW profile
        """
        return self.Fb(x)

    ################################################################################

    def Cl_Gauss(self, R0, M0, Dl, v, l):
        beta0 = R0/Dl
        return (4*GN*M0*v/Dl**2)**2*np.pi/2.*np.exp(-l**2*beta0**2)

    def Cl_Plummer(self, R0, M0, Dl, v, l):
        beta0 = R0/Dl
        return (4*GN*M0*v/Dl**2)**2*np.pi/2.*l**2*beta0**2*kn(1, l*beta0)**2

    def Cl_Point(R0, M0, Dl, v, l):
        return (4*GN*M0*v/Dl**2)**2*np.pi/2.

    ################################################################################

    def Cl_NFW(self, M200, Dl, v, l):
        r_s, rho_s = self.get_rs_rhos_NFW(M200)
        M0 = 4*np.pi*r_s**3*rho_s
        pref = GN**2*v**2*8*np.pi*l**2/Dl**4
        theta_s = r_s/Dl
        if not self.precompute_NFW:
            MjdivM0 = self.MNFWdivM0_integ(theta_s, l)
        else:
            MjdivM0 = 10**self.MjdivM0_integ_interp(np.log10(l), np.log10(theta_s))[0]
        return pref*M0**2*MjdivM0**2

    def MNFWdivM0_integ(self,theta_s, l):
        return mp.quadosc(lambda theta: self.MNFWdivM0(theta/theta_s)*mp.j1(l*theta), [0, mp.inf], period=2*mp.pi/l)

    def procompute_NFW_MjdivM0(self):
        l_min, l_max = 1, 2000
        n_l = 20
        l_ary = np.logspace(np.log10(l_min), np.log10(l_max), n_l)

        theta_s_min, theta_s_max = 0.001, 5
        n_theta_s = 20
        theta_s_ary = np.logspace(np.log10(theta_s_min), np.log10(theta_s_max), n_theta_s)

        MjdivM0_integ_ary = np.zeros((n_theta_s, n_l))

        for itheta_s, theta_s in enumerate(tqdm_notebook(theta_s_ary)):
            for il, l in enumerate((l_ary)):
                MjdivM0_integ_ary[itheta_s, il] = self.MNFWdivM0_integ(theta_s, l)

        self.MjdivM0_integ_interp = interp2d(np.log10(l_ary), np.log10(theta_s_ary), np.log10(MjdivM0_integ_ary), kind='linear')

    def Cl_tNFW(self, M200, Dl, v, l, tau=15.):
        r_s, rho_s = self.get_rs_rhos_NFW(M200)
        M0 = 4*np.pi*r_s**3*rho_s
        pref = GN**2*v**2*8*np.pi*l**2/Dl**4
        theta_s = r_s/Dl
        MjdivM0 = mp.quadosc(lambda theta: self.MtNFWdivM0(theta/theta_s, tau)*mp.j1(l*theta), [0, mp.inf], period=2*mp.pi/l)

        return pref*M0**2*MjdivM0**2

    def Cl_Burk(self, M200, Dl, v, l, p=0.7):
        r_b, rho_b = self.get_rb_rhob_Burk(M200, p)
        M0 = 4*np.pi*r_b**3*rho_b
        pref = GN**2*v**2*8*np.pi*l**2/Dl**4
        theta_b = r_b/Dl
        MjdivM0 = mp.quadosc(lambda theta: self.MBurkdivM0(theta/theta_b)*mp.j1(l*theta), [0, mp.inf], period=2*mp.pi/l)

        return pref*M0**2*MjdivM0**2

    ################################################################################

    def get_rs_rhos_NFW(self, M200):
        """ Get NFW scale radius and density
        """
        c200 = self.c200_SC(M200/M_s)
        r200 = (M200/(4/3.*np.pi*200*rho_c))**(1/3.)
        rho_s = M200/(4*np.pi*(r200/c200)**3*(np.log(1 + c200) - c200/(1 + c200)))
        r_s = r200/c200
        return r_s, rho_s

    def get_rb_rhob_Burk(self, M200, p):
        """ Get Burkert scale radius and density
        """
        c200n = self.c200_SC(M200/M_s)
        c200b = c200n/p
        r200 = (M200/(4/3.*mp.pi*200*rho_c))**(1/3.)
        r_b = r200/c200b
        rho_b = M200/(r_b**3*mp.pi*(-2*mp.atan(c200b) + mp.log((1+c200b)**2*(1+c200b**2))))
        return r_b, rho_b

    ################################################################################

    def R0_VL(self, M0):
        """ Concentration-mass relation for Plummer profile from 1711.03554
        """
        return 1.2*kpc*(M0/(1e8*M_s))**0.5

    def c200_SC(self, M200):
        """ Concentration-mass relation according to Sanchez-Conde&Prada14
        """
        x=np.log(M200*h) # Given in terms of M_s/h in S-C&P paper
        pars=[37.5153, -1.5093, 1.636e-2, 3.66e-4, -2.89237e-5, 5.32e-7][::-1]
        return np.polyval(pars, x)

    ################################################################################