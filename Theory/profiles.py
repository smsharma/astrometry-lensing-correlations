import sys
import os
sys.path.append("../Simulations/")

import numpy as np
import mpmath as mp
from scipy.special import erf, jn, jv, kn
from scipy.interpolate import interp2d
from scipy.misc import derivative
from tqdm import *

from units import *

class Profiles:
    """ Class to calculate expected power spectra from astrometric induced velocities and accelerations
       
        :param precompute: List of profiles to precompute arrays for to speed up computation ['Burk', 'NFW']
    """
    def __init__(self):
        self.c200_model = self.c200_SCP # Set a default concentration model to start

    ##################################################
    # Enclosed masses
    ##################################################

    def MGauss(self, theta, M0, beta0):
        """ Enclosed mass in cylinder, Gaussian profile
        """
        return M0*(1-mp.exp(-theta**2/(2*beta0**2)))

    def MPlumm(self, theta, M0, beta0):
        """ Enclosed mass in cylinder, Plummer profile
        """
        return M0*theta**2/(theta**2 + beta0**2)

    def MNFWdivM0(self, x):
        """ Enclosed mass in cylinder, NFW profile
        """
        return (mp.log(x/2) + self.F(x))

    def MtNFWdivM0(self, x, tau=15.):
        """ Enclosed mass in cylinder, NFW profile
        """
        return self.Ft(x, tau)

    def MBurkdivM0(self, x):
        """ Enclosed mass in cylinder, Burkert profile
        """
        return self.Fb(x)

    ##################################################
    # Helper functions
    ##################################################

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
        """ Helper function for truncated NFW deflection # TODO: cite source Mathematice nb
        """
        return tau**2/(tau**2 + 1)**2*((tau**2 + 1 + 2*(x**2 - 1))*self.F(x) + tau*mp.pi + (tau**2 - 1)*mp.log(tau) + mp.sqrt(tau**2 + x**2)*(-mp.pi + (tau**2 - 1)/tau*self.L(x, tau)))

    def L(self, x, tau):
        """ Helper function for truncated NFW deflection
        """
        return mp.log(x/(mp.sqrt(tau**2 + x**2) + tau)) # TODO: cite source Mathematice nb

    def Fb(self, x):
        """ Helper function for Burkert deflection # TODO: cite source Mathematice nb
        """
        if x > 1:
            return mp.log(x/2.) + mp.pi/4.*(mp.sqrt(x**2 + 1) - 1) + mp.sqrt(x**2 + 1)/2*mp.acoth(mp.sqrt(x**2 + 1)) - 0.5*mp.sqrt(x**2 - 1)*mp.atan(mp.sqrt(x**2 - 1))
        elif x == 1:
            return -mp.log(2.) - mp.pi/4. + 1/(2*mp.sqrt(2))*(mp.pi + mp.log(3 + 2*mp.sqrt(2)))
        elif x < 1:
            return mp.log(x/2.) + mp.pi/4.*(mp.sqrt(x**2 + 1) - 1) + mp.sqrt(x**2 + 1)/2*mp.acoth(mp.sqrt(x**2 + 1)) + 0.5*mp.sqrt(1 - x**2)*mp.atanh(mp.sqrt(1 - x**2))

    ##################################################
    # Functions to precompute enclosed mass integral
    ##################################################

    def MNFWdivM0_integ(self,theta_s, l):
        """ Enclosed mass integral for NFW profile
        """
        return mp.quadosc(lambda theta: self.MNFWdivM0(theta/theta_s)*mp.j1(l*theta), [0, mp.inf], period=2*mp.pi/l)

    def MBurkdivM0_integ(self,theta_b, l):
        """ Enclosed mass integral for Burkert profile
        """
        return mp.quadosc(lambda theta: self.MBurkdivM0(theta/theta_b)*mp.j1(l*theta), [0, mp.inf], period=2*mp.pi/l)

    def precompute_MNFWdivM0(self, n_l=20, n_theta_s=20):
        """ Precompute enclosed mass integral for NFW profile
        """
        if os.path.isfile("arrays/MNFWdivM0_integ_ary_n_l_" + str(n_l) + " n_theta_s " + str(n_theta_s) + ".npz"):
            print("Loading NFW parameters")
            file = np.load("arrays/MNFWdivM0_integ_ary_n_l_" + str(n_l) + " n_theta_s " + str(n_theta_s) + ".npz")
            l_ary = file['l_ary']
            theta_s_ary = file['theta_s_ary']
            MNFWdivM0_integ_ary = file['MNFWdivM0_integ_ary']

        else:
            print("Precomputing NFW parameters")

            l_min, l_max = 1, 2000
            l_ary = np.logspace(np.log10(l_min), np.log10(l_max), n_l)

            theta_s_min, theta_s_max = 0.001, 5
            theta_s_ary = np.logspace(np.log10(theta_s_min), np.log10(theta_s_max), n_theta_s)


            MNFWdivM0_integ_ary = np.zeros((n_theta_s, n_l))

            for itheta_s, theta_s in enumerate(tqdm_notebook(theta_s_ary)):
                for il, l in enumerate((l_ary)):
                    MNFWdivM0_integ_ary[itheta_s, il] = self.MNFWdivM0_integ(theta_s, l)

            np.savez("arrays/MNFWdivM0_integ_ary_n_l_" + str(n_l) + " n_theta_s " + str(n_theta_s) + ".npz",
                l_ary=l_ary, theta_s_ary=theta_s_ary, MNFWdivM0_integ_ary=MNFWdivM0_integ_ary)

        self.MNFWdivM0_integ_interp = interp2d(np.log10(l_ary), np.log10(theta_s_ary), np.log10(MNFWdivM0_integ_ary), kind='linear')
        self.precompute_NFW = True

    def precompute_MBurkdivM0(self, n_l=20, n_theta_b=20):
        """ Precompute enclosed mass integral for Burkert profile
        """
        if os.path.isfile("arrays/MBurkdivM0_integ_ary_n_l_" + str(n_l) + "_n_theta_b_" + str(n_theta_b) + ".npz"):
            print("Loading Burkert parameters")
            file = np.load("arrays/MBurkdivM0_integ_ary_n_l_" + str(n_l) + "_n_theta_b_" + str(n_theta_b) + ".npz")
            l_ary = file['l_ary']
            theta_b_ary = file['theta_b_ary']
            MBurkdivM0_integ_ary = file['MBurkdivM0_integ_ary']

        else:
            print("Precomputing Burkert parameters")

            l_min, l_max = 1, 2000
            l_ary = np.logspace(np.log10(l_min), np.log10(l_max), n_l)

            theta_b_min, theta_b_max = 0.001, 4

            theta_b_ary = np.logspace(np.log10(theta_b_min), np.log10(theta_b_max), n_theta_b)

            MBurkdivM0_integ_ary = np.zeros((n_theta_b, n_l))

            for itheta_b, theta_b in enumerate(tqdm_notebook(theta_b_ary)):
                for il, l in enumerate((l_ary)):
                    MBurkdivM0_integ_ary[itheta_b, il] = self.MBurkdivM0_integ(theta_b, l)

            np.savez("arrays/MBurkdivM0_integ_ary_n_l_" + str(n_l) + "_n_theta_b_" + str(n_theta_b) + ".npz",
                l_ary=l_ary, theta_b_ary=theta_b_ary, MBurkdivM0_integ_ary=MBurkdivM0_integ_ary)

        self.MBurkdivM0_integ_interp = interp2d(np.log10(l_ary), np.log10(theta_b_ary), np.log10(MBurkdivM0_integ_ary), kind='linear')
        self.precompute_Burk = True

    ##################################################
    # Profile properties
    ##################################################

    def get_rs_rhos_NFW(self, M200, **kwargs):
        """ Get NFW scale radius and density
        """
        c200 = self.c200_model(M200, **kwargs)
        r200 = (M200/(4/3.*np.pi*200*rho_c))**(1/3.)
        rho_s = M200/(4*np.pi*(r200/c200)**3*(np.log(1 + c200) - c200/(1 + c200)))
        r_s = r200/c200
        return r_s, rho_s

    def get_rb_rhob_Burk(self, M200, p=0.7):
        """ Get Burkert scale radius and density
        """
        c200n = self.c200_SCP(M200)
        c200b = c200n/p
        r200 = (M200/(4/3.*np.pi*200*rho_c))**(1/3.)
        r_b = r200/c200b
        rho_b = M200/(r_b**3*np.pi*(-2*np.arctan(c200b) + np.log((1+c200b)**2*(1+c200b**2))))
        return r_b, rho_b

    ##################################################
    # Concentration-mass relations
    ##################################################
    # Note to self: be careful with the assumptions 
    #    (e.g. cosmology, subhalo mass function, 
    #    (sub)halo profile) that go into these and  
    #    make sure that onsistent with the rest of 
    #    the analysis.
    ##################################################

    def R0_VL(self, M0):
        """ 'Concentration-mass' relation for a local Plummer profile 
            according to eq. 14 of 1606.04946
        """
        return 1.62*kpc*(M0/(1e8*M_s))**0.5

    def c200_SCP(self, M200):
        """ Concentration-mass relation according to eq. 1 of  Sanchez-Conde & Prada 2014 (1312.1729)
            :param M200: M200 mass of subhalo
        """
        x=np.log(M200/(M_s/h))
        pars=[37.5153, -1.5093, 1.636e-2, 3.66e-4, -2.89237e-5, 5.32e-7][::-1]
        return np.polyval(pars, x)

    def c200_Correa(self, M200, z=0):
        """ Concentration-mass relation according to eq. 19 of 1502.00391
            :param M200: M200 mass of subhalo
        """
        alpha = 1.62774 - 0.2458*(1+z) + 0.01716*(1+z)**2
        beta = 1.66079 + 0.00359*(1+z) - 1.6901*(1+z)**0.00417
        gamma = -0.02049 + 0.0253*(1+z)**-0.1044
        c200_val = 10**(alpha + beta * np.log10(M200/M_s)*(1+gamma*(np.log10(M200/M_s))**2))
        return c200_val

    def c200_Moline(self, M200, xsub=1.):
        """ Concentration-mass relation according to eq. 6 of 1603.04057
            :param M200: M200 mass of subhalo
            :param xsub: distance of subhalo from the center of host halo
                in units of virial radius of host halo
        """
        c0 = 19.9
        a = [-0.195, 0.089, 0.089]
        b = -0.54
        c200_val = c0*(1 + np.sum([(a[i-1]*np.log10(M200/(1e8*M_s/h)))**i for i in range(1,4)])*(1 + b*np.log10(xsub)))
        return c200_val

    ##################################################
    # Spatial distributions and profiles
    ##################################################

    def r2rho_V_NFW(self, r, r_s=18*kpc):
        """ Unnormalized NFW profile density.
            Default scale radius for Milky Way
        """
        return r**2*1/((r/r_s)*(1+r/r_s)**2)

    def r2rho_V_NFW_evolved(self, r, r_s=18*kpc, gamma=1.3):
        """ Unnormalized NFW profile density.
            Default scale radius for Milky Way, gamma reflects stripping
            in subhalo number density population according to 1509.02175
            from Aquarius halo A.
        """
        return r**2*1/((r/r_s)*(1+r/r_s)**2)*r**gamma

    def r2rho_V_ein_EAQ(self, r, r_s=199*kpc, alpha_E=0.678):
        """ Unnormalized Einasto density profile.
            Default parameters reflect stripped subhalo distribution according 
            to 1606.04898 from Aquarius.
        """ 
        return r**2*np.exp(-2/alpha_E*((r/r_s)**alpha_E-1)) 

    ##################################################
    # Mass distributions
    ##################################################

    def rho_M_WDM(self, M, mWDM=3.3*KeV, alpha=-1.9, beta = -1.3):
        """ Halo mass function for WDM (sterile neutrino) cosmologies
            according to Schneider et al. (2012) from 1801.01505
        """
        A0 = 2e8/M_s
        pref = (1 + self.mhm(mWDM)/M)**beta
        return pref*self.rho_M_SI(M, alpha)

    def mhm(self, m):
        """ Half-mode mass (mass scale at which the linear matter power 
            spectrum is reduced by 50 per cent). See 1801.01505 for 
            details.
        """
        return 1e10*(m/KeV)**-3.33*M_s/h

    def rho_M_SI(self, M, alpha=-1.9):
        """ Scale-invariant halo mass function
        """ 
        A0 = 2e8/M_s
        return A0*(M/M_s)**alpha

    ##################################################
    # Velocity distributions
    ##################################################

    def rho_v_SHM(self, vvec, v0=220.*Kmps, vesc=544.*Kmps):
        """ Normalized truncated Maxwellian velocity distribution
        """
        Nesc = erf(vesc/v0) - 2/np.sqrt(np.pi)*vesc/v0*np.exp(-vesc**2/v0**2)
        v = np.linalg.norm(vvec)
        return 1/(Nesc*np.pi**1.5*v0**3)*np.exp(-v**2/v0**2)*(v < vesc)

    def vE(self, t):
        """ Earth velocity at day of year t 
        """
        g = lambda t: omega*t
        nu = lambda g: g + 2*e*np.sin(g) + (5/4.)*e**2*np.sin(2*g)
        lambdaa = lambda t: lambdap + nu(g(t))
        r = lambda t: vE0/omega*(1-e**2)/(1+e*np.cos(nu(g(t))))*(np.sin(lambdaa(t))*e1 - np.cos(lambdaa(t))*e2)
        return derivative(r,t)*Kmps
