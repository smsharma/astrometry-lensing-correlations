## TODO: 
# - Fisher plotting function
# - Which parameters to vary? (minimum subhalo mass)
# - One sided Fisher
# - Limit-setting with Fisher

import sys
import os
sys.path.append("../Simulations/")

import numpy as np
import mpmath as mp
from scipy.special import erf, jn, jv, kn
from scipy.interpolate import interp2d
from scipy.integrate import quad, nquad
from tqdm import *
from skmonaco import mcquad, mcimport, mcmiser

from units import *
from profiles import Profiles

class PowerSpectra(Profiles):
    """ Class to calculate expected power spectra from astrometric induced velocities and accelerations
       
        :param precompute: List of profiles to precompute arrays for to speed up computation ['Burk', 'NFW']
    """
    def __init__(self, precompute=['Burk', 'NFW']):

        Profiles.__init__(self)

        # Precompute arrays to speed up computation
        if 'Burk' in precompute:
            self.precompute_MBurkdivM0()
        if 'NFW' in precompute:
            self.precompute_MNFWdivM0()

    ##################################################
    # Induced velocity/acceleration power spectra
    ##################################################

    def Cl_Gauss(self, R0, M0, Dl, v, l):
        """ Induced velocity power spectrum for Gaussian lens
            :param R0: size of lens
            :param M0: mass of lens
            :param Dl: (physical) distance to lens
            :param v: (physical) velocity of lens in projection transverse to los
            :param l: multipole
        """
        beta0 = R0/Dl
        return (4*GN*M0*v/Dl**2)**2*np.pi/2.*np.exp(-l**2*beta0**2)

    def Cl_Plummer(self, R0, M0, Dl, v, l):
        """ Induced velocity power spectrum for Plummer lens
            :param R0: size of lens
            :param M0: mass of lens
            :param Dl: (physical) distance to lens
            :param v: (physical) velocity of lens in projection transverse to los
            :param l: multipole
        """
        beta0 = R0/Dl
        return (4*GN*M0*v/Dl**2)**2*np.pi/2.*l**2*beta0**2*kn(1, l*beta0)**2

    def Cl_Point(self, M0, Dl, v, l):
        """ Induced velocity power spectrum for point lens
            :param M0: mass of lens
            :param Dl: (physical) distance to lens
            :param v: (physical) velocity of lens in projection transverse to los
            :param l: multipole
        """
        return (4*GN*M0*v/Dl**2)**2*np.pi/2.

    def Cl_NFW(self, M200, Dl, v, l, Rsub=None):
        """ Induced velocity power spectrum for NFW lens
            :param M200: M200 of NFW lens
            :param Dl: (physical) distance to lens
            :param v: (physical) velocity of lens in projection transverse to los
            :param l: multipole
        """
        if self.c200_model is self.c200_Moline:
            kwargs = {'xsub':Rsub/R200_MW}
        else:
            kwargs = {}
        r_s, rho_s = self.get_rs_rhos_NFW(M200, **kwargs)
        M0 = 4*np.pi*r_s**3*rho_s
        pref = GN**2*v**2*8*np.pi*l**2/Dl**4
        theta_s = r_s/Dl
        if not self.precompute_NFW:
            MjdivM0 = self.MNFWdivM0_integ(theta_s, l)
        else:
            MjdivM0 = 10**self.MNFWdivM0_integ_interp(np.log10(l), np.log10(theta_s))[0]
        return pref*M0**2*MjdivM0**2

    def Cl_tNFW(self, M200, Dl, v, l, tau=15.):
        """ Induced velocity power spectrum for truncated NFW lens
            :param M200: M200 of NFW lens
            :param Dl: (physical) distance to lens
            :param v: (physical) velocity of lens in projection transverse to los
            :param l: multipole
            :param tau: ratio of truncation and scale radii
        """
        r_s, rho_s = self.get_rs_rhos_NFW(M200)
        M0 = 4*np.pi*r_s**3*rho_s
        pref = GN**2*v**2*8*np.pi*l**2/Dl**4
        theta_s = r_s/Dl
        self.set_mp()
        MjdivM0 = mp.quadosc(lambda theta: self.MtNFWdivM0(theta/theta_s, tau)*mp.j1(l*theta), [0, mp.inf], period=2*mp.pi/l)
        return pref*M0**2*MjdivM0**2

    def Cl_Burk(self, M200, Dl, v, l, p=0.7):
        """ Induced velocity power spectrum for Burkert lens
            :param M200: M200 of NFW lens
            :param Dl: (physical) distance to lens
            :param v: (physical) velocity of lens in projection transverse to los
            :param l: multipole
            :param p: ratio of NFW and Burkert concentrations
        """
        r_b, rho_b = self.get_rb_rhob_Burk(M200, p)
        M0 = 4*np.pi*r_b**3*rho_b
        pref = GN**2*v**2*8*np.pi*l**2/Dl**4
        theta_b = r_b/Dl
        if not self.precompute_Burk:
            MjdivM0 = self.MBurkdivM0_integ(theta_b, l)
        else:
            MjdivM0 = 10**self.MBurkdivM0_integ_interp(np.log10(l), np.log10(theta_b))[0]
        return pref*M0**2*MjdivM0**2

class PowerSpectraPopulations(PowerSpectra):
    """ Class to calculate expected power spectra from astrometric induced velocities and accelerations
       
        :param precompute: List of profiles to precompute arrays for to speed up computation ['Burk', 'NFW']
    """
    def __init__(self, l_min=1, l_max=2000, n_l=50, R_min=1*kpc, R_max=200*kpc):

        PowerSpectra.__init__(self)

        self.l_min = 1
        self.l_max = 2000
        self.n_l = 50

        self.l_ary = np.arange(self.l_min,self.l_max)
        self.l_ary_calc = np.logspace(np.log10(self.l_min), np.log10(self.l_max), self.n_l)  

        self.calc_v_proj_mean_integrals()

    def integrand_norm(self, x):
        """ Integrand for calculating overall normalization of 
            joint mass-radial distribution pdf
        """ 
        M200, r = np.exp(x[0])*M_s, np.exp(x[1])*kpc
        return 4*np.pi*M200*r*self.rho_M(M200)*self.rho_R(r)

    def integrand_norm_compact(self, x):
        """ Integrand for calculating overall normalization of 
            joint mass-radial distribution pdf
        """ 
        r = np.exp(x[0])*kpc
        return 4*np.pi*r*self.rho_R(r)

    def set_radial_distribution(self, rho_R, R_min, R_max, **kwargs):

        self.rho_R = rho_R
        self.rho_R_kwargs = kwargs

        self.R_min = R_min
        self.R_max = R_max

    def set_mass_distribution(self, rho_M, M_min, M_max, M_min_calib, M_max_calib, N_calib, **kwargs):
        # TODO: Stabilize distributions

        self.rho_M = rho_M
        self.rho_M_kwargs = kwargs

        self.M_min = M_min
        self.M_max = M_max

        self.M_min_calib = M_min_calib
        self.M_max_calib = M_max_calib

        self.N_calib = N_calib

        norm, norm_err = mcquad(self.integrand_norm,npoints=1e6,xl=[np.log(self.M_min_calib/M_s),np.log(self.R_min/kpc)],xu=[np.log(self.M_max_calib/M_s),np.log(self.R_max/kpc)],nprocs=5)

        self.pref = self.N_calib / norm

    def set_mass_distribution_compact(self, M_DM, R0_DM, f_DM, **kwargs):
        # TODO: Stabilize distributions

        self.M_DM = M_DM
        self.R0_DM = R0_DM
        self.f_DM = f_DM

        norm, norm_err = mcquad(self.integrand_norm_compact,npoints=1e6,xl=[np.log(self.R_min/kpc)],xu=[np.log(self.R_max/kpc)],nprocs=5)

        self.pref = self.f_DM*1./(self.M_DM/(1e12*M_s)) / norm

    def set_subhalo_properties(self, c200_model):

        self.c200_model = c200_model

    def calc_v_proj_mean_integrals(self):
        # TODO: CHECK THESE!!
        
        vsun = np.array([11.*Kmps, 232.*Kmps, 7.*Kmps])

        print("Calculating velocity integrals")

        # Mean projected v**2 for velocity integral
        self.vsq_proj_mean = 3.680502364741616e-07
 #nquad(lambda v, theta, phi: (v/2.)**2*v**2*np.sin(theta)*self.rho_v_SHM(v*np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)]) + vsun + self.vE(150)), [[0,950.*Kmps],[0,np.pi],[0,2*np.pi]])[0]

        # Mean projected v**4 for acceleration integral
        # TODO: Need to improve accuracy!
        self.v4_proj_mean = 2.039716181028691e-13
 #nquad(lambda v, theta, phi: (v/2.)**4*v**2*np.sin(theta)*self.rho_v_SHM(v*np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)]) + vsun + self.vE(150)), [[0,950.*Kmps],[0,np.pi],[0,2*np.pi]])[0]

    def integrand(self, x, ell, accel=False):
        
        logR, theta, logm = x[0], x[1], x[2]
        m = np.exp(logm)*M_s
        r = np.exp(logR)*kpc
        
        if accel:
            pref = (3/64)*ell**2/r**2
            units = (1e-6*asctorad/Year**2)**2
        else:
            pref = 1
            units = (1e-6*asctorad/Year)**2

        l = np.sqrt(r**2 + Rsun**2 + 2*r*Rsun*np.cos(theta))
        return pref*r*m*self.Cl_NFW(m, l, 1, ell, r) / units  * self.rho_M(m, **self.rho_M_kwargs) * self.rho_R(r, **self.rho_R_kwargs)

    def integrand_compact(self, x, ell, accel=False):

        logR, theta = x[0], x[1]
        r = np.exp(logR)*kpc
        
        if accel:
            pref = (3/64)*ell**2/r**2
            units = (1e-6*asctorad/Year**2)**2
        else:
            pref = 1
            units = (1e-6*asctorad/Year)**2

        l = np.sqrt(r**2 + Rsun**2 + 2*r*Rsun*np.cos(theta))
        if not self.R0_DM == 0:
            return pref*r*self.Cl_Gauss(self, self.R0_DM, self.M_DM, l, 1, ell) / units  * self.rho_R(r, **self.rho_R_kwargs)
        else:
            return pref*r*self.Cl_Point(self.M_DM, l, 1, ell) / units  * self.rho_R(r, **self.rho_R_kwargs)

    def C_l_total(self, ell, theta_deg_mask = 0, accel=False):
        
        theta_rad_mask = np.deg2rad(theta_deg_mask)

        logR_integ_ary = np.linspace(np.log(self.R_min/kpc), np.log(self.R_max/kpc), 20)
        theta_integ_ary = np.linspace(theta_rad_mask, np.pi-theta_rad_mask, 20)
        logM_integ_ary = np.linspace(np.log(self.M_min/M_s), np.log(self.M_max/M_s), 20)

        measure = (logR_integ_ary[1] - logR_integ_ary[0])*(theta_integ_ary[1] - theta_integ_ary[0])*(logM_integ_ary[1] - logM_integ_ary[0])
        
        integ = 0
        for logR in logR_integ_ary:
            for theta in theta_integ_ary:
                for logM in logM_integ_ary:
                    integ += np.sin(theta)*self.integrand([logR, theta, logM], ell, accel)
        integ *= 2*np.pi*measure

        if accel:
            v_term = self.v4_proj_mean
        else:
            v_term = self.vsq_proj_mean

        return self.pref * integ * v_term

    def C_l_compact_total(self, ell, theta_deg_mask = 0, accel=False):
        
        theta_rad_mask = np.deg2rad(theta_deg_mask)

        logR_integ_ary = np.linspace(np.log(self.R_min/kpc), np.log(self.R_max/kpc), 20)
        theta_integ_ary = np.linspace(theta_rad_mask, np.pi-theta_rad_mask, 20)

        measure = (logR_integ_ary[1] - logR_integ_ary[0])*(theta_integ_ary[1] - theta_integ_ary[0])
        
        integ = 0
        for logR in logR_integ_ary:
            for theta in theta_integ_ary:
                integ += np.sin(theta)*self.integrand_compact([logR, theta], ell, accel)
        integ *= 2*np.pi*measure

        if accel:
            v_term = self.v4_proj_mean
        else:
            v_term = self.vsq_proj_mean

        return self.pref * integ * v_term


    def dC_l_dR_total(self, ell, R, theta_deg_mask = 0, accel=False):
        
        theta_rad_mask = np.deg2rad(theta_deg_mask)

        theta_integ_ary = np.linspace(theta_rad_mask, np.pi-theta_rad_mask, 20)
        logM_integ_ary = np.linspace(np.log(self.M_min/M_s), np.log(self.M_max/M_s), 20)

        measure = (theta_integ_ary[1] - theta_integ_ary[0])*(logM_integ_ary[1] - logM_integ_ary[0])
        
        integ = 0
        for theta in theta_integ_ary:
            for logM in logM_integ_ary:
                integ += np.sin(theta)*self.integrand([np.log(R/kpc), theta, logM], ell, accel)/R
        integ *= 2*np.pi*measure

        if accel:
            v_term = self.v4_proj_mean
        else:
            v_term = self.vsq_proj_mean

        return self.pref * integ * v_term

    def dC_l_dM_total(self, ell, M, theta_deg_mask = 0, accel=False):
        
        theta_rad_mask = np.deg2rad(theta_deg_mask)

        logR_integ_ary = np.linspace(np.log(self.R_min/kpc), np.log(self.R_max/kpc), 20)
        theta_integ_ary = np.linspace(theta_rad_mask, np.pi-theta_rad_mask, 20)

        measure = (logR_integ_ary[1] - logR_integ_ary[0])*(theta_integ_ary[1] - theta_integ_ary[0])
        
        integ = 0
        for logR in logR_integ_ary:
            for theta in theta_integ_ary:
                integ += np.sin(theta)*self.integrand([logR, theta, np.log(M/M_s)], ell, accel)/M
        integ *= 2*np.pi*measure

        if accel:
            v_term = self.v4_proj_mean
        else:
            v_term = self.vsq_proj_mean

        return self.pref * integ * v_term 


    def get_C_l_total_ary(self, theta_deg_mask = 10, accel=False):
        C_l_calc_ary = [self.C_l_total(ell, theta_deg_mask = 10, accel=False) for ell in tqdm_notebook(self.l_ary_calc)]
        self.C_l_ary = 10**np.interp(np.log10(self.l_ary), np.log10(self.l_ary_calc), np.log10(C_l_calc_ary))
        return self.C_l_ary

    def get_C_l_compact_total_ary(self, theta_deg_mask = 10, accel=False):
        self.C_l_ary = len(self.l_ary)*[self.C_l_compact_total(1, theta_deg_mask = 10, accel=False)]
        return np.array(self.C_l_ary)

def integ(M_min, M_max, alpha=-1.9):
    return (M_max**(alpha + 1) - M_min**(alpha + 1))/(alpha + 1)
    if alpha == -1:
        return np.log(M_max) - np.log(M_min)

def integ_M(M_min, M_max, alpha=-1.9):
    if alpha == -2:
        return np.log(M_max) - np.log(M_min)
    return (M_max**(alpha + 2) - M_min**(alpha + 2))/(alpha + 2)