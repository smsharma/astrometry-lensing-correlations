import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, Galactic, Galactocentric, CartesianDifferential
from scipy.misc import derivative

import pdf_sampler
from units import *

class SubhaloSample:
    def __init__(self, r_vir=213.5*kpc, m_host=1.4e12*M_s, m_min=10**-6, m_max=1e-1, alpha_m=1.9, m_min_calib=1e8*M_s, m_max_calib=1e10*M_s, n_calib=None, n_sh=None, sh_m_frac=1, m_delta=None, sh_distrib='Aq1', t=365.25, custom_coords=None, sh_profile='NFW'):
        """ Class for generating a Galactic subhalo sample
        Args:
            r_vir: host virial radius in natural units
            m_min: minimum subhalo mass in units of the host halo mass
            m_max: maximum subhalo mass in units of the host halo mass
            m_host: host mass in natural units
            alpha_m: slope of subhalo mass function, dN/dm ~ m^-alpha
            m_min_calib: minimum subhalo mass for calibration of subhalo number
            m_max_calib: maximum subhalo mass for calibration of subhalo number
            n_calib: number of subhalos between m_min_calib and m_max_calib
            n_sh: number of subhalos, if specified will override n_calib and sh_m_frac
            sh_m_frac: Subhalos mass fraction, if specified will override n_calib
            m_delta: if specified, generate all subhalos at single mass 
            sh_distrib: distribution of subhalos, 'MW' or 'Aq1' 
            t: time of year in days at which to use Earth velocity 
            custom_coords: Custom subhalo coordinates
            sh_profile: ["NFW", "Plummer"]
        """

        self.r_vir = r_vir

        self.sh_distrib = sh_distrib

        if self.sh_distrib == 'Aq1':
            self.r_s = 0.81*r_vir
            self.alpha_E = 0.67
        elif self.sh_distrib == 'MW':
            self.r_s = 18*kpc

        self.m_min_calib = m_min_calib
        self.m_max_calib = m_max_calib

        self.m_host = m_host
        self.m_min = m_min*m_host
        self.m_max = m_max*m_host

        self.m_delta = m_delta

        self.alpha_m = alpha_m

        self.n_calib = n_calib
        self.n_sh = n_sh
        self.sh_m_frac = sh_m_frac

        self.t = t
        self.custom_coords = custom_coords

        self.sh_profile = sh_profile

    def get_sh_sample(self):
        """ Generate sample of subhalos
        """
        if self.n_sh is not None:
            pass
        elif self.sh_m_frac is not None and self.m_delta is not None:
            self.n_sh = np.random.poisson(self.sh_m_frac*self.m_host/self.m_delta)
        elif self.n_calib is not None:
            self.get_n_sh()
        self.get_r_sample()
        self.get_m_sample()
        self.get_sh_prop()
        self.get_v_sample()
        self.get_coords_galactic()

    def get_n_sh(self):
        """ Get number of subhalos calibrated to n_calib between m_min_calib and m_max_calib
        """
        self.n_sh = (self.m_max**(-self.alpha_m+1)-self.m_min**(-self.alpha_m+1))*self.n_calib/(self.m_max_calib**(-self.alpha_m+1)-self.m_min_calib**(-self.alpha_m+1))
        self.n_sh = np.random.poisson(self.n_sh)

    def rho_ein(self, r):
        """Unnormalized Einasto profile density"""
        return np.exp((-2/self.alpha_E)*((r/self.r_s)**self.alpha_E-1))

    def rho_nfw(self, r):
        """Unnormalized NFW profile density"""
        return 1/((r/self.r_s)*(1+r/self.r_s)**2)

    def rho_m(self, m):
        """Unnormalized mass function"""
        return m**-self.alpha_m 

    def get_r_sample(self):
        """ Sample Galactocentric radii
        """
        r_vals = np.linspace(1e-5,1000,10000)*kpc
        if self.sh_distrib == 'Aq1':
            rho_r = self.rho_ein
        elif self.sh_distrib == 'MW':
            rho_r = self.rho_nfw
        rho_vals = r_vals**2*rho_r(r_vals)
        r_dist = pdf_sampler.PDFSampler(r_vals, rho_vals)
        self.r_sample = r_dist(self.n_sh)

    def rho_v(self, v):
        """Unnormalized SHM Maxwellian"""
        v0 = 220*Kmps
        return np.exp(-v**2/v0**2)

    def get_v_sample(self):
        """ Get Galactocentric velocities drawn from a SHM Maxwellian
        """
        v_vals = np.linspace(0,550.,10000)*Kmps # Cutoff at escape velocity
        rho_vals = self.rho_v(v_vals)
        v_dist = pdf_sampler.PDFSampler(v_vals, rho_vals)
        self.v_sample = v_dist(self.n_sh)

        self.coords_vxyz = self.v_sample*self.sample_spherical(self.n_sh) # Sample random vectors

    def get_m_sample(self):
        """ Sample subhalo masses
        """
        if self.m_delta is None:
            m_vals = np.linspace(self.m_min,self.m_max,10000000)
            rho_vals = self.rho_m(m_vals)
            m_dist = pdf_sampler.PDFSampler(m_vals, rho_vals)
            self.m_sample = m_dist(self.n_sh)
        else:
            self.m_sample = self.m_delta*np.ones(self.n_sh)

    def get_sh_prop(self):
        """ Get subhalos properties
        """
        if self.sh_profile == "NFW":
            self.c200_sample = np.array([self.c200(m/M_s) for m in self.m_sample])
            self.r200_sample = np.array([(m/(4/3.*np.pi*200*rho_c))**(1/3.) for m in self.m_sample])
            self.rs_sample = self.r200_sample/self.c200_sample
            self.rho_s_sample = rho_c*(200/3.)*self.c200_sample**3/(np.log(1 + self.c200_sample) - self.c200_sample/(1 + self.c200_sample))
        elif self.sh_profile in ["Plummer", "Gaussian"]:
            self.c200_sample = np.array([self.R0_VL(m) for m in self.m_sample])

    def sample_spherical(self, npoints, ndim=3):
        """ Sample random vectors
        """
        vec = np.random.randn(ndim, npoints)
        vec /= np.linalg.norm(vec, axis=0)
        return vec

    def get_coords_galactic(self):
        """ Convert to Galactic coordinates
        """
        coords_xyz = self.sample_spherical(self.n_sh)  # Sample random vectors
        v_sun = np.array([11., 232., 7.]) # Velocity of the Sub in Galactocentric frame
        v_E = self.vE(self.t) # Earth velocity

        # Rotate about x-axis to ecliptic coordinates
        v_sun_E_ecliptic = CartesianDifferential(np.array([0, np.linalg.norm(v_sun + v_E), 0])*u.km/u.s)

        self.coords_gc = Galactocentric(
                             x=coords_xyz[0]*self.r_sample/kpc*u.kpc,  # Scale vectors by sampled 
                             y=coords_xyz[1]*self.r_sample/kpc*u.kpc,  # distance and convert to 
                             z=coords_xyz[2]*self.r_sample/kpc*u.kpc,  # galactocentric coordinates
                             v_x=self.coords_vxyz[0]/Kmps*u.km/u.s,
                             v_y=self.coords_vxyz[1]/Kmps*u.km/u.s,
                             v_z=self.coords_vxyz[2]/Kmps*u.km/u.s,
                             galcen_v_sun=v_sun_E_ecliptic)

        self.coords_galactic = self.coords_gc.transform_to(Galactic)  # Transform to local coordinates
        self.d_sample = self.coords_galactic.distance.value*kpc

        if self.custom_coords is not None:
            custom_l, custom_b, custom_pm_l_cosb, custom_pm_b, custom_d = self.custom_coords
            self.coords_galactic = Galactic(l=custom_l, b=custom_b, pm_l_cosb=custom_pm_l_cosb, pm_b=custom_pm_b)
            self.d_sample = custom_d

    def c200(self, m200):
        """ Virial concentration according to Sanchez-Conde&Prada14
        """
        x=np.log(m200*h) # Given in terms of M_s/h in S-C&P paper
        pars=[37.5153, -1.5093, 1.636e-2, 3.66e-4, -2.89237e-5, 5.32e-7][::-1]
        return np.polyval(pars, x)

    def R0_VL(self, M0):
        """ "Concentration-mass" relation for Plummer profile from 1711.03554
        """
        return 1.2*kpc*(M0/(1e8*M_s))**0.5


    def vE(self, t):
        """ Earth velocity at day of year t 
        """
        g = lambda t: omega*t
        nu = lambda g: g + 2*e*np.sin(g) + (5/4.)*e**2*np.sin(2*g)
        lambdaa = lambda t: lambdap + nu(g(t))
        r = lambda t: vE0/omega*(1-e**2)/(1+e*np.cos(nu(g(t))))*(np.sin(lambdaa(t))*e1 - np.cos(lambdaa(t))*e2)
        return derivative(r,t)