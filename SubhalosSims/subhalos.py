import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, Galactic, Galactocentric, CartesianDifferential

import pdf_sampler
from units import *

class SubhaloSample:
    def __init__(self, r_vir=213.5*kpc, r_s=0.81, alpha_E=0.67, m_host=1.1e12*M_s, m_min=10**-6, m_max=1e-1, alpha_m=1.9, m_min_calib=1e8*M_s, m_max_calib=1e10*M_s, N_calib=300, get_J=False):
        """ Class for generating a Galactic subhalo sample
        Args:
            r_vir: host virial radius in natural units
            r_s: host scale radius in units of r_vir
            alpha_E: Einasto parameter
            m_min: minimum subhalo mass in units of the host halo mass
            m_max: maximum subhalo mass in units of the host halo mass
            m_host: host mass in natural units
            alpha_m: slope of subhalo mass function, dN/dm ~ m^-alpha
            m_min_calib: minimum subhalo mass for calibration of subhalo number
            m_max_calib: maximum subhalo mass for calibration of subhalo number
            N_calib: number of subhalos between m_min_calib and m_max_calib
        Returns:
            double log likelihood
        """

        self.r_vir = r_vir
        self.r_s = r_s*r_vir
        self.alpha_E = alpha_E

        self.m_min_calib = m_min_calib
        self.m_max_calib = m_max_calib

        self.m_host = m_host
        self.m_min = m_min*m_host
        self.m_max = m_max*m_host

        self.alpha_m = alpha_m

        self.N_calib = N_calib

        self.load_constants()

    def get_sh_sample(self):
        """ Generate sample of subhalos
        """
        self.get_N_sh()
        self.get_r_sample()
        self.get_m_sample()
        self.get_sh_prop()
        self.get_v_sample()
        self.get_coords_galactic()

    def load_constants(self):
        self.h = 0.7
        self.rho_c = 1.8788e-26*self.h**2*Kilogram/Meter**3
        
    def get_N_sh(self):
        """ Get number of subhalos calibrated to N_calib between m_min_calib and m_max_calib
        """
        self.N_sh = (self.m_max**(-self.alpha_m+1)-self.m_min**(-self.alpha_m+1))*self.N_calib/(self.m_max_calib**(-self.alpha_m+1)-self.m_min_calib**(-self.alpha_m+1))
        self.N_sh = np.random.poisson(self.N_sh)

    def rho_ein(self, r):
        """Unnormalized Einasto profile density"""
        return np.exp((-2/self.alpha_E)*((r/self.r_s)**self.alpha_E-1))

    def rho_m(self, m):
        """Unnormalized mass function"""
        return m**-self.alpha_m 

    def get_r_sample(self):
        """ Sample Galactocentric radii
        """
        r_vals = np.linspace(0,1000,10000)*kpc
        rho_vals = r_vals**2*self.rho_ein(r_vals)
        r_dist = pdf_sampler.PDFSampler(r_vals, rho_vals)
        self.r_sample = r_dist(self.N_sh)

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
        self.v_sample = v_dist(self.N_sh)

        self.coords_vxyz = self.v_sample*self.sample_spherical(self.N_sh) # Sample random vectors

    def get_m_sample(self):
        """ Sample subhalo masses
        """
        m_vals = np.linspace(self.m_min,self.m_max,10000)
        rho_vals = self.rho_m(m_vals)
        m_dist = pdf_sampler.PDFSampler(m_vals, rho_vals)
        self.m_sample = m_dist(self.N_sh)

    def get_sh_prop(self):
        """ Get subhalos rvir and rs
        """
        self.c200_sample = np.array([self.c200(m/M_s) for m in self.m_sample])
        self.r200_sample = np.array([(m/(4/3.*np.pi*200*self.rho_c))**(1/3.) for m in self.m_sample])
        self.rs_sample = self.r200_sample/self.c200_sample
        self.rho_s_sample = self.rho_c*(200/3.)*self.c200_sample**3/(np.log(1 + self.c200_sample) - self.c200_sample/(1 + self.c200_sample))

    def sample_spherical(self, npoints, ndim=3):
        """ Sample random vectors
        """
        vec = np.random.randn(ndim, npoints)
        vec /= np.linalg.norm(vec, axis=0)
        return vec

    def get_coords_galactic(self):
        """ Convert to Galactic coordinates
        """
        coords_xyz = self.sample_spherical(self.N_sh)  # Sample random vectors
        self.coords_gc = Galactocentric(x=coords_xyz[0]*self.r_sample/kpc*u.kpc,  # Scale vectors by sampled 
                             y=coords_xyz[1]*self.r_sample/kpc*u.kpc,  # distance and convert to 
                             z=coords_xyz[2]*self.r_sample/kpc*u.kpc,  # galactocentric coordinates
                             v_x=self.coords_vxyz[1]/Kmps*u.km/u.s,
                             v_y=self.coords_vxyz[1]/Kmps*u.km/u.s,
                             v_z=self.coords_vxyz[1]/Kmps*u.km/u.s)
        self.coords_gc_nosunv = Galactocentric(x=coords_xyz[0]*self.r_sample/kpc*u.kpc,  # Scale vectors by sampled 
                             y=coords_xyz[1]*self.r_sample/kpc*u.kpc,  # distance and convert to 
                             z=coords_xyz[2]*self.r_sample/kpc*u.kpc,  # galactocentric coordinates
                             v_x=self.coords_vxyz[1]/Kmps*u.km/u.s,
                             v_y=self.coords_vxyz[1]/Kmps*u.km/u.s,
                             v_z=self.coords_vxyz[1]/Kmps*u.km/u.s,
                             galcen_v_sun=CartesianDifferential(np.array([0,0,0])*u.km/u.s))

        self.coords_galactic =self.coords_gc.transform_to(Galactic)  # Transform to local coordinates
        self.coords_galactic_nosunv =self.coords_gc_nosunv.transform_to(Galactic)  # Transform to local coordinates

    def c200(self, m200):
        """ According to Sanchez-Conde&Prada14
        """
        # x=np.log(m200*self.h) # Given in terms of M_s/h in S-C&P paper
        x=np.log(m200) # Given in terms of M_s/h in S-C&P paper
        pars=[37.5153, -1.5093, 1.636e-2, 3.66e-4, -2.89237e-5, 5.32e-7][::-1]
        return np.polyval(pars, x)

