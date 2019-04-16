import sys
sys.path.append("../theory")

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, Galactic, Galactocentric, CartesianDifferential
from scipy.integrate import quad
from skmonaco import mcquad, mcimport, mcmiser

import pdf_sampler
from units import *
from profiles import Profiles

class SubhaloSample(Profiles):
    def __init__(self, use_v_E=False, t=150., sh_profile='NFW'):
        """ Class for generating a Galactic subhalo sample
        Args:
            t: time of year in days at which to use Earth velocity 
            use_v_E: whether to add on Earth velocity
            sh_profile: ["NFW", "Plummer", "Gaussian"]
        """

        Profiles.__init__(self)

        self.use_v_E = use_v_E
        self.t = t
        self.sh_profile = sh_profile

    def get_sh_sample(self):
        """ Generate sample of subhalos
        """
        self.get_r_sample()
        self.get_M_sample()
        self.get_sh_prop()
        self.get_v_sample()
        self.get_coords_galactic()

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

        # self.N_halos = self.N_calib*quad(lambda M: self.rho_M(M, **self.rho_M_kwargs), M_min, M_max)[0]/quad(lambda M:self.rho_M(M, **self.rho_M_kwargs), M_min_calib, M_max_calib)[0]
        # self.N_halos = np.random.poisson(self.N_halos)

        def integ(logM):
            M = np.exp(logM)*M_s
            return M*self.rho_M(M)

        norm1, _ = mcquad(integ , npoints=1e6,xl=[np.log(self.M_min/M_s)],xu=[np.log(self.M_max/M_s)],nprocs=5)

        norm2, _ = mcquad(integ, npoints=1e6,xl=[np.log(self.M_min_calib/M_s)],xu=[np.log(self.M_max_calib/M_s)],nprocs=5)


        self.N_halos = np.random.poisson(self.N_calib * norm1 / norm2)[0]

        print("Simulating", str(self.N_halos), "subhalos between", str(self.M_min/M_s), "and",  str(self.M_max/M_s))

    def set_subhalo_properties(self, c200_model, distdep=False):

        self.c200_model = c200_model
        self.c200_distdep = distdep # Distance dependent

    def get_r_sample(self):
        """ Sample Galactocentric radii
        """
        r_vals = np.linspace(self.R_min,self.R_max,10000)
        rho_vals = self.rho_R(r_vals)
        r_dist = pdf_sampler.PDFSampler(r_vals, rho_vals)
        self.r_sample = r_dist(self.N_halos)

    def get_v_sample(self):
        """ Get Galactocentric velocities drawn from a SHM Maxwellian
        """
        v_vals = np.linspace(0,550.,10000)*Kmps
        rho_vals = self.rho_v_SHM_scalar(v_vals)
        v_dist = pdf_sampler.PDFSampler(v_vals, rho_vals)
        self.v_sample = v_dist(self.N_halos)

        self.coords_vxyz = self.v_sample*self.sample_spherical(self.N_halos) # Sample random vectors

    def get_M_sample(self):
        """ Sample subhalo masses
        """
        M_vals = np.linspace(self.M_min,self.M_max,10000000)
        rho_vals = self.rho_M(M_vals)
        M_dist = pdf_sampler.PDFSampler(M_vals, rho_vals)
        self.M_sample = M_dist(self.N_halos)

    def get_sh_prop(self):
        """ Get subhalos properties
        """
        if self.sh_profile == "NFW":
            if self.c200_distdep:
                self.c200_sample = self.c200_model(self.M_sample, self.r_sample)
            else:
                self.c200_sample = self.c200_model(self.M_sample)
            self.r200_sample = (self.M_sample/(4/3.*np.pi*200*rho_c))**(1/3.)
            self.rs_sample = self.r200_sample/self.c200_sample
            self.rho_s_sample = rho_c*(200/3.)*self.c200_sample**3/(np.log(1 + self.c200_sample) - self.c200_sample/(1 + self.c200_sample))
        elif self.sh_profile in ["Plummer", "Gaussian"]:
            self.c200_sample = self.R0_VL(self.M_sample)

    def sample_spherical(self, npoints, ndim=3):
        """ Sample random vectors
        """
        vec = np.random.randn(ndim, npoints)
        vec /= np.linalg.norm(vec, axis=0)
        return vec

    def get_coords_galactic(self):
        """ Convert to Galactic coordinates
        """
        coords_xyz = self.sample_spherical(self.N_halos)  # Sample random vectors
        v_sun = np.array([11., 232., 7.]) # Velocity of the Sub in Galactocentric frame

        if self.use_v_E: # If want non-zero Earth velocity
            v_E = self.vE(self.t) 
        else:
            v_E = np.zeros(3)

        # Rotate about x-axis to ecliptic coordinates. CHECK IF THIS IS RIGHT.
        # v_sun_E_ecliptic = CartesianDifferential(np.array([0, np.linalg.norm(v_sun + v_E), 0])*u.km/u.s)
        # v_sun_E_ecliptic = CartesianDifferential((v_sun + v_E)*u.km/u.s)
        v_sun_E_ecliptic = CartesianDifferential(np.zeros(3)*u.km/u.s)

        self.coords_gc = Galactocentric(
                             x=coords_xyz[0]*self.r_sample/kpc*u.kpc,  # Scale vectors by sampled 
                             y=coords_xyz[1]*self.r_sample/kpc*u.kpc,  # distance and convert to 
                             z=coords_xyz[2]*self.r_sample/kpc*u.kpc,  # galactocentric coordinates
                             v_x=self.coords_vxyz[0]/Kmps*u.km/u.s,
                             v_y=self.coords_vxyz[1]/Kmps*u.km/u.s,
                             v_z=self.coords_vxyz[2]/Kmps*u.km/u.s,
                             # galcen_v_sun=)
                             galcen_v_sun=v_sun_E_ecliptic)

        self.coords_galactic = self.coords_gc.transform_to(Galactic)  # Transform to Galactic coordinates
        self.d_sample = self.coords_galactic.distance.value*kpc
