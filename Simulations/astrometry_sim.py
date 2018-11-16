import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import healpy as hp
from astropy import units as u
from astropy.coordinates import SkyCoord
from tqdm import *

from units import *
from subhalo_sim import SubhaloSample
import pdf_sampler

class QuasarSim(SubhaloSample):
    def __init__(self, verbose=True, max_sep=2, data_folder='../data/', save_tag='sample', save=False, save_folder='Output/', *args, **kwargs):
        """ Class for simulating lens-induced astrometric perturbations
            in Gaia DR2 QSOs. 
        
            :param max_sep: maximum seperation in degrees to consider between lenses and quasars
            :param data_folder: local folder with external data
            *args, **kwargs: lens population parameters passed to the SubhaloSample class 
        """
        SubhaloSample.__init__(self, *args, **kwargs)
        
        self.max_sep = max_sep
        self.verbose = verbose
        self.data_folder = data_folder
        self.save_folder = save_folder
        self.save_tag = save_tag
        self.save = save
        
        self.load_gaia_quasars()
        self.analysis_pipeline()

    def analysis_pipeline(self):
        """ Run analysis sequence
        """ 
        self.get_sh_sample()
        self.get_velocities()
        if self.save:
            self.save_products()

    def update_sample(self, *args, **kwargs):
        """ Update lens parameters and redo simulation

            *args, **kwargs: lens population parameters passed to the SubhaloSample class
        """
        SubhaloSample.__init__(self, *args, **kwargs)
        self.analysis_pipeline()

    def load_gaia_quasars(self):
        """ Load quasars and their positions
        """
        pd_qsrs = pd.read_csv(self.data_folder + "quasars_phot.csv")

        ra = pd_qsrs['ra'].values
        dec = pd_qsrs['dec'].values

        coords_qsrs_icrs = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')

        self.coords_qsrs = coords_qsrs_icrs.transform_to('galactic')
        self.n_qsrs = len(pd_qsrs)

    def get_velocities(self):
        """ Get induced velocities of quasars and store them in self.mu_qsrs
        """

        # Galactic coordinates of lenses
        coords_lenses = SkyCoord(frame="galactic", l=self.coords_galactic.l, b=self.coords_galactic.b)

        # Get indices `idxs_qsrs` of objects within `max_sep` of the lenses with indixes `idxs_lenses`
        idxs_lenses, idxs_qsrs, _, _ = self.coords_qsrs.search_around_sky(coords_lenses, self.max_sep*u.deg)  

        ## Loop over lenses and get induced velocities for nearby (within `max_sep`) quasars

        self.mu_qsrs = np.zeros((self.n_qsrs,2))

        for i_lens in tqdm(range(self.n_sh), disable=1-self.verbose):
                        
            # Lens velocity
            self.pm_l_cosb_lens = self.coords_galactic.pm_l_cosb.value[i_lens]
            self.pm_b_lens = self.coords_galactic.pm_b.value[i_lens]

            # Convert lens velocity from mas/yr to natural units
            v_lens = np.array([self.pm_l_cosb_lens, self.pm_b_lens])*1e-3*asctorad/Year

            # Indices of quasars around lens i_lens
            idxs_qsrs_around = idxs_qsrs[idxs_lenses == i_lens] 

            # Lens position
            self.l_lens = self.coords_galactic.l.value[i_lens]
            self.b_lens = self.coords_galactic.b.value[i_lens]
            
            # Angular impact parameters of quasars around lens i_lens
            beta_lens_qsrs_around = np.transpose([self.coords_qsrs.l.value[idxs_qsrs_around] - self.l_lens, 
                self.coords_qsrs.b.value[idxs_qsrs_around] - self.b_lens])*np.pi/180

            # Grab some lens properties
            c200_lens, m_lens, d_lens = self.c200_sample[i_lens], self.m_sample[i_lens], self.d_sample[i_lens]
            
            for i_qsr, idx_qsr in enumerate(idxs_qsrs_around):

                mu_qsr = self.mu(beta_lens_qsrs_around[i_qsr],
                                v_lens, c200_lens, m_lens, d_lens)
                
                self.mu_qsrs[idx_qsr] += mu_qsr

    def save_products(self):
        """ Save sim output
        """

        np.savez('Output/mu_qsrs_' + self.save_tag,
            pos_lens=[self.coords_galactic.l.value, self.coords_galactic.b.value],
            vel_lens=[self.coords_galactic.pm_l_cosb.value, self.coords_galactic.pm_b.value],
            mu_qsrs=np.transpose(self.mu_qsrs*1e3), # in mas/yr
            )

    def mu(self, beta_vec, v_ang_vec, c200_lens, M200_lens, d_lens):
        """ Get lens-induced velocity

            :param beta_vec: angular impact parameter (pointing lens to source) vector in rad
            :param v_ang_vec: lens angular velocity in natural units
            :param c200_lens: lens concentration
            :param M200_lens: lens mass in natural units
            :param d_lens: distance to lens in natural units
            :return: lens-induced velocity in as/yr
        """ 

        b_vec = d_lens*np.array(beta_vec) # Convert angular to physical impact parameter
        v_vec = d_lens*np.array(v_ang_vec) # Convert angular to physical velocity
        b = np.linalg.norm(b_vec) # Impact parameter
        M, dMdb = self.MdMdb(b, c200_lens, M200_lens)
        b_unit_vec = b_vec/b # Convert angular to physical impact parameter
        b_dot_v = np.dot(b_unit_vec, v_vec)
        factor = (dMdb/b*b_unit_vec*b_dot_v 
                    + M/b**2*(v_vec - 2*b_unit_vec*b_dot_v))

        return -factor*4*GN/(asctorad/Year) # Concert to as/yr

    def dFdx(self, x):
        """ Helper function for NFW deflection, from astro-ph/0102341 eq. (49)
        """
        return (1-x**2*self.F(x))/(x*(x**2-1))
        
    def F(self, x):
        """ Helper function for NFW deflection, from astro-ph/0102341 eq. (48)
        """
        if x > 1:
            return np.arctan(np.sqrt(x**2-1))/(np.sqrt(x**2 - 1))
        elif x == 1:
            return 1
        elif x < 1:
            return np.arctanh(np.sqrt(1-x**2))/(np.sqrt(1-x**2))

    def MdMdb(self, b, c, M):
        """ NFW mass and derivative within a cylinder or radius `b`

            :param b: Cylinder radius, in natural units
            :param c: NFW concentration
            :param M: NFW mass
            :return: mass within `b`, derivative of mass within `b` at `b`
        """
        delta_c = (200/3.)*c**3/(np.log(1+c) - c/(1+c)) 
        rho_s = rho_c*delta_c 
        r_s = (M/((4/3.)*np.pi*c**3*200*rho_c))**(1/3.) # NFW scale radius
        x = b/r_s 
        M = 4*np.pi*rho_s*r_s**3*(np.log(x/2.) + self.F(x))
        dMdb = 4*np.pi*rho_s*r_s**2*((1/x) + self.dFdx(x))
        
        return M, dMdb