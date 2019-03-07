from units import *
import matplotlib.pyplot as plt
from astrometry_sim import QuasarSim
import healpy as hp
from pylab import cm as cmaps
from estimator_wholesky import get_vector_alm
from tqdm import *
from astropy import units as u

max_sep = 10
nside = 128
lmax = 3*nside - 1

sim = QuasarSim(max_sep=max_sep, 
                verbose=True, 
                sim_uniform=True, nside=nside, calc_powerspecs=True, 
                do_alpha=True)

sim.set_mass_distribution(sim.rho_M_SI, f_DM=0.5, M_min=1e7*M_s, M_max=1e10*M_s, alpha=-1.9)
sim.set_radial_distribution(sim.r2rho_V_ein_EAQ, R_min=1*kpc, R_max=260*kpc)
sim.set_subhalo_properties(sim.c200_SCP)

sim.analysis_pipeline()