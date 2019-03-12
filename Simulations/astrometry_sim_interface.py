import sys, os
import argparse

from units import *
import matplotlib.pyplot as plt
from astrometry_sim import QuasarSim
import healpy as hp
from pylab import cm as cmaps
from estimator_wholesky import get_vector_alm
from tqdm import *
from astropy import units as u

### Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--imc",
                  action="store", dest="imc", default=0, type=float)

results=parser.parse_args()
imc=results.imc

max_sep = 20
nside = 128
lmax = 3*nside - 1

sim = QuasarSim(max_sep=max_sep, 
                verbose=True, 
                sim_uniform=True, nside=nside, calc_powerspecs=True, 
                do_alpha=True,
                save_tag='lcdm_' + str(imc), save=True, save_dir='/scratch/sm8383/QuasarSims/')

sim.set_mass_distribution(sim.rho_M_SI, M_min=10**6.5*M_s, M_max=0.04*1.1e12*M_s, M_min_calib=1e8*M_s, M_max_calib=1e10*M_s, N_calib=150, alpha=-1.9)

sim.set_radial_distribution(sim.r2rho_V_ein_EAQ, R_min=1e-3*kpc, R_max=260*kpc)
sim.set_subhalo_properties(sim.c200_SCP, distdep=False)

sim.analysis_pipeline()