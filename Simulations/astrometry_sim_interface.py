import sys, os
import argparse

from tqdm import *
import numpy as np

from astrometry_sim import QuasarSim
from units import *

### Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--log10M",
                  action="store", dest="log10M", default=0, type=float)
parser.add_argument("--lcdm",
                  action="store", dest="lcdm", default=1, type=float)
parser.add_argument("--mfrac",
                  action="store", dest="mfrac", default=0, type=float)
parser.add_argument("--imc",
                  action="store", dest="imc", default=0, type=float)


results=parser.parse_args()
log10M=results.log10M
mfrac=results.mfrac
imc=results.imc
lcdm=results.lcdm

nside = 128
max_sep = 3. 


if not lcdm:
	save_tag = 'uniform_log10M_' + str(log10M) + '_mfrac_' + str(np.round(mfrac,2)) + '_mc_' + str(int(imc))
	sim = QuasarSim(sh_m_frac=mfrac, max_sep=max_sep, m_delta=10**log10M*M_s, c200_delta=100000., 
		verbose=True, save=True, save_dir='/scratch/sm8383/QuasarSims/', save_tag=save_tag, sh_distrib='MW', sim_uniform=True, nside=nside)

else:
	save_tag = 'uniform_lcdm_mmin_1e6_mc_' + str(int(imc))
	sim = QuasarSim(alpha_m=1.9, n_calib=150, max_sep=max_sep, verbose=True, sh_distrib='Aq1', m_min=1e-6, sim_uniform=True, nside=nside, save=True, save_dir='/scratch/sm8383/QuasarSims/', save_tag=save_tag)