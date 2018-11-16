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
parser.add_argument("--mfrac",
                  action="store", dest="mfrac", default=0, type=float)
parser.add_argument("--imc",
                  action="store", dest="imc", default=0, type=float)


results=parser.parse_args()
log10M=results.log10M
mfrac=results.mfrac
imc=results.imc

save_tag = 'log10M_' + str(log10M) + '_mfrac_' + str(np.round(mfrac,2)) + '_mc_' + str(int(imc))

sim = QuasarSim(sh_m_frac=1, m_delta=10**log10M*M_s, c200_delta=100000., 
	verbose=True, save=True, save_dir='/scratch/sm8383/QuasarSims/' save_tag=save_tag, sh_distrib='MW')