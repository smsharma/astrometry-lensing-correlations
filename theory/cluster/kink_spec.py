import sys
sys.path.append("../")
sys.path.append("../../")
import argparse

import numpy as np
from tqdm import *
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.optimize import fsolve

from theory.spec_calc import PowerSpectra, PowerSpectraPopulations
from theory.kink import MassFunctionKink
from theory.units import *

# Command line arguments

parser = argparse.ArgumentParser()
parser.add_argument("--nB", action="store", dest="nB", default=1., type=float)
parser.add_argument("--kB", action="store", dest="kB", default=1, type=float)

results=parser.parse_args()

nB = results.nB
kB = results.kB

gen_file = '/Users/smsharma/PycharmProjects/Lensing-PowerSpectra/theory/arrays/pk/generate_Pk_kink.py'
save_dir  = '/scratch/sm8383/QuasarSim'
# Get class instance with custom primordial power spectrum

mfk = MassFunctionKink(gen_file=gen_file)

CLASS_inst_vanilla = mfk.get_CLASS_kink(k_B=kB, n_B=0.9665)
CLASS_inst = mfk.get_CLASS_kink(k_B=kB, n_B=nB)

# Get power spectrum

k_ary = np.logspace(-5, np.log10(400))
pk_ary = np.array([CLASS_inst.pk_lin(k,0) for k in k_ary])

# Get mass function

M_ary = np.logspace(4, 12) * M_s
dndM_ary = np.array([mfk.dn_dM_s(M, CLASS_inst) for M in M_ary])
dndM_interp = interp1d(np.log10(M_ary), np.log10(dndM_ary))

def dndM(M):
    return 10 ** dndM_interp(np.log10(M))

# Calibration

pspec = PowerSpectra(precompute=['NFW', 'Burk'])

# Number calibration ("fudging")

N_calib = 150.
pref = N_calib / quad(lambda M: mfk.dn_dM_s(M, CLASS_inst_vanilla), 1e8 * M_s, 1e10 * M_s, epsabs=0, epsrel=1e-4)[0]
N_calib_new = pref * quad(lambda M: mfk.dn_dM_s(M, CLASS_inst), 1e8 * M_s, 1e10 * M_s, epsabs=0, epsrel=1e-2)[0]

M_sc_calib = 1e11 * M_s
pspecpop = PowerSpectraPopulations(l_max=2000, CLASS_inst=CLASS_inst, fudge_factor_rho_s=1.)
rho_s_new = pspecpop.get_rs_rhos_NFW(M_sc_calib)[1]

M200 = 10 ** fsolve(lambda M200: np.log10(M_sc_calib / M_s) - np.log10(pspecpop.get_M_sc(10 ** M200 * M_s) / M_s), 9.)[0] * M_s
pspecpop = PowerSpectraPopulations(l_max=2000)
rho_s_old = pspecpop.get_rs_rhos_NFW(M200)[1]

fudge_factor_rho_s = rho_s_old / rho_s_new

# Get power spectrum

pspecpop = PowerSpectraPopulations(l_max=2000, CLASS_inst=CLASS_inst, fudge_factor_rho_s=fudge_factor_rho_s)

pspecpop.set_radial_distribution(pspecpop.r2rho_V_NFW, R_min=1e-2*kpc, R_max=260*kpc)
pspecpop.set_mass_distribution(dndM, M_min=1e4*M_s, M_max=0.01*1.1e12*M_s,
                               M_min_calib=1e8*M_s, M_max_calib=1e10*M_s, N_calib=N_calib_new)
pspecpop.set_subhalo_properties(pspecpop.c200_Moline)

C_l_mu_new = pspecpop.get_C_l_total_ary()

np.savez(save_dir + '/' + str(kB) + '_' + str(nB) + ".npz",
         k_ary=k_ary,
         pk_ary=pk_ary,
         M_ary=M_ary,
         dndM_ary=dndM_ary,
         C_l_mu_new=C_l_mu_new
         )
