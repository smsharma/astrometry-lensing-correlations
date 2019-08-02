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
from theory.kink import MassFunctionKink, Sigma
from theory.units import *

sys.path.append('/group/hepheno/smsharma/heptools/colossus/')

from colossus.cosmology import cosmology
from colossus.lss import mass_function

# Command line arguments

parser = argparse.ArgumentParser()
parser.add_argument("--nB", action="store", dest="nB", default=1., type=float)
parser.add_argument("--kB", action="store", dest="kB", default=1, type=float)

results=parser.parse_args()

nB = results.nB
kB = results.kB

pk_dir = '/group/hepheno/smsharma/Lensing-PowerSpectra/theory/arrays/pk/'
save_dir  = '/group/hepheno/smsharma/Lensing-PowerSpectra/theory/cluster/cluster_out/'

# Get class instance with custom primordial power spectrum

mfk = MassFunctionKink(gen_file=pk_dir + 'generate_Pk_kink.py')

CLASS_inst = mfk.get_CLASS_kink(k_B=kB, n_B=nB, k_max=1e2)
CLASS_inst_vanilla = mfk.get_CLASS_kink(k_B=kB, n_B=0.9665, k_max=1e2)

for idnx, inst in enumerate([CLASS_inst_vanilla, CLASS_inst]):
    k_ary = np.logspace(-6, np.log10(1e2), 10000)
    Pk_ary = np.array([inst.pk_lin(k, 0) for k in k_ary])

    log10_k_interp_ary = np.linspace(-6, 7, 10000)
    log10_P_interp = interp1d(np.log10(k_ary * h), np.log10(Pk_ary / h ** 3), bounds_error=False,
                              fill_value='extrapolate')
    log10_P_interp_ary = (log10_P_interp)(log10_k_interp_ary)

    if idnx == 1:
        file_kinked = pk_dir + 'pk' + str(kB) + '_' + str(nB) + '.dat'
        np.savetxt(file_kinked,
           np.transpose([log10_k_interp_ary, log10_P_interp_ary]),
           delimiter='\t')
    else:
        file_base = pk_dir + 'pk' + str(kB) + '_' + str(nB) + '_base.dat'
        np.savetxt(file_base,
           np.transpose([log10_k_interp_ary, log10_P_interp_ary]),
           delimiter='\t')

# Get mass function from colossus

cosmo = cosmology.setCosmology('planck18')

M_ary = np.logspace(3.5, 12, 50)

dndlnM_vanilla_ary = mass_function.massFunction(M_ary, 0.0, mdef = '200m', model = 'tinker08', q_in='M', q_out = 'dndlnM', ps_args={'model': mfk.randomword(5), 'path':file_base})
dndlnM_ary = mass_function.massFunction(M_ary, 0.0, mdef = '200m', model = 'tinker08', q_in='M', q_out = 'dndlnM', ps_args={'model': mfk.randomword(5), 'path':file_kinked})

dndlnM_vanilla_interp = interp1d(np.log10(M_ary * M_s), np.log10(dndlnM_vanilla_ary / M_ary))
dndlnM_interp = interp1d(np.log10(M_ary * M_s), np.log10(dndlnM_ary / M_ary))

# Calibrate to number density at high masses
N_calib = 150.

pref = N_calib / quad(lambda M: 10 ** dndlnM_vanilla_interp(np.log10(M)), 1e8 * M_s, 1e10 * M_s, epsabs=0, epsrel=1e-4)[0]
N_calib_new = pref * quad(lambda M: 10 ** dndlnM_interp(np.log10(M)), 1e8 * M_s, 1e10 * M_s, epsabs=0, epsrel=1e-4)[0]

# Get c200

sig = Sigma(log10_P_interp)

M_calib = 1e12 * M_s

M_ary_conc = np.logspace(7, 13, 10) * M_s
c200_ary = [sig.c200_zcoll(M, C=100., f=0.02)[0] for M in tqdm(M_ary_conc)]

c200_interp = interp1d(np.log10(M_ary_conc), np.log10(c200_ary), bounds_error=False,
                              fill_value='extrapolate')

def dndM(M):
    return 10 ** dndlnM_interp(np.log10(M))

def c200_custom(M):
    return 10 ** c200_interp(np.log10(M))

pspecpop = PowerSpectraPopulations(l_max=10000)

pspecpop.set_radial_distribution(pspecpop.r2rho_V_NFW, R_min=1e-2*kpc, R_max=260*kpc)
pspecpop.set_mass_distribution(dndM, M_min=1e4*M_s, M_max=0.01*1.1e12*M_s,
                               M_min_calib=1e8*M_s, M_max_calib=1e10*M_s, N_calib=N_calib_new)
pspecpop.set_subhalo_properties(c200_custom)

C_l_mu_new = pspecpop.get_C_l_total_ary(l_los_min=pspecpop.l_cutoff)
C_l_alpha_new = pspecpop.get_C_l_total_ary(l_los_min=pspecpop.l_cutoff, accel=True)

np.savez(save_dir + '/calib2_' + str(kB) + '_' + str(nB) + ".npz",
         C_l_mu_new=C_l_mu_new,
         C_l_alpha_new=C_l_alpha_new
         )
