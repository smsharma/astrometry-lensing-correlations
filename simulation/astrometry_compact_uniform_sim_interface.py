import sys
sys.path.append("../")
sys.path.append("../../")
import argparse

from theory.units import *
from simulation.astrometry_sim import QuasarSim

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--imc", action="store", dest="imc", default=0, type=int)

results=parser.parse_args()

max_sep = 20
nside = 128
lmax = 3*nside - 1
imc=results.imc

sim = QuasarSim(max_sep=max_sep,
                verbose=False,
                sim_uniform=True,
                nside=nside,
                calc_powerspecs=True,
                do_alpha=True,
                save=True,
                save_dir='/mnt/hepheno/smsharma/QuasarSim/',  # '/scratch/sm8383/QuasarSim',
                save_tag='gaussian_uniform_galNFW_f0p1_M1e8_R1e1_nside128_sep20_mc' + str(imc),
                sh_profile='Gaussian',
                f_sub=0.1,
                R0=10 * pc)

sim.set_mass_distribution(sim.rho_M_SI, M_min=1e8*M_s, M_max=1e10*M_s, M_min_calib=1e8*M_s, M_max_calib=1e10*M_s, N_calib=150, alpha=-1.9)
sim.set_radial_distribution(sim.r2rho_V_NFW, R_min=1e-3*kpc, R_max=260*kpc)
sim.set_subhalo_properties(sim.c200_SCP, distdep=False)

sim.analysis_pipeline()