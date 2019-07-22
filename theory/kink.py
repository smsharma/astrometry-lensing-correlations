import numpy as np
import matplotlib.pyplot as plt
from tqdm import *
from astropy.cosmology import Planck15
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.misc import derivative
from classy import Class

from theory.spec_calc import PowerSpectra, PowerSpectraPopulations
from theory.astrometry_forecast import Parameter, AstrometryObservation, FisherForecast
from theory.units import *

class MassFunctionKink:
    def __init__(self, A_s=2.105 / 1e9, n_s=0.9665, gen_file='/Users/smsharma/PycharmProjects/Lensing-PowerSpectra/theory/arrays/pk/generate_Pk_kink.py'):
        self.A_s = A_s
        self.n_s = n_s
        self.gen_file = gen_file

    def get_CLASS_kink(self, k_B=0.1, n_B=0.9665):
        common_settings = {  # Background parameters
            'H0': 67.32117,
            'omega_b': 0.02238280,
            'N_ur': 2.03066666667,
            'omega_cdm': 0.1201075,
            'N_ncdm': 1,
            'omega_ncdm': 0.0006451439,
            'YHe': 0.2454006,
            'tau_reio': 0.05430842,
            'modes': 's',

            # Output settings
            'output': 'mPk',
            'P_k_max_1/Mpc': 400.0,
            'P_k_ini type': 'external_Pk',
            'command': "python " + str(self.gen_file),
            'custom1': 0.05,
            'custom2': self.A_s,
            'custom3': self.n_s,
            'custom4': k_B,
            'custom5': n_B
        }

        CLASS_inst = Class()
        CLASS_inst.set(common_settings)
        CLASS_inst.compute()

        return CLASS_inst

    def dn_dM_s(self, M_sc, CLASS_inst):
        R = (M_sc / (4 / 3. * np.pi * rho_m)) ** (1 / 3.)
        sigma = CLASS_inst.sigma(R / Mpc, 0)
        sigma_log_deriv = np.abs(
            M_sc * derivative(lambda M: np.log(CLASS_inst.sigma((M / (4 / 3. * np.pi * rho_m)) ** (1 / 3.) / Mpc, 0)),
                              x0=M_sc, dx=(0.9) * M_sc))
        return np.sqrt(2 / np.pi) * rho_m / M_sc ** 2 * delta_c / sigma * sigma_log_deriv * np.exp(
            -delta_c ** 2 / (2 * sigma ** 2))