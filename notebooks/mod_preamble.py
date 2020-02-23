import numpy as np
import sys, os
import multiprocessing
from joblib import Parallel, delayed
from time import sleep
import matplotlib.pyplot as plt
from matplotlib import colors, ticker
import matplotlib
import matplotlib.gridspec as gridspec
import scipy as sp
from scipy.integrate import quad, nquad, odeint
from scipy.interpolate import interp1d, interp2d
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import brentq
from astropy import units as u
from astropy.coordinates import SkyCoord
#from PyAstronomy import pyaC
from tqdm import *
import pandas as pd
from scipy.stats import chi2
from scipy.special import sph_harm, gamma
from scipy.special import lpmv
from scipy.special import sici
from mpmath import spherharm
from mpmath import legenp
from mpmath import odefun
import mpmath as mp
import mpmath as mp
import healpy as hp
from healpy.sphtfunc import map2alm
from scipy.constants import golden

a_eq = 1/3250;
GeV = 1.0; eV = 10**-9 * GeV;
MeV = 10**6 * eV; keV = 10**3 * eV; meV = 10**-3 * eV; mueV = 10**-6 * eV; 
kg = 5.6096 * 10**26 * GeV; gram = 10**-3 * kg;
meter = 1/(0.1973 * 10**-15) * GeV**(-1); centimeter = 10**-2 * meter; kilometer = 10**3 * meter; mm = 10**-3 * meter; fm = 10**-15 * meter; nm = 10**-9 * meter; angstrom = 10**-10 * meter; 
second = 299792458. * meter; Hz = second**-1;
hour = 3600 * second; day = 24 * hour; year = 365 * day; 
kelvin = 8.6 * 10**-5 * eV;
joule = kg * meter**2 * second**-2; erg = 10**-7 * joule; watt = joule * second**-1;  newton = kg * meter * second**-2;

Hubble_eq = 2.68*10**(-28) * eV;
Hubble_0 = 1.4473*10**(-33) * eV; #in eV
mPlanck = 2.43534*10**18 * GeV; # in GeV
GN = 1/ (mPlanck**2 * 8 * np.pi); 
Rho_DM0 = 0.26395 * 3 * Hubble_0**2 * mPlanck**2;
Rho_M0 = 0.3 * 3 * Hubble_0**2 * mPlanck**2;
M_Solar = 1.1107*10**(57) * GeV; # in GeV
TCMB = 2.73 * kelvin; 
pc = 1.56261*10**(32) * GeV**-1;
kpc = 10**3 * pc; Mpc = 10**6 * pc; Gpc = 10**9 * pc;
degree = np.pi / 180; arcmin = degree / 60; arcsec = arcmin / 60; mas = 10**-3 * arcsec; muas = 10**-6 * arcsec;
masy = mas / year; muasy = muas / year; masyy = mas / year**2; muasyy = muas / year**2; 