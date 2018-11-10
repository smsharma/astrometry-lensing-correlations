"""
Functions to perform Helmholtz decomposition for vector fields on a sphere

Resources
---------
Bessel functions in SciPy
    https://www.johndcook.com/blog/bessel_python/
Spherical Harmonic
    http://functions.wolfram.com/HypergeometricFunctions/SphericalHarmonicYGeneral/


Convention
----------
cartesian coordinates
    :(x, y, z):
spherical coordinates
    :(r, theta, phi): where
        :r:     radial coordinate; must be in (0, oo);
        :theta: polar coordinate; must be in [0, pi];
        :phi:   azimuthal coordinate; must be in [0, 2*pi];
"""


import sys, os

import numpy as np
from scipy.special import sph_harm
import healpy as hp
from tqdm import *
from mpmath import spherharm

def Ylm(l, m, theta, phi): 
    """
    Redefine spherical harmonics from scipy.special
    to match physics convention.
    
    Parameters
    ----------
    l : int, array_like
        Degree of the harmonic (int); ``l >= 0``.
    m : int, array_like
        Order of the harmonic (int); ``|m| <= l``.
    theta : array_like
        Polar (colatitudinal) coordinate; must be in ``[0, pi]``.
    phi : array_like
        Azimuthal (longitudinal) coordinate; must be in ``[0, 2*pi]``.
    
    Returns
    -------
    Ylm : complex float
       The harmonic Ylm sampled at ``theta`` and ``phi``.
    """
    if np.abs(m) > l:
        Ylm = 0
    else:
        # Ylm = sph_harm(m, l, phi, theta) # Perform redefinition
        Ylm = spherharm(l, m, theta, phi) # Perform redefinition
    return Ylm

def del_Ylm(l, m, r, theta, phi):
    dYdtheta = (m*(1/np.tan(theta))*Ylm(l, m, theta, phi) 
                + np.sqrt((l-m)*(l+m+1))*np.exp(-1.j*theta)*Ylm(l, m+1, theta, phi))
    dYdphi = 1.j*m*Ylm(l, m, theta, phi)

    return np.array([0, 
            (1/r)*dYdtheta,
            (1/(r*np.sin(theta)))*dYdphi
            ])

def Ylm_vec(l, m, r, theta, phi): 
    """ theta polar, phi azimuthal
    """
    return np.array([r, 0, 0])*Ylm(l, m, theta, phi)

def Psilm_vec(l, m, r, theta, phi):
    """ theta polar, phi azimuthal
    """
    return r*del_Ylm(l, m, r, theta, phi)

def Philm_vec(l, m, r, theta, phi):
    """ theta polar, phi azimuthal
    """
#     return np.cross([r, theta, phi], del_Ylm(l, m, r, theta, phi))
    return np.cross([1, 0, 0], Psilm_vec(l, m, r, theta, phi))

def map2alm(the_map, mask, l, m,comp=0):
    if comp==0:
        pref = 1
        comp = Ylm_vec
    elif comp==1:
        pref = (1/(l*(l+1)))
        comp = Psilm_vec
    elif comp==2:
        pref = (1/(l*(l+1)))
        comp = Philm_vec
    else:
        print("Component should be set to 0, 1 or 2!")
    nside = hp.npix2nside(len(the_map))
    alm = 0
    domega = hp.nside2pixarea(nside)
    good_pix = np.where(mask==0)[0]
    for ipix in (good_pix):
        theta, phi = hp.pix2ang(nside, ipix)
        alm += np.dot(the_map[ipix], np.conjugate(comp(l, m, 1, theta, phi)))*domega
    return pref*alm

def alm2Cell(alm, l):
    return(1/(2*l+1))*np.sum([np.absolute(alm[i])**2 for i in range(len(alm))])

# def precompute_comps(lmax, mask=None, comp=0):
#     if comp==0:
#         pref = 1
#         comp = Ylm_vec
#     elif comp==1:
#         pref = (1/(l*(l+1)))
#         comp = Psilm_vec
#     elif comp==2:
#         pref = (1/(l*(l+1)))
#         comp = Philm_vec
#     else:
#         print("Component should be set to 0, 1 or 2!") 
#     good_pix = np.where(mask==0)[0]
#     comps = np.zeros((lmax+1, lmax+1, len(good_pix), 3), dtype=np.complex_)
#     nside = hp.npix2nside(len(mask))
#     for il in tqdm_notebook(range(lmax + 1)):
#         for im in range(il + 1):
#             for ipix in (range(len(good_pix))):
#                 theta, phi = hp.pix2ang(nside, ipix)
#                 comps[il][im][ipix] = comp(il, im, 1, theta, phi)

def map2Cell(the_map, l, mask=None, comp=0):
    if mask is None:
        mask = np.zeros_like(np.transpose(the_map)[0])

    alms = [map2alm(the_map, mask, l, m, comp) for m in tqdm_notebook(range(-l, l+1))]
    Cell = alm2Cell(alms, l)
    return Cell