import sys, os

import numpy as np
from scipy.special import sph_harm
import healpy as hp
from tqdm import *
from mpmath import spherharm
from itertools import groupby
from operator import itemgetter
# from libsharp import normalized_associated_legendre_table
from healpy.sphtfunc import map2alm
from healpy.sphtfunc import Alm

def Ylm_map(lmax, nside):
    npix = hp.nside2npix(nside)
    coords = hp.pix2ang(nside, range(npix))
    seq = list(np.transpose(coords))
    seq.sort(key = itemgetter(0))
    groups = groupby(seq, itemgetter(0))
    idxs = np.cumsum([len(list(data)) for (key, data) in groups])

    tmp_map = np.zeros((npix,lmax + 1, lmax + 1),dtype=np.complex)

    for im, m in enumerate(range(lmax + 1)):
        unique_coords = np.unique(coords[0])
        Ylm_uniq = np.zeros((len(unique_coords), lmax + 1))
        Ylm_uniq[:,im:] = normalized_associated_legendre_table(lmax,m,unique_coords)
        idx_min = 0
        for i, idx in enumerate(range(len(idxs))):
            idx_max = idxs[i]
            tmp_map[idx_min:idx_max,:,im] = np.ones((idx_max-idx_min, lmax + 1))*Ylm_uniq[i]
            idx_min = idx_max
        for il in range(im, lmax + 1):
            tmp_map[:,il,im] *= np.exp(1.j*m*coords[1])
    return tmp_map

def alm_getidx(lmax,l,m):
    if m >= 0:
        return Alm.getidx(lmax,l,m)
    else:
        return round(0.5*lmax**2 + 1.5*lmax + 1) - (lmax+1) + Alm.getidx(lmax,l,abs(m))
    
def alm_getlm(lmax,idx):
    if idx < round(0.5*lmax**2 + 1.5*lmax + 1):
        return Alm.getlm(lmax,idx)
    else:
        lm_pos =  Alm.getlm(lmax, idx - round(0.5*lmax**2 + 1.5*lmax + 1) + (lmax+1))
        return (lm_pos[0], -lm_pos[1])

def complexmap2alm(map):
    alm_real = map2alm(np.real(map))
    alm_imag = map2alm(np.imag(map))
    alm_total = alm_real + alm_imag*1.j
    for idx_lower in range(lmax+1, len(alm_real)):
        alm_total = np.concatenate(( alm_total, [(-1)**Alm.getlm(lmax,idx_lower)[1] * (alm_real[idx_lower]-alm_imag[idx_lower]*1.j)] ))    
    return alm_total

def mat_Fisher(weights_map):
    mat_amp = mat_Fisher_amp(weights_map)
    return mat_amp * np.conjugate(mat_amp)

def mat_Fisher_amp(weights_map):
    mat_f = np.zeros(((lmax+1)**2,(lmax+1)**2), dtype=np.complex64)
    Yharm_map = Ylm_map(lmax,nside)
    for idx in tqdm(range((lmax+1)**2)):
        l = alm_getlm(lmax,idx)[0].item()
        m = alm_getlm(lmax,idx)[1].item()
        mat_f[idx] += complexmap2alm(Yharm_map[:,l,m]*weights_map)

    return mat_f

# nside = 32
# npix = hp.nside2npix(nside)
# lmax = 3*nside - 1
# Ylm_map(lmax,nside)
# print(Ylm_map(lmax,nside)[546,22,15])
# mat_Fisher(np.ones(npix))