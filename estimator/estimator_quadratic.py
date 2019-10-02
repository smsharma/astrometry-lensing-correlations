import numpy as np
import sys, os
import matplotlib.pyplot as plt
from scipy.integrate import quad, nquad
from scipy.interpolate import interp1d
from astropy import units as u
from astropy.coordinates import SkyCoord
from tqdm import *
import pandas as pd
from scipy.stats import chi2
from scipy.special import sph_harm 
from scipy.special import lpmv
import healpy as hp
from healpy.sphtfunc import map2alm
from healpy.sphtfunc import Alm

class QuadraticEstimator():
    def __init__(self, nside=16):
        self.n_pix = hp.nside2npix(nside)
        self.l_max = 3 * nside - 1
        self.pix_coords = np.asarray(hp.pixelfunc.pix2ang(nside,np.arange(self.n_pix))).transpose()

    def _Y_harm(self, l, m, theta, phi):
        return sph_harm(m, l, phi, theta)

    def _Y_harm_map(self, l, m):
        """spherical harmonic map over healpix with nside defined"""
        tmp_map = np.zeros(self.n_pix, dtype=np.complex)
        if l >= 0:  # To avoid getting nan outputs when there are l = -1 calls
            for i in np.arange(self.n_pix):
                tmp_map[i] += self._Y_harm(l, m, self.pix_coords[i,0], self.pix_coords[i,1])
        return tmp_map

    def _c_coeffs(self, c_idx,l,m):
        if c_idx == 1:
            return np.sqrt((l + 1) * (l - m) * (l + m) / ( l * (2 * l - 1) * (2 * l + 1) + 1e-10) )
        elif c_idx == 2:
            return -np.sqrt(l * (l - m + 1) * (l + m + 1) / ((l + 1) * (2 * l + 3) * (2 * l + 1) + 1e-10))
        elif c_idx == 3:
            return 1.j * m / np.sqrt(l * (l + 1) + 1e-10)
        
    def _sin_map(self):
        return np.sin(self.pix_coords[:, 0])

    def _sin_inv_map(self):
        tmp_map = np.zeros(self.n_pix, dtype=np.complex)
        for i in np.arange(self.n_pix):
            sin_val = np.sin(self.pix_coords[i,0])
            if np.abs(sin_val) > 0:
                tmp_map[i]= 1 / sin_val
            else:
                tmp_map[i]= 1e40    
        return tmp_map

    def _A_harm_map(self, l,m):
        tmp_map = np.zeros(self.n_pix,dtype=np.complex)
        if l > 0:  # To avoid getting nan outputs when there are l = -1 calls
            if l > np.abs(m):
                for i in np.arange(self.n_pix):
                    tmp_map[i] += self._c_coeffs(1, l, m) * self._Y_harm(l - 1, m, self.pix_coords[i,0], self.pix_coords[i, 1]) + self._c_coeffs(2, l, m) * self._Y_harm(l + 1, m, self.pix_coords[i,0], self.pix_coords[i,1])
            elif l == abs(m):  # To avoid getting nan outputs when l-1 < abs(m)
                for i in np.arange(self.n_pix):
                    tmp_map[i] += self._c_coeffs(2, l, m) * self._Y_harm(l + 1, m, self.pix_coords[i, 0], self.pix_coords[i, 1])
        elif l==0:
            for i in np.arange(self.n_pix):
                tmp_map[i] += self._c_coeffs(2, l, m) * self._Y_harm(l + 1, m, self.pix_coords[i,0], self.pix_coords[i,1])
        return tmp_map

    def _B_harm_map(self, l,m):
        tmp_map = np.zeros(self.n_pix,dtype=np.complex)
        if l > 0:  # To avoid dividing by zero
            for i in np.arange(self.n_pix):
                tmp_map[i] += self._c_coeffs(3, l, m) * self._Y_harm(l, m, self.pix_coords[i,0], self.pix_coords[i,1])
        return tmp_map

    def _Psi_harm_map(self, l, m):
        """Psi harmonic map over healpix with nside defined"""
        theta_comp_map = -self._sin_inv_map() * self._A_harm_map(l, m)
        phi_comp_map = -self._sin_inv_map() * self._B_harm_map(l, m)
        return np.transpose([theta_comp_map, phi_comp_map])

    def _alm_get_idx(self, l_max, l, m):
        if m >= 0:
            return Alm.getidx(l_max, l, m)
        else:
            return round(0.5 * l_max ** 2 + 1.5 * l_max + 1) - (l_max + 1) + Alm.getidx(l_max, l, abs(m))
        
    def _alm_get_lm(self, l_max, idx):
        if idx < round(0.5 * l_max ** 2 + 1.5 * l_max + 1):
            return Alm.getlm(l_max, idx)
        else:
            lm_pos =  Alm.getlm(l_max, idx - round(0.5 * l_max ** 2 + 1.5 * l_max + 1) + (l_max + 1))
            return (lm_pos[0], -lm_pos[1])
        
    def _complexmap2alm(self, map):
        alm_real = map2alm(np.real(map))
        alm_imag = map2alm(np.imag(map))
        alm_total = alm_real + alm_imag*1j
        
        for idx_lower in range(self.l_max + 1, alm_real.size):
            alm_total = np.concatenate(( alm_total, [(-1)**Alm.getlm(self.l_max, idx_lower)[1] * (alm_real[idx_lower] - alm_imag[idx_lower]*1j)] ))    
            
        return alm_total

    def _mat_Fisher1(self, weights_map):
        mat_P_tilde = self._mat_P_tilde(weights_map)
        return 0.5 * np.real(mat_P_tilde * np.transpose(mat_P_tilde))

    def _mat_Fisher1_inv(self, mat_fisher1):
        matfish_block = mat_fisher1[1:, 1:]
        matfish_block_inv = np.real(np.linalg.inv(matfish_block))
        return np.pad(matfish_block_inv, ((1, 0),(1, 0)), 'constant')

    def _mat_Psi_tilde(self, weights_map):
        prefactor_map = -self._sin_inv_map() * weights_map
        mat_Psitilde_1 = np.zeros(((self.l_max + 1) ** 2, (self.l_max + 1) ** 2),dtype='complex')
        mat_Psitilde_2 = np.zeros(((self.l_max + 1) ** 2, (self.l_max + 1) ** 2),dtype='complex')
        
        for idx in tqdm_notebook(range((self.l_max + 1)**2)):
            l, m = self._alm_get_lm(self.l_max, idx)[0].item(), self._alm_get_lm(self.l_max, idx)[1].item()
            psiharm_map1, psiharm_map2 = self._Psi_harm_map(l, m)[:, 0], self._Psi_harm_map(l, m)[:, 1]
            mat_Psitilde_1[idx] = self._complexmap2alm(prefactor_map * psiharm_map1)
            mat_Psitilde_2[idx] = self._complexmap2alm(prefactor_map * psiharm_map2)
            
        return mat_Psitilde_1, mat_Psitilde_2

    def _mat_P_tilde(self, weights_map):
        
        mat_Psitilde_1, mat_Psitilde_2 = self._mat_Psi_tilde(weights_map)
        c1vec = np.zeros(self.l_max ** 2, dtype='complex')  # Create coefficient vectors
        c2vec = np.zeros(self.l_max ** 2, dtype='complex')
        c3vec = np.zeros(self.l_max ** 2, dtype='complex')
        
        for idx in range(self.l_max ** 2):  # Only go up to lmax-1 to avoid getting out of range on the l+1 recursion
            l, m = self._alm_get_lm(self.l_max - 1, idx)[0].item(), self._alm_get_lm(self.l_max - 1, idx)[1].item() 
            c1vec[idx] = self._c_coeffs(1, l, m) * np.min([l,1])  # Multiplier to handle l==0 exception
            c2vec[idx], c3vec[idx] =  self._c_coeffs(2, l, m), self._c_coeffs(3, l, np.abs(m))  # Why abs(m)?
            
        vec_idx_zero = np.zeros(self.l_max ** 2, dtype='int')  # Create permutation vectors to select parts of Psitilde matrices
        vec_idx_minus = np.zeros(self.l_max ** 2, dtype='int')
        vec_idx_plus = np.zeros(self.l_max ** 2, dtype='int')
        
        for idx in range(self.l_max ** 2):
            l, m = self._alm_get_lm(self.l_max - 1, idx)[0].item(), self._alm_get_lm(self.l_max - 1, idx)[1].item()
            vec_idx_zero[idx] = self._alm_get_idx(self.l_max, l, m)
            vec_idx_minus[idx] = self._alm_get_idx(self.l_max, l - 1, m)
            vec_idx_plus[idx] = self._alm_get_idx(self.l_max, l + 1, m)
            
        mat_Psitilde_2_zero = -np.transpose(c3vec * np.transpose(mat_Psitilde_2[vec_idx_zero][:, vec_idx_zero]))
        mat_Psitilde_1_minus = np.transpose(c1vec * np.transpose(mat_Psitilde_1[vec_idx_zero][:, vec_idx_minus]))
        mat_Psitilde_1_plus = np.transpose(c2vec * np.transpose(mat_Psitilde_1[vec_idx_zero][:, vec_idx_plus]))
        mat_Ptilde = mat_Psitilde_1_minus + mat_Psitilde_1_plus + mat_Psitilde_2_zero
        return mat_Ptilde

    def _mat_num1_sig(self, data_map_b, data_map_l, weights_map):
        dtilde_map_b = self._sin_inv_map() * weights_map * data_map_b
        dtilde_map_l = self._sin_inv_map() * weights_map * data_map_l
        dtilde_b_lm = self._complexmap2alm(dtilde_map_b)
        dtilde_l_lm = self._complexmap2alm(dtilde_map_l)
        c1vec = np.zeros(self.l_max ** 2, dtype='complex')  # Create coefficient vectors
        c2vec = np.zeros(self.l_max ** 2, dtype='complex')
        c3vec = np.zeros(self.l_max ** 2, dtype='complex')
        for idx in range(self.l_max ** 2):  # Only go up to lmax - 1 to avoid getting out of range on the l + 1 recursion
            l, m = self._alm_get_lm(self.l_max - 1, idx)[0].item(), self._alm_get_lm(self.l_max - 1, idx)[1].item() 
            c1vec[idx] = self._c_coeffs(1, l, m) * np.min([l, 1])  # Multiplier to handle l==0 exception
            c2vec[idx], c3vec[idx] =  self._c_coeffs(2, l, m), self._c_coeffs(3, l, m)
        vec_idx_zero = np.zeros(self.l_max ** 2, dtype='int')  # Create permutation vectors to select parts of A and B matrices
        vec_idx_minus = np.zeros(self.l_max ** 2, dtype='int')
        vec_idx_plus = np.zeros(self.l_max ** 2, dtype='int')
        for idx in range(self.l_max ** 2):
            l, m = self._alm_get_lm(self.l_max - 1, idx)[0].item(), self._alm_get_lm(self.l_max - 1, idx)[1].item()
            vec_idx_zero[idx] = self._alm_get_idx(self.l_max, l, m)
            vec_idx_minus[idx] = self._alm_get_idx(self.l_max, l - 1, m)
            vec_idx_plus[idx] = self._alm_get_idx(self.l_max, l + 1, m)
        dtilde_l_lm_zero  = dtilde_l_lm[vec_idx_zero]
        dtilde_b_lm_minus = dtilde_b_lm[vec_idx_minus]
        dtilde_b_lm_plus  = dtilde_b_lm[vec_idx_plus]
        amp = c1vec * dtilde_b_lm_minus + c2vec * dtilde_b_lm_plus - c3vec * dtilde_l_lm_zero
        return np.real(amp * np.conjugate(amp))

    def _arr_pow_spec(self, pow_spec):  # Converts idx components into (l,m) array
        l_max = (np.sqrt(pow_spec.shape[0]) - 1).astype(int)
        tmp_mat = np.zeros((l_max + 1, l_max + 1)) + 1e-100
        for l in range(l_max + 1):
            for m in range(l + 1):
                idx = self._alm_get_idx(l_max, l, m)
                tmp_mat[m, l] += pow_spec[idx]
        return tmp_mat