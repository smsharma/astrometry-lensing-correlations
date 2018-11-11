import sys, os

from tqdm import *
import numpy as np
import healpy as hp
from scipy.special import legendre
from multiprocessing import Pool
import argparse
from pathlib import Path

# ### Command line arguments
# parser = argparse.ArgumentParser()
# parser.add_argument("--lmin",
#                   action="store", dest="lmin", default=0, type=int)
# parser.add_argument("--lmax",
#                   action="store",dest='lmax', default=100,type=int)

# results=parser.parse_args()
# lmin=results.lmin
# lmax=results.lmax

### Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--i",
                  action="store", dest="i", default=0, type=int)

results=parser.parse_args()
i=results.i

dp = np.load('data/dp.npy')
Cinv = np.load('data/Cinv.npy')

# print("Going from", lmin, "to", lmax)

# def get_fisher(l_1):
#     P_l_1 = P_l(l_1)
#     m1 = np.multiply(np.diag(Cinv)[:,None], P_l_1)
#     for l_2 in range(l_1 + 1):
#         filename = "output/f_"+str(l_1)+"_"+str(l_2)+".npy"
#         the_file = Path(filename)
#         if the_file.is_file():
#             print("Exists", l_1, l_2)
#             continue
#         print("Doing", l_1, l_2)
#         np.save(filename, 0.5*np.einsum("ij,ji",m1,np.multiply(np.diag(Cinv)[:,None], P_l(l_2))))

def get_fisher(llp):
    l_1, l_2 = llp
    filename = "output/f_"+str(l_1)+"_"+str(l_2)+".npy"
    the_file = Path(filename)
    if the_file.is_file():
        print("Exists", l_1, l_2)
    else:
        P_l_1 = np.load("/scratch/sm8383/P_l/P_l_"+str(l_1)+".npy")
        P_l_2 = np.load("/scratch/sm8383/P_l/P_l_"+str(l_2)+".npy")
        np.save(filename, 0.5*np.einsum("ij,ji",np.multiply(np.diag(Cinv)[:,None], P_l_1),np.multiply(np.diag(Cinv)[:,None], P_l_2)))

llp_ary = np.load("data/llp_ary.npy")[i]
pool = Pool(10)                         # Create a multiprocessing Pool
pool.map(get_fisher, llp_ary)  # process data_inputs iterable with pool