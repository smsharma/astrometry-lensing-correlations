
import sys, os
import random
import numpy as np

batch='''#!/bin/bash

#SBATCH -N 1   # node count
#SBATCH --ntasks-per-node=4
#SBATCH -t 3:00:00
#SBATCH --mem=12GB

source ~/.bashrc
conda activate venv_smsharma 

cd /home/sm8383/Lensing-PowerSpectra/simulation


'''

for imc in range(100,200):
    batchn = batch  + "\n"
    batchn += "python astrometry_sim_interface.py --imc " + str(imc)
    fname = "batch/mc_" + str(imc) + ".batch" 
    f=open(fname, "w")
    f.write(batchn)
    f.close()
    os.system("chmod +x " + fname);
    os.system("sbatch " + fname);
