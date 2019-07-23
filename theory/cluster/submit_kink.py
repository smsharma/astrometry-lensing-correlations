import sys, os
import random
import numpy as np

batch='''#!/bin/bash

#SBATCH -N 1   # node count
#SBATCH --ntasks-per-node=1
#SBATCH -t 00:59:00
#SBATCH --mem=3GB

source ~/.bashrc
conda activate

cd /home/sm8383/Lensing-PowerSpectra/theory/cluster

'''

n_B_ary = [0.9665] + list(np.linspace(1, 6, 11))
k_B_ary = np.logspace(np.log10(1), np.log10(100), 11)

for n_B in n_B_ary:
    for k_B in k_B_ary:
        batchn = batch  + "\n"
        batchn += "python kink_spec.py --kB " + str(k_B) + " --nB " + str(n_B)
        fname = "batch/" + str(k_B) + "_" + str(n_B) + ".batch"
        f=open(fname, "w")
        f.write(batchn)
        f.close()
        os.system("chmod +x " + fname);
        os.system("sbatch " + fname);
