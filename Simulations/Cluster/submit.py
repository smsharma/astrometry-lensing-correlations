
import sys, os
import random
import numpy as np

batch='''#!/bin/bash

#SBATCH -N 1   # node count
#SBATCH --ntasks-per-node=1
#SBATCH -t 3:00:00
#SBATCH --mem=4GB

source activate

cd /home/sm8383/Lensing-PowerSpectra/Simulations

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/sm8383/anaconda3/lib/

'''

for imc in range(10):
    for log10M in np.arange(8,12):
    # for log10M in [7]:
        for mfrac in np.arange(0.1,1.1,0.1):

            batchn = batch  + "\n"
            batchn += "python astrometry_sim_interface.py --log10M " + str(log10M) + ' --mfrac ' + str(mfrac) + ' --imc ' + str(imc)
            fname = "batch/" + 'log10M_' + str(log10M) + '_mfrac_' + str(np.round(mfrac,2)) + '_mc_' + str(imc) + ".batch" 
            f=open(fname, "w")
            f.write(batchn)
            f.close()
            os.system("chmod +x " + fname);
            os.system("sbatch " + fname);
