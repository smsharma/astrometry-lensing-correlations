
import sys, os
import random
import numpy as np

batch='''#!/bin/bash

#SBATCH -N 1   # node count
#SBATCH --ntasks-per-node=10
#SBATCH -t 24:00:00
#SBATCH --mem=50GB

source activate

cd /home/sm8383/Lensing-PowerSpectra/Fisher

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/sm8383/anaconda3/lib/

'''

# l_ranges = list(np.arange(10,100,10))
# l_ranges = [0, 10]

# for lmin, lmax in zip(l_ranges[:-1],l_ranges[1:]):
# 	batchn = batch  + "\n"
# 	batchn += "python fisher.py --lmin " + str(lmin) + " --lmax " + str(lmax)
# 	fname = "batch/" + str(lmin) + "_" + str(lmax) + ".batch" 
# 	f=open(fname, "w")
# 	f.write(batchn)
# 	f.close()
# 	os.system("chmod +x " + fname);
# 	os.system("sbatch " + fname);

for i in range(25):
	batchn = batch  + "\n"
	batchn += "python fisher.py --i " + str(i) 
	fname = "batch/" + str(i) + ".batch" 
	f=open(fname, "w")
	f.write(batchn)
	f.close()
	os.system("chmod +x " + fname);
	os.system("sbatch " + fname);
