import sys, os
import random
import numpy as np

batch='''#!/bin/bash
#SBATCH -N 1   # node count
#SBATCH --ntasks-per-node=1
#SBATCH -t 1:00:00
#SBATCH --mem=4GB
##SBATCH --mail-type=begin
##SBATCH --mail-type=end
##SBATCH --mail-user=sm8383@nyu.edu
#SBATCH -p hepheno

# MPI compilers
export MKL_LP64_ILP64=ilp64
source /opt/intel/compilers_and_libraries_2016.2.181/linux/bin/compilervars.sh intel64
source /opt/intel/impi/5.0.3.048/bin64/mpivars.sh

source ~/.bashrc
source /group/hepheno/smsharma/heptools/anaconda3/etc/profile.d/conda.sh
conda activate 

cd /group/hepheno/smsharma/Lensing-PowerSpectra/theory/cluster/

'''

n_B_ary = [0.9665] + list(np.linspace(1., 3, 11))
k_B_ary = np.logspace(np.log10(5), np.log10(50), 11)

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
