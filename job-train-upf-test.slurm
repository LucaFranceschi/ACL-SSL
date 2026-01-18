#!/bin/bash

#SBATCH --job-name=acl-train
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 1
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --error=outputs/slurm-%j.err
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=luca.franceschi01@estudiant.upf.edu

set -e

env_name="acl-ssl"

module load Anaconda3
source activate $env_name

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/lfranceschi/.conda/envs/$env_name/lib

# rsync -avz --exclude-from=myutils/excluded-directories.txt ~/repos/ACL-SSL $SCRATCH

# cd $SCRATCH/ACL-SSL

./SingleGPU_Experiment.sh

# rsync -avz Visual_results $DATA
# rsync -avz Train_record $DATA