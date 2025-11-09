#!/bin/bash

#SBATCH --job-name=test_flickr
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --partition gpu
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --error=outputs/slurm-%j.err
#SBATCH --time=0-00:30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=luca.franceschi01@estudiant.upf.edu

# ###      #asdfasdf SBITCH --gres=gpu:2

module load conda
conda activate acl_ssl24

rsync -avz --exclude-from=excluded-directories.txt ~/repos/ACL-SSL $SCRATCH

cd $SCRATCH/ACL-SSL

./Test_PTModels.sh
# srun Test_PTModels.sh

# copiar resultados a $DATA?