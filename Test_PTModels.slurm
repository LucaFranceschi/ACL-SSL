#!/bin/bash

#SBATCH --job-name=test_flickr
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --partition std
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --error=outputs/slurm-%j.err
#SBATCH --time=0-00:10:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=luca.franceschi01@estudiant.upf.edu

# ###      #asdfasdf SBITCH --gres=gpu:2

module load conda
conda activate acl_ssl8

rsync -avz --exclude-from=excluded-directories.txt ~/repos/ACL-SSL $SCRATCH

cd $SCRATCH/ACL-SSL

srun Test_PTModels.sh

# copiar resultados a $DATA?