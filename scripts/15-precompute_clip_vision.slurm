#!/bin/bash

#SBATCH --job-name=precomp
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 4
#SBATCH --mem-per-cpu 16G
#SBATCH --gres=gpu:1
#SBATCH --exclude="node025,node032"
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --error=outputs/slurm-%j.err
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=luca.franceschi01@estudiant.upf.edu

set -e

nvidia-smi
echo SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST

singularity run --nv --bind /home/lfranceschi/repos/ACL-SSL container.sif \
/home/lfranceschi/repos/ACL-SSL/scripts/14-precompute_clip_vision.sh