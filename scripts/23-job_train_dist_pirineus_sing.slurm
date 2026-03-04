#!/bin/bash

#SBATCH --job-name=acl-train3
#SBATCH --partition gpu
#SBATCH --gres=gpu:h100:2
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 2
#SBATCH --cpus-per-task 32
#SBATCH --time=10-00:00:00
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --error=outputs/slurm-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=luca.franceschi01@estudiant.upf.edu

set -e

module load singularity

EXPERIMENT_VERSION=$1

nvidia-smi
echo SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST

# remember the .sif needs to be extracted beforehand with

singularity run --userns --nv --bind /home/lfranceschi/repos/ACL-SSL,$DATA $DATA/container-sandbox/ \
/home/lfranceschi/repos/ACL-SSL/scripts/3-distributed_train_pirineus.sh $EXPERIMENT_VERSION

if (( $? == 0 )); then
    rsync -avz $DATA/train_outputs/$SLURM_JOBID /home/lfranceschi/repos/ACL-SSL/train_outputs/$SLURM_JOBID
fi