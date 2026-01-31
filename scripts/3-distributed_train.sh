#!/bin/bash

EXPERIMENT_VERSION=$1

echo "SLURM_VISIBLE_DEVICES: $SLURM_JOB_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

nvidia-smi

DATA="/home/lfranceschi/repos/ACL-SSL"

cd $DATA

mkdir -p $DATA/train_outputs/$SLURM_JOBID

set -a; source config/.env; set +a

python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 --master_port 12345 \
Train_ACL_on_vggsound.py \
--model_name ACL_ViT16 \
--model_path $DATA/pretrain \
--exp_name aclifa_1gpu \
--train_config $EXPERIMENT_VERSION \
--vggss_path $DATA/VGGSS \
--flickr_path $DATA/Flickr \
--avs_path $DATA/AVSBench/AVS1 \
--vggsound_path $DATA/vggsound \
--san_path $DATA/silence_and_noise/audio \
--save_path $DATA/train_outputs/$SLURM_JOBID \
--wandb_logging
