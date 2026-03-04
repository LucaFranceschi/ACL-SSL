#!/bin/bash

EXPERIMENT_VERSION=$1

echo "SLURM_VISIBLE_DEVICES: $SLURM_JOB_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

nvidia-smi

REPO="/home/lfranceschi/repos/ACL-SSL"
DATASETS=$DATA/datasets
SAVE_PATH=$DATA/train_outputs/$SLURM_JOBID

cd $REPO

mkdir -p $SAVE_PATH

set -a
source config/.env
WANDB_MODE=offline
WANDB_PATH_LOGS=$SAVE_PATH
set +a

torchrun --nnodes=1 --nproc_per_node=2 --master_port 12345 \
train_ACL.py \
--model_name ACL_ViT16 \
--model_path $REPO/pretrain \
--exp_name aclifa_2gpu \
--train_config $EXPERIMENT_VERSION \
--vggss_path $DATASETS/VGGSS \
--flickr_path $DATASETS/Flickr \
--avs_path $DATASETS/AVSBench/AVS1 \
--vggsound_path $DATASETS/vggsound \
--san_path $DATASETS/silence_and_noise/audio \
--save_path $SAVE_PATH \
--wandb_logging

# --recover_from $REPO/train_outputs/2059437/Train_record/ACL_ViT16_aclifa_2gpu/Param_5.pth \