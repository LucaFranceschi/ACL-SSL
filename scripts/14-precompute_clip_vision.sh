#!/bin/bash

echo "SLURM_VISIBLE_DEVICES: $SLURM_JOB_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

nvidia-smi

REPO="/home/lfranceschi/repos/ACL-SSL"
DATA=$REPO/datasets
SAVE_PATH=$DATA/vggsound/precomputed_frames

cd $REPO

mkdir -p $SAVE_PATH

set -a; source config/.env; set +a

python precompute_CLIP.py \
--model_name ACL_ViT16 \
--model_path $REPO/pretrain \
--train_config Exp_ACL_v2 \
--vggsound_path $DATA/vggsound \
--save_path $SAVE_PATH