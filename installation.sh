#! /bin/bash

# try to replicate the steps in the repo but with versions that will work, since nobody had the great idea to share an environment.yaml file

set -euo pipefail

module load conda
conda create -y -n acl_ssl8
conda activate acl_ssl8

conda install -y -c nvidia cudatoolkit=11.7
conda install -y -c conda-forge cudnn
conda install -y python=3.10
pip install --no-cache-dir torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 numpy==1.24.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install --no-cache-dir tensorboard==2.11.2
pip install --no-cache-dir transformers==4.25.1
pip install --no-cache-dir opencv-python==4.7.0.72
pip install --no-cache-dir tqdm==4.65.0
pip install --no-cache-dir scikit-learn==1.2.2
pip install --no-cache-dir six==1.16.0