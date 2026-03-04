#!/bin/bash

# ==========================================================
# RQ-VAE Training Script for MuonRec
# Assume this script is executed from the project root (MuonRec/)
# ==========================================================

# 1. Select the dataset category
CATEGORY="Industrial_and_Scientific"

# 2. Define standard paths
DATA_PATH="./minionerec_engine/data/Amazon/index/${CATEGORY}.emb-qwen-td.npy"
CKPT_DIR="./outputs/rqvae"

echo "----------------------------------------------------"
echo "Starting RQ-VAE training for: ${CATEGORY}"
echo "Model checkpoints will be saved to: ${CKPT_DIR}/${CATEGORY}"
echo "----------------------------------------------------"

# 3. Execute training from the project root
python minionerec_engine/rq/rqvae.py \
      --dataset "${CATEGORY}" \
      --data_path "${DATA_PATH}" \
      --ckpt_dir "${CKPT_DIR}" \
      --lr 1e-3 \
      --epochs 10000 \
      --batch_size 20480 \
      --eval_step 50 \
      --device "cuda:0"

echo "----------------------------------------------------"
echo "✓ Training script execution finished."
echo "----------------------------------------------------"