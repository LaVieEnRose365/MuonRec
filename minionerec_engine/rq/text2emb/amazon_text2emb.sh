#!/bin/bash

# ==========================================================
# Text Embedding Generation Script for MuonRec
# Assume this script is executed from the project root (MuonRec/)
# ==========================================================

# Step 1: Set your local PLM path (e.g., Qwen2-1.5B-Instruct)
# Replace this placeholder with the actual directory of your model
MODEL_PATH="./models/Qwen2-1.5B-Instruct"

# Step 2: Define categories to process
# You can add more categories to the list below
CATEGORIES=("Industrial_and_Scientific")

for category in "${CATEGORIES[@]}"; do
    echo "----------------------------------------------------"
    echo "Processing Category: ${category}"
    echo "----------------------------------------------------"

    python minionerec_engine/rq/text2emb/amazon_text2emb.py \
        --dataset "${category}" \
        --root "./minionerec_engine/data/Amazon" \
        --plm_name "qwen2-1.5b" \
        --plm_checkpoint "${MODEL_PATH}" \
        --gpu_id 0 \
        --max_sent_len 1024
done

echo "----------------------------------------------------"
echo "✓ Embedding generation completed for all categories."
echo "----------------------------------------------------"