#!/bin/bash

# ==========================================================
# Data Preprocessing Script for MuonRec
# Assume this script is executed from the project root (MuonRec/)
# ==========================================================

# Step 1: Define paths to your raw Amazon dataset JSON files
# Replace these with the actual locations on your machine
METADATA_FILE="./raw_data/meta_Industrial_and_Scientific.json"
REVIEWS_FILE="./raw_data/Industrial_and_Scientific_5.json"

# Step 2: Run the preprocessing script
python minionerec_engine/data/amazon18_data_process.py \
    --dataset Industrial_and_Scientific \
    --metadata_file ${METADATA_FILE} \
    --reviews_file ${REVIEWS_FILE} \
    --user_k 5 \
    --item_k 5 \
    --st_year 2017 \
    --st_month 10 \
    --ed_year 2018 \
    --ed_month 11 \
    --output_path ./minionerec_engine/data/Amazon