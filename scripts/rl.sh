#!/bin/bash
export NCCL_IB_DISABLE=1        # Disable IB/RoCE for certain cluster environments
export HF_ENDPOINT=https://hf-mirror.com  # Use mirror for HuggingFace if needed

# ==========================================
# 1. Experiment Configuration
# ==========================================
# [Options] "muon" or "adam"
OPTIMIZER="muon" 

# [Options] "Industrial_and_Scientific" or "Office_Products"
CATEGORY="Industrial_and_Scientific" 

# Path to the SFT checkpoint (dynamically connects to run_sft.sh output)
SFT_MODEL_PATH="outputs/generative/sft_${OPTIMIZER}/${CATEGORY}/final_checkpoint"

# Training hyperparameters
TRAIN_BATCH_SIZE=64
EVAL_BATCH_SIZE=128
NUM_PROCESSES=4
GRAD_ACCUMULATION_STEPS=2
NUM_EPOCHS=2

# Optimizer hyperparameters
ADAM_LR=1e-5
ADAM_WD=0.0
MUON_LR=1e-3
MUON_WD=1e-5

# ==========================================
# 2. Path Resolution (Assume running from project root)
# ==========================================
DATA_DIR="minionerec_engine/data/Amazon"
OUTPUT_DIR="outputs/generative/rl_${OPTIMIZER}/${CATEGORY}"

# Dynamically locate the dataset files
train_file=$(ls -f ${DATA_DIR}/train/${CATEGORY}*.csv | head -n 1)
eval_file=$(ls -f ${DATA_DIR}/valid/${CATEGORY}*11.csv | head -n 1)
info_file=$(ls -f ${DATA_DIR}/info/${CATEGORY}*.txt | head -n 1)

echo "=========================================="
echo "           MuonRec RL Pipeline            "
echo "=========================================="
echo "Dataset Category : ${CATEGORY}"
echo "Optimizer Mode   : ${OPTIMIZER^^}"
echo "Base Model (SFT) : ${SFT_MODEL_PATH}"
echo "Output Directory : ${OUTPUT_DIR}"
echo "=========================================="

# ==========================================
# 3. Dynamic Script & Argument Routing
# ==========================================
if [ "$OPTIMIZER" = "muon" ]; then
    SCRIPT_PATH="experiments/generative/rl_muon.py"
    OPT_ARGS="--learning_rate ${ADAM_LR} --adam_weight_decay ${ADAM_WD} --muon_learning_rate ${MUON_LR} --muon_weight_decay ${MUON_WD}"
elif [ "$OPTIMIZER" = "adam" ]; then
    SCRIPT_PATH="experiments/generative/rl_adam.py"
    OPT_ARGS="--learning_rate ${ADAM_LR}"
else
    echo "Error: Unknown OPTIMIZER set. Please choose 'muon' or 'adam'."
    exit 1
fi

# ==========================================
# 4. Distributed Training Execution (Accelerate)
# ==========================================
accelerate launch \
    --config_file minionerec_engine/config/zero2_opt.yaml \
    --num_processes ${NUM_PROCESSES} \
    --main_process_port 29503 \
    ${SCRIPT_PATH} \
    --model_path ${SFT_MODEL_PATH} \
    --train_batch_size ${TRAIN_BATCH_SIZE} \
    --eval_batch_size ${EVAL_BATCH_SIZE} \
    --num_train_epochs ${NUM_EPOCHS} \
    --gradient_accumulation_steps ${GRAD_ACCUMULATION_STEPS} \
    --train_file "${train_file}" \
    --eval_file "${eval_file}" \
    --info_file "${info_file}" \
    --category "${CATEGORY}" \
    --sample_train False \
    --eval_step 0.0999 \
    --reward_type ranking \
    --num_generations 16 \
    --mask_all_zero False \
    --dynamic_sampling False \
    --sync_ref_model True \
    --beam_search True \
    --test_during_training False \
    --temperature 1.0 \
    --add_gt False \
    --beta 0.1 \
    --dapo False \
    --output_dir ${OUTPUT_DIR} \
    --wandb_run_name "${CATEGORY}-rl-${OPTIMIZER}-$(date +%Y%m%d)" \
    --sid_index_path ${DATA_DIR}/index/${CATEGORY}.index.json \
    --item_meta_path ${DATA_DIR}/index/${CATEGORY}.item.json \
    ${OPT_ARGS}

echo ""
echo "========================================"
echo "✓ RL Training (${OPTIMIZER^^}) Completed!"
echo "========================================"