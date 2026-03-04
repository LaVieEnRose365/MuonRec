#!/bin/bash
export NCCL_IB_DISABLE=1        # Disable IB/RoCE for certain cluster environments

# ==========================================
# 1. Experiment Configuration
# ==========================================
# [Options] "muon" or "adam"
OPTIMIZER="muon" 

# [Options] "Industrial_and_Scientific" or "Office_Products"
CATEGORY="Industrial_and_Scientific" 

# Set to "Qwen/Qwen2.5-3B-Instruct" or your local model path
BASE_MODEL="Qwen/Qwen2.5-3B-Instruct" 

# Training hyperparameters
GLOBAL_BATCH_SIZE=1024
MICRO_BATCH_SIZE=8
SEED=42

# Optimizer hyperparameters
ADAM_LR=1e-4
ADAM_WD=0
MUON_LR=1e-4
MUON_WD=1e-5

# ==========================================
# 2. Path Resolution (Assume running from project root)
# ==========================================
DATA_DIR="minionerec_engine/data/Amazon"
OUTPUT_DIR="outputs/generative/sft_${OPTIMIZER}/${CATEGORY}"

# Dynamically locate the dataset files
train_file=$(ls -f ${DATA_DIR}/train/${CATEGORY}*11.csv | head -n 1)
eval_file=$(ls -f ${DATA_DIR}/valid/${CATEGORY}*11.csv | head -n 1)
test_file=$(ls -f ${DATA_DIR}/test/${CATEGORY}*11.csv | head -n 1)
info_file=$(ls -f ${DATA_DIR}/info/${CATEGORY}*.txt | head -n 1)

echo "=========================================="
echo "          MuonRec SFT Pipeline            "
echo "=========================================="
echo "Dataset Category : ${CATEGORY}"
echo "Optimizer Mode   : ${OPTIMIZER^^}"
echo "Train Data       : ${train_file}"
echo "Output Directory : ${OUTPUT_DIR}"
echo "=========================================="

# ==========================================
# 3. Dynamic Script & Argument Routing
# ==========================================
if [ "$OPTIMIZER" = "muon" ]; then
    SCRIPT_PATH="experiments/generative/sft_muon.py"
    OPT_ARGS="--adam_lr ${ADAM_LR} --adam_wd ${ADAM_WD} --muon_lr ${MUON_LR} --muon_wd ${MUON_WD}"
elif [ "$OPTIMIZER" = "adam" ]; then
    SCRIPT_PATH="experiments/generative/sft_adam.py"
    # sft_adam.py uses the default huggingface trainer arguments for adam
    OPT_ARGS="--learning_rate ${ADAM_LR} --weight_decay ${ADAM_WD}"
else
    echo "Error: Unknown OPTIMIZER set. Please choose 'muon' or 'adam'."
    exit 1
fi

# ==========================================
# 4. Distributed Training Execution
# ==========================================
torchrun --nproc_per_node 8 \
        ${SCRIPT_PATH} \
        --base_model ${BASE_MODEL} \
        --batch_size ${GLOBAL_BATCH_SIZE} \
        --micro_batch_size ${MICRO_BATCH_SIZE} \
        --train_file ${train_file} \
        --eval_file ${eval_file} \
        --output_dir ${OUTPUT_DIR} \
        --category ${CATEGORY} \
        --train_from_scratch False \
        --seed ${SEED} \
        --sid_index_path ${DATA_DIR}/index/${CATEGORY}.index.json \
        --item_meta_path ${DATA_DIR}/index/${CATEGORY}.item.json \
        --freeze_LLM False \
        ${OPT_ARGS}