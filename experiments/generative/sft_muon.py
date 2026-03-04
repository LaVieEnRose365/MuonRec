import os
import sys
import json
import math
import random
from typing import List, Any, Dict, Optional, Tuple, Union
from functools import partial

import numpy as np 
import fire
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset, concatenate_datasets, Dataset as HFDataset
from transformers import EarlyStoppingCallback, AutoConfig, TrainerCallback, AutoModelForCausalLM, AutoTokenizer, set_seed
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import ConcatDataset
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from minionerec_engine.data import (
    SFTData, 
    SidSFTDataset, 
    SidItemFeatDataset, 
    FusionSeqRecDataset, 
    TitleHistory2SidSFTDataset
)
from Muon.muon import MuonWithAuxAdam
os.environ['WANDB_DISABLED'] = 'true'

class OfflinePrinterCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            if "loss" in logs:
                print(f">>> [Step {state.global_step}] Train Loss: {logs['loss']:.6f} | LR: {logs.get('learning_rate', 0):.2e}")
            if "eval_loss" in logs:
                print(f"\n{'='*40}")
                print(f"*** EVALUATION [Step {state.global_step}] ***")
                print(f"*** Validation Loss: {logs['eval_loss']:.6f}")
                print(f"{'='*40}\n")

    def on_save(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            print(f"\n[Checkpoint] Model saved at step {state.global_step}")

class TokenExtender:
    def __init__(self, data_path, dataset, index_file=".index.json"):
        self.data_path = data_path
        self.dataset = dataset
        self.index_file = index_file
        self.indices = None
        self.new_tokens = None
        
    def _load_data(self):
        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)
    
    def get_new_tokens(self):
        if self.new_tokens is not None: 
            return self.new_tokens
        if self.indices is None: 
            self._load_data()
        self.new_tokens = set()
        for index in self.indices.values():
            for token in index: 
                self.new_tokens.add(token)
        self.new_tokens = sorted(list(self.new_tokens))
        return self.new_tokens

def _get_cosine_schedule_with_warmup_lr_lambda(current_step, *, num_warmup_steps, num_training_steps, num_cycles):
    if current_step < num_warmup_steps:
        return max(0.1, float(current_step) / float(max(1, num_warmup_steps)))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.1, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles: float = 0.5, last_epoch: int = -1):
    lr_lambda = partial(_get_cosine_schedule_with_warmup_lr_lambda, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, num_cycles=num_cycles)
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def train(
    base_model: str = "",
    train_file: str = "",
    eval_file: str = "",
    output_dir: str = "",
    sample: int = -1,
    seed: int = 42,
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 10,
    cutoff_len: int = 512,
    group_by_length: bool = False,
    freeze_LLM: bool = False,
    wandb_project: str = "",
    wandb_run_name: str = "",
    resume_from_checkpoint: str = None,
    category: str = "",
    train_from_scratch: bool = False,
    sid_index_path: str = "",
    item_meta_path: str = "",
    # Hyperparameters aligned with shell scripts
    adam_lr: float = 1e-4,
    adam_wd: float = 0.0,
    muon_lr: float = 0.02,
    muon_wd: float = 0.0,
):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    set_seed(42, deterministic=True)
    
    category_dict = {
        "Industrial_and_Scientific": "industrial and scientific items", 
        "Office_Products": "office products", 
        "Toys_and_Games": "toys and games", 
        "Sports": "sports and outdoors", 
        "Books": "books"
    }
    print(f"Dataset Category: {category}")
    category = category_dict[category]
    assert base_model, "Please specify a --base_model"
    
    gradient_accumulation_steps = batch_size // micro_batch_size
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    if not train_from_scratch:
        model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16)
    else:
        config = AutoConfig.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_config(config)
        print("Training from scratch!")
        
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    
    original_vocab_size = len(tokenizer)
    
    if sid_index_path and os.path.exists(sid_index_path):
        print(f"Loading index from {sid_index_path}")
        token_extender = TokenExtender(data_path=os.path.dirname(sid_index_path), dataset=os.path.basename(sid_index_path).split('.')[0])
        new_tokens = token_extender.get_new_tokens()
        if new_tokens:
            print(f"Adding {len(new_tokens)} new tokens to tokenizer")
            tokenizer.add_tokens(new_tokens)
            model.resize_token_embeddings(len(tokenizer))

    # Freeze LLM logic
    if freeze_LLM:
        print("Freezing LLM parameters, only training new token embeddings")
        for param in model.parameters(): 
            param.requires_grad = False
            
        if sid_index_path and os.path.exists(sid_index_path) and new_tokens:
            embedding_layer = model.get_input_embeddings()
            if embedding_layer.weight.shape[0] > original_vocab_size:
                embedding_layer.weight.requires_grad = True
                def mask_grad(grad):
                    grad[:original_vocab_size].zero_()
                    return grad
                embedding_layer.weight.register_hook(mask_grad)
                print(f"Unfrozen {len(new_tokens)} new token embeddings")
        else:
            print("Warning: freeze_LLM=True but no new tokens added.")
        
    # Data Loading
    train_datasets = []
    train_datasets.append(SidSFTDataset(train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, seed=seed, category=category))
    train_datasets.append(SidItemFeatDataset(item_file=item_meta_path, index_file=sid_index_path, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, seed=seed, category=category))
    train_datasets.append(FusionSeqRecDataset(train_file=train_file, item_file=item_meta_path, index_file=sid_index_path, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, seed=seed, category=category))
    train_datasets.append(SFTData(train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, seed=seed, category=category))
    train_datasets.append(TitleHistory2SidSFTDataset(train_file=train_file, item_file=item_meta_path, index_file=sid_index_path, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, seed=seed, category=category))
    
    train_data = ConcatDataset(train_datasets)
    val_data = SidSFTDataset(train_file=eval_file, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, seed=seed, category=category)
    print("LOAD DATA FINISHED")    
    
    sample_frac = 1.0
    hf_train_dataset = HFDataset.from_dict({k: [v[k] for v in train_data] for k in train_data[0].keys()})
    hf_train_dataset = hf_train_dataset.shuffle(seed=42).select(range(int(sample_frac * len(hf_train_dataset))))
    hf_val_dataset = HFDataset.from_dict({k: [v[k] for v in val_data] for k in val_data[0].keys()}).shuffle(seed=seed)
    
    steps_per_epoch = len(hf_train_dataset) // batch_size
    total_steps = steps_per_epoch * num_epochs
    
    print(f"\n=== Training Configuration ===")
    print(f"Dataset size: {len(hf_train_dataset)}")
    print(f"Global batch size: {batch_size}")
    print(f"Micro batch size per device: {micro_batch_size}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total training steps: {total_steps}")
    print(f"Number of epochs: {num_epochs}")
    print(f"==============================\n")
    
    # Configure Muon + Adam Hybrid Optimizer (Qwen2.5)
    print(f"\n=== Configuring Muon Optimizer for Qwen2.5 ===")
    muon_params = []
    adam_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad: 
            continue
        
        # 1. Embeddings, LM Head, Norms, Bias -> Adam
        if "embed_tokens" in name or "lm_head" in name:
            adam_params.append(param)
        elif "norm" in name: 
            adam_params.append(param)
        elif param.ndim < 2: # Bias (1D)
            adam_params.append(param)
        
        # 2. 2D Projections (Attention & MLP) -> Muon
        elif param.ndim == 2:
            if any(target in name for target in ["proj"]):
                muon_params.append(param)
            else:
                adam_params.append(param)
        else:
            adam_params.append(param)

    print(f"Muon params (Attention/MLP Weights): {len(muon_params)} tensors")
    print(f"Adam params (Embeds/Head/Norms/Bias): {len(adam_params)} tensors")
    
    # Group parameters for the custom MuonWithAuxAdam optimizer
    optimizer_grouped_parameters = []

    if muon_params:
        optimizer_grouped_parameters.append({
            "params": muon_params, 
            "use_muon": True, 
            "lr": muon_lr, 
            "weight_decay": muon_wd,
            "momentum": 0.95,  
        })

    if adam_params:
        optimizer_grouped_parameters.append({
            "params": adam_params, 
            "use_muon": False, 
            "lr": adam_lr, 
            "weight_decay": adam_wd,
            "betas": (0.9, 0.95), 
            "eps": 1e-8
        })
    
    # Initialize the imported hybrid optimizer
    optimizer = MuonWithAuxAdam(optimizer_grouped_parameters)
    
    # Initialize scheduler
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=20, num_training_steps=total_steps, num_cycles=0.5)
    
    # Print configuration
    print("\n" + "="*50)
    print("=== Optimizer & Scheduler Configuration ===")
    print("="*50)
    print(f"Optimizer Type: MuonWithAuxAdam")
    print(f"[Group 1] Muon (Linear Layers):")
    print(f"    - LR: {muon_lr}")
    print(f"    - Weight Decay: {muon_wd}")
    print(f"[Group 2] Adam (Embed/Norm/Bias):")
    print(f"    - LR: {adam_lr}")
    print(f"    - Weight Decay: {adam_wd}")
    print("="*50 + "\n")
    
    eval_step = 0.05
    eval_steps_per_epoch = int(1.0 / eval_step)
    early_stopping_patience = eval_steps_per_epoch
    
    print(f"Early stopping patience: {early_stopping_patience} eval steps (= 1 epoch)")
    
    trainer = transformers.Trainer(
        model=model,
        train_dataset=hf_train_dataset,
        eval_dataset=hf_val_dataset,
        args=transformers.TrainingArguments(
            run_name=wandb_run_name,
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=20,
            num_train_epochs=num_epochs,
            learning_rate=adam_lr, 
            bf16=True,
            logging_steps=1,
            optim="adamw_torch",
            eval_strategy="steps",
            eval_steps=eval_step, 
            save_strategy="steps",
            save_steps=eval_step,
            output_dir=output_dir,
            save_total_limit=1,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="none", 
            metric_for_best_model="eval_loss", 
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=early_stopping_patience),
            OfflinePrinterCallback() 
        ],
        optimizers=(optimizer, lr_scheduler) 
    )
    model.config.use_cache = False
    
    # Start training
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Post-training Statistics
    metrics = train_result.metrics
    
    best_step = "Unknown"
    if trainer.state.best_model_checkpoint:
        try:
            best_step = int(trainer.state.best_model_checkpoint.split("-")[-1])
        except ValueError:
            best_step = trainer.state.best_model_checkpoint

    final_train_loss = metrics.get("train_loss", "N/A")
    best_valid_loss = float('inf')
    for log in trainer.state.log_history:
        if "eval_loss" in log:
            if log["eval_loss"] < best_valid_loss:
                best_valid_loss = log["eval_loss"]
    
    total_params = sum(p.numel() for p in model.parameters())
    flops_per_sample = 6 * total_params * cutoff_len
    
    print("\n" + "="*50)
    print(" FINAL TRAINING RESULTS")
    print("="*50)
    print(f"1. Train Loss:        {final_train_loss}")
    print(f"2. Valid Loss (Best): {best_valid_loss:.6f}")
    print(f"3. Convergence Step:  {best_step}")
    print(f"4. FLOPs per Sample:  {flops_per_sample:.4e}")
    print("-" * 50)
    print(f"(Model Params: {total_params:,}, Seq Len: {cutoff_len})")
    print("="*50 + "\n")
    
    trainer.save_model(output_dir)
    output_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    fire.Fire(train)