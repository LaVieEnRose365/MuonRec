import os
import sys
import json
import math
import random
import torch
import torch.nn as nn
import numpy as np
import fire
import transformers
from datasets import Dataset as HFDataset
from torch.utils.data import ConcatDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, AutoConfig, set_seed
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
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
os.environ['WANDB_DISABLED'] = 'true'

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


def train(
    # model/data params
    base_model: str = "",  
    train_file: str = "",
    eval_file: str = "",
    output_dir: str = "",
    sample: int = -1,
    seed: int = 42,
    
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    cutoff_len: int = 512,
    
    # llm hyperparams
    group_by_length: bool = False,
    freeze_LLM: bool = False,
    use_adam: bool = False,  # Set to True to use custom Adam optimizer instead of AdamW

    # logging & state params
    wandb_project: str = "",
    wandb_run_name: str = "",
    resume_from_checkpoint: str = None,
    category: str = "",
    train_from_scratch: bool = False,
    sid_index_path: str = "",
    item_meta_path: str = "",
):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    set_seed(42, deterministic=True)
    os.environ['WANDB_PROJECT'] = wandb_project
    
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
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
        )
    else:
        config = AutoConfig.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_config(config)
        print("Training from scratch!")
        
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    
    # Save original vocab size to handle freezing logic when adding new tokens
    original_vocab_size = len(tokenizer)
    new_tokens = []
    
    if sid_index_path and os.path.exists(sid_index_path):
        print(f"Loading index from {sid_index_path}")
        token_extender = TokenExtender(
            data_path=os.path.dirname(sid_index_path),
            dataset=os.path.basename(sid_index_path).split('.')[0]
        )
        new_tokens = token_extender.get_new_tokens()
        if new_tokens:
            print(f"Adding {len(new_tokens)} new tokens to tokenizer")
            tokenizer.add_tokens(new_tokens)
            model.resize_token_embeddings(len(tokenizer))

    # Freeze LLM parameters if required
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

                print(f"Unfrozen {len(new_tokens)} new token embeddings "
                      f"(indices {original_vocab_size} to {len(tokenizer)-1})")
        else:
            print("Warning: freeze_LLM=True but no new tokens added. All parameters are frozen!")

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters (with grad-mask): {trainable_params:,} / "
              f"{total_params:,} ({100*trainable_params/total_params:.2f}%)")
        
    # Dataset preparation
    train_datasets = []
    train_datasets.append(SidSFTDataset(train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, seed=seed, category=category))
    train_datasets.append(SidItemFeatDataset(item_file=item_meta_path, index_file=sid_index_path, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, seed=seed, category=category))
    train_datasets.append(FusionSeqRecDataset(train_file=train_file, item_file=item_meta_path, index_file=sid_index_path, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, seed=seed, category=category))
    train_datasets.append(SFTData(train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, seed=seed, category=category))
    train_datasets.append(TitleHistory2SidSFTDataset(train_file=train_file, item_file=item_meta_path, index_file=sid_index_path, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, seed=seed, category=category))
    
    train_data = ConcatDataset(train_datasets)
    val_data = SidSFTDataset(train_file=eval_file, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, seed=seed, category=category)
    print("LOAD DATA FINISHED")    
    
    if resume_from_checkpoint:
        checkpoint_name = os.path.join(resume_from_checkpoint, "pytorch_model.bin")

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True
    
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
    print(f"Number of devices: {world_size}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total training steps: {total_steps}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Initial learning rate: {learning_rate}")
    print(f"Weight Decay: {weight_decay}")
    print(f"==============================\n")
    
    # Calculate evaluation steps for early stopping
    eval_step = 0.05
    eval_steps_per_epoch = int(1.0 / eval_step)
    early_stopping_patience = eval_steps_per_epoch  
    print(f"Early stopping patience: {early_stopping_patience} eval steps (= 1 epoch)")
    
    # Initialize TrainingArguments
    training_args = transformers.TrainingArguments(
        run_name=wandb_run_name,
        per_device_train_batch_size=micro_batch_size,
        per_device_eval_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=20,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,     
        bf16=True,
        logging_steps=1,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        eval_strategy="steps",
        eval_steps=eval_step, 
        save_strategy="steps",
        save_steps=eval_step,
        output_dir=output_dir,
        save_total_limit=1,
        load_best_model_at_end=True,
        ddp_find_unused_parameters=False if ddp else None,
        group_by_length=group_by_length,
        report_to=None,
        metric_for_best_model="eval_loss",
    )

    # Debugging: Verify critical training arguments
    print("\n" + "="*40)
    print("DEBUG CHECK: Verifying Training Arguments")
    print(f"--> Received Learning Rate: {training_args.learning_rate}")
    print(f"--> Received Weight Decay:  {training_args.weight_decay}")
    print(f"--> Received BF16 setting:  {training_args.bf16}")
    print("="*40 + "\n")

    # Prepare optimizer and scheduler if using custom Adam
    trainer_kwargs = {
        "model": model,
        "train_dataset": hf_train_dataset,
        "eval_dataset": hf_val_dataset,
        "args": training_args,
        "data_collator": transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        "callbacks": [
            EarlyStoppingCallback(
                early_stopping_patience=eval_steps_per_epoch
            )
        ],
    }

    if use_adam:
        print("Using custom Adam optimizer instead of AdamW")
        optimizer = Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=weight_decay
        )

        # Create learning rate scheduler (cosine annealing with warmup)
        warmup_steps = 20
        total_steps = steps_per_epoch * num_epochs

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        lr_scheduler = LambdaLR(optimizer, lr_lambda)
        trainer_kwargs["optimizers"] = (optimizer, lr_scheduler)

        print("DEBUG CHECK: Adam Optimizer Parameter Groups")
        for i, group in enumerate(optimizer.param_groups):
            print(f"Group {i} - LR: {group['lr']}, WD: {group['weight_decay']}, Params: {len(group['params'])}")
        print("="*40 + "\n")
    else:
        print("Using default AdamW optimizer from transformers")
        print("="*40 + "\n")

    # Initialize Trainer
    trainer = transformers.Trainer(**trainer_kwargs)

    model.config.use_cache = False
    
    # Start training
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(output_dir)
    
    output_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    fire.Fire(train)