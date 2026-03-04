import os
import sys
import random
import numpy as np
import torch
import math
import json
import pickle
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed, AutoConfig
from torch.utils.data import ConcatDataset
from sklearn.metrics import ndcg_score
from fire import Fire
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from minionerec_engine.data import (
    D3Dataset, 
    SidDataset, 
    RLTitle2SidDataset, 
    RLSeqTitle2SidDataset, 
    RLSid2TitleDataset, 
    RLSidhis2TitleDataset
)
from minionerec_engine.minionerec_trainer import ReReTrainer
from experiments.traditional.sasrec_adam import SASRec
from Muon.muon import MuonWithAuxAdam
os.environ['WANDB_DISABLED'] = 'true'

class MuonReReTrainer(ReReTrainer):
    def __init__(self, muon_lr=3e-5, muon_wd=1e-5, adam_wd=0.0, *args, **kwargs):
        self.muon_lr = muon_lr
        self.muon_wd = muon_wd
        self.adam_wd = adam_wd
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        """
        Override create_optimizer to safely initialize Muon inside the Trainer,
        bypassing potential DeepSpeed optimizer hijacking.
        """
        if self.optimizer is None:
            print(f"\n=== Configuring Muon Optimizer inside Trainer (Rank {self.args.local_rank}) ===")
            
            muon_params = []
            adam_params = []
            
            model_to_optimize = self.model
            
            for name, param in model_to_optimize.named_parameters():
                if not param.requires_grad: 
                    continue
                
                # Core Parameter Grouping Logic 
                # Robust against DeepSpeed dimensionality flattening

                # 1. Embeddings and LM Head -> Adam
                if "embed_tokens" in name or "lm_head" in name:
                    adam_params.append(param)
                # 2. Normalization Layers -> Adam
                elif "norm" in name: 
                    adam_params.append(param)
                # 3. Projection/FC Layers -> Muon 
                elif any(target in name for target in ["proj", "fc", "linear", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]):
                    muon_params.append(param)
                # 4. Fallback for uncaptured 2D tensors -> Muon
                elif param.ndim >= 2:
                    muon_params.append(param)
                # 5. Remaining scalars/biases -> Adam
                else:
                    adam_params.append(param)

            print(f"Muon params count: {len(muon_params)}")
            print(f"Adam params count: {len(adam_params)}")
            
            if len(muon_params) == 0:
                print(" WARNING: Muon param list is empty! Please check parameter naming logic.")

            muon_lr = self.muon_lr
            adam_lr = self.args.learning_rate 
            muon_wd = self.muon_wd
            adam_wd = self.adam_wd
            
            print(f"-> Muon LR: {muon_lr}, Muon WD: {muon_wd}")
            print(f"-> Adam LR: {adam_lr}, Adam WD: {adam_wd}")
            
            
            # Construct optimizer groups ensuring strict compatibility 
            # with MuonWithAuxAdam signature requirements.
        
            muon_group = {
                "params": muon_params,
                "use_muon": True,
                "lr": muon_lr,
                "weight_decay": muon_wd,
                "momentum": 0.95 
            }
            
            adam_group = {
                "params": adam_params,
                "use_muon": False,
                "lr": adam_lr,
                "weight_decay": adam_wd,
                "betas": (self.args.adam_beta1, self.args.adam_beta2),
                "eps": self.args.adam_epsilon
            }
            
            optimizer_grouped_parameters = []
            if len(muon_params) > 0:
                optimizer_grouped_parameters.append(muon_group)
            if len(adam_params) > 0:
                optimizer_grouped_parameters.append(adam_group)
            
            self.optimizer = MuonWithAuxAdam(optimizer_grouped_parameters)
            print("Muon Optimizer (External) created successfully.\n")
            
        return self.optimizer

class OfflineRLPrinterCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            step = state.global_step
            log_items = []
            if "loss" in logs:
                log_items.append(f"Loss: {logs['loss']:.4f}")
            if "reward_mean" in logs:
                log_items.append(f"Reward: {logs['reward_mean']:.4f}")
            if "kl" in logs:
                log_items.append(f"KL: {logs['kl']:.4f}")
            
            if log_items:
                print(f">>> [Step {step}] " + " | ".join(log_items))

    def on_save(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            print(f"\n[Checkpoint] RL Model saved at step {state.global_step}")

def train(
    # model/data params
    model_path: str = "",
    seed: int = 42,
    train_file: str = "",
    eval_file: str = "",
    info_file: str = "",
    category: str = "",
    
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    
    # training hyperparams
    output_dir: str = "",
    train_batch_size: int = 32,
    eval_batch_size: int = 32,
    gradient_accumulation_steps: int = 1,
    temperature: float = 1.0,
    add_gt: bool = False,
    eval_step: float = 0.199,
    num_generations: int = 16,
    num_train_epochs: int = 1,
    learning_rate: float = 1e-5, 
    muon_learning_rate: float = 3e-5,
    muon_weight_decay: float = 1e-5,
    adam_weight_decay: float = 0.0,
    beta: float = 0.1,
    beam_search: bool = False,
    test_during_training: bool = True,
    dynamic_sampling: bool = False,
    mask_all_zero: bool = False,
    sync_ref_model: bool = False,
    test_beam: int = 20,
    reward_type: str = "rule",
    sample_train: bool = False,
    ada_path: str = "",
    cf_path: str = "",
    sid_index_path: str = "",
    item_meta_path: str = "",
    dapo: bool = False,
    gspo: bool = False,
):
    torch.backends.cuda.enable_flash_sdp(False)  
    torch.backends.cuda.enable_mem_efficient_sdp(False)
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
    
    with open(info_file, 'r') as f:
        info = f.readlines()
        item_name = [_.split('\t')[0].strip() for _ in info]
        item2id = {name: i for i, name in enumerate(item_name)}

    sample = -1
    train_datasets = []
    train_datasets.append(SidDataset(train_file, category=category_dict[category], sample=sample))
    train_datasets.append(RLTitle2SidDataset(item_file=item_meta_path, index_file=sid_index_path, category=category_dict[category], sample=sample))
    train_datasets.append(RLSeqTitle2SidDataset(train_file, category=category_dict[category], sample=10000))
    train_data = ConcatDataset(train_datasets)
    eval_data = SidDataset(eval_file, category=category_dict[category], sample=sample)

    train_dataset = Dataset.from_dict({k : [elm[k] for elm in train_data] for k in train_data[0].keys()})
    train_dataset = train_dataset.shuffle(seed=seed) 
    if sample_train and "sft" in model_path:
        train_dataset = train_dataset.select(range(int(0.2 * len(train_dataset)), len(train_dataset)))
    eval_dataset = Dataset.from_dict({k : [elm[k] for elm in eval_data] for k in eval_data[0].keys()})
    eval_dataset = eval_dataset.shuffle(seed=seed)
    
    prompt2history = {}
    history2target = {}
    
    for dataset in train_datasets:
        if hasattr(dataset, 'prompt2history'):
            prompt2history.update(dataset.prompt2history)
        if hasattr(dataset, 'history2target'):
            history2target.update(dataset.history2target)
    
    if hasattr(eval_data, 'prompt2history'):
        prompt2history.update(eval_data.prompt2history)
    if hasattr(eval_data, 'history2target'):
        history2target.update(eval_data.history2target)

    print("train_dataset: ", train_dataset)
    print("eval_dataset: ", eval_dataset)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Allow Reward model to load on a specified device
    device = torch.device("cuda:7" if torch.cuda.device_count() > 7 else "cuda:0")
    
    len_seq = 10 
    item_num = len(item_name)

    if reward_type == "sasrec":
        model = SASRec(64, item_num, len_seq, 0.1, device) 
        model.to(device)
        model.load_state_dict(torch.load(cf_path))
        model.eval()
    if reward_type == "semantic":
        with open(ada_path, "rb") as f:
            item_ada_embd = pickle.load(f)
        item_ada_embd = torch.tensor(item_ada_embd).to(device)

    ndcg_rewards = [-1.0/math.log2(i+2) for i in range(num_generations)]
    ndcg_rewards = [-elm/sum(ndcg_rewards) for elm in ndcg_rewards]

    def ndcg_rule_reward(prompts, completions):
        history = [prompt2history[prompt] for prompt in prompts]
        targets = [history2target[elm] for elm in history]
        repeat = num_generations
        rewards = []
        flag = False
        lis = []
        for i, completion in enumerate(completions):
            if completion.strip("\n\"") == targets[i].strip("\n\""):
                flag = True
                lis.append(0.0)
            else:
                lis.append(ndcg_rewards[i%num_generations])
            if (i+1)%num_generations == 0:
                if flag: rewards.extend(lis)
                else: rewards.extend([0.0] * num_generations)
                flag = False
                lis = []
        return rewards

    def rule_reward(prompts, completions):
        history = [prompt2history[prompt] for prompt in prompts]
        targets = [history2target[elm] for elm in history]
        rewards = []
        for i, completion in enumerate(completions):
            if completion.strip("\n\" ") == targets[i].strip("\n\" "):
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        return rewards
    
    def semantic_reward(prompts, completions):
        history = [prompt2history[prompt] for prompt in prompts]
        targets = [history2target[elm] for elm in history]
        target_ids = [item2id[elm.strip("\"\n")] for elm in targets]
        completion_ids = []
        for i, completion in enumerate(completions):
            clean_c = completion.strip("\"\n")
            if clean_c in item2id:
                completion_ids.append(item2id[clean_c])
            else:
                completion_ids.append(random.randint(0, item_num-1))
        
        rewards = torch.cosine_similarity(item_ada_embd[target_ids], item_ada_embd[completion_ids], dim=-1)
        return rewards

    def cf_reward(prompts, completions):
        history = [prompt2history[prompt] for prompt in prompts]
        history_list = [elm.split("::") for elm in history]
        pred_ids = []
        for i, elm in enumerate(completions):
            elm = elm.strip("\n\"")
            if elm not in item_name:
                pred_ids.append(random.randint(0, item_num-1))
            else:
                pred_ids.append(item2id[elm])
        
        len_lis = []
        history_ids = []
        for his in history_list:
            his = [item2id[elm] for elm in his]
            len_lis.append(len(his))
            if len(his) < len_seq: 
                his = his + [item_num] * (len_seq - len(his))
            else:
                his = his[-len_seq:] 
            history_ids.append(his)
        
        seq = torch.LongTensor(history_ids).to(device)
        pred = torch.LongTensor(pred_ids).to(device)    
        
        with torch.no_grad():
            predictions = model.forward_eval(seq, torch.tensor(np.array(len_lis)).to(device))
            scores = torch.gather(predictions, 1,  pred.view(-1, 1)).view(-1)
        return scores
    
    reward_fun = None
    if reward_type == "rule":
        reward_fun = rule_reward
    elif reward_type == "ranking":
        reward_fun = [rule_reward, ndcg_rule_reward]
    elif reward_type == "sasrec":
        reward_fun = cf_reward
    elif reward_type == "semantic":
        reward_fun = semantic_reward

    training_args = GRPOConfig(output_dir=output_dir,
                                save_steps=0.1,
                                save_total_limit=5,
                                eval_strategy="steps",
                                max_completion_length=128,
                                num_generations=num_generations,
                                temperature=temperature,
                                sync_ref_model=sync_ref_model,
                                per_device_eval_batch_size=eval_batch_size,
                                per_device_train_batch_size=train_batch_size,
                                gradient_accumulation_steps=gradient_accumulation_steps,  
                                eval_steps=eval_step, 
                                logging_steps=1, 
                                learning_rate=learning_rate,
                                beta=beta,
                                warmup_ratio=0.1,
                                max_grad_norm= 0.3,
                                num_train_epochs=num_train_epochs,
                                bf16=True,
                                optim="adamw_torch",
                                lr_scheduler_type="cosine", 
                                save_strategy="steps",
                                report_to="none", 
                                run_name=wandb_run_name,
                            )
    
    trainer = MuonReReTrainer(
        model=model_path, 
        base_model=model_path,
        muon_lr=muon_learning_rate,
        muon_wd=muon_weight_decay,
        adam_wd=adam_weight_decay,
        dapo=dapo,
        gspo=gspo,
        add_gt=add_gt,
        dynamic_sampling=dynamic_sampling,
        beam_search=beam_search,
        test_during_training=test_during_training,
        test_beam=test_beam,
        info_file=info_file,
        prompt2history=prompt2history,
        history2target=history2target,
        reward_funcs=reward_fun,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        callbacks=[OfflineRLPrinterCallback()]
    )
    
    print("\n" + "="*60)
    print("RL TRAINING START (Muon Enabled - External Implementation)")
    print("="*60)

    train_result = trainer.train()

    metrics = train_result.metrics
    total_flos = metrics.get("total_flos", 0)
    
    config = AutoConfig.from_pretrained(model_path)
    total_params = 1.54e9 
    
    avg_gen_len = 64
    total_samples_processed = len(train_dataset) * num_train_epochs
    estimated_flops = 6 * total_params * avg_gen_len * total_samples_processed
    
    if total_flos == 0: total_flos = estimated_flops
    flops_per_sample = total_flos / total_samples_processed if total_samples_processed > 0 else 0

    print("\n" + "="*50)
    print("RL TRAINING STATISTICS")
    print("="*50)
    print(f"Train Loss (Final): {metrics.get('train_loss', 'N/A')}")
    
    best_reward = -float('inf')
    for log in trainer.state.log_history:
        if "reward_mean" in log and log["reward_mean"] > best_reward:
            best_reward = log["reward_mean"]
            
    print(f"Best Reward (Mean): {best_reward:.4f}")
    print(f"Best Step: {trainer.state.global_step}") 
    print(f"FLOPs per Sample (Est): {flops_per_sample:.4e}")
    print("="*50 + "\n")

    trainer.save_model(output_dir)
    output_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
if __name__ == "__main__":
    Fire(train)