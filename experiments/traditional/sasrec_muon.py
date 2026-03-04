import numpy as np
import pandas as pd
import argparse
import torch
from torch import nn
import torch.nn.functional as F
import os
import sys
import logging
import time as Time
from collections import Counter
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import random
import json
import copy
import ast
import wandb
from transformers import set_seed
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from Muon.muon import SingleDeviceMuonWithAuxAdam, MuonWithAuxAdam

try:
    from SASRecModules_ori import *
except ImportError:
    print("Warning: SASRecModules_ori not found. Assuming MultiHeadAttention/PositionwiseFeedForward are defined elsewhere or this is a dry run.")

logging.getLogger().setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description="Run supervised GRU/SASRec/Caser with Muon Optimizer.")
    
    parser.add_argument('--epoch', type=int, default=500, help='Number of max epochs.')
    parser.add_argument('--data', nargs='?', default='Industrial_and_Scientific', help='Dataset name')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=64, help='Embedding size.')
    parser.add_argument('--num_filters', type=int, default=16, help='num_filters')
    parser.add_argument('--filter_sizes', nargs='?', default='[2,3,4]', help='filter_sizes')
    parser.add_argument('--r_click', type=float, default=0.2, help='reward for click')
    parser.add_argument('--r_buy', type=float, default=1.0, help='reward for purchase')
    
    # Optimizer Hyperparameters
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate for Adam parts (Embeddings, Bias, LN).')
    parser.add_argument('--muon_lr', type=float, default=0.02, help='Learning rate for Muon parts (Matrix weights). Default is 0.02.')
    parser.add_argument('--l2_decay', type=float, default=1e-5, help='l2 loss reg coef.')
    parser.add_argument('--muon_wd', type=float, default=0.0, help='Weight decay for Muon parts. Try 0.01 if needed.')
    
    parser.add_argument('--save_flag', type=int, default=1, help='0: Disable saver, 1: Activate')
    parser.add_argument('--cuda', type=int, default=1, help='cuda device ID (e.g., 0, 1, 2).')
    parser.add_argument('--alpha', type=float, default=0, help='dro alpha.')
    parser.add_argument('--beta', type=float, default=1.0, help='for robust radius')
    parser.add_argument("--model", type=str, default="SASRec", help='Model name: SASRec, GRU, Caser')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='dropout')
    parser.add_argument("--early_stop", type=int, default=10, help='early stop epoch')
    parser.add_argument("--eval_num", type=int, default=1, help='evaluate every eval_num epoch')
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--result_json_path", type=str, default="./result_temp/temp.json")
    parser.add_argument("--sample_num", type=int, default=65536)
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--loss_type", type=str, default="bce")
    
    return parser.parse_args()

# ================= MODEL DEFINITIONS =================

class GRU(nn.Module):
    def __init__(self, hidden_size, item_num, state_size, gru_layers=1):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.item_num = item_num
        self.state_size = state_size
        self.item_embeddings = nn.Embedding(num_embeddings=item_num + 1, embedding_dim=self.hidden_size)
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)
        self.gru = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=gru_layers, batch_first=True)
        self.s_fc = nn.Linear(self.hidden_size, self.item_num)

    def forward(self, states, len_states):
        emb = self.item_embeddings(states)
        emb_packed = torch.nn.utils.rnn.pack_padded_sequence(emb, len_states.cpu(), batch_first=True, enforce_sorted=False)
        emb_packed, hidden = self.gru(emb_packed)
        hidden = hidden.view(-1, hidden.shape[2])
        supervised_output = self.s_fc(hidden)
        return supervised_output

    def forward_eval(self, states, len_states):
        return self.forward(states, len_states)

class Caser(nn.Module):
    def __init__(self, hidden_size, item_num, state_size, num_filters, filter_sizes, dropout_rate):
        super(Caser, self).__init__()
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.state_size = state_size
        self.filter_sizes = eval(filter_sizes)
        self.num_filters = num_filters
        self.dropout_rate = dropout_rate
        self.item_embeddings = nn.Embedding(num_embeddings=item_num + 1, embedding_dim=self.hidden_size)
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)
        
        # Note: Muon automatically handles 4D weights (Conv2d) by flattening them to 2D for orthogonalization
        self.horizontal_cnn = nn.ModuleList([nn.Conv2d(1, self.num_filters, (i, self.hidden_size)) for i in self.filter_sizes])
        for cnn in self.horizontal_cnn:
            nn.init.xavier_normal_(cnn.weight)
            nn.init.constant_(cnn.bias, 0.1)
        self.vertical_cnn = nn.Conv2d(1, 1, (self.state_size, 1))
        nn.init.xavier_normal_(self.vertical_cnn.weight)
        nn.init.constant_(self.vertical_cnn.bias, 0.1)
        self.num_filters_total = self.num_filters * len(self.filter_sizes)
        final_dim = self.hidden_size + self.num_filters_total
        self.s_fc = nn.Linear(final_dim, item_num)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, states, len_states):
        input_emb = self.item_embeddings(states)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1)
        input_emb *= mask
        input_emb = input_emb.unsqueeze(1)
        pooled_outputs = []
        for cnn in self.horizontal_cnn:
            h_out = nn.functional.relu(cnn(input_emb))
            h_out = h_out.squeeze()
            p_out = nn.functional.max_pool1d(h_out, h_out.shape[2])
            pooled_outputs.append(p_out)
        h_pool = torch.cat(pooled_outputs, 1)
        h_pool_flat = h_pool.view(-1, self.num_filters_total)
        v_out = nn.functional.relu(self.vertical_cnn(input_emb))
        v_flat = v_out.view(-1, self.hidden_size)
        out = torch.cat([h_pool_flat, v_flat], 1)
        out = self.dropout(out)
        supervised_output = self.s_fc(out)
        return supervised_output

    def forward_eval(self, states, len_states):
        return self.forward(states, len_states)

class SASRec(nn.Module):
    def __init__(self, hidden_size, item_num, state_size, dropout, device, num_heads=1):
        super(SASRec, self).__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.item_embeddings = nn.Embedding(num_embeddings=item_num + 1, embedding_dim=hidden_size)
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)
        self.positional_embeddings = nn.Embedding(num_embeddings=state_size, embedding_dim=hidden_size)
        self.emb_dropout = nn.Dropout(dropout)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.ln_3 = nn.LayerNorm(hidden_size)
        self.mh_attn = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size, hidden_size, dropout)
        self.s_fc = nn.Linear(hidden_size, item_num)

    def forward(self, states, len_states):
        inputs_emb = self.item_embeddings(states)
        inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(self.device)
        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        indices = (len_states - 1).view(-1, 1, 1).repeat(1, 1, self.hidden_size)
        state_hidden = torch.gather(ff_out, 1, indices)
        supervised_output = self.s_fc(state_hidden).squeeze()
        return supervised_output

    def forward_eval(self, states, len_states):
        return self.forward(states, len_states)

# ================= UTILS & EVALUATION =================

def evaluate_games(model, test_file, device, topk, data_dir, item_num, seq_size, save_logits=False, eval_type="test"):
    
    def calculate_hit_games_cuda(prediction, topk_list, target, hit_all, ndcg_all):
        rank_list = (prediction.shape[1] - 1 - torch.argsort(torch.argsort(prediction)))
        target_rank = torch.gather(rank_list, 1, target.view(-1, 1)).view(-1).clone()
        ndcg_temp_full = 1 / torch.log2(target_rank + 2)
        
        for i, top_k in enumerate(topk_list):
            mask = (target_rank < top_k).float()
            recall_temp = mask.sum()
            ndcg_temp = (ndcg_temp_full * mask).sum()
            hit_all[i] += recall_temp.cpu().item()
            ndcg_all[i] += ndcg_temp.cpu().item()
        return hit_all, ndcg_all

    eval_seqs = pd.read_csv(os.path.join(data_dir, test_file))
    eval_seqs = eval_seqs[['history_item_id', 'item_id']]
    eval_seqs = eval_seqs.rename(columns={'history_item_id': 'seq', 'item_id': 'next'})
    eval_seqs['seq'] = eval_seqs['seq'].apply(ast.literal_eval)
    eval_seqs['len_seq'] = eval_seqs['seq'].apply(lambda x: len(x))
    eval_seqs['seq'] = eval_seqs['seq'].apply(lambda x: x + [item_num] * (seq_size - len(x)))

    batch_size = 1024
    hit_all = [0] * len(topk)
    ndcg_all = [0] * len(topk)
    
    total_samples = len(eval_seqs)
    total_batch_num = int(np.ceil(total_samples / batch_size))
    
    sasrec_logits = []
    
    model.eval()
    with torch.no_grad():
        for i in range(total_batch_num):
            begin = i * batch_size
            end = min((i + 1) * batch_size, total_samples)
            batch = eval_seqs[begin:end]
            
            seq = torch.LongTensor(list(batch['seq'])).to(device)
            target = torch.LongTensor(list(batch['next'])).to(device)
            len_seq = torch.tensor(list(batch['len_seq'])).to(device)

            if isinstance(model, GRU):
                len_seq_cpu = len_seq.cpu()
                prediction = model.forward_eval(seq, len_seq_cpu)
            else:
                prediction = model.forward_eval(seq, len_seq)
            
            if save_logits:
                sasrec_logits.append(prediction)
            
            hit_all, ndcg_all = calculate_hit_games_cuda(prediction, topk, target, hit_all, ndcg_all)

    if save_logits and isinstance(model, SASRec):
        sasrec_logits = torch.cat(sasrec_logits, dim=0)
    
    hr_list = [h / total_samples for h in hit_all]
    ndcg_list = [n / total_samples for n in ndcg_all]
    
    return ndcg_list[-1], hr_list, ndcg_list

def calculate_valid_loss(model, dataloader, device, loss_type, model_loss, item_num, args):
    model.eval()
    total_loss = 0
    steps = 0
    with torch.no_grad():
        for seq, len_seq, target in dataloader:
            target_neg = []
            target_np = target.numpy()
            for idx in range(len(len_seq)):
                neg = np.random.randint(item_num)
                while neg == target_np[idx]:
                    neg = np.random.randint(item_num)
                target_neg.append(neg)
            
            seq = seq.to(device)
            target = target.to(device)
            len_seq = len_seq.to(device)
            target_neg = torch.LongTensor(target_neg).to(device)

            if isinstance(model, GRU):
                model_output = model(seq, len_seq.cpu())
            else:
                model_output = model(seq, len_seq)

            target = target.view((-1, 1))
            target_neg = target_neg.view((-1, 1))

            if loss_type == "bce":
                pos_scores = torch.gather(model_output, 1, target)
                neg_scores = torch.gather(model_output, 1, target_neg)
                scores = torch.cat((pos_scores, neg_scores), 0)
                labels = torch.cat((torch.ones_like(pos_scores), torch.zeros_like(neg_scores)), 0)
                loss = model_loss(scores, labels)
            elif loss_type == "ce":
                loss = model_loss(model_output, target.squeeze(-1).long())
            
            total_loss += loss.item()
            steps += 1
            
    return total_loss / steps if steps > 0 else 0.0

def calcu_propensity_score(buffer, item_num):
    items = list(buffer['next'])
    freq = Counter(items)
    for i in range(item_num):
        if i not in freq: freq[i] = 0
    pop = np.array([freq[i] for i in range(item_num)])
    ps = pop + 1
    ps = ps / np.sum(ps)
    return np.power(ps, 0.05)

class RecDataset(Dataset):
    def __init__(self, data_df):
        self.data = data_df
    def __getitem__(self, i):
        temp = self.data.iloc[i]
        seq = torch.tensor(temp['seq'])
        len_seq = torch.tensor(temp['len_seq'])
        next_item = torch.tensor(temp['next'])
        return seq, len_seq, next_item
    def __len__(self):
        return len(self.data)

# ================= MAIN =================

def main(topk, data_file_train, data_file_test, data_file_valid, data_dirs):
    if not args.debug:
        wandb.init(project="Rec", name=f"{args.data}_{args.model}_Muon", mode="disabled")
    else:
        wandb.init(mode="disabled")

    with open(os.path.join(data_dirs['info'], data_file_info), 'r') as f:
        lines = f.readlines()
        item_num = len(lines)
    
    print(f"Detected Item Num: {item_num}")
    seq_size = 10 

    # Initialize Model
    if args.model=='SASRec':
        model = SASRec(args.hidden_factor, item_num, seq_size, args.dropout_rate, device)
    elif args.model=="GRU":
        model = GRU(args.hidden_factor, item_num, seq_size)
    elif args.model=="Caser":
        model = Caser(args.hidden_factor, item_num, seq_size, args.num_filters, args.filter_sizes, args.dropout_rate)

    model.to(device)
    
    # =========================================================================
    # Parameter Grouping and Initialization
    # =========================================================================
    muon_params = []
    adam_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        
        # Embedding Layers (Lookup Tables) -> Adam
        # Embeddings in RecSys typically do not require orthogonalization and have sparse updates.
        if 'embedding' in name.lower():
            adam_params.append(p)
        
        # Low-dimensional parameters (Bias, LayerNorm weight, 1D vectors) -> Adam
        # 1D vectors cannot be matrix-decomposed.
        elif p.ndim < 2:
            adam_params.append(p)
            
        # Core Compute Layers (Linear, GRU Weights, Conv Kernels) -> Muon
        # SASRec's Attention/FFN weights (2D)
        # GRU's weight_ih, weight_hh (2D)
        # Caser's Conv2d Filters (4D) -> Muon internal logic handles flattening.
        else:
            muon_params.append(p)

    print(f"Total Params: {len(list(model.parameters()))}")
    print(f"Muon Params (Matrix/Conv): {len(muon_params)}")
    print(f"Adam Params (Emb/Bias/LN): {len(adam_params)}")

    # Build parameter groups list
    # SingleDeviceMuonWithAuxAdam reads the 'use_muon' key to determine the update strategy.
    param_groups = [
        {
            'params': muon_params, 
            'use_muon': True, 
            'lr': args.muon_lr,
            'momentum': 0.95,
            'weight_decay': args.muon_wd
        },
        {
            'params': adam_params, 
            'use_muon': False, 
            'lr': args.lr,
            'weight_decay': args.l2_decay
        }
    ]

    # Initialize Hybrid Optimizer 
    optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
    
    # Scheduler 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-6)

    if args.loss_type == "bce":
        model_loss = nn.BCEWithLogitsLoss()
    elif args.loss_type == "ce":
        model_loss = nn.CrossEntropyLoss()

    # Data Loading
    def prepare_data(fname, folder):
        df = pd.read_csv(os.path.join(folder, fname))
        df = df[['history_item_id', 'item_id']].rename(columns={'history_item_id': 'seq', 'item_id': 'next'})
        df['seq'] = df['seq'].apply(ast.literal_eval)
        df['len_seq'] = df['seq'].apply(len)
        df['seq'] = df['seq'].apply(lambda x: x + [item_num] * (seq_size - len(x)))
        return df

    print("Loading Training Data...")
    train_df = prepare_data(data_file_train, data_dirs['train'])
    train_dataset = RecDataset(train_df)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    print("Loading Validation Data...")
    valid_df = prepare_data(data_file_valid, data_dirs['valid'])
    valid_dataset = RecDataset(valid_df)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    best_epoch = 0
    ndcg_max = 0
    early_stop_cnt = 0
    best_model = None
    
    final_best_metrics = {
        "train_loss": 0.0,
        "valid_loss": 0.0,
        "hr_list": [],
        "ndcg_list": []
    }

    print("Start Training...")
    for i in range(args.epoch):
        model.train()
        epoch_losses = []
        
        # Train Step
        for seq, len_seq, target in tqdm(train_loader, desc=f"Epoch {i+1}/{args.epoch}", leave=False):
            optimizer.zero_grad()
            
            target_neg = []
            target_np = target.numpy()
            for idx in range(len(len_seq)):
                neg = np.random.randint(item_num)
                while neg == target_np[idx]:
                    neg = np.random.randint(item_num)
                target_neg.append(neg)
            
            seq = seq.to(device)
            target = target.to(device)
            len_seq = len_seq.to(device)
            target_neg = torch.LongTensor(target_neg).to(device)
            
            if isinstance(model, GRU):
                model_output = model(seq, len_seq.cpu())
            else:
                model_output = model(seq, len_seq)

            target = target.view((-1, 1))
            target_neg = target_neg.view((-1, 1))

            pos_scores = torch.gather(model_output, 1, target)
            neg_scores = torch.gather(model_output, 1, target_neg)
            
            scores = torch.cat((pos_scores, neg_scores), 0)
            labels = torch.cat((torch.ones((len(len_seq), 1)), torch.zeros((len(len_seq), 1))), 0).to(device)

            if args.loss_type == "bce":
                loss = model_loss(scores, labels)
            else:
                loss = model_loss(model_output, target.squeeze(-1).long())

            loss.backward()
            # The hybrid optimizer automatically handles Muon and Adam parameters separately.
            optimizer.step() 
            epoch_losses.append(loss.item())

        scheduler.step()
        
        avg_train_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0

        if (i + 1) % args.eval_num == 0:
            current_lrs = [group['lr'] for group in optimizer.param_groups]
            
            ndcg_last, val_hr, val_ndcg = evaluate_games(
                model, data_file_valid, device, topk, data_dirs['valid'], item_num, seq_size, eval_type="val"
            )
            current_valid_loss = calculate_valid_loss(model, valid_loader, device, args.loss_type, model_loss, item_num, args)
            
            print(f"Epoch {i+1}: LRs {current_lrs} | Train Loss {avg_train_loss:.4f} | Valid Loss {current_valid_loss:.4f} | Valid NDCG {ndcg_last:.4f}")

            if ndcg_last > ndcg_max:
                ndcg_max = ndcg_last
                best_epoch = i
                early_stop_cnt = 0
                best_model = copy.deepcopy(model)
                final_best_metrics["train_loss"] = avg_train_loss
                final_best_metrics["valid_loss"] = current_valid_loss
                final_best_metrics["hr_list"] = val_hr
                final_best_metrics["ndcg_list"] = val_ndcg
            else:
                early_stop_cnt += 1
                if early_stop_cnt > args.early_stop:
                    print(f"Early stop at epoch {i+1}")
                    break
    
    return best_model, final_best_metrics, item_num, seq_size

if __name__ == '__main__':
    args = parse_args()
    topk = [1, 3, 5, 10, 20]

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    set_seed(args.seed, deterministic=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Path Setup (Adapted to new minionerec_engine architecture)
    amazon_data_dir = os.path.join(project_root, 'minionerec_engine', 'data', 'Amazon')
    data_dirs = {
        'train': os.path.join(amazon_data_dir, 'train'),
        'test': os.path.join(amazon_data_dir, 'test'),
        'valid': os.path.join(amazon_data_dir, 'valid'),
        'info': os.path.join(amazon_data_dir, 'info')
    }
    
    try:
        data_file_train = [f for f in os.listdir(data_dirs['train']) if args.data in f and f.endswith('.csv')][0]
        data_file_test = [f for f in os.listdir(data_dirs['test']) if args.data in f and f.endswith('.csv')][0]
        data_file_valid = [f for f in os.listdir(data_dirs['valid']) if args.data in f and f.endswith('.csv')][0]
        data_file_info = [f for f in os.listdir(data_dirs['info']) if args.data in f and f.endswith('.txt')][0]
    except IndexError:
        print("Error: Could not find dataset files containing name '{}'".format(args.data))
        exit(1)

    best_model, best_metrics, item_num, seq_size = main(topk, data_file_train, data_file_test, data_file_valid, data_dirs)

    print("\nCalculating Final Test Metrics...")
    _, test_hr, test_ndcg = evaluate_games(
        best_model, data_file_test, device, topk, data_dirs['test'], item_num, seq_size, save_logits=True, eval_type="test"
    )

    print("\n" + "="*20 + " RESULT " + "="*20)
    print(f"Train loss: {best_metrics['train_loss']:.6f}")
    print(f"Valid loss: {best_metrics['valid_loss']:.6f}")
    
    print(f"recall@1: {test_hr[0]:.6f}")
    print(f"recall@3: {test_hr[1]:.6f}")
    print(f"recall@5: {test_hr[2]:.6f}")
    print(f"recall@10: {test_hr[3]:.6f}")
    
    print(f"NDCG@3: {test_ndcg[1]:.6f}")
    print(f"NDCG@5: {test_ndcg[2]:.6f}")
    print(f"NDCG@10: {test_ndcg[3]:.6f}")
    print("="*48 + "\n")

    output_dir = os.path.dirname(args.result_json_path)
    os.makedirs(output_dir, exist_ok=True)

    result_dict = {
        "config": vars(args),
        "train_loss": best_metrics['train_loss'],
        "valid_loss": best_metrics['valid_loss'],
        "test_metrics": {
            f"HR@{k}": v for k, v in zip(topk, test_hr)
        }
    }
    result_dict["test_metrics"].update({f"NDCG@{k}": v for k, v in zip(topk, test_ndcg)})
    
    with open(args.result_json_path, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, indent=4)
    print(f"Saved results to: {args.result_json_path}")
    
    model_filename = f"best_model_{args.data}_{args.model}_Muon.pth"
    model_save_path = os.path.join(output_dir, model_filename)
    torch.save(best_model.state_dict(), model_save_path)
    print(f"Saved model to:   {model_save_path}")