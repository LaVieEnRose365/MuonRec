import argparse
import random
import torch
import numpy as np
import logging
import os
import sys
from torch.utils.data import DataLoader

# =====================================================
# Dynamically add project root to sys.path
# =====================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
# Current script is at: project_root/minionerec_engine/rq/
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Standardized imports from the engine
from minionerec_engine.rq.datasets import EmbDataset
from minionerec_engine.rq.models.rqvae import RQVAE
from minionerec_engine.rq.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="RQ-VAE Training for MuonRec")

    # Training Hyperparameters
    parser.add_argument('--dataset', type=str, default='Industrial_and_Scientific')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=5000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--eval_step', type=int, default=50, help='evaluation frequency')
    parser.add_argument('--learner', type=str, default="AdamW", help='optimizer type')
    parser.add_argument('--lr_scheduler_type', type=str, default="constant")
    parser.add_argument('--warmup_epochs', type=int, default=50)
    
    # Data Paths - Aligned with the new MuonRec structure
    parser.add_argument("--data_path", type=str,
                        default="./minionerec_engine/data/Amazon/index/Industrial_and_Scientific.emb-qwen-td.npy",
                        help="Path to pre-computed item embeddings.")
    parser.add_argument("--ckpt_dir", type=str, default="./outputs/rqvae", 
                        help="Directory to save the best models.")

    # Model Configuration
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--dropout_prob", type=float, default=0.0)
    parser.add_argument("--bn", type=bool, default=False, help="Use BatchNorm")
    parser.add_argument("--loss_type", type=str, default="mse", help="mse or l1")
    parser.add_argument("--kmeans_init", type=bool, default=True)
    parser.add_argument("--kmeans_iters", type=int, default=100)
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.0, 0.0, 0.0], help="Sinkhorn epsilons")
    parser.add_argument("--sk_iters", type=int, default=50)

    # RQ-VAE Specifics
    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[256, 256, 256], help='Codebook sizes per layer')
    parser.add_argument('--e_dim', type=int, default=32, help='Quantization embedding size')
    parser.add_argument('--quant_loss_weight', type=float, default=1.0, help='Commitment loss weight')
    parser.add_argument("--beta", type=float, default=0.25, help="Beta for commitment loss")
    parser.add_argument('--layers', type=int, nargs='+', default=[2048, 1024, 512, 256, 128, 64], help='MLP hidden layers')

    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--save_limit', type=int, default=5)

    return parser.parse_args()

if __name__ == '__main__':
    # Fix the random seed for reproducibility
    seed = 2024
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parse_args()
    
    # Auto-adjust data path based on dataset name if not default
    if args.dataset != 'Industrial_and_Scientific' and "Industrial_and_Scientific" in args.data_path:
        args.data_path = f"./minionerec_engine/data/Amazon/index/{args.dataset}.emb-qwen-td.npy"

    print("="*50)
    print(f"Starting RQ-VAE Training for: {args.dataset}")
    print(args)
    print("="*50)

    logging.basicConfig(level=logging.INFO)

    # 1. Build Dataset and Model
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Embedding file not found: {args.data_path}. Run text2emb script first.")
        
    data = EmbDataset(args.data_path)
    model = RQVAE(in_dim=data.dim,
                  num_emb_list=args.num_emb_list,
                  e_dim=args.e_dim,
                  layers=args.layers,
                  dropout_prob=args.dropout_prob,
                  bn=args.bn,
                  loss_type=args.loss_type,
                  quant_loss_weight=args.quant_loss_weight,
                  beta=args.beta,
                  kmeans_init=args.kmeans_init,
                  kmeans_iters=args.kmeans_iters,
                  sk_epsilons=args.sk_epsilons,
                  sk_iters=args.sk_iters)
    
    print(model)

    # 2. Data Loader
    data_loader = DataLoader(data, num_workers=args.num_workers,
                             batch_size=args.batch_size, shuffle=True,
                             pin_memory=True)

    # 3. Setup Trainer
    # Ensure dataset-specific output folder exists
    args.ckpt_dir = os.path.join(args.ckpt_dir, args.dataset)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    
    trainer = Trainer(args, model, len(data_loader))
    
    # 4. Train
    print("Beginning model fitting...")
    best_loss, best_collision_rate = trainer.fit(data_loader)

    print("\n" + "="*50)
    print(f"TRAINING COMPLETED FOR {args.dataset}")
    print(f"Best Reconstruction Loss: {best_loss:.6f}")
    print(f"Best Collision Rate:       {best_collision_rate:.4%}")
    print(f"Checkpoint saved to:       {args.ckpt_dir}")
    print("="*50)