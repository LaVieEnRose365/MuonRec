import collections
import json
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys
import fire

# =====================================================
# Dynamically add project root to sys.path
# to ensure absolute imports work correctly
# =====================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
# Current script is at: project_root/minionerec_engine/rq/models/
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Standardize imports from the engine
from minionerec_engine.rq.datasets import EmbDataset
from minionerec_engine.rq.models.rqvae import RQVAE

def check_collision(all_indices_str):
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    return tot_item == tot_indice

def get_indices_count(all_indices_str):
    indices_count = collections.defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count

def get_collision_item(all_indices_str):
    index2id = {}
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)
    collision_item_groups = []
    for index in index2id:
        if len(index2id[index]) > 1:
            collision_item_groups.append(index2id[index])
    return collision_item_groups

def generate(
    dataset: str = "Industrial_and_Scientific",
    ckpt_path: str = "",
    device_id: int = 0,
    output_dir: str = "./minionerec_engine/data/Amazon/index"
):
    """
    Main function to generate item indices based on a trained RQ-VAE model.
    """
    if not ckpt_path:
        raise ValueError("Please provide a valid --ckpt_path for the RQ-VAE model.")

    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    args = ckpt["args"]
    state_dict = ckpt["state_dict"]

    # Load data from the centralized engine data directory
    data = EmbDataset(args.data_path)

    # Initialize Model
    model = RQVAE(in_dim=data.dim,
                  num_emb_list=args.num_emb_list,
                  e_dim=args.e_dim,
                  layers=args.layers,
                  dropout_prob=args.dropout_prob,
                  bn=args.bn,
                  loss_type=args.loss_type,
                  quant_loss_weight=args.quant_loss_weight,
                  kmeans_init=args.kmeans_init,
                  kmeans_iters=args.kmeans_iters,
                  sk_epsilons=args.sk_epsilons,
                  sk_iters=args.sk_iters)

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(f"Loaded RQ-VAE model from {ckpt_path}")

    data_loader = DataLoader(data, num_workers=args.num_workers,
                             batch_size=64, shuffle=False,
                             pin_memory=True)

    all_indices = []
    all_indices_str = []
    prefix = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]

    # First pass: Get initial indices
    for d in tqdm(data_loader, desc="Initial Indexing"):
        d = d.to(device)
        indices = model.get_indices(d, use_sk=False)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for index in indices:
            code = [prefix[i].format(int(ind)) for i, ind in enumerate(index)]
            all_indices.append(code)
            all_indices_str.append(str(code))

    all_indices = np.array(all_indices)
    all_indices_str = np.array(all_indices_str)

    # Resolve collisions using Sinkhorn-Knopp if enabled in the model config
    for vq in model.rq.vq_layers[:-1]:
        vq.sk_epsilon = 0.0
    if model.rq.vq_layers[-1].sk_epsilon == 0.0:
        model.rq.vq_layers[-1].sk_epsilon = 0.003

    tt = 0
    while tt < 20:
        if check_collision(all_indices_str):
            break

        collision_item_groups = get_collision_item(all_indices_str)
        print(f"Round {tt}: {len(collision_item_groups)} collision groups found.")
        
        for collision_items in collision_item_groups:
            d = data[collision_items].to(device)
            indices = model.get_indices(d, use_sk=True)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for item, index in zip(collision_items, indices):
                code = [prefix[i].format(int(ind)) for i, ind in enumerate(index)]
                all_indices[item] = code
                all_indices_str[item] = str(code)
        tt += 1

    # Statistics
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    print("\n" + "="*30)
    print(f"Indexing Complete for {dataset}")
    print(f"Total Items: {tot_item}")
    print(f"Collision Rate: {(tot_item-tot_indice)/tot_item:.4%}")
    print("="*30)

    # Save to the centralized index directory
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{dataset}.index.json")
    
    all_indices_dict = {item: list(indices) for item, indices in enumerate(all_indices.tolist())}
    with open(output_file, 'w') as fp:
        json.dump(all_indices_dict, fp)
    
    print(f"Indices saved to: {output_file}")

if __name__ == '__main__':
    fire.Fire(generate)