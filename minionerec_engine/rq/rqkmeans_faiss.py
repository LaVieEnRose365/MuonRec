#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FAISS ResidualQuantizer + Sinkhorn-based Uniform Semantic Mapping
Standardized for MuonRec Engine.
"""

import argparse
import json
import os
import sys
import numpy as np
from tqdm import tqdm

# =====================================================
# Dynamically add project root to sys.path
# =====================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
# Current script is at: project_root/minionerec_engine/rq/
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Standard imports
import faiss

def pairwise_sq_dists_batch(X, C, C_norm2=None):
    if C_norm2 is None:
        C_norm2 = np.sum(C * C, axis=1)
    X_norm2 = np.sum(X * X, axis=1, keepdims=True)
    dots = X @ C.T
    return X_norm2 + C_norm2[None, :] - 2.0 * dots

def train_faiss_rq(data, num_levels=3, codebook_size=256, verbose=True):
    N, d = data.shape
    if verbose:
        print(f"Training FAISS ResidualQuantizer: N={N}, D={d}, Levels={num_levels}, Codebook={codebook_size}")

    nbits = int(np.log2(codebook_size))
    rq = faiss.ResidualQuantizer(d, num_levels, nbits)
    rq.train_type = faiss.ResidualQuantizer.Train_default
    rq.max_beam_size = 1

    rq.train(np.ascontiguousarray(data.astype(np.float32)))
    if verbose:
        print("  Training completed.\n")
    return rq

def encode_with_rq(rq, data, verbose=True):
    data = np.ascontiguousarray(data.astype(np.float32))
    if verbose:
        print(f"Encoding {data.shape[0]} vectors...")
    codes = rq.compute_codes(data)
    if codes.ndim == 1:
        codes = codes.reshape(-1, rq.M)
    codes = codes.astype(np.int32)
    if verbose:
        print(f"  Encoding done. Codes shape: {codes.shape}\n")
    return codes

def get_rq_codebooks(rq):
    M, d = rq.M, rq.d
    nbits_val = get_first_nbits(rq)
    K = 1 << nbits_val
    cb_flat = faiss.vector_to_array(rq.codebooks).astype(np.float32)
    return cb_flat.reshape(M, K, d)

def sinkhorn_balance_level(residuals, centroids, capacities=None, *,
                           iters=30, tau=None, verbose=True, seed=42):
    import ot # Python Optimal Transport
    rng = np.random.RandomState(seed)
    N, d = residuals.shape
    K = centroids.shape[0]

    if capacities is None:
        capacities = np.full(K, N // K, dtype=np.int64)
        capacities[: (N % K)] += 1
    
    a = np.ones(N) / N
    b = capacities / float(N)
    Cn2 = np.sum(centroids * centroids, axis=1)

    D_full = pairwise_sq_dists_batch(residuals, centroids, Cn2).astype(np.float64)
    
    if tau is None:
        # Simple heuristic for tau
        tau = 0.01 

    P = ot.sinkhorn(a, b, D_full, tau, numItermax=iters)

    # Deterministic assignment based on transport plan
    assign = np.argmax(P, axis=1).astype(np.int32)
    
    if verbose:
        unique, counts = np.unique(assign, return_counts=True)
        print(f"    Level balanced. Unique codes: {len(unique)}, Count range: [{counts.min()}, {counts.max()}]")

    return assign

def sinkhorn_uniform_mapping(rq, data, codes, iters=30, verbose=True, seed=42):
    codebooks = get_rq_codebooks(rq)
    N, M = codes.shape
    K = codebooks.shape[1]

    codes_bal = codes.copy()
    for l in range(M):
        if verbose:
            print(f"=== Applying Sinkhorn Uniform Mapping: Level {l+1}/{M} ===")
        
        # Compute residuals
        res = np.ascontiguousarray(data.astype(np.float32)).copy()
        for prev_l in range(l):
            res -= codebooks[prev_l][codes_bal[:, prev_l]]

        codes_bal[:, l] = sinkhorn_balance_level(
            res, codebooks[l], iters=iters, verbose=verbose, seed=seed+l
        )
    return codes_bal

def save_indices_json(codes, path):
    tpl = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]
    idx = {i: [tpl[j].format(int(c)) for j, c in enumerate(code)] for i, code in enumerate(codes)}
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(idx, f, indent=2)
    print(f"Indices saved to: {path}")

def get_first_nbits(rq):
    if isinstance(rq.nbits, int):
        return rq.nbits
    return int(faiss.vector_to_array(rq.nbits).ravel()[0])

def main():
    parser = argparse.ArgumentParser(description="FAISS-RQ + Sinkhorn uniform mapping")
    parser.add_argument("--dataset", default="Industrial_and_Scientific")
    parser.add_argument("--num_levels", type=int, default=3)
    parser.add_argument("--codebook_size", type=int, default=256)
    parser.add_argument("--uniform", action="store_true", help="Enable Sinkhorn uniform mapping")
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--root", default="./minionerec_engine/data/Amazon/index")
    args = parser.parse_args()

    # Align paths with MuonRec structure
    data_path = os.path.join(args.root, f"{args.dataset}.emb-qwen-td.npy")
    out_dir = os.path.join(args.root, "faiss_indices")
    os.makedirs(out_dir, exist_ok=True)
    
    out_json = os.path.join(out_dir, f"{args.dataset}.faiss-rq.index.json")

    print(f"Loading embeddings from: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Embedding file not found: {data_path}")
        
    data = np.load(data_path)

    # 1. Train RQ
    rq = train_faiss_rq(data, args.num_levels, args.codebook_size)
    codes_raw = encode_with_rq(rq, data)

    # 2. Optional Balancing
    if args.uniform:
        codes_final = sinkhorn_uniform_mapping(rq, data, codes_raw, iters=args.iters)
    else:
        codes_final = codes_raw

    # 3. Save
    save_indices_json(codes_final, out_json)

if __name__ == "__main__":
    main()