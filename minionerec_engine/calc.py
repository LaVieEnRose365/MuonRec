import os
import fire
import math
import json
import pandas as pd
import numpy as np
from tqdm import tqdm

def gao(path, item_path):
    if not isinstance(path, list):
        path = [path]
    
    if not os.path.exists(item_path) and not item_path.endswith(".txt"):
        item_path = f"{item_path}.txt"
        
    if not os.path.exists(item_path):
        raise FileNotFoundError(f"Cannot find item info file: {item_path}")
        
    with open(item_path, 'r', encoding='utf-8') as f:
        items = f.readlines()
        
    item_names = [_.split('\t')[0].strip() for _ in items]
    item_ids = [_ for _ in range(len(item_names))]
    item_dict = dict()
    for i in range(len(item_names)):
        if item_names[i] not in item_dict:
            item_dict[item_names[i]] = [item_ids[i]]
        else:   
            item_dict[item_names[i]].append(item_ids[i])

    topk_list = [1, 3, 5, 10, 20, 50]
    
    for p in path:
        if not os.path.exists(p):
            print(f"Warning: Result file {p} not found.")
            continue

        with open(p, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        text = [[_.strip("\"\n").strip() for _ in sample["predict"]] for sample in test_data]
        
        n_beam = -1
        CC = 0
        
        for index_outer, sample in tqdm(enumerate(text), total=len(text)):
            if n_beam == -1:
                n_beam = len(sample)
                valid_topk = [k for k in topk_list if k <= n_beam]
                ALLNDCG = np.zeros(len(valid_topk))
                ALLHR = np.zeros(len(valid_topk))
            
            output_val = test_data[index_outer]['output']
            if isinstance(output_val, list):
                target_item = output_val[0].strip("\"").strip(" ")
            else:
                target_item = str(output_val).strip(" \n\"")
                
            minID = 1000000
            for i in range(len(sample)):
                if sample[i] == target_item:
                    minID = i
                    break
            
            for k_idx, topk in enumerate(valid_topk):
                if minID < topk:
                    ALLNDCG[k_idx] = ALLNDCG[k_idx] + (1 / math.log(minID + 2))
                    ALLHR[k_idx] = ALLHR[k_idx] + 1
        
        print(f"\nResults for: {p}")
        print(f"Beam size: {n_beam}")
        print(f"NDCG:\t{ALLNDCG / len(text) / (1.0 / math.log(2))}")
        print(f"HR\t{ALLHR / len(text)}")

if __name__=='__main__':
    fire.Fire(gao)