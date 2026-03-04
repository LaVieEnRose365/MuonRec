#!/usr/bin/env python3
"""
Convert dataset to ReRe format with semantic IDs
Standardized for MuonRec Engine.
"""

import json
import pandas as pd
import numpy as np
import os
import sys
from typing import Dict, List, Any
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.abspath(os.path.join(current_dir, "../"))

def load_dataset(data_dir: str, dataset_name: str) -> Dict[str, Any]:
    """Load all dataset files"""
    data = {}
    
    item_json_path = os.path.join(data_dir, f'{dataset_name}.item.json')
    with open(item_json_path, 'r', encoding='utf-8') as f:
        data['items'] = json.load(f)
    
    index_json_path = os.path.join(data_dir, f'{dataset_name}.index.json')
    with open(index_json_path, 'r', encoding='utf-8') as f:
        data['item_to_semantic'] = json.load(f)
    
    # Load train/valid/test splits
    splits = {}
    for split in ['train', 'valid', 'test']:
        split_file = os.path.join(data_dir, f'{dataset_name}.{split}.inter')
        if os.path.exists(split_file):
            with open(split_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()[1:]  # Skip header
                splits[split] = [line.strip().split('\t') for line in lines if line.strip()]
    
    data['splits'] = splits
    return data

def semantic_tokens_to_id(tokens: List[str]) -> str:
    """Convert semantic tokens list to concatenated string with brackets preserved"""
    return ''.join(tokens)

def create_item_info_file(items: Dict[str, Dict], item_to_semantic: Dict[str, List], output_path: str):
    """Create item info file (sid -> title -> item_id mapping)"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for item_id, item_data in items.items():
            if item_id in item_to_semantic:
                semantic_tokens = item_to_semantic[item_id]
                semantic_id = semantic_tokens_to_id(semantic_tokens)
                item_title = item_data.get('title', f'Item_{item_id}')
                f.write(f"{semantic_id}\t{item_title}\t{item_id}\n")

def convert_interactions_to_csv(splits: Dict[str, List], items: Dict[str, Dict], 
                               item_to_semantic: Dict[str, List], output_dir: str, category: str,
                               max_valid_samples: int = None, max_test_samples: int = None, seed: int = 42,
                               keep_longest_only: bool = True):
    """Convert interaction data to ReRe CSV format using semantic IDs"""
    import random
    random.seed(seed)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name, split_data in splits.items():
        rows = []
        user_to_longest = {}
        
        for line in split_data:
            if len(line) != 3: continue
                
            user_id, item_sequence, target_item = line
            history_item_ids = [int(x) for x in item_sequence.split()] if item_sequence.strip() else []
            target_item_id = int(target_item)
            
            # Map item_ids to semantic_ids
            history_semantic_ids = []
            for item_id in history_item_ids:
                if str(item_id) in item_to_semantic:
                    history_semantic_ids.append(semantic_tokens_to_id(item_to_semantic[str(item_id)]))
            
            target_semantic_id = None
            if str(target_item_id) in item_to_semantic:
                target_semantic_id = semantic_tokens_to_id(item_to_semantic[str(target_item_id)])
            
            if target_semantic_id is None: continue
            
            history_item_titles = [items[str(tid)].get('title', f'Item_{tid}') for tid in history_item_ids if str(tid) in items]
            target_title = items.get(str(target_item_id), {}).get('title', f'Item_{target_item_id}')
            
            row = {
                'user_id': f'A{user_id}',
                'history_item_title': history_item_titles,
                'item_title': target_title,
                'history_item_id': history_item_ids,
                'item_id': target_item_id,
                'history_item_sid': history_semantic_ids,
                'item_sid': target_semantic_id
            }
            
            if split_name == 'train' and keep_longest_only:
                if user_id not in user_to_longest or len(history_item_ids) > len(user_to_longest[user_id]['history_item_id']):
                    user_to_longest[user_id] = row
            else:
                rows.append(row)
        
        if split_name == 'train' and keep_longest_only:
            rows = list(user_to_longest.values())
        
        # Sampling logic
        if split_name == 'valid' and max_valid_samples and len(rows) > max_valid_samples:
            rows = random.sample(rows, max_valid_samples)
        elif split_name == 'test' and max_test_samples and len(rows) > max_test_samples:
            rows = random.sample(rows, max_test_samples)
        
        if rows:
            df = pd.DataFrame(rows)
            output_file = os.path.join(output_dir, f'{category}_5_2016-10-2018-11.csv')
            df.to_csv(output_file, index=False)
            print(f"Created {split_name} file: {output_file} with {len(rows)} rows")

def main():
    parser = argparse.ArgumentParser(description='Convert dataset to ReRe format with semantic IDs')
    
    parser.add_argument('--data_dir', type=str, 
                       default=os.path.join(project_root, 'minionerec_engine/data/Amazon/Industrial_and_Scientific'),
                       help='Path to dataset directory')
    parser.add_argument('--dataset_name', type=str, default='Industrial_and_Scientific',
                       help='Dataset name')
    parser.add_argument('--output_dir', type=str,
                       default=os.path.join(project_root, 'minionerec_engine/data/Amazon'),
                       help='Output directory for processed data')
    parser.add_argument('--category', type=str, default=None)
    parser.add_argument('--max_valid_samples', type=int, default=None)
    parser.add_argument('--max_test_samples', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--keep_longest_only', action='store_true', default=False)
    
    args = parser.parse_args()
    if args.category is None: args.category = args.dataset_name
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Input directory {args.data_dir} not found.")
        return

    print(f"Loading {args.dataset_name} data from {args.data_dir}")
    dataset_data = load_dataset(args.data_dir, args.dataset_name)
    
    # Create subdirectories
    for subdir in ['train', 'valid', 'test', 'info']:
        os.makedirs(os.path.join(args.output_dir, subdir), exist_ok=True)
    
    # Create item info file
    info_file = os.path.join(args.output_dir, 'info', f'{args.category}_5_2016-10-2018-11.txt')
    create_item_info_file(dataset_data['items'], dataset_data['item_to_semantic'], info_file)
    
    # Convert splits
    for split_name in ['train', 'valid', 'test']:
        if split_name in dataset_data['splits']:
            split_output_dir = os.path.join(args.output_dir, split_name)
            convert_interactions_to_csv(
                {split_name: dataset_data['splits'][split_name]}, 
                dataset_data['items'],
                dataset_data['item_to_semantic'],
                split_output_dir,
                args.category,
                max_valid_samples=args.max_valid_samples,
                max_test_samples=args.max_test_samples,
                seed=args.seed,
                keep_longest_only=args.keep_longest_only
            )
    
    print(f"Conversion completed! Data saved to {args.output_dir}")

if __name__ == '__main__':
    main()