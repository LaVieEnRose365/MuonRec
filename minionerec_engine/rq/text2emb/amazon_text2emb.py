import argparse
import json
import os
import random
import re
import sys
import torch
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModel

# =====================================================
# Dynamically add project root to sys.path
# =====================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
# Current script is at: project_root/minionerec_engine/rq/text2emb/
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Standardize import from engine utilities
# Assuming utils.py is in minionerec_engine or rq/text2emb
try:
    from minionerec_engine.rq.text2emb.utils import clean_text, load_json
except ImportError:
    # Fallback for local execution
    from utils import clean_text, load_json

def load_data(args):
    """Load item feature data."""
    item2feature_path = os.path.join(args.root, args.dataset, f'{args.dataset}.item.json')
    if not os.path.exists(item2feature_path):
        raise FileNotFoundError(f"Item feature file not found: {item2feature_path}")
    
    print(f"Loading data from: {item2feature_path}")
    return load_json(item2feature_path)

def generate_text(item2feature, features):
    """Concatenate specified features into text strings."""
    item_text_list = []
    for item in item2feature:
        data = item2feature[item]
        text = []
        for meta_key in features:
            if meta_key in data:
                val = clean_text(data[meta_key])
                text.append(val.strip())
        item_text_list.append([int(item), text])
    return item_text_list

def generate_item_embedding(args, item_text_list, tokenizer, model, word_drop_ratio=-1):
    """Generate dense embeddings using a Pre-trained Language Model (Qwen)."""
    print(f"Generating Embeddings for {args.dataset} using {args.plm_name}...")

    items, texts = zip(*item_text_list)
    order_texts = [None] * len(items)
    for item, text in zip(items, texts):
        order_texts[item] = text
    
    embeddings = []
    start, batch_size = 0, 1
    
    model.eval()
    with torch.no_grad():
        while start < len(order_texts):
            if (start + 1) % 500 == 0:
                print(f"  Processed {start + 1}/{len(order_texts)} items...")
            
            field_texts = order_texts[start: start + batch_size]
            field_texts = zip(*field_texts)
    
            field_embeddings = []
            for sentences in field_texts:
                sentences = list(sentences)
                
                # Optional: Word dropout for augmentation
                if word_drop_ratio > 0:
                    new_sentences = []
                    for sent in sentences:
                        words = sent.split(' ')
                        filtered = [w for w in words if random.random() > word_drop_ratio]
                        new_sentences.append(' '.join(filtered))
                    sentences = new_sentences
                
                # Tokenization
                encoded = tokenizer(sentences, max_length=args.max_sent_len,
                                    truncation=True, return_tensors='pt', 
                                    padding="longest").to(args.device)
                
                # Model Inference
                outputs = model(input_ids=encoded.input_ids,
                                attention_mask=encoded.attention_mask)
    
                # Mean Pooling over last hidden states
                masked_output = outputs.last_hidden_state * encoded['attention_mask'].unsqueeze(-1)
                mean_output = masked_output.sum(dim=1) / encoded['attention_mask'].sum(dim=-1, keepdim=True)
                field_embeddings.append(mean_output.detach().cpu())
    
            # Average across different text fields (e.g., Title and Description)
            field_mean_emb = torch.stack(field_embeddings, dim=0).mean(dim=0)
            embeddings.append(field_mean_emb)
            start += batch_size

    embeddings = torch.cat(embeddings, dim=0).numpy()
    print(f"Final Embedding Shape: {embeddings.shape}")

    output_file = os.path.join(args.root, args.dataset, f"{args.dataset}.emb-{args.plm_name}.npy")
    np.save(output_file, embeddings)
    print(f"Embeddings saved to: {output_file}")

def load_qwen_model(model_path, device):
    """Initialize PLM and Tokenizer."""
    print(f"Initializing PLM from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path, 
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to(device)
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer, model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Industrial_and_Scientific')
    parser.add_argument('--root', type=str, default="./minionerec_engine/data/Amazon", help='Root data directory')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--plm_name', type=str, default='qwen2-1.5b')
    parser.add_argument('--plm_checkpoint', type=str, required=True, help='Local path to Qwen model')
    parser.add_argument('--max_sent_len', type=int, default=1024)
    parser.add_argument('--word_drop_ratio', type=float, default=-1)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    args.device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # 1. Preprocess and Load Data
    item_text_list = generate_text(load_data(args), ['title', 'description'])

    # 2. Load Model
    tokenizer, model = load_qwen_model(args.plm_checkpoint, args.device)

    # 3. Generate and Save
    generate_item_embedding(args, item_text_list, tokenizer, model, 
                            word_drop_ratio=args.word_drop_ratio)