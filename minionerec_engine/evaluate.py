import pandas as pd
import fire
import torch
import json
import os
import sys
import numpy as np
import random
from tqdm import tqdm
from transformers import GenerationConfig, AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
try:
    from minionerec_engine.data import EvalD3Dataset, EvalSidDataset
    from minionerec_engine.LogitProcessor import ConstrainedLogitsProcessor
except ImportError:
    from data import EvalD3Dataset, EvalSidDataset
    from LogitProcessor import ConstrainedLogitsProcessor

def get_hash(x):
    x = [str(_) for _ in x]
    return '-'.join(x)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def main(
    base_model: str = "",
    train_file: str = "",
    info_file: str = "",
    category: str = "",
    test_data_path: str = "",
    result_json_data: str = "",
    batch_size: int = 4,
    K: int = 0,
    seed: int = 42,
    length_penalty: float = 0.0,
    max_new_tokens: int = 256,
    num_beams: int = 50,
    device_id: int = 0 
):
    set_seed(seed)
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    category_dict = {
        "Industrial_and_Scientific": "industrial and scientific items", 
        "Office_Products": "office products", 
        "Toys_and_Games": "toys and games", 
        "Sports": "sports and outdoors", 
        "Books": "books"
    }
    category_str = category_dict.get(category, category)
    print(f"Evaluating category: {category_str}")

    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, device_map={"": device})
    model.eval()

    with open(info_file, 'r', encoding='utf-8') as f:
        info = f.readlines()
        semantic_ids = [line.split('\t')[0].strip() + "\n" for line in info]
        item_titles = [line.split('\t')[1].strip() + "\n" for line in info if len(line.split('\t')) >= 2]
        
        info_semantic = [f'''### Response:\n{_}''' for _ in semantic_ids]
        info_titles = [f'''### Response:\n{_}''' for _ in item_titles]

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    if base_model.lower().find("llama") > -1:
        prefixID = [tokenizer(_).input_ids[1:] for _ in info_semantic]
        prefixTitleID = [tokenizer(_).input_ids[1:] for _ in info_titles]
    else:
        prefixID = [tokenizer(_).input_ids for _ in info_semantic]
        prefixTitleID = [tokenizer(_).input_ids for _ in info_titles]
    
    prefix_index = 4 if base_model.lower().find("gpt2") > -1 else 3
    
    hash_dict = dict()
    for index, ID in enumerate(prefixID):
        ID.append(tokenizer.eos_token_id)
        for i in range(prefix_index, len(ID)):
            hash_number = get_hash(ID[:i]) if i == prefix_index else get_hash(ID[prefix_index:i])
            if hash_number not in hash_dict:
                hash_dict[hash_number] = set()
            hash_dict[hash_number].add(ID[i])

    hash_dict_title = dict()
    for index, ID in enumerate(prefixTitleID):
        ID.append(tokenizer.eos_token_id)
        for i in range(prefix_index, len(ID)):
            hash_number = get_hash(ID[:i]) if i == prefix_index else get_hash(ID[prefix_index:i])
            if hash_number not in hash_dict_title:
                hash_dict_title[hash_number] = set()
            hash_dict_title[hash_number].add(ID[i])

    for key in hash_dict.keys(): hash_dict[key] = list(hash_dict[key])
    for key in hash_dict_title.keys(): hash_dict_title[key] = list(hash_dict_title[key])

    def prefix_allowed_tokens_fn_semantic(batch_id, input_ids):
        hash_number = get_hash(input_ids)
        return hash_dict.get(hash_number, [])
        
    prefix_allowed_tokens_fn = prefix_allowed_tokens_fn_semantic
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    val_dataset = EvalSidDataset(train_file=test_data_path, tokenizer=tokenizer, max_len=2560, category=category_str, test=True, K=K, seed=seed)
    encodings = [val_dataset[i] for i in range(len(val_dataset))]
    test_data = val_dataset.get_all()

    model.config.pad_token_id = model.config.eos_token_id = tokenizer.eos_token_id

    def evaluate_batch(encs, num_beams=10, max_new_tokens=64, length_penalty=1.0, **kwargs):
        maxLen = max([len(_["input_ids"]) for _ in encs])
        padding_ids = []
        attention_mask = []

        for _ in encs:
            L = len(_["input_ids"])
            padding_ids.append([tokenizer.pad_token_id] * (maxLen - L) + _["input_ids"])
            attention_mask.append([0] * (maxLen - L) + [1] * L) 
        
        generation_config = GenerationConfig(
            num_beams=num_beams,
            length_penalty=length_penalty,
            num_return_sequences=num_beams,
            pad_token_id = tokenizer.pad_token_id,
            eos_token_id = tokenizer.eos_token_id,
            max_new_tokens = max_new_tokens,
            **kwargs
        )
        
        with torch.no_grad():
            clp = ConstrainedLogitsProcessor(
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                num_beams=num_beams,
                base_model=base_model
            )
            logits_processor = LogitsProcessorList([clp])

            generation_output = model.generate(
                torch.tensor(padding_ids).to(device),
                attention_mask=torch.tensor(attention_mask).to(device),
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                logits_processor=logits_processor,
            )
       
        batched_completions = generation_output.sequences[:, maxLen:]
        output = tokenizer.batch_decode(batched_completions, skip_special_tokens=True)
        output = [_.split("Response:\n")[-1].strip() for _ in output]
        return [output[i * num_beams: (i + 1) * num_beams] for i in range(len(output) // num_beams)]
    
    outputs = []
    BLOCKS = [encodings[i : i + batch_size] for i in range(0, len(encodings), batch_size)]
    
    for batch in tqdm(BLOCKS, desc="Generating Predictions"):
        outputs.extend(evaluate_batch(batch, max_new_tokens=max_new_tokens, num_beams=num_beams, length_penalty=length_penalty))
       
    for i, test in enumerate(test_data):
        test["predict"] = outputs[i]
        if 'dedup' in test: test.pop('dedup')  

    os.makedirs(os.path.dirname(result_json_data), exist_ok=True)
    with open(result_json_data, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=4)
    print(f"Results saved to: {result_json_data}")

if __name__ == '__main__':
    fire.Fire(main)