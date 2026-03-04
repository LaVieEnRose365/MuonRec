import fire
from loguru import logger
import json
from tqdm import tqdm
import random
import datetime
import csv
import os
import sys

# =====================================================
# Dynamically add project root to sys.path
# =====================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def get_timestamp_start(year, month):
    """Convert year and month to unix timestamp."""
    return int(datetime.datetime(year=year, month=month, day=1, hour=0, minute=0, second=0, microsecond=0).timestamp())

def run_preprocessing(
    category, 
    metadata_file: str, 
    reviews_file: str, 
    K: int = 5, 
    st_year: int = 2017, 
    st_month: int = 10, 
    ed_year: int = 2018, 
    ed_month: int = 11, 
    output_path: str = "./minionerec_engine/data/Amazon"
):
    """
    Main preprocessing function with recursive time range expansion.
    """
    if st_year < 1996:
        logger.error("Reached minimum supported year (1996). Stopping expansion.")
        return

    start_timestamp = get_timestamp_start(st_year, st_month)
    end_timestamp = get_timestamp_start(ed_year, ed_month)
    logger.info(f"Processing range: {st_year}-{st_month} to {ed_year}-{ed_month}")

    # Load raw data
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    if not os.path.exists(reviews_file):
        raise FileNotFoundError(f"Reviews file not found: {reviews_file}")

    with open(metadata_file, 'r') as f:
        metadata = [json.loads(line) for line in f]
    with open(reviews_file, 'r') as f:
        reviews = [json.loads(line) for line in f]

    logger.info(f"Loaded {len(metadata)} metadata items and {len(reviews)} reviews for {category}.")
    
    # Initial set of valid items based on metadata quality
    remove_items = set()
    id_title = {}
    for meta in tqdm(metadata, desc="Cleaning metadata"):
        if ('title' not in meta) or (meta['title'].find('<span id') > -1):
            remove_items.add(meta['asin'])
            continue
        
        # Clean title strings
        meta['title'] = meta["title"].replace("&quot;", "\"").replace("&amp;", "&").strip(" ").strip("\"")
        if 1 < len(meta['title']) and len(meta['title'].split(" ")) <= 20:
            id_title[meta['asin']] = meta['title']
        else:
            remove_items.add(meta['asin'])

    # K-core filtering loop
    remove_users = set()
    while True:
        flag = False
        total_inters = 0
        user_counts = {}
        item_counts = {}
        valid_reviews = []

        for review in tqdm(reviews, desc="Iterative K-core filtering"):
            ts = int(review["unixReviewTime"])
            if ts < start_timestamp or ts > end_timestamp:
                continue
            
            asin = review['asin']
            uid = review['reviewerID']
            if uid in remove_users or asin in remove_items or asin not in id_title:
                continue
            
            user_counts[uid] = user_counts.get(uid, 0) + 1
            item_counts[asin] = item_counts.get(asin, 0) + 1
            total_inters += 1
            valid_reviews.append(review)
            
        # Check K-core constraints
        for user, count in user_counts.items():
            if count < K:
                remove_users.add(user)
                flag = True
        for item, count in item_counts.items():
            if count < K:
                remove_items.add(item)
                flag = True

        logger.info(f"Current Stats - Users: {len(user_counts)}, Items: {len(item_counts)}, Interactions: {total_inters}")
        
        # Condition to stop or expand time range
        if not flag:
            break
        if st_year > 1996 and len(item_counts) < 3000:
            logger.warning(f"Item count {len(item_counts)} < 3000, expanding time range recursively...")
            return run_preprocessing(
                category, metadata_file, reviews_file, K, st_year - 1, st_month, ed_year, ed_month, output_path
            )

    # Final assignment and output preparation
    logger.info(f"Final Filtering - Removed Users: {len(remove_users)}, Removed Items: {len(remove_items)}")
    
    # Map items to IDs
    final_items = sorted(list(item_counts.keys()))
    random.seed(42)
    random.shuffle(final_items)
    item2id = {item: i for i, item in enumerate(final_items)}
    
    # Ensure category output directories exist
    cat_out_dir = os.path.join(output_path, category)
    for folder in ['info', 'train', 'valid', 'test']:
        os.makedirs(os.path.join(cat_out_dir, folder), exist_ok=True)
    
    # Save item information mapping
    info_path = os.path.join(cat_out_dir, "info", f"{category}_item_map.txt")
    with open(info_path, 'w') as f:
        for item in final_items:
            f.write(f"{id_title[item]}\t{item2id[item]}\n")
    
    # Build interaction sequences (Sliding window of 10)
    user_interact = {}
    for review in valid_reviews:
        uid, asin, ts, score = review['reviewerID'], review['asin'], int(review['unixReviewTime']), float(review['overall'])
        if uid not in user_interact:
            user_interact[uid] = []
        user_interact[uid].append((asin, score, ts))
    
    interaction_list = []
    for uid, inters in tqdm(user_interact.items(), desc="Building sequences"):
        inters.sort(key=lambda x: x[2]) # Sort by timestamp
        
        asins, scores, timestamps = zip(*inters)
        item_ids = [item2id[a] for a in asins]
        titles = [id_title[a] for a in asins]
        
        for i in range(1, len(item_ids)):
            start_idx = max(i - 10, 0)
            interaction_list.append([
                uid, 
                asins[start_idx:i], asins[i], 
                item_ids[start_idx:i], item_ids[i],
                titles[start_idx:i], titles[i],
                scores[start_idx:i], scores[i],
                timestamps[start_idx:i], timestamps[i]
            ])

    # Sort sequences by chronological order (target timestamp)
    interaction_list.sort(key=lambda x: int(x[-1]))
    
    # 8:1:1 Split and Save
    num_total = len(interaction_list)
    train_end, valid_end = int(num_total * 0.8), int(num_total * 0.9)
    
    headers = ['user_id', 'item_asins', 'item_asin', 'history_item_id', 'item_id', 
               'history_item_title', 'item_title', 'history_rating', 'rating', 
               'history_timestamp', 'timestamp']
    
    datasets = {
        'train': interaction_list[:train_end],
        'valid': interaction_list[train_end:valid_end],
        'test': interaction_list[valid_end:]
    }
    
    for split_name, data in datasets.items():
        file_path = os.path.join(cat_out_dir, split_name, f"{category}_split.csv")
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(data)
        logger.info(f"Saved {split_name} split: {len(data)} sequences to {file_path}")

    logger.info("Preprocessing Pipeline Completed Successfully!")

if __name__ == '__main__':
    fire.Fire(run_preprocessing)