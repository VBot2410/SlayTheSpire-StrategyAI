import pandas as pd
import numpy as np
import orjson
import lightgbm as lgb
import os
import spire_strategy.data.pipeline as config

def load_and_process_act_data(csv_path, card_vocab_size, relic_vocab_size, floor_min, floor_max):
    """Loads and processes data for a specific floor range with explicit intra-group variations."""
    print(f"Streaming and filtering floors {floor_min}-{floor_max} into RAM...")
    
    chunks = []
    for chunk in pd.read_csv(csv_path, chunksize=100000):
        filtered_chunk = chunk[(chunk["floor"] >= floor_min) & (chunk["floor"] <= floor_max)]
        chunks.append(filtered_chunk)
    df = pd.concat(chunks, ignore_index=True)
    
    if len(df) == 0:
        return None, None, None

    X_metrics = df[["floor", "character_class", "ascension_level", "gold", "hp_ratio"]].to_numpy(dtype=np.float32)
    
    # Multi-Hot Matrices
    relic_matrix = np.zeros((len(df), relic_vocab_size + 1), dtype=np.int8)
    for idx, raw_seq in enumerate(df["relic_seq"]):
        tokens = orjson.loads(raw_seq)
        for t in tokens:
            if t > 0: relic_matrix[idx, t] = 1
                
    deck_matrix = np.zeros((len(df), card_vocab_size + 1), dtype=np.int16)
    for idx, raw_seq in enumerate(df["deck_seq"]):
        tokens = orjson.loads(raw_seq)
        for t in tokens:
            if t > 0: deck_matrix[idx, t] += 1
                
    candidate_ids = df["candidate_card_id"].to_numpy(dtype=np.int64)
    candidate_matrix = np.zeros((len(df), card_vocab_size + 1), dtype=np.int8)
    for idx, cand_id in enumerate(candidate_ids):
        candidate_matrix[idx, cand_id] = 1

    # Contextual Interaction Fields
    candidate_deck_count = np.zeros((len(df), 1), dtype=np.float32)
    is_skip_row = (candidate_ids == 0).astype(np.float32).reshape(-1, 1)
    
    for idx in range(len(df)):
        cand_id = candidate_ids[idx]
        if cand_id > 0:
            candidate_deck_count[idx, 0] = float(deck_matrix[idx, cand_id])

    # Advanced Synergy Engines (Breaks Intra-Group Uniformity)
    total_relics_owned = relic_matrix.sum(axis=1, keepdims=True).astype(np.float32)
    cand_relic_count_interaction = candidate_matrix.astype(np.float32) * total_relics_owned
    
    total_deck_size = deck_matrix.sum(axis=1, keepdims=True).astype(np.float32)
    cand_deck_size_interaction = candidate_matrix.astype(np.float32) * total_deck_size

    # Build final feature array
    X = np.hstack([
        X_metrics, 
        is_skip_row,
        candidate_deck_count, 
        candidate_matrix, 
        relic_matrix, 
        deck_matrix,
        cand_relic_count_interaction,
        cand_deck_size_interaction
    ])
    
    y = df["target"].to_numpy(dtype=np.int8)
    groups = df["group_id"].to_numpy(dtype=np.int32)
    
    return X, y, groups

def train_act_model(act_name, csv_path, card_vocab_size, relic_vocab_size, floor_min, floor_max):
    X_subset, y_subset, groups_subset = load_and_process_act_data(
        csv_path, card_vocab_size, relic_vocab_size, floor_min, floor_max
    )
    
    if X_subset is None:
        print(f"Skipping {act_name}: No data found.")
        return 0.0, 0
        
    print(f"\n=== Training Specialized Ranker for {act_name} ===")
    
    unique_groups = np.unique(groups_subset)
    np.random.seed(42)
    np.random.shuffle(unique_groups)
    
    split_idx = int(0.85 * len(unique_groups))
    train_group_set = set(unique_groups[:split_idx])
    
    train_mask = np.isin(groups_subset, list(train_group_set))
    val_mask = ~train_mask
    
    X_train, y_train, groups_train = X_subset[train_mask], y_subset[train_mask], groups_subset[train_mask]
    X_val, y_val, groups_val = X_subset[val_mask], y_subset[val_mask], groups_subset[val_mask]
    
    del X_subset, y_subset, groups_subset
    
    train_sort = np.argsort(groups_train)
    X_train, y_train, groups_train = X_train[train_sort], y_train[train_sort], groups_train[train_sort]
    
    val_sort = np.argsort(groups_val)
    X_val, y_val, groups_val = X_val[val_sort], y_val[val_sort], groups_val[val_sort]
    
    _, train_group_counts = np.unique(groups_train, return_counts=True)
    _, val_group_counts = np.unique(groups_val, return_counts=True)
    
    print(f"[{act_name}] Split completed. Train screens: {len(train_group_counts)} | Val screens: {len(val_group_counts)}")
    
    train_data = lgb.Dataset(X_train, label=y_train, group=train_group_counts)
    val_data = lgb.Dataset(X_val, label=y_val, group=val_group_counts, reference=train_data)
    
    # UNIFIED HIGH-REGULARIZATION SINGLE-ROUND SETTINGS
    params = {
        "objective": "lambdarank",       
        "metric": "ndcg",                
        "ndcg_eval_at":[1],          
        "boosting_type": "gbdt",
        "label_gain":[0, 1],            
        "lambdarank_truncation_level": 4, 
        "learning_rate": 0.05,          
        "num_leaves": 31,                
        "max_depth": 6,                  
        "min_data_in_leaf": 100,         
        "feature_fraction": 0.60,        
        "bagging_fraction": 0.80,        
        "bagging_freq": 1,               
        "reg_alpha": 1.0,                
        "reg_lambda": 5.0,              
        "verbosity": -1,
        "n_jobs": 3                      
    }

    # Lock to 1 boost round to secure the structural baseline peak
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1, 
        valid_sets=[val_data],
        callbacks=[lgb.log_evaluation(1)]
    )
    
    # Vectorized Validation
    val_preds = model.predict(X_val)
    eval_df = pd.DataFrame({'group': groups_val, 'pred': val_preds, 'target': y_val})
    idx_max_pred = eval_df.groupby('group')['pred'].idxmax()
    chosen_targets = eval_df.loc[idx_max_pred, 'target']
    
    top1_accuracy = (chosen_targets == 1).mean() * 100
    print(f"-> {act_name} Evaluation | Total Screens: {len(idx_max_pred)} | Top-1 Accuracy: {top1_accuracy:.2f}%")
    
    model_filename = f"spire_lgb_{act_name.lower().replace(' ', '_')}.txt"
    model.save_model(model_filename)
    print(f"Saved {model_filename}")
    
    return top1_accuracy, len(idx_max_pred)

def train_split_pipeline(csv_path, card_vocab_size, relic_vocab_size):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")

    # Process each Act independently to maintain a clean RAM profile
    acc_1, screens_1 = train_act_model("Act 1", csv_path, card_vocab_size, relic_vocab_size, 1, 17)
    acc_2, screens_2 = train_act_model("Act 2", csv_path, card_vocab_size, relic_vocab_size, 18, 34)
    acc_3, screens_3 = train_act_model("Act 3", csv_path, card_vocab_size, relic_vocab_size, 35, 100)
    
    total_screens = screens_1 + screens_2 + screens_3
    if total_screens > 0:
        weighted_accuracy = ((acc_1 * screens_1) + (acc_2 * screens_2) + (acc_3 * screens_3)) / total_screens
        print(f"\n=======================================================")
        print(f"FINAL OPTIMIZED SPIRE ACCURACY: {weighted_accuracy:.2f}%")
        print(f"=======================================================")

if __name__ == "__main__":
    CARD_SIZE = len(config.ALL_VANILLA_CARDS) 
    RELIC_SIZE = len(config.ALL_VANILLA_RELICS)
    train_split_pipeline("slay_the_spire_transformer.csv", CARD_SIZE, RELIC_SIZE)
