import pandas as pd
import numpy as np
import orjson
import lightgbm as lgb
import os
from spire_strategy.data import ALL_VANILLA_CARDS, ALL_VANILLA_RELICS
from spire_strategy import PACKAGE_ROOT

MODELS_DIR = PACKAGE_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

def load_and_process_act_data(csv_path, card_vocab_size, relic_vocab_size, floor_min, floor_max):
    """Loads and processes data for a specific floor range into a single-row multiclass structure."""
    print(f"Streaming and filtering floors {floor_min}-{floor_max} into RAM...")
    
    chunks = []
    for chunk in pd.read_csv(csv_path, chunksize=100000):
        filtered_chunk = chunk[(chunk["floor"] >= floor_min) & (chunk["floor"] <= floor_max)]
        chunks.append(filtered_chunk)
    df = pd.concat(chunks, ignore_index=True)
    
    if len(df) == 0:
        return None, None, None

    # =========================================================================
    # FIX: CHRONOLOGICAL RE-ALIGNMENT & SCREEN-AGGREGATION
    # =========================================================================
    print("Re-sorting dataframe to secure screen and run cohesion...")
    # Ensure the dataframe is sorted so candidate screens sit perfectly back-to-back
    df = df.sort_values(by=["character_class", "ascension_level", "group_id", "floor"]).reset_index(drop=True)

    # 1. Group rows by choice screen to extract single-row metrics
    grouped = df.groupby('group_id')
    
    # Base environment metrics remain exactly the same per screen
    X_metrics = grouped[["floor", "character_class", "ascension_level", "gold", "hp_ratio"]].first().to_numpy(dtype=np.float32)
    
    # 2. Extract multi-hot relic and deck counts for each screen
    # Since relics/decks are identical across sibling rows, we can pull the first element safely
    def process_json_matrix(series, vocab_size, is_count=False):
        matrix = np.zeros((len(series), vocab_size + 1), dtype=np.int16)
        for idx, raw_seq in enumerate(series):
            tokens = orjson.loads(raw_seq)
            for t in tokens:
                if t > 0:
                    if is_count: matrix[idx, t] += 1
                    else: matrix[idx, t] = 1
        return matrix

    relic_matrix = process_json_matrix(grouped["relic_seq"].first(), relic_vocab_size, is_count=False)
    deck_matrix = process_json_matrix(grouped["deck_seq"].first(), card_vocab_size, is_count=True)

    # 3. Create fixed-shape choice matrices (Assuming max 4 options per screen: 3 cards + 1 skip)
    # Target value will represent the integer index position of the winning choice
    num_screens = len(grouped)
    candidate_cards = np.zeros((num_screens, 4), dtype=np.int32)
    candidate_deck_counts = np.zeros((num_screens, 4), dtype=np.float32)
    is_skip_mask = np.zeros((num_screens, 4), dtype=np.float32)
    y_multiclass = np.zeros(num_screens, dtype=np.int8)

    # Map raw rows into fixed 4-slot screen structures
    screen_idx = 0
    for group_id, frame in grouped:
        frame_rows = frame.reset_index(drop=True)
        for slot in range(min(4, len(frame_rows))):
            cand_id = int(frame_rows.loc[slot, "candidate_card_id"])
            candidate_cards[screen_idx, slot] = cand_id
            is_skip_mask[screen_idx, slot] = 1.0 if cand_id == 0 else 0.0
            
            # Context deck counts matching target shifts
            shifted_id = cand_id + 1
            if cand_id > 0 and 0 < shifted_id <= card_vocab_size:
                candidate_deck_counts[screen_idx, slot] = float(deck_matrix[screen_idx, shifted_id])
                
            # If this specific card slot was the one picked, lock its index as our label target
            if int(frame_rows.loc[slot, "target"]) == 1:
                y_multiclass[screen_idx] = slot
        screen_idx += 1

    # Calculate aggregate footprints for advanced synergy processing
    total_relics_owned = relic_matrix.sum(axis=1, keepdims=True).astype(np.float32)
    total_deck_size = deck_matrix.sum(axis=1, keepdims=True).astype(np.float32)

    # Build the flat training matrix
    X = np.hstack([
        X_metrics,                  # 5 features
        is_skip_mask,               # 4 features
        candidate_deck_counts,      # 4 features
        candidate_cards,            # 4 features
        relic_matrix,               # relic_vocab_size + 1 features
        deck_matrix,                # card_vocab_size + 1 features
        total_relics_owned,         # 1 feature
        total_deck_size             # 1 feature
    ])
    
    # Regenerate a clean, unique screen sequence for the run tracker loop
    # We pass the group_id array back so train_act_model can still track run signatures!
    groups = df.groupby('group_id')["group_id"].first().to_numpy(dtype=np.int32)
    
    return X, y_multiclass, groups

def train_act_model(act_name, csv_path, card_vocab_size, relic_vocab_size, floor_min, floor_max):
    X_subset, y_subset, groups_subset = load_and_process_act_data(
        csv_path, card_vocab_size, relic_vocab_size, floor_min, floor_max
    )
    
    if X_subset is None:
        print(f"Skipping {act_name}: No data found.")
        return 0.0, 0
        
    print(f"\n=== Training Specialized Ranker for {act_name} ===")
    
    # =========================================================================
    # FIXED: INSULATED GROUP RESETS (Prevents Sibling Row Fragmentation)
    # =========================================================================
    print(f"[{act_name}] Compiling unique screen intervals...")
    
    # Track the exact boundary changes between different reward screens
    is_new_screen = np.zeros(len(X_subset), dtype=bool)
    is_new_screen[0] = True
    is_new_screen[1:] = (groups_subset[1:] != groups_subset[:-1])
    
    # Compress our fields down to only look at one row per screen
    screen_floors = X_subset[is_new_screen, 0]
    screen_classes = X_subset[is_new_screen, 1]
    screen_ascensions = X_subset[is_new_screen, 2]
    
    # Track true macro run resets across independent screen groups
    is_run_reset = np.zeros(len(screen_floors), dtype=bool)
    is_run_reset[0] = True # Initial screen starts the first run
    
    # A true run reset happens when the floor drops or class variables shift
    is_run_reset[1:] |= (screen_floors[1:] < screen_floors[:-1])
    is_run_reset[1:] |= (screen_classes[1:] != screen_classes[:-1])
    is_run_reset[1:] |= (screen_ascensions[1:] != screen_ascensions[:-1])
    
    # Generate sequential IDs for each unique run block
    unique_run_ids = np.cumsum(is_run_reset)
    
    # Expand the run IDs back out to match every raw row in your data matrix
    run_ids_subset = np.zeros(len(X_subset), dtype=np.int32)
    run_ids_subset[is_new_screen] = unique_run_ids
    
    # Forward-fill the assignments across all sibling row items
    for idx in range(1, len(run_ids_subset)):
        if not is_new_screen[idx]:
            run_ids_subset[idx] = run_ids_subset[idx - 1]
            
    print(f"[{act_name}] True Footprint: Detected {run_ids_subset[-1]} completely decoupled game runs.")

    # =========================================================================
    # LEAKAGE-PROOF SPLIT USING GENERATED RUN SIGNATURES
    # =========================================================================
    unique_runs = np.unique(run_ids_subset)
    np.random.seed(42)
    np.random.shuffle(unique_runs)
    
    # Allocate 85% of full runs to training, 15% to validation
    split_idx = int(0.85 * len(unique_runs))
    train_run_set = set(unique_runs[:split_idx])
    
    # Map the unique runs allocation mask down to flat matrix rows
    train_mask = np.isin(run_ids_subset, list(train_run_set))
    val_mask = ~train_mask
    
    # Separate data arrays cleanly
    X_train, y_train, groups_train = X_subset[train_mask], y_subset[train_mask], groups_subset[train_mask]
    X_val, y_val, groups_val = X_subset[val_mask], y_subset[val_mask], groups_subset[val_mask]
    
    del X_subset, y_subset, groups_subset, run_ids_subset
    
    # -------------------------------------------------------------------------
    # RESTORE ORDER: Sort by group_id so sibling options stay perfectly grouped
    # -------------------------------------------------------------------------
    train_sort = np.argsort(groups_train)
    X_train, y_train, groups_train = X_train[train_sort], y_train[train_sort], groups_train[train_sort]
    
    val_sort = np.argsort(groups_val)
    X_val, y_val, groups_val = X_val[val_sort], y_val[val_sort], groups_val[val_sort]
    
    _, train_group_counts = np.unique(groups_train, return_counts=True)
    _, val_group_counts = np.unique(groups_val, return_counts=True)
    
    print(f"[{act_name}] Split completed. Train screens: {len(train_group_counts)} | Val screens: {len(val_group_counts)}")

    
    # Identify the exact column index of character_class
    # Based on X_metrics ordering: ["floor", "character_class", "ascension_level", "gold", "hp_ratio"]
    # character_class is at index position 1.
    categorical_indices = [1]

    # Strip away the group assignments so LightGBM processes the rows independently
    train_data = lgb.Dataset(
        X_train, 
        label=y_train, 
        categorical_feature=categorical_indices
    )
    val_data = lgb.Dataset(
        X_val, 
        label=y_val, 
        reference=train_data, 
        categorical_feature=categorical_indices
    )
    
        # DEEP-DENSITY BINARY PROBABILITY ENGINE
    params = {
        "objective": "multiclass",       # Switched to multi-choice optimization
        "num_class": 4,                  # 4 distinct candidate slot channels
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        
        "learning_rate": 0.02,          
        "num_leaves": 63,                
        "max_depth": 8,                  
        "min_data_in_leaf": 25,         
        "feature_fraction": 0.70,        
        "bagging_fraction": 0.85,        
        "bagging_freq": 1,               
        "reg_alpha": 0.2,                
        "reg_lambda": 2.0,              
        "verbosity": -1,
        "n_jobs": 4                      
    }


    callbacks = [
        lgb.early_stopping(stopping_rounds=30, first_metric_only=True, verbose=True),
        lgb.log_evaluation(period=25)    # Log every 25 rounds for a cleaner console view
    ]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1500,            # Raised from 300
        valid_sets=[val_data],
        callbacks=callbacks
    )

    
    # =========================================================================
    # UPGRADED INDEX-INSULATED ACCURACY EVALUATION
    # =========================================================================
    # Process predictions (Will return an array of shapes: [num_screens, 4])
    val_preds = model.predict(X_val)
    
    # The highest value index across the horizontal axes represents the choice
    chosen_slots = np.argmax(val_preds, axis=1)
    
    # Calculate performance accuracy by checking if the chosen index matches the label
    top1_accuracy = (chosen_slots == y_val).mean() * 100
    print(f" -> {act_name} Evaluation | Total Screens: {len(y_val)} | Top-1 Accuracy: {top1_accuracy:.2f}%")

    
    model_filename = f"spire_lgb_{act_name.lower().replace(' ', '_')}.txt"
    model_path = MODELS_DIR / model_filename
    model.save_model(model_path)
    print(f"Saved {model_filename} to package path: {MODELS_DIR}")
    
    return top1_accuracy, len(y_val)


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
    CARD_SIZE = len(ALL_VANILLA_CARDS) 
    RELIC_SIZE = len(ALL_VANILLA_RELICS)
    train_split_pipeline("slay_the_spire_transformer.csv", CARD_SIZE, RELIC_SIZE)
