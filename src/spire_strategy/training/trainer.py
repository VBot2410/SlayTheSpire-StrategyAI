import pandas as pd
import numpy as np
import orjson
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
import spire_strategy.data.pipeline as pipeline
import os

CARD_INDICES = {name: idx for idx, name in enumerate(pipeline.ALL_VANILLA_CARDS)}
RELIC_INDICES = {name: idx for idx, name in enumerate(pipeline.ALL_VANILLA_RELICS)}

# =========================================================================
# DOMAIN FEATURE CONFIGURATION (RESTORED SPELLING)
# =========================================================================
# 1. Targets exact custom spelling markers: "striker", "strikeg", "strikeb", etc.
STRIKE_IDS = {idx for name, idx in CARD_INDICES.items() if "strike" in name or "striker" in name}
DEFEND_IDS = {idx for name, idx in CARD_INDICES.items() if "defend" in name or "defendr" in name}

# 2. Archetype systems matching specific vanilla token subsets
POWER_FORM_IDS = {idx for name, idx in CARD_INDICES.items() if "form" in name or "echoform" in name or "demonform" in name or "devaform" in name}
EXHAUST_SYS_IDS = {idx for name, idx in CARD_INDICES.items() if "exhaust" in name or "corruption" in name or "fiendfire" in name or "feelnopain" in name or "darkembrace" in name}
POISON_SYS_IDS  = {idx for name, idx in CARD_INDICES.items() if "poison" in name or "noxiousfumes" in name or "catalyst" in name or "bouncingflask" in name}
ORB_SYS_IDS     = {idx for name, idx in CARD_INDICES.items() if "orb" in name or "defragment" in name or "biasedcognition" in name or "electrodynamics" in name or "zap" in name}
STANCE_SYS_IDS  = {idx for name, idx in CARD_INDICES.items() if "stance" in name or "eruption" in name or "vigilance" in name or "tantrum" in name or "calm" in name or "wrath" in name}

# 3. High-Cost Cards matching your specific array tokens
HIGH_COST_IDS = {idx for name, idx in CARD_INDICES.items() if name in [
    "bludgeon", "demonform", "carnage", "uppercut", "clothesline", "whirlwind", "immolate", "feed", 
    "nightmare", "wraithform", "bouncingflask", "legsweep", "corpsesplosion", "grandfinale", "bullettime",
    "meteorstrike", "echoform", "sunder", "hyperbeam", "electrodynamics", "creativeai", "buffer",
    "alpha", "wish", "ragnarok", "wheelkick", "conjureblade", "scrawl", "devaform"
]}

# 4. Critical Relic Vocabulary ID Identifiers (+1 is handled at array runtime lookup)
SNECKO_EYE_ID = RELIC_INDICES.get("sneckoeye", -1)
MUMMIFIED_HAND_ID = RELIC_INDICES.get("mummifiedhand", -1)

def prepare_data_for_lightgbm(csv_path, card_vocab_size, relic_vocab_size):
    print("Loading data into RAM for advanced LightGBM preprocessing...")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")
        
    df = pd.read_csv(csv_path)
    num_rows = len(df)
    
    # 1. Base Tabular Features
    X_base = df[["floor", "character_class", "ascension_level", "gold", "hp_ratio"]].to_numpy(dtype=np.float32)
    floors = X_base[:, 0]
    
    print("Flattening sequences into multi-hot array matrices...")
    # 2. Multi-Hot Relic Matrix
    relic_matrix = np.zeros((num_rows, relic_vocab_size + 1), dtype=np.int8)
    for idx, raw_seq in enumerate(df["relic_seq"]):
        tokens = orjson.loads(raw_seq)
        for t in tokens:
            if t > 0:
                relic_matrix[idx, t] = 1
                
    # 3. Frequency Count Deck Matrix
    deck_matrix = np.zeros((num_rows, card_vocab_size + 1), dtype=np.int16)
    for idx, raw_seq in enumerate(df["deck_seq"]):
        tokens = orjson.loads(raw_seq)
        for t in tokens:
            if t > 0:
                deck_matrix[idx, t] += 1

    # 4. Candidate Identification Matrix
    print("Creating One-Hot matrix for row candidate identity...")
    candidate_ids = df["candidate_card_id"].to_numpy(dtype=np.int64)
    candidate_matrix = np.zeros((num_rows, card_vocab_size + 1), dtype=np.int8)
    for idx, cand_id in enumerate(candidate_ids):
        candidate_matrix[idx, cand_id] = 1

    # 5. Core Duplicate Interaction Column
    print("Computing candidate-card ownership interaction layer...")
    already_owned = np.zeros((num_rows, 1), dtype=np.float32)
    for idx in range(num_rows):
        cand_id = candidate_ids[idx]
        if cand_id > 0 and deck_matrix[idx, cand_id] > 0:
            already_owned[idx, 0] = 1.0

    # =========================================================================
    # MATRIX-VECTORIZED DOMAIN FEATURE ENGINEERING (RESTORED CODES)
    # =========================================================================
    print("Vectorizing archetypes and legendary relic interactions...")
    
    # Compute continuous macro dimensions across your deck matrix
    total_deck_size = np.sum(deck_matrix, axis=1, keepdims=True).astype(np.float32)
    
    # Generate aggregate archetype scaling counts instantly using our updated vocabulary sets
    strike_counts  = np.sum(deck_matrix[:, list(STRIKE_IDS)], axis=1, keepdims=True).astype(np.float32) if STRIKE_IDS else np.zeros((num_rows, 1), dtype=np.float32)
    defend_counts  = np.sum(deck_matrix[:, list(DEFEND_IDS)], axis=1, keepdims=True).astype(np.float32) if DEFEND_IDS else np.zeros((num_rows, 1), dtype=np.float32)
    power_counts   = np.sum(deck_matrix[:, list(POWER_FORM_IDS)], axis=1, keepdims=True).astype(np.float32) if POWER_FORM_IDS else np.zeros((num_rows, 1), dtype=np.float32)
    exhaust_counts = np.sum(deck_matrix[:, list(EXHAUST_SYS_IDS)], axis=1, keepdims=True).astype(np.float32) if EXHAUST_SYS_IDS else np.zeros((num_rows, 1), dtype=np.float32)
    poison_counts  = np.sum(deck_matrix[:, list(POISON_SYS_IDS)], axis=1, keepdims=True).astype(np.float32) if POISON_SYS_IDS else np.zeros((num_rows, 1), dtype=np.float32)
    orb_counts     = np.sum(deck_matrix[:, list(ORB_SYS_IDS)], axis=1, keepdims=True).astype(np.float32) if ORB_SYS_IDS else np.zeros((num_rows, 1), dtype=np.float32)
    stance_counts  = np.sum(deck_matrix[:, list(STANCE_SYS_IDS)], axis=1, keepdims=True).astype(np.float32) if STANCE_SYS_IDS else np.zeros((num_rows, 1), dtype=np.float32)
    
    # Act-Specific Game Logic Indicators
    is_act_1 = (floors <= 16).astype(np.float32).reshape(-1, 1)
    is_act_3 = (floors >= 34).astype(np.float32).reshape(-1, 1)
    
    # Legendary Relic Interaction Flag Calculations
    snecko_interaction = np.zeros((num_rows, 1), dtype=np.float32)
    mummified_interaction = np.zeros((num_rows, 1), dtype=np.float32)
    
    # Pull presence vector arrays (+1 accounts for token shifting boundary rules)
    has_snecko = relic_matrix[:, SNECKO_EYE_ID + 1] == 1 if SNECKO_EYE_ID != -1 else np.zeros(num_rows, dtype=bool)
    has_mummy  = relic_matrix[:, MUMMIFIED_HAND_ID + 1] == 1 if MUMMIFIED_HAND_ID != -1 else np.zeros(num_rows, dtype=bool)
    
    for idx in range(num_rows):
        cand_id = candidate_ids[idx]
        # Align candidate token indexes cleanly
        if has_snecko[idx] and (cand_id - 1 in HIGH_COST_IDS):
            snecko_interaction[idx, 0] = 1.0
        if has_mummy[idx] and (cand_id - 1 in POWER_FORM_IDS):
            mummified_interaction[idx, 0] = 1.0

    # Stack ALL engineered features cleanly into a single unified input array
    X = np.hstack([
        X_base, 
        is_act_1, is_act_3, total_deck_size,
        strike_counts, defend_counts, power_counts, exhaust_counts, poison_counts, orb_counts, stance_counts,
        snecko_interaction, mummified_interaction,
        already_owned, 
        candidate_matrix, 
        relic_matrix, 
        deck_matrix
    ])
    
    y = df["target"].to_numpy(dtype=np.int8)
    groups = df["group_id"].to_numpy(dtype=np.int32)
    
    return X, y, groups


def train_lightgbm_ranking(csv_path, card_vocab_size, relic_vocab_size):
    X, y, groups = prepare_data_for_lightgbm(csv_path, card_vocab_size, relic_vocab_size)
    
    # CRITICAL FOR LAMBDARANK: Data MUST be strictly sorted sequentially by group_id
    print("Sorting data by group boundaries for Lambdarank list alignment...")
    sort_indices = np.argsort(groups)
    X, y, groups = X[sort_indices], y[sort_indices], groups[sort_indices]
    
    # Generate clean 85/15 deterministic split points based on sorted group boundaries
    unique_groups = np.unique(groups)
    np.random.seed(42)
    np.random.shuffle(unique_groups)
    
    split_idx = int(0.85 * len(unique_groups))
    train_group_set = set(unique_groups[:split_idx])
    
    train_idx = [i for i, g in enumerate(groups) if g in train_group_set]
    val_idx = [i for i, g in enumerate(groups) if g not in train_group_set]
    
    X_train, y_train, groups_train = X[train_idx], y[train_idx], groups[train_idx]
    X_val, y_val, groups_val = X[val_idx], y[val_idx], groups[val_idx]
    
    # CALCULATE GROUP LENGTHS: Lambdarank requires the row count per screen block
    _, train_group_counts = np.unique(groups_train, return_counts=True)
    _, val_group_counts = np.unique(groups_val, return_counts=True)
    
    print(f"Dataset split completed. Train screens: {len(train_group_counts)} | Val screens: {len(val_group_counts)}")
    
    # Initialize Lambdarank datasets
    train_data = lgb.Dataset(X_train, label=y_train, group=train_group_counts)
    val_data = lgb.Dataset(X_val, label=y_val, group=val_group_counts, reference=train_data)
    
    # Optimized tree hyperparameters for sequence-extracted data
    params = {
        "objective": "lambdarank",       # List-wise ranking framework
        "metric": "ndcg",                # Normalized Discounted Cumulative Gain
        "ndcg_eval_at":[1, 2, 3],        # Evaluates 1, 2, and 3 during training for smoother convergence gradients
        "boosting_type": "gbdt",
        "learning_rate": 0.02,           # Slower rate for deep step optimization convergence
        "num_leaves": 127,                # Wide trees for deep card/relic synergy captures
        "max_depth": 10,
        # # === THE ANTI-OVERFITTING FIXES ===
        # "min_data_in_leaf": 100,         # CRITICAL: A leaf MUST apply to at least 100 rows to exist. Blocks memorizing single runs!
        "feature_fraction": 0.8,         # Forces trees to ignore dominant blocks and evaluate secondary card synergies smoothly
        # "bagging_fraction": 0.8,         # Trains each tree on a random 80% subsample of rows to prevent overfitting
        # "bagging_freq": 1,
        "verbosity": -1,
        "n_jobs": -1                     # Parallelize over all available CPU threads in RAM
    }
    
    print("Training LightGBM Lambdarank List-Wise Ranker...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=150,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30), # Increased runway to coast past local plateaus
            lgb.log_evaluation(25)
        ]
    )
    
    # =========================================================================
    # VALIDATION PHASE: COMPUTE TRUE TOP-1 CHOICE SELECTION ACCURACY
    # =========================================================================
    print("\nCalculating true Top-1 choice selection accuracy on validation screens...")
    val_preds = model.predict(X_val)
    
    group_max_score = {}
    group_winning_target = {}
    
    # Iterate across rows to map out screen choices
    for idx in range(len(val_preds)):
        g_id = int(groups_val[idx])
        score = float(val_preds[idx])
        is_pick = int(y_val[idx])
        
        # Identify the single highest ranking element prediction score in this decision block
        if g_id not in group_max_score or score > group_max_score[g_id]:
            group_max_score[g_id] = score
            group_winning_target[g_id] = is_pick
            
    total_screens = len(group_winning_target)
    correct_selections = sum(1 for g in group_winning_target if group_winning_target[g] == 1)
    
    if total_screens > 0:
        top1_accuracy = (correct_selections / total_screens) * 100
        print(f"-> Lambdarank Evaluation | Total Screens Checked: {total_screens} | Top-1 Accuracy: {top1_accuracy:.2f}%")
    else:
        print("-> Evaluation Error: No validation groups detected.")
        
    model.save_model("spire_lgb_recommender.txt")
    print("LightGBM model weights successfully written to disk!")

if __name__ == "__main__":
    # Ensure the dimensions align exactly with dictionary lookups
    CARD_SIZE = len(pipeline.ALL_VANILLA_CARDS) 
    RELIC_SIZE = len(pipeline.ALL_VANILLA_RELICS)
    
    train_lightgbm_ranking("slay_the_spire_transformer.csv", CARD_SIZE, RELIC_SIZE)