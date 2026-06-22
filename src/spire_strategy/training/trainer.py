import json
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit

MODEL_FILE = "spire_ranker.txt"

def train_model():
    """Trains the LightGBM Ranker utilizing Group-Based Validation Splits and Early Stopping."""
    if not os.path.exists("dataset.csv"):
        print("Error: dataset.csv missing. Please run parse_runs.py first.")
        return None

    print("Loading compiled spreadsheet dataset...")
    df = pd.read_csv("dataset.csv")
    
    # Force sort to keep group chunks uniform
    df = df.sort_values("group_id").reset_index(drop=True)
    
    # 1. GROUP-AWARE VALIDATION SPLIT
    # Standard random splits break ranking groups. We must keep all 4 rows of an event together.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, val_idx = next(gss.split(df, groups=df["group_id"]))
    
    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    
    # Calculate group sizing arrays for both sectors
    train_groups = train_df.groupby("group_id").size().to_numpy()
    val_groups = val_df.groupby("group_id").size().to_numpy()
    
    # Isolate targets and features
    y_train = train_df["target"].to_numpy()
    X_train = train_df.drop(columns=["group_id", "target"])
    
    y_val = val_df["target"].to_numpy()
    X_val = val_df.drop(columns=["group_id", "target"])
    
    print(f"Data split finalized: {len(train_groups)} training scenarios | {len(val_groups)} validation scenarios.")
    
    # 2. CONSTRUCT MODEL WITH EXTENDED TREE CAP
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=1000,       # Set high; early stopping will halt training when optimal
        learning_rate=0.03,      # Lower learning rate yields superior tree patterns
        max_depth=6,
        verbose=-1
    )
    
    # 3. FIT WITH EARLY STOPPING CALLBACKS
    print("Training ranking engine with live validation tracking...")
    ranker.fit(
        X_train, 
        y_train, 
        group=train_groups, 
        eval_set=[(X_val, y_val)],
        eval_group=[val_groups],
        categorical_feature=["character_class"],
        callbacks=[
            # Stops if validation NDCG score fails to improve for 20 straight trees
            lgb.early_stopping(stopping_rounds=20, verbose=True), 
            lgb.log_evaluation(period=10)
        ]
    )
    
    # Print diagnostic validation capabilities
    best_iter = ranker.booster_.best_iteration
    print(f"Model training converged successfully at iteration: {best_iter}")
    
    ranker.booster_.save_model(MODEL_FILE)
    print(f"--> Exported optimal model weights successfully to '{MODEL_FILE}'!")
    return ranker

def load_existing_model():
    """Loads a pre-trained LightGBM Booster model if it exists on disk."""
    if os.path.exists(MODEL_FILE):
        print(f"Found existing model weights at '{MODEL_FILE}'. Loading...")
        bst = lgb.Booster(model_file=MODEL_FILE)
        print("Model loaded successfully! Skipping training phase.")
        return bst
    return None