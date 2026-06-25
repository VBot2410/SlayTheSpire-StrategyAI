import lightgbm as lgb
import numpy as np
import pandas as pd
from spire_strategy.data import ALL_VANILLA_CARDS, ALL_VANILLA_RELICS

def generate_feature_names(card_vocab_size, relic_vocab_size):
    """Reconstructs exact string labels matching your training horizontal stack order."""
    # 1. Base Environment Metrics (5 features)
    feature_names = ["floor", "character_class", "ascension_level", "gold", "hp_ratio"]
    
    # 2. Skip and Deck Context Flags (2 features)
    feature_names += ["is_virtual_skip", "candidate_deck_count"]
    
    # 3. Candidate Card One-Hot Matrix (card_vocab_size + 1 features)
    for i in range(card_vocab_size + 1):
        if i == 0:
            feature_names.append("candidate_is_SKIP_index_0")
        else:
            card_name = ALL_VANILLA_CARDS[i - 1] if (i - 1) < len(ALL_VANILLA_CARDS) else f"unknown_card_{i}"
            feature_names.append(f"candidate_is_{card_name.upper()}")

    # 4. Relic Multi-Hot Vector (relic_vocab_size + 1 features)
    for i in range(relic_vocab_size + 1):
        relic_name = ALL_VANILLA_RELICS[i - 1] if (i - 1) < len(ALL_VANILLA_RELICS) else f"unknown_relic_{i}"
        feature_names.append(f"owned_relic_{relic_name.upper()}")

    # 5. Deck Frequency Vector (card_vocab_size + 1 features)
    for i in range(card_vocab_size + 1):
        card_name = ALL_VANILLA_CARDS[i - 1] if (i - 1) < len(ALL_VANILLA_CARDS) else f"unknown_card_{i}"
        feature_names.append(f"deck_count_of_{card_name.upper()}")

    # 6. Candidate * Total Relics Interaction Array (card_vocab_size + 1 features)
    for i in range(card_vocab_size + 1):
        if i == 0:
            feature_names.append("interaction_SKIP_x_total_relics")
        else:
            card_name = ALL_VANILLA_CARDS[i - 1] if (i - 1) < len(ALL_VANILLA_CARDS) else f"unknown_card_{i}"
            feature_names.append(f"interaction_{card_name.upper()}_x_total_relics")

    # 7. Candidate * Total Deck Size Interaction Array (card_vocab_size + 1 features)
    for i in range(card_vocab_size + 1):
        if i == 0:
            feature_names.append("interaction_SKIP_x_deck_size")
        else:
            card_name = ALL_VANILLA_CARDS[i - 1] if (i - 1) < len(ALL_VANILLA_CARDS) else f"unknown_card_{i}"
            feature_names.append(f"interaction_{card_name.upper()}_x_deck_size")

    return feature_names

def profile_act_models(model_dir="."):
    card_vocab_size = len(ALL_VANILLA_CARDS)
    relic_vocab_size = len(ALL_VANILLA_RELICS)
    
    # Generate the comprehensive map keys
    feature_names = generate_feature_names(card_vocab_size, relic_vocab_size)
    
    acts = {
        "Act 1 (Floors 1-17)": "spire_lgb_act_1.txt",
        "Act 2 (Floors 18-34)": "spire_lgb_act_2.txt",
        "Act 3 (Floors 35+)": "spire_lgb_act_3.txt"
    }
    
    print(f"[PROFILER] Calculated exact vector footprint: {len(feature_names)} features.")

    for act_label, file_name in acts.items():
        if not os.path.exists(file_name):
            print(f"⚠️ Model file missing: {file_name}. Skipping profile.")
            continue
            
        print(f"\n=======================================================")
        print(f"🔮 TOP SPLIT DRIVERS FOR: {act_label.upper()}")
        print(f"=======================================================")
        
        # Load booster and pull split frequency counts
        bst = lgb.Booster(model_file=file_name)
        importance_splits = bst.feature_importance(importance_type="split")
        
        # Guard against zero-split trees (if a model completely flattened)
        if np.sum(importance_splits) == 0:
            print(" -> This model tree is empty or contains no splits.")
            continue
            
        # Create structured framework for sorting
        df_importance = pd.DataFrame({
            "Feature_Index": np.arange(len(importance_splits)),
            "Feature_Name": feature_names[:len(importance_splits)], # Clip safety guard
            "Split_Count": importance_splits
        })
        
        # Filter down to only nodes that actually triggered a branch split
        active_splits = df_importance[df_importance["Split_Count"] > 0].sort_values(by="Split_Count", ascending=False)
        
        for idx, row in active_splits.head(12).iterrows():
            print(f" Rank {idx+1:<2} | Node idx: {row['Feature_Index']:<4} | Splits: {int(row['Split_Count']):<3} | Driver: {row['Feature_Name']}")

if __name__ == "__main__":
    import os
    profile_act_models()
