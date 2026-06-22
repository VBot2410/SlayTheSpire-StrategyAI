import os
import json
import glob
import pandas as pd

# --- CONFIGURATION FILTERS ---
# Update this path to match your local system
RUNS_DIR = os.path.expanduser("~/.steam/steam/steamapps/common/SlayTheSpire/runs")

MIN_FLOOR_THRESHOLD = 6      # Drops extremely short or abandoned runs
MIN_ASCENSION_LEVEL = 10     # Only trains on high-difficulty/optimized choices (Set 0 for all runs)

# Fixed mapping for character classes
CHARACTER_MAP = {
    "IRONCLAD": 0,
    "THE_SILENT": 1,
    "DEFECT": 2,
    "WATCHER": 3
}

def build_vocabularies():
    """Scans all JSON files once to build a fixed ID mapping for cards and relics."""
    all_cards = {"SKIP"}
    all_relics = set()
    
    json_files = glob.glob(os.path.join(RUNS_DIR, "**/*.json"), recursive=True)
    if not json_files:
        print(f"No run files found in {RUNS_DIR}. Please check your path.")
        return {}, {}

    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Apply filters during vocabulary building to remain consistent
                if data.get("floor_reached", 1) < MIN_FLOOR_THRESHOLD:
                    continue
                if data.get("ascension_level", 0) < MIN_ASCENSION_LEVEL:
                    continue
                    
                for relic in data.get("relics", []):
                    all_relics.add(relic)
                for choice in data.get("card_choices", []):
                    all_cards.add(choice.get("picked", "SKIP") or "SKIP")
                    for unpicked in choice.get("not_picked", []):
                        all_cards.add(unpicked)
        except Exception:
            continue

    card_vocab = {card: idx for idx, card in enumerate(sorted(all_cards))}
    relic_vocab = {relic: idx for idx, relic in enumerate(sorted(all_relics))}
    
    with open("card_vocab.json", "w") as f:
        json.dump(card_vocab, f, indent=2)
    with open("relic_vocab.json", "w") as f:
        json.dump(relic_vocab, f, indent=2)
        
    print(f"Vocabularies saved! Found {len(card_vocab)} unique cards and {len(relic_vocab)} unique relics.")
    return card_vocab, relic_vocab

def extract_choice_rows():
    """Flattens choices into groups of 4 rows, applying filters and progress-scaled weighting."""
    if not os.path.exists("card_vocab.json"):
        card_vocab, relic_vocab = build_vocabularies()
    else:
        with open("card_vocab.json", "r") as f:
            card_vocab = json.load(f)
        with open("relic_vocab.json", "r") as f:
            relic_vocab = json.load(f)

    if not card_vocab:
        return

    flattened_data = []
    group_id = 0
    skipped_runs_count = 0

    json_files = glob.glob(os.path.join(RUNS_DIR, "**/*.json"), recursive=True)
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                run = json.load(f)
                
                # 1. APPLY GLOBAL FILTERS
                floor_reached = run.get("floor_reached", 1)
                ascension_level = run.get("ascension_level", 0)
                
                if floor_reached < MIN_FLOOR_THRESHOLD or ascension_level < MIN_ASCENSION_LEVEL:
                    skipped_runs_count += 1
                    continue
                
                char_string = run.get("character_chosen", "").upper()
                if char_string not in CHARACTER_MAP:
                    continue  
                char_id = CHARACTER_MAP[char_string]
                
                # 2. AUTOMATIC PROGRESS-WEIGHT CALCULATION
                victory = run.get("victory", False)
                if victory:
                    # Winning runs get max structural choice score authority
                    score_modifier = 1.0  
                else:
                    # Smoothly scale weight between 0.1 and 0.8 based on survival distance.
                    # Max standard floor is 57.
                    score_modifier = 0.1 + (min(floor_reached, 57) / 57.0) * 0.7
                
                current_relics = set()
                deck_counts = {card: 0 for card in card_vocab}
                
                for relic in run.get("relics", []):
                    current_relics.add(relic)

                for choice in run.get("card_choices", []):
                    floor = choice.get("floor", 1)
                    picked_card = choice.get("picked", "") or "SKIP"
                    options = list(choice.get("not_picked", []))
                    
                    if picked_card != "SKIP" and picked_card in card_vocab:
                        options.append(picked_card)
                    
                    if len(options) != 3:
                        continue 
                    
                    # Build Base Context Features
                    context = {
                        "group_id": group_id, 
                        "floor": floor,
                        "character_class": char_id
                    }
                    for c_name, c_idx in card_vocab.items():
                        context[f"deck_{c_idx}"] = deck_counts[c_name]
                    for r_name, r_idx in relic_vocab.items():
                        context[f"relic_{r_idx}"] = 1 if r_name in current_relics else 0

                    # Generate rows for the 3 physical cards
                    for card_option in options:
                        if card_option not in card_vocab:
                            continue
                        row = context.copy()
                        row["candidate_card_id"] = card_vocab[card_option]
                        row["is_virtual_skip"] = 0
                        # Multiply binary pick by our dynamic run quality score modifier
                        row["target"] = score_modifier if card_option == picked_card else 0.0
                        flattened_data.append(row)

                    # Generate row for the Virtual Skip action
                    skip_row = context.copy()
                    skip_row["candidate_card_id"] = card_vocab["SKIP"]
                    skip_row["is_virtual_skip"] = 1
                    skip_row["target"] = score_modifier if picked_card == "SKIP" else 0.0
                    flattened_data.append(skip_row)

                    group_id += 1
                    
                    if picked_card != "SKIP" and picked_card in deck_counts:
                        deck_counts[picked_card] += 1
        except Exception:
            continue

    if flattened_data:
        df = pd.DataFrame(flattened_data)
        df.to_csv("dataset.csv", index=False)
        print(f"Dataset compiled into dataset.csv with {group_id} decision scenarios.")
        print(f"Filtered out {skipped_runs_count} runs based on Floor (<{MIN_FLOOR_THRESHOLD}) or Ascension (<{MIN_ASCENSION_LEVEL}) limits.")
    else:
        print("Dataset creation failed: No runs matched your specified filtering criteria thresholds.")

if __name__ == "__main__":
    extract_choice_rows()
