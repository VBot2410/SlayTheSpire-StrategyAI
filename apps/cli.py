import json
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from spire_strategy.training.trainer import load_existing_model, train_model

CHARACTER_MAP = {"IRONCLAD": 0, "THE_SILENT": 1, "DEFECT": 2, "WATCHER": 3}

def update_inventory(current_set, input_string, vocab_dict, type_label):
    """Parses modifications (+Item, -Item) or resets inventory entirely."""
    input_string = input_string.strip()
    if not input_string:
        return
    if input_string.lower() == "clear":
        current_set.clear()
        print(f"Cleared all {type_label}.")
        return

    items = input_string.split(",")
    for item in items:
        item = item.strip()
        if not item:
            continue
        if item.startswith("+"):
            name = item[1:].strip()
            if name in vocab_dict:
                current_set.append(name) if isinstance(current_set, list) else current_set.add(name)
                print(f"Added {type_label}: {name}")
            else:
                print(f"Warning: '{name}' not found in historical vocabulary.")
        elif item.startswith("-"):
            name = item[1:].strip()
            if name in current_set:
                current_set.remove(name)
                print(f"Removed {type_label}: {name}")
            else:
                print(f"Warning: '{name}' is not in your current inventory.")
        else:
            if items.index(item) == 0:
                current_set.clear()
            name = item
            if name in vocab_dict:
                current_set.append(name) if isinstance(current_set, list) else current_set.add(name)
            else:
                print(f"Warning: '{name}' not found in historical vocabulary.")

def run_cli(model):
    """Launches the persistent-state interactive terminal helper with auto-confirmation."""
    with open("card_vocab.json", "r") as f:
        card_vocab = json.load(f)
    with open("relic_vocab.json", "r") as f:
        relic_vocab = json.load(f)
        
    feature_cols = pd.read_csv("dataset.csv", nrows=1).drop(columns=["group_id", "target"]).columns.tolist()

    live_deck = []
    live_relics = set()
    char_id = 0
    floor = 1

    print("\n" + "="*50)
    print("   SLAY THE SPIRE PERSISTENT AI STATE ENGINE")
    print("="*50)
    
    print("Select Character (IRONCLAD, THE_SILENT, DEFECT, WATCHER):")
    char_input = input(">> ").strip().upper()
    char_id = CHARACTER_MAP.get(char_input, 0)
    
    if char_input == "IRONCLAD":
        live_deck = ["Strike_R"]*5 + ["Defend_R"]*4 + ["Bash"]
        live_relics = {"Burning Blood"}
    elif char_input == "THE_SILENT":
        live_deck = ["Strike_G"]*5 + ["Defend_G"]*5 + ["Neutralize", "Survivor"]
        live_relics = {"Ring of the Snake"}
    elif char_input == "DEFECT":
        live_deck = ["Strike_B"]*4 + ["Defend_B"]*4 + ["Zap", "Dualcast"]
        live_relics = {"Cracked Core"}
    elif char_input == "WATCHER":
        live_deck = ["Strike_P"]*4 + ["Defend_P"]*4 + ["Eruption", "Vigilance"]
        live_relics = {"Pure Water"}

    while True:
        print("\n" + "="*50)
        print(f" CURRENT STATUS | Floor: {floor} | Cards in Deck: {len(live_deck)} | Relics: {len(live_relics)}")
        print("="*50)
        print("Commands available on text screens: 'shop' (to modify deck/relics manually), 'exit' (to quit)")
        
        try:
            # Gather card options first to keep the layout fast
            print("\nEnter the 3 offered card reward choices (comma separated, or type 'shop' / 'exit'):")
            choice_input = input(">> ").strip()
            
            if choice_input.lower() == 'exit':
                break
            
            # Allow manual inventory management menu for shops, events, and campfires
            if choice_input.lower() == 'shop':
                print(f"\n[SHOP/EVENT MODE] Deck adjustments? (Press Enter to skip, '+Card' to add, '-Card' to lose):")
                deck_in = input(">> ")
                update_inventory(live_deck, deck_in, card_vocab, "Card")
                        
                print(f"[SHOP/EVENT MODE] Relic adjustments? (Press Enter to skip, '+Relic' to add, '-Relic' to lose):")
                relic_in = input(">> ")
                update_inventory(live_relics, relic_in, relic_vocab, "Relic")
                
                floor_in = input(f"Update Floor Number? [Current: {floor}] (Press Enter to keep): ").strip()
                if floor_in:
                    floor = int(floor_in)
                continue

            screen_choices = [c.strip() for c in choice_input.split(",") if c.strip() in card_vocab]
            if len(screen_choices) != 3:
                print(f"Warning: Requires exactly 3 recognized cards. Found {len(screen_choices)}.")
                continue

            # Construct structural feature rows
            base_row = {col: 0 for col in feature_cols}
            base_row["floor"] = floor
            base_row["character_class"] = char_id
            
            for card in live_deck:
                col_name = f"deck_{card_vocab[card]}"
                if col_name in base_row:
                    base_row[col_name] += 1
            for relic in live_relics:
                col_name = f"relic_{relic_vocab[relic]}"
                if col_name in base_row:
                    base_row[col_name] = 1

            eval_rows = []
            for choice in screen_choices:
                row = base_row.copy()
                row["candidate_card_id"] = card_vocab[choice]
                row["is_virtual_skip"] = 0
                eval_rows.append(row)
                
            skip_row = base_row.copy()
            skip_row["candidate_card_id"] = card_vocab["SKIP"]
            skip_row["is_virtual_skip"] = 1
            eval_rows.append(skip_row)

            eval_df = pd.DataFrame(eval_rows)[feature_cols]
            scores = model.predict(eval_df)
            
            # Display score rankings mapped to structural input indexes
            print("\n" + "-"*45 + "\n AI RECOMENDED CARD UTILITY RANKINGS\n" + "-"*45)
            results = []
            for i, choice in enumerate(screen_choices):
                results.append((choice, scores[i], i + 1))
            results.append(("SKIP CARD REWARD SCREEN", scores[3], "skip"))
            
            # Sort descending for display purposes
            display_results = sorted(results, key=lambda x: x[1], reverse=True)
            for rank, (name, score, selection_id) in enumerate(display_results, 1):
                print(f"{rank}. [{selection_id}] {name:<30} | Utility: {score:.4f}")
            print("-"*45)

            # --- AUTOMATIC DECK CONFIRMATION INTERACTIVE PROMPT ---
            while True:
                print(f"\nWhich selection did you pick in-game? (Options: 1, 2, 3, or 'skip')")
                pick_input = input(">> ").strip().lower()
                
                if pick_input == 'skip':
                    print("--> You skipped the reward. Deck remains unchanged.")
                    break
                elif pick_input in ['1', '2', '3']:
                    chosen_card_name = screen_choices[int(pick_input) - 1]
                    live_deck.append(chosen_card_name)
                    print(f"--> Confirmed! Automatically added '{chosen_card_name}' to your deck state.")
                    break
                else:
                    print("Invalid input choice. Please enter 1, 2, 3, or 'skip'.")

            # Advance game state automatically to keep loops smooth
            floor += 1

        except Exception as e:
            print(f"Processing anomaly: {e}")

if __name__ == "__main__":
    # If you ever want to force retrain, simply delete or rename 'spire_ranker.txt'
    ranker_model = load_existing_model()
    if ranker_model is None:
        ranker_model = train_model()
    if ranker_model:
        run_cli(ranker_model)
