import os
import csv
import random
import subprocess
import orjson

# --- CONFIGURATION PARAMETERS ---
SEVENZ_ARCHIVE_PATH = os.getenv("SPIRE_DATA_ARCHIVE", "archive.7z")
OUTPUT_CSV = "dataset.csv"
TEMP_RAW_ROWS = "dataset.tmp"

MIN_FLOOR_THRESHOLD = 6
MIN_ASCENSION_LEVEL = 10

# Scale this up as high as you want (even 40000+)
TARGET_RUNS_LIMIT = 4 
QUALIFICATION_SAFETY_MULTIPLIER = 2.0

CHARACTER_MAP = {"IRONCLAD": 0, "THE_SILENT": 1, "DEFECT": 2, "WATCHER": 3}
STARTING_DECKS = {
    0: ["Strike_R"] * 5 + ["Defend_R"] * 4 + ["Bash"],
    1: ["Strike_G"] * 5 + ["Defend_G"] * 5 + ["Neutralize", "Survivor"],
    2: ["Strike_B"] * 4 + ["Defend_B"] * 4 + ["Zap", "Dualcast"],
    3: ["Strike_P"] * 4 + ["Defend_P"] * 4 + ["Eruption", "Vigilance"]
}

def find_working_7zip_command():
    for cmd in ["7z", "7zz", "7zip"]:
        try:
            subprocess.run([cmd, "-h"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return cmd
        except FileNotFoundError:
            continue
    raise RuntimeError("Could not find a valid 7-Zip installation.")

SEVEN_ZIP_CMD = find_working_7zip_command()

def get_archive_file_list(archive_path):
    result = subprocess.run(
        [SEVEN_ZIP_CMD, "l", "-ba", "-slt", archive_path], 
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8"
    )
    filenames = []
    for line in result.stdout.splitlines():
        if line.startswith("Path = ") and line.endswith(".json"):
            filenames.append(line.split("Path = ", 1)[1].strip())
    return filenames

def execute_scalable_pipeline():
    all_files = get_archive_file_list(SEVENZ_ARCHIVE_PATH)
    raw_sample_size = min(int(TARGET_RUNS_LIMIT * QUALIFICATION_SAFETY_MULTIPLIER), len(all_files))
    
    random.shuffle(all_files)
    sample_targets = all_files[:raw_sample_size] 
    
    card_vocab = {"SKIP": 0}
    relic_vocab = {}
    processed_count = 0
    group_id = 0
        
    print(f"Starting memory-safe streaming pipeline. Target: {TARGET_RUNS_LIMIT} qualified runs.")
    
    # Open intermediate scratch file to stream data incrementally to disk
    with open(TEMP_RAW_ROWS, mode="w", newline="", encoding="utf-8") as temp_file:
        writer = csv.writer(temp_file)
        
        # Process target files one by one using 7z stdout streaming to prevent RAM bloat
        for file_path in sample_targets:
            if processed_count >= TARGET_RUNS_LIMIT:
                break
                
            try:
                cmd = [SEVEN_ZIP_CMD, "e", SEVENZ_ARCHIVE_PATH, file_path, "-so", "-y"]
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                
                if proc.returncode != 0 or not proc.stdout:
                    continue
                
                # Parse raw bytes into JSON memory array
                file_items = orjson.loads(proc.stdout)
                del proc 
                
                # Handle unexpected structures gracefully (ensure file contains a list layout)
                if not isinstance(file_items, list):
                    file_items = [file_items]
                
                # Loop through individual nested run events inside the array wrapper
                for item in file_items:
                    if processed_count >= TARGET_RUNS_LIMIT:
                        break
                        
                    data = item.get("event") if isinstance(item, dict) and "event" in item else item
                    if not isinstance(data, dict):
                        continue
                    
                    # FILTER RUNS IN RAM: Checks requirements BEFORE committing writing spikes onto storage disk
                    if data.get("floor_reached", 1) < MIN_FLOOR_THRESHOLD: continue
                    if data.get("ascension_level", 0) < MIN_ASCENSION_LEVEL: continue
                    char_str = data.get("character_chosen")
                    if char_str not in CHARACTER_MAP: continue
                    char_id = CHARACTER_MAP[char_str]
                    card_choices = data.get("card_choices", [])
                    if not card_choices: continue
                    
                    processed_count += 1
                    
                    current_relics = set()
                    relics_by_floor = {int(r["floor"]): r["key"] for r in data.get("relics_obtained", []) if "floor" in r and r.get("floor") is not None}
                    running_deck = STARTING_DECKS.get(char_id, []).copy()
                    card_choices.sort(key=lambda x: x.get("floor", 0))
                    
                    current_floor = 1
                    for choice in card_choices:
                        floor = int(choice.get("floor", 0))
                        for f_idx in range(current_floor, floor + 1):
                            if f_idx in relics_by_floor:
                                r_name = relics_by_floor[f_idx]
                                current_relics.add(r_name)
                                relic_vocab[r_name] = relic_vocab.get(r_name, len(relic_vocab))
                        current_floor = floor
                        
                        # CRITICAL CHANGE: Strip upgrades using split("+")[0] to find the base name
                        raw_picked = choice.get("picked", "SKIP")
                        picked = raw_picked.split("+")[0] if raw_picked != "SKIP" else "SKIP"
                        
                        not_picked = [card.split("+")[0] for card in choice.get("not_picked", [])]
                        
                        if picked not in card_vocab: card_vocab[picked] = len(card_vocab)
                        for npc in not_picked:
                            if npc not in card_vocab: card_vocab[npc] = len(card_vocab)
                        for card in running_deck:
                            if card not in card_vocab: card_vocab[card] = len(card_vocab)

                        deck_snapshot = {}
                        for card in running_deck:
                            deck_snapshot[card] = deck_snapshot.get(card, 0) + 1
                        
                        writer.writerow([group_id, floor, char_id, orjson.dumps(list(current_relics)).decode('utf-8'), orjson.dumps(deck_snapshot).decode('utf-8'), picked, 0, 1 if picked != "SKIP" else 0])
                        for npc in not_picked:
                            writer.writerow([group_id, floor, char_id, orjson.dumps(list(current_relics)).decode('utf-8'), orjson.dumps(deck_snapshot).decode('utf-8'), npc, 0, 0])
                        if picked == "SKIP":
                            writer.writerow([group_id, floor, char_id, orjson.dumps(list(current_relics)).decode('utf-8'), orjson.dumps(deck_snapshot).decode('utf-8'), "SKIP", 1, 1])
                            
                        if picked != "SKIP":
                            running_deck.append(picked)
                            
                    group_id += 1
                    
                    if processed_count % 10 == 0:
                        print(f" -> Logged {processed_count} qualified runs to disk storage...")
                        
            except Exception:
                continue

    print("Saving vocabulary maps...")
    with open("card_vocab.json", "wb") as f: f.write(orjson.dumps(card_vocab))
    with open("relic_vocab.json", "wb") as f: f.write(orjson.dumps(relic_vocab))

    # =========================================================================
    # PASS 2: STREAM COMPUTE FLAT MATRIX ROWS
    # =========================================================================
    print(f"\nVectorizing intermediate file into final dataset layout...")
    sorted_relics = sorted(relic_vocab, key=relic_vocab.get)
    sorted_cards = sorted(card_vocab, key=card_vocab.get)

    headers = ["group_id", "floor", "character_class"]
    headers += [f"relic_{relic_vocab[idx]}" for idx in sorted_relics]
    headers += [f"deck_{card_vocab[idx]}" for idx in sorted_cards]
    headers += ["candidate_card_id", "is_virtual_skip", "target"]

    relic_indices = {name: idx for idx, name in enumerate(sorted_relics)}
    card_indices = {name: idx for idx, name in enumerate(sorted_cards)}
    
    num_relics = len(relic_vocab)
    num_cards = len(card_vocab)

    with open(TEMP_RAW_ROWS, "r", encoding="utf-8") as temp_in, open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csv_out:
        reader = csv.reader(temp_in)
        writer = csv.writer(csv_out)
        
        writer.writerow(headers)
        
        for row in reader:
            group_id, floor, char_id, relics_json, deck_json, candidate, is_virtual, target = row
            
            current_relics = orjson.loads(relics_json)
            deck_snapshot = orjson.loads(deck_json)
            
            relic_vector = [0] * num_relics
            for r in current_relics:
                if r in relic_indices:
                    relic_vector[relic_indices[r]] = 1
                    
            deck_vector = [0] * num_cards
            for card, count in deck_snapshot.items():
                if card in card_indices:
                    deck_vector[card_indices[card]] = count
                    
            candidate_id = card_indices.get(candidate, -1)
            
            flat_row = [group_id, floor, char_id] + relic_vector + deck_vector + [candidate_id, int(is_virtual), int(target)]
            writer.writerow(flat_row)

    if os.path.exists(TEMP_RAW_ROWS):
        os.remove(TEMP_RAW_ROWS)
        
    print(f"Pipeline finished! Final dataset stored at: {OUTPUT_CSV}")

if __name__ == "__main__":
    execute_scalable_pipeline()
