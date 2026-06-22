import os
import csv
import random
import subprocess
import orjson

# --- CONFIGURATION PARAMETERS ---
SEVENZ_ARCHIVE_PATH = os.getenv("SPIRE_DATA_ARCHIVE", "archive.7z")
OUTPUT_CSV = "dataset.csv"
TEMP_RAW_ROWS = "dataset.tmp"
TRANSFORMER_OUTPUT_CSV = "slay_the_spire_transformer.csv"

MIN_FLOOR_THRESHOLD = 6
MIN_ASCENSION_LEVEL = 10

# --- SHAPE CONSTRAINTS FOR PYTORCH TENSORS ---
# Caps ensure consistent static sequence lengths for mini-batching
MAX_DECK_LEN = 60   
MAX_RELIC_LEN = 30

# Scale this up as high as you want!
TARGET_RUNS_LIMIT = 200
QUALIFICATION_SAFETY_MULTIPLIER = 2.0

CHARACTER_MAP = {"ironclad": 0, "thesilent": 1, "defect": 2, "watcher": 3}
STARTING_DECKS = {
    0: ["striker"] * 5 + ["defendr"] * 4 + ["bash"],
    1: ["strikeg"] * 5 + ["defendg"] * 5 + ["neutralize", "survivor"],
    2: ["strikeb"] * 4 + ["defendb"] * 4 + ["zap", "dualcast"],
    3: ["strikep"] * 4 + ["defendp"] * 4 + ["eruption", "vigilance"]
}

# =========================================================================
# COMPREHENSIVE STATIC GAME VOCABULARY MAPPINGS (Vanilla Base Content)
# =========================================================================
# Explicitly defining cards and relics prevents mid-run column drift bottlenecks.
ALL_VANILLA_CARDS = [
    "skip", "striker", "defendr", "bash", "strikeg", "defendg", "neutralize", "survivor",
    "strikeb", "defendb", "zap", "dualcast", "strikep", "defendp", "eruption", "vigilance",
    "anger", "armaments", "bodyslam", "clash", "cleave", "clothesline", "flex", "havoc", "headbutt",
    "heavyblade", "ironwave", "perfectedstrike", "pommelstrike", "shrugitoff", "swordboomerang", 
    "thunderclap", "twinstrike", "wildstrike", "battletrance", "bloodletting", "burningbarrier", 
    "carnage", "combust", "darkembrace", "disarm", "dualwield", "entrench", "evolve", "feelnopain",
    "firebreathing", "flamebarrier", "ghostlyarmor", "hemokinesis", "immolate", "inflame", 
    "intimidate", "metallicize", "powerthrough", "pummel", "rage", "rampage", "recklesscharge", 
    "rupture", "searingblow", "secondwind", "seeingred", "shockwave", "spotweakness", "uppercut", 
    "whirlwind", "barricade", "berserk", "bludgeon", "brutality", "corruption", "demonform", 
    "doubletap", "exhume", "feed", "fiendfire", "impervious", "juggernaut", "limitbreak", 
    "offering", "reaper", "bane", "daggerspray", "daggerthrow", "deadlypoison", "deflect", 
    "dodgeandroll", "flyingknee", "outmaneuver", "piercingwail", "poisonedstab", "prepared", 
    "quickslash", "slice", "sneakystrike", "suckerpunch", "acrobatics", "backflip", "blur", 
    "bouncingflask", "calculatedgamble", "caltrops", "catalyst", "choke", "dash", "distraction", 
    "endlessagony", "eviscerate", "expertise", "flechettes", "footwork", "heelhook", "infiniteblades", 
    "legsweep", "malaise", "masterfulstab", "noxiousfumes", "predator", "riddlewithholes", 
    "skewer", "terror", "welllaidplans", "athousandcuts", "adrenaline", "afterimage", "alchemize", 
    "bullettime", "burst", "diediedie", "doppelganger", "envenom", "glassknife", "grandfinale", 
    "phantasmalkiller", "nightmare", "toolsofthetrade", "stormofsteel", "unload", "wraithform", 
    "balllightning", "barrage", "beamcell", "chargebattery", "claw", "coldsnap", "compiledriver", 
    "coolheaded", "gash", "gofortheeyes", "hologram", "leap", "rebound", "steambarrier", 
    "sweepingbeam", "turbo", "aggregate", "autoshields", "blizzard", "bootsequence", "chill", 
    "consume", "defragment", "doomandgloom", "doubleenergy", "electrodynamics", "ftl", 
    "forcefield", "fusion", "geneticalgorithm", "heatsinks", "helloworld", "loop", "melter", 
    "overclock", "recycle", "reinforcedbody", "reprogram", "riptide", "scrape", "skim", 
    "staticdischarge", "storm", "sunder", "tempest", "whitenoise", "allforone", "amplify", 
    "biasedcognition", "buffer", "coresurge", "creativeai", "echoform", "fission", "hyperbeam", 
    "machinelearning", "meteorstrike", "multicast", "rainbow", "reboot", "seek", "thunderstrike", 
    "bow", "crushjoints", "flurryofblows", "flyingsleeves", "followup", "justlucky", "sashwhip", 
    "thirdeye", "cutthroughfate", "emptybody", "emptyfist", "evaluating", "proclaim", 
    "pressurepoints", "protect", "crescendo", "tranquility", "alpha", "battlehymn", "carvereality", 
    "collect", "conclude", "deceivereality", "devotion", "foreigninfluence", "indignation", 
    "innerpeace", "likewater", "meditation", "mentalfortress", "nirvana", "perseverance", 
    "pray", "reachheaven", "regretfulfists", "sanctity", "simmering_fury", "swivel", 
    "talktothehand", "tantrum", "waveofthehand", "wavesofthehand", "weave", "windmillstrike", 
    "worship", "wreathofflame", "blasphemy", "brilliance", "conjureblade", "deusexmachina", 
    "devaform", "establishment", "fasting", "lessonlearned", "masterreality", "omniscience", 
    "ragnarok", "scrawl", "spiritshield", "wish", "apparition", "bite", "blind", "darkshackles", 
    "deepbreath", "discovery", "dramaticentrance", "enlightenment", "finesse", "flashofsteel", 
    "forethought", "goodinstincts", "handofgreed", "impatience", "jackofalltrades", "madness", 
    "mindblast", "panacea", "panache", "purity", "sadisticnature", "secrettechnique", 
    "secretweapon", "swiftstrike", "thinkingahead", "transmutation", "violence", "chrysalis", 
    "magnetism", "mayhem", "metamorphosis", "masterofstrategy", "thebomb", "clumsy", "decay", 
    "doubt", "injury", "normality", "pain", "regret", "shame", "writhe", "parasite", 
    "necronomicurse", "curseofthebell", "ascendersbane", "slimed", "void", "dazed", "wound", 
    "burn", "jax", "seversoul", "warcry", "burningpact", "bloodforblood", "sentinel", 
    "truegrit", "dropkick", "wheelkick", "infernalblade", "streamline", "steam", "stack", 
    "conservebattery", "glacier", "capacitor", "selfrepair", "chaos", "darkness", "ripandtear", 
    "steampower", "redo", "alloutattack", "concentrate", "bladedance", "cripplingpoison", 
    "underhandedstrike", "backstab", "reflex", "escapeplan", "finisher", "lockon", "apotheosis", 
    "undo", "accuracy", "cloakanddagger", "setup", "corpseexplosion", "venomology", "tactician", 
    "singingbowl", "nightterror", "fearnoevil", "pathtovictory", "wireheading", "study", 
    "bowlingbash", "signaturemove", "halt", "clearthemind", "wallop", "unknowncard"
]

ALL_VANILLA_RELICS = [
    "burningblood", "ringofthesnake", "crackedcore", "purewater", "akabeko", "anchor", 
    "ancientteaset", "artofwar", "bagofmarbles", "bagofpreparation", "bloodvial", 
    "bronzescales", "centennialpuzzle", "ceramicfish", "dreamcatcher", "happyflower", 
    "juzubracelet", "lantern", "mawbank", "mealticket", "nunchaku", "oddlysmoothstone", 
    "omamori", "orichalcum", "pennib", "potionbelt", "preservedinsect", "regalpillow", 
    "smilingmask", "strawberry", "boot", "toyornithopter", "vajra", "warpaint", "bluecandle", 
    "bottledflame", "bottledlightning", "bottledtornado", "darkstoneperiapt", "eternalfeather", 
    "frozenegg", "horncleat", "inkbottle", "kunai", "letteropener", "matryoshka", "meatonthebone", 
    "mercuryhourglass", "moltenegg", "mummifiedhand", "ornamentalfan", "pantograph", 
    "pear", "questioncard", "shovel", "singingbowl", "strikedummy", "sundial", "toxicegg", 
    "whitebeaststatue", "birdfacedurn", "calipers", "captainswheel", "championbelt", 
    "charonsashes", "cloakclasp", "deadbranch", "duvudoll", "fossilizedhelix", "gamblingchip", 
    "ginger", "girya", "goldeneye", "icecream", "incenseburner", "lizardtail", "magicflower", 
    "mango", "oldcoin", "peacepipe", "pocketwatch", "prayerwheel", "shuriken", "stonecalendar", 
    "thecourier", "torii", "toughbandages", "tingsha", "turnip", "unceasingtop", "wingboots", 
    "astrolabe", "blackblood", "blackstar", "bustedcrown", "callingbell", "coffeedripper", 
    "cursedkey", "ectoplasm", "emptycage", "fusionhammer", "hoveringkite", "inversiondynamo", 
    "markofpain", "nuclearbattery", "pandorasbox", "philosophersstone", "runicdome", 
    "runicpyramid", "sacredbark", "slaverscollar", "sneckoeye", "sozu", "velvetchoker", 
    "wristblade", "cauldron", "chemicalx", "dollysmirror", "frozeneye", "leeswaffle", 
    "medicalkit", "membershipcard", "orangepellets", "orrery", "prismaticshard", 
    "slingofcourage", "strangespoon", "sling", "thespecimen", "niloscodex", "enchiridion", 
    "necronomicon", "mutagenicstrength", "goldenidol", "bloodyidol", "redmask", "spiritpoop", 
    "cultistheadpiece", "faceofcleric", "ssserpenthead", "gremlinmask", "nlothsgift", 
    "nlothshungryface", "warpedtongs", "oddmushroom", "neowslament", "circlet", "damaru", 
    "datadisk", "redskull", "sneckoskull", "goldplatedcables", "paperkrane", "paperphrog", 
    "teardroplocket", "emotionchip", "threadandneedle", "tungstenrod", "frozencore", 
    "holywater", "ringoftheserpent", "runiccapacitor", "runiccube", "tinyhouse", "handdrill", 
    "clockworksouvenir", "markofthebloom", "jax", "unknownrelic"
]

# Generate fast index lookups mapping string keys to fixed dense column vectors
CARD_INDICES = {name: idx for idx, name in enumerate(ALL_VANILLA_CARDS)}
RELIC_INDICES = {name: idx for idx, name in enumerate(ALL_VANILLA_RELICS)}

NUM_RELICS = len(ALL_VANILLA_RELICS)
NUM_CARDS = len(ALL_VANILLA_CARDS)

def normalize_game_string(input_str: str) -> str:
    """Standardizes game names by removing upgrades, casing, spaces, and dashes."""
    if not input_str:
        return ""
    # Strip card upgrade suffix if present (safe to run on relics too)
    base_name = input_str.split("+")[0]
    # Lowercase, remove spaces, underscores, and dashes
    return base_name.lower().replace(" ", "").replace("_", "").replace("-", "").strip()


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

def run_transformer_pipeline():
    if os.path.exists(TEMP_RAW_ROWS):
        os.remove(TEMP_RAW_ROWS)
    all_files = get_archive_file_list(SEVENZ_ARCHIVE_PATH)
    raw_sample_size = min(int(TARGET_RUNS_LIMIT * QUALIFICATION_SAFETY_MULTIPLIER), len(all_files))
    
    random.shuffle(all_files)
    sample_targets = all_files[:raw_sample_size] 
    
    processed_count = 0
    group_id = 0  # Default fallback for clean/new files
        
    # --- DYNAMICALLY RESOLVE STARTING GROUP_ID ---
    if os.path.exists(TRANSFORMER_OUTPUT_CSV) and os.path.getsize(TRANSFORMER_OUTPUT_CSV) > 0:
        try:
            with open(TRANSFORMER_OUTPUT_CSV, "rb") as f:
                # Seek to the end of the file to scan backward efficiently
                f.seek(0, os.SEEK_END)
                end_pos = f.tell()
                buffer_size = 1024
                
                # Slide backward through the byte buffer to locate the true final newline
                if end_pos > buffer_size:
                    f.seek(end_pos - buffer_size)
                    bytes_data = f.read(buffer_size)
                else:
                    f.seek(0)
                    bytes_data = f.read()
                    
                lines = bytes_data.split(b"\n")
                # Drop trailing empty line splits if they exist
                last_line = lines[-1] if lines[-1] else lines[-2]
                
                # Split the raw CSV string columns to grab the very first token (group_id)
                last_group_id_str = last_line.split(b",")[0].decode('utf-8')
                
                # Ensure the parsed item isn't the string header text block "group_id"
                if last_group_id_str.isdigit():
                    group_id = int(last_group_id_str) + 1
                    print(f"Resuming pipeline cleanly. Last found group_id was {group_id - 1}. Set next start index to: {group_id}")
                else:
                    print("Found file header but no rows. Starting group_id at: 0")
        except Exception as e:
            print(f"Warning: Failed reading tail metadata context ({e}). Defaulting group_id to: 0")
    else:
        print("No prior dataset detected or file is empty. Initializing brand new group_id sequence at: 0")

        
    print(f"Starting Transformer Sequence Pipeline. Target: {TARGET_RUNS_LIMIT} archive units.")
    
    # -------------------------------------------------------------------------
    # PASS 1: EXTRACT RUN CHRONOLOGY AND LOG SEMANTIC DENSE STRINGS
    # -------------------------------------------------------------------------
    with open(TEMP_RAW_ROWS, mode="w", newline="", encoding="utf-8") as temp_file:
        writer = csv.writer(temp_file)
        
        for file_path in sample_targets:
            if processed_count >= TARGET_RUNS_LIMIT:
                break
                
            try:
                # 1. Decompress a single file out to stdout (processed directly in RAM)
                cmd = [SEVEN_ZIP_CMD, "e", SEVENZ_ARCHIVE_PATH, file_path, "-so", "-y"]
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                if proc.returncode != 0 or not proc.stdout:
                    continue
                    
                file_items = orjson.loads(proc.stdout)
                del proc  # Free process memory immediately
                
                if not isinstance(file_items, list):
                    file_items = [file_items]
                    
                for item in file_items:
                    if processed_count >= TARGET_RUNS_LIMIT:
                        break

                    # Isolate the run event payload
                    data = item.get("event") if isinstance(item, dict) and "event" in item else item
                    if not isinstance(data, dict):
                        continue
                        
                    # Filter: Minimum floor reached
                    if data.get("floor_reached", 1) < MIN_FLOOR_THRESHOLD:
                        continue
                        
                    # Filter: Ascension level threshold
                    ascension = int(data.get("ascension_level", 0))
                    if ascension < MIN_ASCENSION_LEVEL:
                        continue
                        
                    # Filter: Valid character check
                    raw_char_str = data.get("character_chosen", "")
                    char_str = normalize_game_string(raw_char_str)
                    
                    if char_str not in CHARACTER_MAP:
                        continue
                        
                    char_id = CHARACTER_MAP[char_str]
                    card_choices = data.get("card_choices", [])
                    if not card_choices:
                        continue

                    processed_count += 1
                    
                    # Timelines metrics
                    gold_timeline = data.get("gold_per_floor", [])
                    hp_timeline = data.get("hp_per_floor", [])
                    max_hp_timeline = data.get("max_hp_per_floor", [])
                        
                    current_relics = set()
                    relics_by_floor = {
                        int(r["floor"]): normalize_game_string(r.get("key", "")) 
                        for r in data.get("relics_obtained", []) 
                        if "floor" in r and r.get("floor") is not None
                    }
                    
                    # --- Optimization: O(1) Deck Tracking Setup ---
                    # Instead of a flat list, we maintain a live-updating counter map
                    deck_snapshot = {}
                    for card in STARTING_DECKS.get(char_id, []):
                        deck_snapshot[card] = deck_snapshot.get(card, 0) + 1
                    
                    card_choices.sort(key=lambda x: x.get("floor", 0))
                    current_floor = 1
                    
                    # --- Optimization: RAM CSV Buffer ---
                    # Accumulate rows in memory per file, writing in one single bulk operation
                    file_rows = []
                    
                    for choice in card_choices:
                        floor = int(choice.get("floor", 0))
                        
                        # Catch up on any relics acquired between choices
                        for f_idx in range(current_floor, floor + 1):
                            if f_idx in relics_by_floor:
                                current_relics.add(relics_by_floor[f_idx])
                        current_floor = floor
                        
                        # Process metrics indexes securely
                        timeline_idx = floor - 1
                        gold_val = gold_timeline[timeline_idx] if timeline_idx < len(gold_timeline) else 99
                        curr_hp = hp_timeline[timeline_idx] if timeline_idx < len(hp_timeline) else 70
                        max_hp = max_hp_timeline[timeline_idx] if timeline_idx < len(max_hp_timeline) else 70
                        hp_ratio = round(float(curr_hp) / float(max_hp), 3) if max_hp > 0 else 1.0
                        
                        # Sanitize card identifiers (strip upgrade level '+1')
                        raw_picked = choice.get("picked", "skip")
                        picked = normalize_game_string(raw_picked) if raw_picked != "skip" else "skip"
                        not_picked = [normalize_game_string(card) for card in choice.get("not_picked", [])]
                        
                        # --- Optimization: JSON-serialize exactly once per floor ---
                        relics_str = orjson.dumps(list(current_relics)).decode('utf-8')
                        deck_str = orjson.dumps(deck_snapshot).decode('utf-8')
                        
                        # Construct flattened ML rows for modeling
                        if picked != "skip":
                            file_rows.append([group_id, floor, char_id, ascension, gold_val, hp_ratio, relics_str, deck_str, picked, 0, 1])
                            for npc in not_picked:
                                file_rows.append([group_id, floor, char_id, ascension, gold_val, hp_ratio, relics_str, deck_str, npc, 0, 0])
                            file_rows.append([group_id, floor, char_id, ascension, gold_val, hp_ratio, relics_str, deck_str, "skip", 1, 0])
                            
                            # --- Optimization: Fast O(1) deck update ---
                            deck_snapshot[picked] = deck_snapshot.get(picked, 0) + 1
                        else:
                            for npc in not_picked:
                                file_rows.append([group_id, floor, char_id, ascension, gold_val, hp_ratio, relics_str, deck_str, npc, 0, 0])
                            file_rows.append([group_id, floor, char_id, ascension, gold_val, hp_ratio, relics_str, deck_str, "skip", 1, 1])
                            
                        group_id += 1
                    
                    # --- Optimization: Single SSD Write Operation per Run File ---
                    if file_rows:
                        writer.writerows(file_rows)
                    
                if processed_count % 10 == 0:
                    print(f" -> Logged {processed_count} JSON entries...")
                    
            except Exception as e:
                # Silently catch malformed file errors to keep parser moving
                continue

    # -------------------------------------------------------------------------
    # PASS 2: CONVERT TO TRANSfORMER SEQUENCE TOKENS AND WRITE FINAL DATASET
    # -------------------------------------------------------------------------
    print(f"\nVectorizing intermediate file into Transformer sequence tokens...")
    
    headers = ["group_id", "floor", "character_class", "ascension_level", "gold", "hp_ratio", "relic_seq", "deck_seq", "candidate_card_id", "is_virtual_skip", "target"]
    
    # Check if the output file already exists and has data inside it
    file_exists_and_not_empty = os.path.exists(TRANSFORMER_OUTPUT_CSV) and os.path.getsize(TRANSFORMER_OUTPUT_CSV) > 0

    with open(TEMP_RAW_ROWS, "r", encoding="utf-8") as temp_in, open(TRANSFORMER_OUTPUT_CSV, "a", newline="", encoding="utf-8") as csv_out:
        reader = csv.reader(temp_in)
        writer = csv.writer(csv_out)
        # Only write headers if the file is empty/new
        if not file_exists_and_not_empty:
            writer.writerow(headers)
        
        for row in reader:
            group_id, floor, char_id, ascension, gold_val, hp_ratio, relics_json, deck_json, candidate, is_virtual, target = row
            
            current_relics = orjson.loads(relics_json)
            deck_snapshot = orjson.loads(deck_json)
            
            # --- TOKENIZE RELICS WITH ID SHIFT (0 = Padding Token) ---
            # If item is found, assign token = index + 1. Otherwise skip.
            unknown_relic_idx = RELIC_INDICES["unknownrelic"]
            relic_seq = [(RELIC_INDICES[r] + 1) if r in RELIC_INDICES else (unknown_relic_idx + 1) for r in current_relics][:MAX_RELIC_LEN]
            # Pad array with zeros to match static matrix size requirement
            relic_seq += [0] * (MAX_RELIC_LEN - len(relic_seq))
            
            # --- UNPACK BAG-OF-WORDS DICT BACK INTO A SEQUENTIAL LIST ---
            raw_deck_list = []
            for card, count in deck_snapshot.items():
                if card in CARD_INDICES:
                    card_token = CARD_INDICES[card] + 1  # ID Shift for padding
                    raw_deck_list.extend([card_token] * count)
            
            deck_seq = raw_deck_list[:MAX_DECK_LEN]
            deck_seq += [0] * (MAX_DECK_LEN - len(deck_seq))
            
            # --- TOKENIZE CANDIDATE ITEM ---
            unknown_idx = CARD_INDICES["unknowncard"]
            if candidate not in CARD_INDICES:
                print(f"DEBUG: Unmapped/Non-Vanilla Card encountered: '{candidate}'")
            candidate_id = CARD_INDICES.get(candidate, unknown_idx) + 1  # Shifts fallback default to 0
            
            # Stringify lists to protect the interior comma tokens during CSV writing
            relic_seq_str = orjson.dumps(relic_seq).decode('utf-8')
            deck_seq_str = orjson.dumps(deck_seq).decode('utf-8')
            
            flat_row = [
                group_id, floor, char_id, int(ascension), int(gold_val), float(hp_ratio),
                relic_seq_str, deck_seq_str, candidate_id, int(is_virtual), int(target)
            ]
            writer.writerow(flat_row)
            
    if os.path.exists(TEMP_RAW_ROWS):
        os.remove(TEMP_RAW_ROWS)
        
    print(f"Pipeline finished safely! Transformer dataset stored at: {TRANSFORMER_OUTPUT_CSV}")

if __name__ == "__main__":
    run_transformer_pipeline()