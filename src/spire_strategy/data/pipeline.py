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
TARGET_RUNS_LIMIT = 50000
QUALIFICATION_SAFETY_MULTIPLIER = 4.0

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
    """
    Single-pass architecture:
    1. Extract 7z entries in memory.
    2. Parse JSON -> Tokenize -> Write directly to final CSV.
    No intermediate temporary files.
    """
    
    # --- PREPARE TARGET PATHS AND INPUTS ---
    if not os.path.exists(SEVENZ_ARCHIVE_PATH):
        raise FileNotFoundError(f"Archive not found: {SEVENZ_ARCHIVE_PATH}")
        
    all_files = get_archive_file_list(SEVENZ_ARCHIVE_PATH)
    if not all_files:
        print("No JSON files found in archive.")
        return

    # Sample target files cleanly
    raw_sample_size = min(int(TARGET_RUNS_LIMIT * QUALIFICATION_SAFETY_MULTIPLIER), len(all_files))
    random.shuffle(all_files)
    sample_targets = all_files[:raw_sample_size]
    
    # Determine if we need to write structural CSV headers
    write_header = not os.path.exists(TRANSFORMER_OUTPUT_CSV) or os.path.getsize(TRANSFORMER_OUTPUT_CSV) == 0
    
    # Get final group_id to resume sequence if file already has historical records
    current_group_id = 0
    if not write_header:
        try:
            with open(TRANSFORMER_OUTPUT_CSV, "rb") as f:
                f.seek(0, os.SEEK_END)
                end_pos = f.tell()
                buffer_size = 1024
                if end_pos > buffer_size:
                    f.seek(end_pos - buffer_size)
                    bytes_data = f.read(buffer_size)
                else:
                    f.seek(0)
                    bytes_data = f.read()
                
                lines = bytes_data.split(b"\n")
                last_line = lines[-1] if lines[-1] else lines[-2]
                last_group_id_str = last_line.split(b",")[0].decode('utf-8')
                
                if last_group_id_str.isdigit():
                    current_group_id = int(last_group_id_str) + 1
                    print(f"Resuming pipeline. Last group_id was {current_group_id - 1}. Starting at: {current_group_id}")
                else:
                    print("Found header only. Starting group_id at: 0")
        except Exception as e:
            print(f"Warning: Could not resume group_id ({e}). Starting at: 0")

    total_runs_processed = 0
    print(f"Starting Single-Pass Transformer Pipeline. Target: {TARGET_RUNS_LIMIT} runs.")
    print(f"Source files to process: {len(sample_targets)} (shuffled sample).")

    # -------------------------------------------------------------------------
    # PERSISTENT FILE SCOPE: Open once, write continuous rows from RAM streams
    # -------------------------------------------------------------------------
    with open(TRANSFORMER_OUTPUT_CSV, "a", newline="", encoding="utf-8") as csv_out:
        writer = csv.writer(csv_out)
        
        if write_header:
            headers = ["group_id", "floor", "character_class", "ascension_level", "gold", "hp_ratio", "relic_seq", "deck_seq", "candidate_card_id", "is_virtual_skip", "target"]
            writer.writerow(headers)
            print("Written new CSV headers.")

        # Main iteration path over all individual archive entries
        for file_path in sample_targets:
            if total_runs_processed >= TARGET_RUNS_LIMIT:
                print(f"Reached target limit ({TARGET_RUNS_LIMIT} runs). Stopping.")
                break
                
            proc = None
            try:
                # 1. Stream single file to RAM using Popen to prevent process blocking locks
                cmd = [SEVEN_ZIP_CMD, "e", SEVENZ_ARCHIVE_PATH, file_path, "-so", "-y"]
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                
                stdout_data, _ = proc.communicate()
                if proc.returncode != 0 or not stdout_data:
                    continue
                    
                file_items = orjson.loads(stdout_data)
                del stdout_data  # Instantly flush raw bytes from memory
                
                if not isinstance(file_items, list):
                    file_items = [file_items]
                    
                # 2. Process each run in the extracted file structure
                for item in file_items:
                    if total_runs_processed >= TARGET_RUNS_LIMIT:
                        break

                    data = item.get("event") if isinstance(item, dict) and "event" in item else item
                    if not isinstance(data, dict):
                        continue
                        
                    # Pre-flight game filters
                    floor_reached = int(data.get("floor_reached", 1))
                    if floor_reached < MIN_FLOOR_THRESHOLD:
                        continue
                        
                    ascension = int(data.get("ascension_level", 0))
                    if ascension < MIN_ASCENSION_LEVEL:
                        continue
                        
                    raw_char_str = data.get("character_chosen", "")
                    char_str = normalize_game_string(raw_char_str)
                    if char_str not in CHARACTER_MAP:
                        continue
                        
                    char_id = CHARACTER_MAP[char_str]
                    card_choices = data.get("card_choices", [])
                    if not card_choices:
                        continue

                    # WIN-GATE: Filter out sub-optimal play. Only keep elite wins or deep runs!
                    is_victory = bool(data.get("victory", False))
                    if not is_victory and floor_reached < 50:
                        continue

                    # Mark this run as qualified
                    total_runs_processed += 1

                    if total_runs_processed % 100 == 0 and total_runs_processed > 0:
                        print(f" -> Processed {total_runs_processed} total run logs...")
                    
                    # --- RUN STATE SETUP ---
                    gold_timeline = data.get("gold_per_floor", [])
                    hp_timeline = data.get("hp_per_floor", [])
                    max_hp_timeline = data.get("max_hp_per_floor", [])
                    
                    current_relics = set()
                    relics_by_floor = {
                        int(r["floor"]): normalize_game_string(r.get("key", "")) 
                        for r in data.get("relics_obtained", []) 
                        if "floor" in r and r.get("floor") is not None
                    }
                    
                    deck_snapshot = {}
                    for card in STARTING_DECKS.get(char_id, []):
                        deck_snapshot[card] = deck_snapshot.get(card, 0) + 1

                    if ascension >= 10:
                        deck_snapshot["ascendersbane"] = 1
                    
                    card_choices.sort(key=lambda x: x.get("floor", 0))
                    current_floor = 1
                    
                    # Cache compiled text rows for this individual log asset file
                    file_rows = []
                    
                    # --- PROCESS CHOICES FOR THIS RUN ---
                    for choice in card_choices:
                        floor = int(choice.get("floor", 0))
                        
                        # Fix: Prevent Floor 0 from resetting current_floor backward below 1
                        # This protects relic tracking sequence ranges from corrupting or duplicating
                        start_range = min(current_floor, floor)
                        for f_idx in range(start_range, floor + 1):
                            if f_idx in relics_by_floor:
                                current_relics.add(relics_by_floor[f_idx])
                        current_floor = max(current_floor, floor)
                        
                        # Process metrics indexes securely (Guards against Floor 0 negative indexes)
                        timeline_idx = floor - 1
                        
                        if 0 <= timeline_idx < len(gold_timeline):
                            gold_val = gold_timeline[timeline_idx]
                        else:
                            gold_val = 99

                        if 0 <= timeline_idx < len(hp_timeline):
                            curr_hp = hp_timeline[timeline_idx]
                        else:
                            curr_hp = 70

                        if 0 <= timeline_idx < len(max_hp_timeline):
                            max_hp = max_hp_timeline[timeline_idx]
                        else:
                            max_hp = 70
                            
                        hp_ratio = round(float(curr_hp) / float(max_hp), 3) if max_hp > 0 else 1.0
                        
                        # Candidate Cards
                        raw_picked = choice.get("picked", "skip")
                        picked = normalize_game_string(raw_picked) if raw_picked != "skip" else "skip"
                        not_picked = [normalize_game_string(card) for card in choice.get("not_picked", [])]
                        
                        # --- TOKENIZE IMMEDIATELY (Single Pass Logic) ---
                        # 1. Relic Sequence Embedding Vectors (+1 handles PyTorch padding alignment)
                        unknown_relic_idx = RELIC_INDICES["unknownrelic"]
                        relic_seq = [(RELIC_INDICES[r] + 1) if r in RELIC_INDICES else (unknown_relic_idx + 1) for r in current_relics][:MAX_RELIC_LEN]
                        relic_seq += [0] * (MAX_RELIC_LEN - len(relic_seq))
                        relic_seq_str = orjson.dumps(relic_seq).decode('utf-8')
                        
                        # 2. Deck Sequence Embedding Vectors
                        raw_deck_list = []
                        for card, count in deck_snapshot.items():
                            if card in CARD_INDICES:
                                card_token = CARD_INDICES[card] + 1
                                raw_deck_list.extend([card_token] * count)
                        deck_seq = raw_deck_list[:MAX_DECK_LEN]
                        deck_seq += [0] * (MAX_DECK_LEN - len(deck_seq))
                        deck_seq_str = orjson.dumps(deck_seq).decode('utf-8')
                        
                        # --- CONSTRUCT FLATTENED GENERATIVE ROWS ---
                        if picked != "skip":
                            picked_id = CARD_INDICES.get(picked, 0) + 1
                            file_rows.append([current_group_id, floor, char_id, ascension, gold_val, hp_ratio, relic_seq_str, deck_seq_str, picked_id, 0, 1])
                            
                            for npc in not_picked:
                                npc_id = CARD_INDICES.get(npc, 0) + 1
                                file_rows.append([current_group_id, floor, char_id, ascension, gold_val, hp_ratio, relic_seq_str, deck_seq_str, npc_id, 0, 0])
                            
                            file_rows.append([current_group_id, floor, char_id, ascension, gold_val, hp_ratio, relic_seq_str, deck_seq_str, 0, 1, 0]) # Virtual skip row
                            
                            # Increment deck counter snapshot properties live
                            deck_snapshot[picked] = deck_snapshot.get(picked, 0) + 1
                        else:
                            for npc in not_picked:
                                npc_id = CARD_INDICES.get(npc, 0) + 1
                                file_rows.append([current_group_id, floor, char_id, ascension, gold_val, hp_ratio, relic_seq_str, deck_seq_str, npc_id, 0, 0])
                            
                            file_rows.append([current_group_id, floor, char_id, ascension, gold_val, hp_ratio, relic_seq_str, deck_seq_str, 0, 1, 1])
                            
                        # Increment tracking choice screens identifiers
                        current_group_id += 1
                        
                    # Bulk flush rows for this file straight to SSD OS cache buffers
                    if file_rows:
                        writer.writerows(file_rows)
                    
            except (orjson.JSONDecodeError, KeyError, ValueError):
                continue
            except Exception as system_err:
                if proc:
                    proc.kill()
                raise system_err

    print(f"\nProcessing complete. No temp files used. Output written cleanly to: {TRANSFORMER_OUTPUT_CSV}")

if __name__ == "__main__":
    run_transformer_pipeline()