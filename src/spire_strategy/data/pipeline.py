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
TARGET_RUNS_LIMIT = 40000
QUALIFICATION_SAFETY_MULTIPLIER = 2.0

CHARACTER_MAP = {"IRONCLAD": 0, "THE_SILENT": 1, "DEFECT": 2, "WATCHER": 3}
STARTING_DECKS = {
    0: ["Strike_R"] * 5 + ["Defend_R"] * 4 + ["Bash"],
    1: ["Strike_G"] * 5 + ["Defend_G"] * 5 + ["Neutralize", "Survivor"],
    2: ["Strike_B"] * 4 + ["Defend_B"] * 4 + ["Zap", "Dualcast"],
    3: ["Strike_P"] * 4 + ["Defend_P"] * 4 + ["Eruption", "Vigilance"]
}

# =========================================================================
# COMPREHENSIVE STATIC GAME VOCABULARY MAPPINGS (Vanilla Base Content)
# =========================================================================
# Explicitly defining cards and relics prevents mid-run column drift bottlenecks.
ALL_VANILLA_CARDS = [
    "SKIP", "Strike_R", "Defend_R", "Bash", "Strike_G", "Defend_G", "Neutralize", "Survivor",
    "Strike_B", "Defend_B", "Zap", "Dualcast", "Strike_P", "Defend_P", "Eruption", "Vigilance",
    "Anger", "Armaments", "BodySlam", "Clash", "Cleave", "Clothesline", "Flex", "Havoc", "Headbutt",
    "HeavyBlade", "IronWave", "PerfectedStrike", "PommelStrike", "ShrugItOff", "SwordBoomerang", 
    "ThunderClap", "TwinStrike", "WildStrike", "BattleTrance", "Bloodletting", "BurningBarrier", 
    "Carnage", "Combust", "DarkEmbrace", "Disarm", "DualWield", "Entrench", "Evolve", "FeelNoPain", 
    "FireBreathing", "FlameBarrier", "GhostlyArmor", "Hemokinesis", "Immolate", "Inflame", 
    "Intimidate", "Metallicize", "PowerThrough", "Pummel", "Rage", "Rampage", "RecklessCharge", 
    "Rupture", "SearingBlow", "SecondWind", "SeeingRed", "Shockwave", "SpotWeakness", "Uppercut", 
    "Whirlwind", "Barricade", "Berserk", "Bludgeon", "Brutality", "Corruption", "DemonForm", 
    "DoubleTap", "Exhume", "Feed", "FiendFire", "Immolate", "Impervious", "Juggernaut", "LimitBreak", 
    "Offering", "Reaper", "Bane", "DaggerSpray", "DaggerThrow", "DeadlyPoison", "Deflect", 
    "DodgeAndRoll", "FlyingKnee", "Outmaneuver", "PiercingWail", "PoisonedStab", "Prepared", 
    "QuickSlash", "Slice", "SneakyStrike", "SuckerPunch", "Acrobatics", "Backflip", "Blur", 
    "BouncingFlask", "CalculatedGamble", "Caltrops", "Catalyst", "Choke", "Dash", "Distraction", 
    "EndlessAgony", "Eviscerate", "Expertise", "Flechettes", "Footwork", "HeelHook", "InfiniteBlades", 
    "LegSweep", "Malaise", "MasterfulStab", "NoxiousFumes", "Outmaneuver", "Predator", "RiddleWithHoles", 
    "Skewer", "Terror", "WellLaidPlans", "A Thousand Cuts", "Adrenaline", "AfterImage", "Alchemize", 
    "BulletTime", "Burst", "DieDieDie", "Doppelganger", "Envenom", "GlassKnife", "GrandFinale", 
    "Grand Finale", "Malaise", "PhantasmalKiller", "Nightmare", "ToolsOfTheTrade", "StormOfSteel", 
    "Unload", "WraithForm", "BallLightning", "Barrage", "BeamCell", "ChargeBattery", "Claw", 
    "ColdSnap", "CompilingDriver", "Coolheaded", "Gash", "GoForTheEyes", "Hologram", "Leap", 
    "Rebound", "SteamBarrier", "SweepingBeam", "TURBO", "Aggregate", "AutoShields", "Blizzard", 
    "BootSequence", "Chill", "Consume", "Defragment", "DoomAndGloom", "DoubleEnergy", "Electrodynamics", 
    "FTL", "ForceField", "Fusion", "GeneticAlgorithm", "Heatsinks", "HelloWorld", "Loop", 
    "Melter", "Overclock", "Recycle", "ReinforcedBody", "Reprogram", "Riptide", "Scrape", 
    "Skim", "StaticDischarge", "Storm", "Sunder", "Tempest", "WhiteNoise", "AllForOne", 
    "Amplify", "BiasedCognition", "Buffer", "CoreSurge", "CreativeAI", "EchoForm", "Fission", 
    "Hyperbeam", "MachineLearning", "MeteorStrike", "Multi-Cast", "Rainbow", "Reboot", "Seek", 
    "ThunderStrike", "Bow", "CrushJoints", "FlurryOfBlows", "FlyingSleeves", "FollowUp", 
    "JustLucky", "SashWhip", "ThirdEye", "CutThroughFate", "EmptyBody", "EmptyFist", "Evaluating", 
    "Proclaim", "PressurePoints", "Protect", "Crescendo", "Tranquility", "Alpha", "BattleHymn", 
    "CarveReality", "Collect", "Conclude", "DeceiveReality", "Devotion", "ForeignInfluence", 
    "Indignation", "InnerPeace", "LikeWater", "Meditation", "MentalFortress", "Nirvana", 
    "Perseverance", "Pray", "ReachHeaven", "RegretfulFists", "Sanctity", "SimmeringFury", 
    "Swivel", "TalkToTheHand", "Tantrum", "WaveOfTheHand", "WavesOfTheHand", "Weave", "WindmillStrike", 
    "Worship", "WreathOfFlame", "Blasphemy", "Brilliance", "ConjureBlade", "DeusExMachina", 
    "DevaForm", "Establishment", "Fasting", "LessonLearned", "MasterReality", "Omniscience", 
    "Ragnarok", "Scrawl", "SpiritShield", "Wish", "Apparition", "Bite", "Blind", "DarkShackles", 
    "DeepBreath", "Discovery", "DramaticEntrance", "Enlightenment", "Finesse", "FlashOfSteel", 
    "Forethought", "GoodInstincts", "HandOfGreed", "Impatience", "JackOfAllTrades", "Madness", 
    "MindBlast", "Panacea", "Panache", "Purity", "SadisticNature", "SecretTechnique", "SecretWeapon", 
    "SwiftStrike", "ThinkingAhead", "Transmutation", "Violence", "Chrysalis", "Magnetism", 
    "Mayhem", "Metamorphosis", "MasterOfStrategy", "TheBomb", "Clumsy", "Decay", "Doubt", 
    "Injury", "Normality", "Pain", "Regret", "Shame", "Writhe", "Parasite", "Necronomicurse", 
    "CurseOfTheBell", "AscendersBane", "Slimed", "Void", "Dazed", "Wound", "Burn", "J.A.X.", "UNKNOWN_CARD"
]

ALL_VANILLA_RELICS = [
    "Burning Blood", "Ring of the Snake", "Cracked Core", "Pure Water", "Akabeko", "Anchor", 
    "Ancient Tea Set", "Art of War", "Bag of Marbles", "Bag of Preparation", "Blood Vial", 
    "Bronze Scales", "Centennial Puzzle", "Ceramic Fish", "Dream Catcher", "Happy Flower", 
    "Juzu Bracelet", "Lantern", "Maw Bank", "Meal Ticket", "Nunchaku", "Oddly Smooth Stone", 
    "Omamori", "Orichalcum", "Pen Nib", "Potion Belt", "Preserved Insect", "Regal Pillow", 
    "Smiling Mask", "Strawberry", "Boot", "Toy Ornithopter", "Vajra", "War Paint", "Blue Candle", 
    "Bottled Flame", "Bottled Lightning", "Bottled Tornado", "Darkstone Periapt", "Eternal Feather", 
    "Frozen Egg", "Horn Cleat", "InkBottle", "Kunai", "Letter Opener", "Matryoshka", "Meat on the Bone", 
    "Mercury Hourglass", "Molten Egg", "Mummified Hand", "Ornamental Fan", "Pantograph", 
    "Pear", "Question Card", "Shovel", "Singing Bowl", "Strike Dummy", "Sundial", "Toxic Egg", 
    "White Beast Statue", "Bird-Faced Urn", "Calipers", "Captain's Wheel", "Champion Belt", 
    "Charon's Ashes", "Cloak Clasp", "Dead Branch", "Du-Vu Doll", "Fossilized Helix", "Gambling Chip", 
    "Ginger", "Girya", "Golden Eye", "Ice Cream", "Incense Burner", "Lizard Tail", "Magic Flower", 
    "Mango", "Old Coin", "Peace Pipe", "Pocketwatch", "Prayer Wheel", "Shuriken", "Stone Calendar", 
    "The Courier", "Torii", "Tough Bandages", "Tingsha", "Turnip", "Unceasing Top", "Wing Boots", 
    "Astrolabe", "Black Blood", "Black Star", "Busted Crown", "Calling Bell", "Coffee Dripper", 
    "Cursed Key", "Ectoplasm", "Empty Cage", "Fusion Hammer", "Hovering Kite", "Inversion Dynamo", 
    "Mark of Pain", "Nuclear Battery", "Pandora's Box", "Philosopher's Stone", "Runic Dome", 
    "Runic Pyramid", "Sacred Bark", "SlaversCollar", "Slaver's Collar", "Snecko Eye", "Sozu", 
    "Velvet Choker", "Wrist Blade", "Cauldron", "Chemical X", "Dolly's Mirror", "Frozen Eye", 
    "Lee's Waffle", "Medical Kit", "Membership Card", "Orange Pellets", "Orrery", "Prismatic Shard", 
    "Sling of Courage", "Strange Spoon", "Sling", "The Specimen", "Tough Bandages", "Nilo's Codex", 
    "Enchiridion", "Necronomicon", "Mutagenic Strength", "Golden Idol", "Bloody Idol", "Red Mask", 
    "Spirit Poop", "Cultist Headpiece", "Face of Cleric", "Ssserpent Head", "Gremlin Mask", "Nloth's Gift", 
    "Nloth's Hungry Face", "Warped Tongs", "Odd Mushroom", "Red Mask", "Neow's Lament", "Circlet", "UNKNOWN_RELIC"
]

# Generate fast index lookups mapping string keys to fixed dense column vectors
CARD_INDICES = {name: idx for idx, name in enumerate(ALL_VANILLA_CARDS)}
RELIC_INDICES = {name: idx for idx, name in enumerate(ALL_VANILLA_RELICS)}

NUM_RELICS = len(ALL_VANILLA_RELICS)
NUM_CARDS = len(ALL_VANILLA_CARDS)

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
    
    processed_count = 0
    group_id = 0
        
    print(f"Starting memory-safe streaming pipeline. Target: {TARGET_RUNS_LIMIT} archive units.")
    
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

                    if data.get("floor_reached", 1) < MIN_FLOOR_THRESHOLD:
                        continue
                        
                    # Capture global Ascension Level feature
                    ascension = int(data.get("ascension_level", 0))
                    if ascension < MIN_ASCENSION_LEVEL:
                        continue
                        
                    char_str = data.get("character_chosen")
                    if char_str not in CHARACTER_MAP:
                        continue
                        
                    char_id = CHARACTER_MAP[char_str]
                    card_choices = data.get("card_choices", [])
                    if not card_choices:
                        continue

                    processed_count += 1
                    
                    # Capture historical floor timelines for tracking dynamic variables
                    gold_timeline = data.get("gold_per_floor", [])
                    hp_timeline = data.get("hp_per_floor", [])
                    max_hp_timeline = data.get("max_hp_per_floor", [])
                        
                    current_relics = set()
                    relics_by_floor = {int(r["floor"]): r["key"] for r in data.get("relics_obtained", []) if "floor" in r and r.get("floor") is not None}
                    
                    # Ensure starting deck items are baseline cleaned from the start
                    running_deck = [card.split("+")[0] for card in STARTING_DECKS.get(char_id, []).copy()]
                    card_choices.sort(key=lambda x: x.get("floor", 0))
                    current_floor = 1
                    
                    for choice in card_choices:
                        floor = int(choice.get("floor", 0))
                        for f_idx in range(current_floor, floor + 1):
                            if f_idx in relics_by_floor:
                                r_name = relics_by_floor[f_idx]
                                current_relics.add(r_name)
                        current_floor = floor
                        
                        # --- TIME-TRAVEL CALCULATOR FOR RUN STATE ---
                        # Run arrays are 0-indexed, meaning Floor 1 status is at index 0
                        timeline_idx = floor - 1
                        
                        # Safe fallbacks protect against corrupted timeline arrays
                        gold_val = gold_timeline[timeline_idx] if timeline_idx < len(gold_timeline) else 99
                        curr_hp = hp_timeline[timeline_idx] if timeline_idx < len(hp_timeline) else 70
                        max_hp = max_hp_timeline[timeline_idx] if timeline_idx < len(max_hp_timeline) else 70
                        
                        # Calculate a normalized health ratio feature (Value between 0.0 and 1.0)
                        hp_ratio = round(float(curr_hp) / float(max_hp), 3) if max_hp > 0 else 1.0
                        
                        # Clean card upgrades immediately using split("+")
                        raw_picked = choice.get("picked", "SKIP")
                        picked = raw_picked.split("+")[0] if raw_picked != "SKIP" else "SKIP"
                        not_picked = [card.split("+")[0] for card in choice.get("not_picked", [])]
                        
                        deck_snapshot = {}
                        for card in running_deck:
                            deck_snapshot[card] = deck_snapshot.get(card, 0) + 1
                            
                        relics_str = orjson.dumps(list(current_relics)).decode('utf-8')
                        deck_str = orjson.dumps(deck_snapshot).decode('utf-8')
                        
                        # --- CLEAN INTERLOCKING CONTRASTIVE PICK LOGIC ---
                        if picked != "SKIP":
                            # Player picked a card
                            writer.writerow([group_id, floor, char_id, ascension, gold_val, hp_ratio, relics_str, deck_str, picked, 0, 1])
                            for npc in not_picked:
                                writer.writerow([group_id, floor, char_id, ascension, gold_val, hp_ratio, relics_str, deck_str, npc, 0, 0])
                            writer.writerow([group_id, floor, char_id, ascension, gold_val, hp_ratio, relics_str, deck_str, "SKIP", 1, 0])
                            
                            running_deck.append(picked)
                        else:
                            # Player picked SKIP
                            for npc in not_picked:
                                writer.writerow([group_id, floor, char_id, ascension, gold_val, hp_ratio, relics_str, deck_str, npc, 0, 0])
                            writer.writerow([group_id, floor, char_id, ascension, gold_val, hp_ratio, relics_str, deck_str, "SKIP", 1, 1])
                            
                        # Increment group_id per choice event reward screen
                        group_id += 1
                    
                if processed_count % 10 == 0:
                    print(f" -> Logged {processed_count} JSON file entries to disk storage...")
                    
            except Exception:
                continue
                
    # =========================================================================
    # PASS 2: STREAM COMPUTE FLAT MATRIX ROWS
    # =========================================================================
    print(f"\nVectorizing intermediate file into final dataset layout...")
    
    # Structural modification tracking injected variables at specific indexed slots
    headers = ["group_id", "floor", "character_class", "ascension_level", "gold", "hp_ratio"]
    headers += [f"relic_{idx}" for idx in range(NUM_RELICS)]
    headers += [f"deck_{idx}" for idx in range(NUM_CARDS)]
    headers += ["candidate_card_id", "is_virtual_skip", "target"]
    
    with open(TEMP_RAW_ROWS, "r", encoding="utf-8") as temp_in, open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csv_out:
        reader = csv.reader(temp_in)
        writer = csv.writer(csv_out)
        writer.writerow(headers)
        
        for row in reader:
            group_id, floor, char_id, ascension, gold_val, hp_ratio, relics_json, deck_json, candidate, is_virtual, target = row
            
            current_relics = orjson.loads(relics_json)
            deck_snapshot = orjson.loads(deck_json)
            
            relic_vector = [0] * NUM_RELICS
            unknown_relic_idx = RELIC_INDICES["UNKNOWN_RELIC"]
            for r in current_relics:
                if r in RELIC_INDICES:
                    relic_vector[RELIC_INDICES[r]] = 1
                else:
                     relic_vector[unknown_relic_idx] = 1
                    
            deck_vector = [0] * NUM_CARDS
            for card, count in deck_snapshot.items():
                if card in CARD_INDICES:
                    deck_vector[CARD_INDICES[card]] = count
                    
            unknown_idx = CARD_INDICES["UNKNOWN_CARD"]
            candidate_id = CARD_INDICES.get(candidate, unknown_idx)
            
            # Combine everything cleanly into the wide format vector row
            flat_row = [group_id, floor, char_id, int(ascension), int(gold_val), float(hp_ratio)] + relic_vector + deck_vector + [candidate_id, int(is_virtual), int(target)]
            writer.writerow(flat_row)
            
    if os.path.exists(TEMP_RAW_ROWS):
        os.remove(TEMP_RAW_ROWS)
        
    print(f"Pipeline finished safely! Final dataset stored at: {OUTPUT_CSV}")



if __name__ == "__main__":
    execute_scalable_pipeline()