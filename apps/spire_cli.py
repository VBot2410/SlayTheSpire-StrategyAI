import sys
import json
import os
import numpy as np
import lightgbm as lgb

# Import existing recommender class and config
import spire_strategy.data.pipeline as config
from apps.live_spire_recommender import LiveSpireRecommender

def normalize_game_string(input_str: str) -> str:
    """
    Standardizes game names by removing upgrades, casing, spaces, and dashes.
    COPIED DIRECTLY FROM DATASET BUILDER FOR PERFECT MATRIX ALIGNMENT.
    """
    if not input_str:
        return ""
    base_name = input_str.split("+")[0]
    return base_name.lower().replace(" ", "").replace("_", "").replace("-", "").strip()

class SpireCLISession:
    def __init__(self):
        print("\n=== Initializing Slay the Spire AI Assistant ===")
        self.recommender = LiveSpireRecommender()
        
        # Build clean, normalized reverse lookup maps using your exact cleaning function
        self.card_to_id = {normalize_game_string(name): idx for idx, name in enumerate(config.ALL_VANILLA_CARDS)}
        self.relic_to_id = {normalize_game_string(name): idx for idx, name in enumerate(config.ALL_VANILLA_RELICS)}
        
        # Exact schema mapping conventions
        self.char_mapping = {"ironclad": 0, "thesilent": 1, "defect": 2, "watcher": 3}
        
        self.starting_decks = {
            0: ["striker"] * 5 + ["defendr"] * 4 + ["bash"],
            1: ["strikeg"] * 5 + ["defendg"] * 5 + ["neutralize", "survivor"],
            2: ["strikeb"] * 4 + ["defendb"] * 4 + ["zap", "dualcast"],
            3: ["strikep"] * 4 + ["defendp"] * 4 + ["eruption", "vigilance"]
        }
        
        self.starting_relics = {
            0: "burningblood",
            1: "ringofthesnake",
            2: "crackedcore",
            3: "purewater"
        }

        # Track session state
        self.active_deck = []    
        self.active_relics = []  
        self.metrics = {
            "floor": 0.0,
            "character_class": 0.0,
            "ascension_level": 20.0,
            "gold": 99.0,
            "hp_ratio": 1.0
        }

    def parse_item(self, text, lookup_dict, item_type="item"):
        """
        Safely parses input strings. If the item cannot be found, it maps 
        it directly to the 'unknowncard' or 'unknownrelic' token.
        """
        clean_text = normalize_game_string(text)
        
        # 1. Standard structural match check
        if clean_text in lookup_dict:
            return lookup_dict[clean_text]
            
        # 2. Integer fallback check
        try:
            return int(clean_text)
        except ValueError:
            pass

        # 3. Dynamic Unknown Token Routing Fallback
        # Look up the exact default string key based on the asset type
        unknown_fallback_key = "unknowncard" if item_type == "card" else "unknownrelic"
        
        if unknown_fallback_key in lookup_dict:
            fallback_id = lookup_dict[unknown_fallback_key]
            print(f"⚠️ Unrecognized {item_type} '{text}'. Mapping to generic '{unknown_fallback_key}' (ID: {fallback_id}).")
            return fallback_id
            
        # Hard failover safeguard if your config doesn't have the string mapped
        print(f"❌ Error: Could not map '{text}' and no '{unknown_fallback_key}' token found in vocabulary configuration.")
        return None

    def setup_initial_run(self):
        """Interactive wizard with automated defaults and a manual custom start override."""
        print("\n--- Run Configuration Wizard  (Floor 0 - Neow) ---")
        print("Tip: You can also bypass this wizard by typing 'load <filename>' at the main prompt.")
        
        raw_char = input("Enter Character (ironclad / thesilent / defect / watcher / custom): ").strip()
        clean_char = normalize_game_string(raw_char)
        if clean_char == "silent": clean_char = "thesilent"
            
        is_custom_start = (clean_char == "custom")
        
        if is_custom_start:
            print("\n⚙️ CUSTOM START SELECTED: You will configure your starting items completely from scratch.")
            char_choice = input("Base Character Class for Model (ironclad/thesilent/defect/watcher): ").strip().lower()
            if char_choice == "silent": char_choice = "thesilent"
            char_idx = self.char_mapping.get(normalize_game_string(char_choice), 0)
        else:
            while clean_char not in self.char_mapping:
                print("❌ Character not recognized. Please use your defined character strings.")
                raw_char = input("Enter Character (ironclad / thesilent / defect / watcher / custom): ").strip()
                clean_char = normalize_game_string(raw_char)
                if clean_char == "silent": clean_char = "thesilent"
                if clean_char == "custom":
                    is_custom_start = True
                    break
            
            if not is_custom_start:
                char_idx = self.char_mapping[clean_char]

        self.metrics["character_class"] = float(char_idx)
        
        asc = input("Enter Ascension Level (0-20) [Default 20]: ").strip()
        asc_level = int(asc) if asc else 20
        self.metrics["ascension_level"] = float(asc_level)
        
        if is_custom_start:
            self.metrics["gold"] = float(input("Starting Gold [Default 99]: ").strip() or 99.0)
            self.metrics["hp_ratio"] = float(input("Starting HP Ratio 0.0-1.0 [Default 1.0]: ").strip() or 1.0)
            self.metrics["floor"] = float(input("Starting Floor [Default 0]: ").strip() or 0.0)

        if not is_custom_start:
            raw_starting_cards = list(self.starting_decks[char_idx])
            starting_relic_str = self.starting_relics[char_idx]

            if asc_level >= 10:
                raw_starting_cards.append("ascendersbane")

            for card_str in raw_starting_cards:
                cid = self.parse_item(card_str, self.card_to_id, "card")
                if cid is not None: self.active_deck.append(cid)
                
            rid = self.parse_item(starting_relic_str, self.relic_to_id, "relic")
            if rid is not None: self.active_relics.append(rid)
        else:
            print("\n--- Custom Inventory Setup ---")
            print("Enter your initial cards separated by commas:")
            custom_cards = input("Cards: ").strip()
            if custom_cards:
                for token in custom_cards.split(","):
                    cid = self.parse_item(token, self.card_to_id, "card")
                    if cid is not None: self.active_deck.append(cid)
                    
            print("\nEnter your initial relics separated by commas:")
            custom_relics = input("Relics: ").strip()
            if custom_relics:
                for token in custom_relics.split(","):
                    rid = self.parse_item(token, self.relic_to_id, "relic")
                    if rid is not None: self.active_relics.append(rid)

        print(f"\n✅ Configuration complete! Session initialized.")

    def save_session(self, filename):
        """Exports the active tracking workspace to a JSON snapshot payload."""
        if not filename.endswith('.json'):
            filename += '.json'
        
        payload = {
            "metrics": self.metrics,
            "active_deck": self.active_deck,
            "active_relics": self.active_relics
        }
        try:
            with open(filename, 'w') as f:
                json.dump(payload, f, indent=4)
            print(f"💾 Session safely exported and written to disk: '{filename}'")
        except Exception as e:
            print(f"❌ Error writing save payload to disk: {e}")

    def load_session(self, filename):
        """Imports an existing tracking snapshot payload, overriding active memory tables."""
        if not filename.endswith('.json'):
            filename += '.json'
            
        if not os.path.exists(filename):
            print(f"❌ Error: File path target structural reference not found: '{filename}'")
            return

        try:
            with open(filename, 'r') as f:
                payload = json.load(f)
            self.metrics = payload["metrics"]
            self.active_deck = payload["active_deck"]
            self.active_relics = payload["active_relics"]
            print(f"📂 Session loaded! Active matrix states updated from '{filename}'")
        except Exception as e:
            print(f"❌ Critical exception parsing file payload configuration parameters: {e}")

    def print_status(self):
        """Displays your active game state cleanly in the console console."""
        print("\n================== LIVE SESSION STATUS ==================")
        print(f" Floor: {int(self.metrics['floor'])} | Gold: {int(self.metrics['gold'])} | HP Ratio: {self.metrics['hp_ratio']:.2f}")
        print(f" Character ID: {int(self.metrics['character_class'])} | Ascension: {int(self.metrics['ascension_level'])}")
        
        deck_names = [config.ALL_VANILLA_CARDS[i] for i in self.active_deck if i < len(config.ALL_VANILLA_CARDS)]
        relic_names = [config.ALL_VANILLA_RELICS[i] for i in self.active_relics if i < len(config.ALL_VANILLA_RELICS)]
        
        print(f" Deck ({len(deck_names)} cards): {', '.join(deck_names)}")
        print(f" Relics: {', '.join(relic_names)}")
        print("=========================================================")

    def execute_recommendation_flow(self, card_rewards_str):
        """Unified, reusable card evaluation and automated deck inventory appending loop."""
        if not card_rewards_str:
            print("❌ Error: Missing screen choice options. Usage: rec card1, card2")
            return
        
        candidates = []
        for token in card_rewards_str.split(","):
            cid = self.parse_item(token, self.card_to_id, "card")
            if cid is not None:
                candidates.append(cid)
        
        if candidates:
            advice, recommended_id = self.recommender.get_recommendation(
                current_run_metrics=self.metrics,
                active_deck_list=self.active_deck,
                active_relics_list=self.active_relics,
                card_candidates=candidates
            )
            print(f"\n{advice}\n")
            
            chosen_action = input("Did you pick a card? (y/n/skip): ").strip().lower()
            
            if chosen_action == 'y':
                if recommended_id > 0:
                    self.active_deck.append(recommended_id)
                    card_name = config.ALL_VANILLA_CARDS[recommended_id]
                    print(f"🤖 Automated Input: Added '{card_name}' into your active deck.")
                else:
                    print("ℹ️ Note: AI advised a SKIP. No cards were added to your deck.")
                    
            elif chosen_action == 'skip':
                print("ℹ️ Manual Input: Choice skipped.")
            else:
                print("⚠️ Timeline tracking paused. Remember to manually update variables if needed.")

    def run_repl_loop(self):
        """Core terminal console read-eval-print loop interface configuration."""
        self.setup_initial_run()
        
        while True:
            self.print_status()
            print("\nAvailable Commands:")
            print("  rec <card1, card2, ...>  -> Request selection recommendation calculations")
            print("  combat                   -> Log post-combat gold rewards and HP changes")
            print("  shop                     -> Open interactive merchant menu for bulk buys/removals")
            print("  next                     -> Advance automated timeline tracking to next Floor Node")
            print("  status                   -> Manually modify metrics (Floor, Gold, or Current/Max HP)")
            print("  add-card / remove-card   -> Manually alter deck tracking arrays")
            print("  add-relic <relic_name>   -> Add a new relic into tracking state")
            print("  remove-relic <relic_name>-> Remove a relic from tracking state")
            print("  save / load <filename>   -> Export or import tracking snapshot payloads")
            print("  exit                     -> Terminate application")
            
            cmd_input = input("\n(Spire-AI) >>> ").strip()
            if not cmd_input:
                continue
                
            parts = cmd_input.split(" ", 1)
            action = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""

            if action == "rec":
                self.execute_recommendation_flow(args)

            elif action == "shop":
                print("\n🏪 --- WELCOME TO THE SHOP MERCHANT INTERFACE ---")
                # 1. Update gold pool immediately
                try:
                    new_gold = input(f"Current Gold is {int(self.metrics['gold'])}. Enter new Gold amount (or press Enter to keep): ").strip()
                    if new_gold:
                        self.metrics["gold"] = float(new_gold)
                except ValueError:
                    print("❌ Error: Invalid gold amount format.")
                    continue

                # 2. Bulk Card Purchasing Loop
                print("\nEnter cards you purchased (separated by commas). Press Enter to skip.")
                cards_bought = input("Bought Cards: ").strip()
                if cards_bought:
                    for token in cards_bought.split(","):
                        cid = self.parse_item(token, self.card_to_id, "card")
                        if cid is not None:
                            self.active_deck.append(cid)
                            print(f"  🛒 Added card to deck: {config.ALL_VANILLA_CARDS[cid]}")

                # 3. Bulk Relic Purchasing Loop
                print("\nEnter relics you purchased (separated by commas). Press Enter to skip.")
                relics_bought = input("Bought Relics: ").strip()
                if relics_bought:
                    for token in relics_bought.split(","):
                        rid = self.parse_item(token, self.relic_to_id, "relic")
                        if rid is not None:
                            self.active_relics.append(rid)
                            print(f"  💎 Added relic to inventory: {config.ALL_VANILLA_RELICS[rid]}")

                # 4. Merchant Card Removal Service
                print("\nDid you pay the merchant to remove a card from your deck?")
                card_removed = input("Removed Card Name: ").strip()
                if card_removed:
                    cid = self.parse_item(card_removed, self.card_to_id, "card")
                    if cid in self.active_deck:
                        self.active_deck.remove(cid)
                        print(f"  ❌ Removed card from deck: {config.ALL_VANILLA_CARDS[cid]}")
                    elif cid is not None:
                        print("  ❌ Error: That card was not found in your current active deck vector.")

                print("🏪 --- LEAVING SHOP MERCHANT INTERFACE ---")

            elif action == "combat":
                print("\n⚔️ --- COMBAT ENCOUNTER MANAGER ---")
                
                # 1. Update HP values to calculate the precise ratio
                try:
                    hp_input = input("Enter current HP and max HP separated by a slash (e.g., 45/75 or press Enter to skip changes): ").strip()
                    if hp_input and "/" in hp_input:
                        curr_hp_str, max_hp_str = hp_input.split("/", 1)
                        curr_hp = float(curr_hp_str.strip())
                        max_hp = float(max_hp_str.strip())
                        
                        if max_hp > 0:
                            # Clamp between 0.0 and 1.0 for model data stability
                            self.metrics["hp_ratio"] = max(0.0, min(1.0, curr_hp / max_hp))
                            print(f"  ❤️ HP Ratio updated: {self.metrics['hp_ratio']:.2f}")
                        else:
                            print("  ❌ Error: Max HP must be greater than 0.")
                except ValueError:
                    print("  ❌ Error: Invalid HP format. Please use 'current/max' format.")

                # 2. Update Gold with combat earnings
                try:
                    gold_earned = input(f"Current Gold: {int(self.metrics['gold'])}. Enter gold gained from reward screen (or press Enter to skip): ").strip()
                    if gold_earned:
                        self.metrics["gold"] += float(gold_earned)
                        print(f"  🪙 Gold updated to: {int(self.metrics['gold'])}")
                except ValueError:
                    print("  ❌ Error: Invalid gold number format.")

                # 3. Check for Combat Relic Drops (Common for Elites/Bosses)
                relic_dropped = input("\nDid this combat drop a relic? Enter name (or press Enter to skip): ").strip()
                if relic_dropped:
                    rid = self.parse_item(relic_dropped, self.relic_to_id, "relic")
                    if rid is not None:
                        self.active_relics.append(rid)
                        print(f"  💎 Added relic to inventory: {config.ALL_VANILLA_RELICS[rid]}")
                
                # 4. Integrated Post-Combat Card Drop Rewards Screen
                print("\n--- Card Reward Screen ---")
                card_rewards = input("Enter the card reward options offered (separated by commas, or press Enter to skip): ").strip()
                
                print("\n⚔️ --- LEAVING COMBAT ENCOUNTER MANAGER ---")
                
                if card_rewards:
                    # REUSE CODE SAFELY: Calls helper instantly before loop iteration terminates
                    self.execute_recommendation_flow(card_rewards)             

            elif action == "status":
                try:
                    self.metrics["floor"] = float(input(f"Current Floor ({self.metrics['floor']}): ") or self.metrics["floor"])
                    self.metrics["gold"] = float(input(f"Current Gold ({self.metrics['gold']}): ") or self.metrics["gold"])
                    hp_in = input(f"Current HP Info (Ratio is {self.metrics['hp_ratio']:.2f}). Enter as current/max (e.g., 55/75 or hit Enter to keep): ").strip()
                    if hp_in and "/" in hp_in:
                        curr_hp_str, max_hp_str = hp_in.split("/", 1)
                        curr_hp = float(curr_hp_str.strip())
                        max_hp = float(max_hp_str.strip())
                        
                        if max_hp > 0:
                            # Clamp values between 0.0 and 1.0 to preserve LightGBM model stability
                            self.metrics["hp_ratio"] = max(0.0, min(1.0, curr_hp / max_hp))
                            print(f"  ❤️ HP Ratio calculated and set to: {self.metrics['hp_ratio']:.2f}")
                        else:
                            print("  ❌ Error: Max HP must be greater than 0.")
                except ValueError:
                    print("❌ Error: Invalid numeric input format.")

            elif action == "add-card":
                cid = self.parse_item(args, self.card_to_id, "card")
                if cid is not None:
                    self.active_deck.append(cid)
                    print(f"✅ Added {config.ALL_VANILLA_CARDS[cid]} to deck list mapping layer.")

            elif action == "remove-card":
                cid = self.parse_item(args, self.card_to_id, "card")
                if cid in self.active_deck:
                    self.active_deck.remove(cid)
                    print(f"❌ Removed {config.ALL_VANILLA_CARDS[cid]} from session array storage.")
                else:
                    print("❌ Error: Card not found in your current tracked deck.")

            elif action == "add-relic":
                rid = self.parse_item(args, self.relic_to_id, "relic")
                if rid is not None:
                    self.active_relics.append(rid)
                    print(f"✅ Added {config.ALL_VANILLA_RELICS[rid]} to relic arrays.")
            
            elif action == "remove-relic":
                if not args:
                    print("❌ Error: Missing relic name parameter. Usage: remove-relic <relic_name>")
                    continue
                
                rid = self.parse_item(args, self.relic_to_id, "relic")
                if rid in self.active_relics:
                    self.active_relics.remove(rid)
                    print(f"❌ Removed {config.ALL_VANILLA_RELICS[rid]} from relic tracking arrays.")
                else:
                    print("❌ Error: That relic was not found in your current active inventory vector.")
            
            elif action == "next":
                self.metrics["floor"] += 1.0
                print(f"⏩ Map Node Cleared. Advanced tracking to Floor {int(self.metrics['floor'])}")

            elif action == "save":
                if not args:
                    print("❌ Error: Save requires a target file name parameter constraint. Usage: save my_run")
                    continue
                self.save_session(args)

            elif action == "load":
                if not args:
                    print("❌ Error: Load requires a source file path string context. Usage: load my_run")
                    continue
                self.load_session(args)

            elif action == "exit":
                print("Goodbye, Defier of the Spire!")
                sys.exit(0)
                
            else:
                print(f"❌ Unknown command option: '{action}'")


if __name__ == "__main__":
    session = SpireCLISession()
    session.run_repl_loop()
