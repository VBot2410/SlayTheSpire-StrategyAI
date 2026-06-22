import lightgbm as lgb
import numpy as np

# IMPORT YOUR DEFINITIVE VOCABULARY ARRAYS SO SHAPES NEVER MISMATCH
import spire_strategy.data.pipeline as config

class LiveSpireRecommender:
    def __init__(self, model_path="spire_lgb_recommender.txt"):
        """
        Dynamically configures vector shapes based on your core script arrays.
        """
        # Load the LightGBM booster weights file directly into RAM memory
        self.bst = lgb.Booster(model_file=model_path)
        
        # DYNAMIC VOCABULARY ALIGNMENT: Never hardcode these array limits again!
        self.card_vocab_size = len(config.ALL_VANILLA_CARDS)
        self.relic_vocab_size = len(config.ALL_VANILLA_RELICS)
        
        print(f"[AI] Model initialized. Card Vector Size: {self.card_vocab_size} | Relic Vector Size: {self.relic_vocab_size}")

    def get_recommendation(self, current_run_metrics, active_deck_list, active_relics_list, card_candidates):
        """
        Args:
            current_run_metrics: dict matching keys: ["floor", "character_class", "ascension_level", "gold", "hp_ratio"]
            active_deck_list: list of integer card IDs currently owned in the deck
            active_relics_list: list of integer relic IDs currently owned
            card_candidates: list of integer card IDs offered on-screen
        """
        # 1. Build the multi-hot Relic context vector using the dynamic size limit
        relic_vector = np.zeros(self.relic_vocab_size + 1, dtype=np.int8)
        for r in active_relics_list:
            if r > 0 and r <= self.relic_vocab_size:
                relic_vector[r] = 1

        # 2. Shift deck item counts by +1 to match the dataset builder's deck_seq formatting
        deck_vector = np.zeros(self.card_vocab_size + 1, dtype=np.int16)
        for c in active_deck_list:
            # c is the raw vocab index from the CLI (e.g. 1 for striker)
            shifted_deck_id = c + 1
            if 0 < shifted_deck_id <= self.card_vocab_size:
                deck_vector[shifted_deck_id] += 1

        # Reconstruct our baseline continuous metrics profile slice array
        base_metrics = [
            float(current_run_metrics["floor"]),
            float(current_run_metrics["character_class"]),
            float(current_run_metrics["ascension_level"]),
            float(current_run_metrics["gold"]),
            float(current_run_metrics["hp_ratio"])
        ]

        # 3. Assemble competing decision options (The cards on screen + 1 virtual skip option)
        all_options = list(card_candidates) + [0]
        rows = []

        for cand_id in all_options:
            # cand_id is either a raw vocab index (e.g., 1) or 0 (for skip)
            # FIX: Compute the exact shifted ID used by the training script
            if cand_id == 0:
                shifted_cand_id = 0  # Skip stays 0
                is_virtual_skip = 1.0
            else:
                shifted_cand_id = cand_id + 1  # Standard cards get +1 shift
                is_virtual_skip = 0.0
            
            # Reconstruct the candidate ownership interaction flag using the shifted ID
            already_owned = 1.0 if (shifted_cand_id > 0 and shifted_cand_id <= self.card_vocab_size and deck_vector[shifted_cand_id] > 0) else 0.0
            
            # Reconstruct One-Hot candidate matrix using shifted bounds
            candidate_row = np.zeros(self.card_vocab_size + 1, dtype=np.int8)
            if 0 <= shifted_cand_id <= self.card_vocab_size:
                candidate_row[shifted_cand_id] = 1

            # Combine everything horizontally for perfect LightGBM alignment
            full_row = np.hstack([base_metrics, already_owned, candidate_row, relic_vector, deck_vector])
            rows.append(full_row)

        # 4. Run inference matrix cross-blocks
        X_infer = np.vstack(rows)
        scores = self.bst.predict(X_infer)
        
        best_idx = np.argmax(scores)
        chosen_option = all_options[best_idx] # Returns the clean raw vocab index (or 0) back to CLI

        if chosen_option == 0:
            return f"❌ AI ADVICE: SKIP ALL CARDS (Skip Rank Utility: {scores[best_idx]:.4f})", 0
        else:
            chosen_name = config.ALL_VANILLA_CARDS[chosen_option]
            return f"✅ AI ADVICE: CHOOSE '{chosen_name.upper()}' (Card ID: {chosen_option} | Prediction Score: {scores[best_idx]:.4f})", chosen_option