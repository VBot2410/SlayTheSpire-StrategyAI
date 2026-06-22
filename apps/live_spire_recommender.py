import lightgbm as lgb
import numpy as np
import os

# IMPORT YOUR DEFINITIVE VOCABULARY ARRAYS SO SHAPES NEVER MISMATCH
import spire_strategy.data.pipeline as config

class LiveSpireRecommender:
    def __init__(self, model_dir="."):
        """Loads and initializes all three Act-Specific LightGBM models."""
        self.models = {
            "act1": lgb.Booster(model_file=os.path.join(model_dir, "spire_lgb_act_1.txt")),
            "act2": lgb.Booster(model_file=os.path.join(model_dir, "spire_lgb_act_2.txt")),
            "act3": lgb.Booster(model_file=os.path.join(model_dir, "spire_lgb_act_3.txt"))
        }
        
        # Dynamic Vocabulary Alignment matching your exact training configuration
        self.card_vocab_size = len(config.ALL_VANILLA_CARDS)
        self.relic_vocab_size = len(config.ALL_VANILLA_RELICS)
        
        print(f"[AI] Act-Split Models Successfully Loaded into Memory!")
        print(f"[AI] Constraints -> Cards Vocab: {self.card_vocab_size} | Relics Vocab: {self.relic_vocab_size}")

    def _select_model(self, floor):
        """Routes the query to the correct specialized ranker based on game progression."""
        if floor <= 17:
            return self.models["act1"]
        elif floor <= 34:
            return self.models["act2"]
        else:
            return self.models["act3"]

    def get_recommendation(self, current_run_metrics, active_deck_list, active_relics_list, card_candidates):
        """Processes real-time state parameters into perfectly shape-aligned matrices."""
        floor = int(current_run_metrics["floor"])
        bst = self._select_model(floor)

        # 1. Build the multi-hot Relic context vector (Mirroring training loop setup)
        relic_vector = np.zeros(self.relic_vocab_size + 1, dtype=np.int8)
        for r in active_relics_list:
            if r > 0 and r <= self.relic_vocab_size:
                relic_vector[r] = 1

        # 2. Reconstruct deck frequency count vector with training shift (+1 format)
        deck_vector = np.zeros(self.card_vocab_size + 1, dtype=np.int16)
        for c in active_deck_list:
            shifted_deck_id = c + 1
            if 0 < shifted_deck_id <= self.card_vocab_size:
                deck_vector[shifted_deck_id] += 1

        # Calculate base structural inventory aggregate sums
        total_relics_owned = float(relic_vector.sum())
        total_deck_size = float(deck_vector.sum())

        # Base environment metrics
        base_metrics = [
            float(floor),
            float(current_run_metrics["character_class"]),
            float(current_run_metrics["ascension_level"]),
            float(current_run_metrics["gold"]),
            float(current_run_metrics["hp_ratio"])
        ]

        # 3. Assemble competing choices (The candidates on screen + 1 virtual skip option)
        all_options = list(card_candidates) + [0]
        rows = []

        for cand_id in all_options:
            # FIX: Training script uses raw cand_id for candidate_matrix indexing!
            # It maps skip directly to index 0.
            is_virtual_skip = 1.0 if cand_id == 0 else 0.0
            
            # Contextual Feature: How many copies of THIS card are already in our deck?
            # Shifted ID is used ONLY when querying the frequency count vector
            shifted_cand_id = cand_id + 1
            if cand_id > 0 and 0 < shifted_cand_id <= self.card_vocab_size:
                candidate_deck_count = float(deck_vector[shifted_cand_id])
            else:
                candidate_deck_count = 0.0

            # Reconstruct One-Hot Candidate Row vector using raw cand_id (matching training matrix layout)
            candidate_row = np.zeros(self.card_vocab_size + 1, dtype=np.int8)
            if 0 <= cand_id <= self.card_vocab_size:
                candidate_row[cand_id] = 1

            # Compute advanced shape-aligned interaction matrices
            cand_relic_count_interaction = candidate_row.astype(np.float32) * total_relics_owned
            cand_deck_size_interaction = candidate_row.astype(np.float32) * total_deck_size

            # FIX: Adjusted horizontal sequence stack order to match the training script's array exactly
            full_row = np.hstack([
                base_metrics, 
                [is_virtual_skip],
                [candidate_deck_count], 
                candidate_row, 
                relic_vector, 
                deck_vector,
                cand_relic_count_interaction,
                cand_deck_size_interaction
            ])
            rows.append(full_row)

        # 4. Process predictions
        X_infer = np.vstack(rows)
        scores = bst.predict(X_infer)

        # DEBUG LOGGING: Print everything the model is thinking
        print("\n--- [AI CHOICE EVALUATION] ---")
        for idx, opt_id in enumerate(all_options):
            if opt_id == 0:
                print(f" -> Option [SKIP]            | Utility Score: {scores[idx]:.4f}")
            else:
                c_name = config.ALL_VANILLA_CARDS[opt_id]
                print(f" -> Option [{c_name.upper():<17}] | Utility Score: {scores[idx]:.4f}")
        print("-------------------------------\n")
        
        best_idx = np.argmax(scores)
        chosen_option = all_options[best_idx] 

        if chosen_option == 0:
            return f"❌ AI ADVICE: SKIP ALL CARDS (Skip Rank Utility: {scores[best_idx]:.4f})", 0
        else:
            chosen_name = config.ALL_VANILLA_CARDS[chosen_option]
            return f"✅ AI ADVICE: CHOOSE '{chosen_name.upper()}' (Card ID: {chosen_option} | Prediction Score: {scores[best_idx]:.4f})", chosen_option
