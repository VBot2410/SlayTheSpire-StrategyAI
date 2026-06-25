import lightgbm as lgb
import numpy as np
import os
# IMPORT YOUR DEFINITIVE VOCABULARY ARRAYS SO SHAPES NEVER MISMATCH
from spire_strategy.data import ALL_VANILLA_CARDS, ALL_VANILLA_RELICS
from spire_strategy import PACKAGE_ROOT

MODELS_DIR = PACKAGE_ROOT / "models/"

class LiveSpireRecommender:
    def __init__(self, model_dir=MODELS_DIR):
        """
        Loads and initializes all three Act-Specific Multiclass LightGBM models.
        """
        self.models = {
            "act1": lgb.Booster(model_file=os.path.join(model_dir, "spire_lgb_act_1.txt")),
            "act2": lgb.Booster(model_file=os.path.join(model_dir, "spire_lgb_act_2.txt")),
            "act3": lgb.Booster(model_file=os.path.join(model_dir, "spire_lgb_act_3.txt"))
        }
        
        # Dynamic Vocabulary Alignment matching your training dataset footprint
        self.card_vocab_size = len(ALL_VANILLA_CARDS)
        self.relic_vocab_size = len(ALL_VANILLA_RELICS)
        
        print(f"[AI] Multiclass Engine Init. Cards: {self.card_vocab_size} | Relics: {self.relic_vocab_size}")

    def _select_model(self, floor):
        """Routes the query to the correct specialized ranker based on game progression."""
        if floor <= 17:
            return self.models["act1"]
        elif floor <= 34:
            return self.models["act2"]
        else:
            return self.models["act3"]

    def get_recommendation(self, current_run_metrics, active_deck_list, active_relics_list, card_candidates):
        """
        Translates real-time state metrics into a single horizontal multiclass vector.
        Natively aligned to a pure 0-indexed architecture.
        """
        floor = int(current_run_metrics["floor"])
        bst = self._select_model(floor)

        # 1. Reconstruct baseline multi-hot relics and frequency-count decks (Direct 0-indexing)
        relic_vector = np.zeros(self.relic_vocab_size + 1, dtype=np.int16)
        for r in active_relics_list:
            if 0 <= r <= self.relic_vocab_size: 
                relic_vector[r] = 1

        deck_vector = np.zeros(self.card_vocab_size + 1, dtype=np.int16)
        for c in active_deck_list:
            if 0 <= c <= self.card_vocab_size: 
                deck_vector[c] += 1

        total_relics = float(relic_vector.sum())
        total_deck_size = float(deck_vector.sum())

        # 2. Build our environmental slice matching X_metrics
        base_metrics = [
            float(floor),
            float(current_run_metrics["character_class"]),
            float(current_run_metrics["ascension_level"]),
            float(current_run_metrics["gold"]),
            float(current_run_metrics["hp_ratio"])
        ]

        # 3. Form our fixed 4-slot choice window layout (3 card candidates max + 1 skip option)
        # Any card candidate offered on screen goes into slots 0-2. Index 0 is natively the skip token code.
        all_options = list(card_candidates)
        while len(all_options) < 3:
            all_options.append(0) # Pad empty options with native 0-index skip tokens
        all_options.append(0)    # Slot index 3 is always explicitly reserved for the Skip All action
        
        # Clip down to exact max boundary constraints
        final_slots = all_options[:4]

        candidate_cards = np.zeros(4, dtype=np.int32)
        candidate_deck_counts = np.zeros(4, dtype=np.float32)
        is_skip_mask = np.zeros(4, dtype=np.float32)

        for slot, cand_id in enumerate(final_slots):
            candidate_cards[slot] = cand_id
            is_skip_mask[slot] = 1.0 if cand_id == 0 else 0.0
            
            # Direct vocabulary lookup matching your exact dataset builder grouping format
            if 0 <= cand_id <= self.card_vocab_size:
                candidate_deck_counts[slot] = float(deck_vector[cand_id])

        # 4. Perfectly align horizontally matching your stacked hstack training pipeline shape
        full_row = np.hstack([
            base_metrics,               # 5 features
            is_skip_mask,               # 4 features
            candidate_deck_counts,      # 4 features
            candidate_cards,            # 4 features
            relic_vector,               # relic_vocab_size + 1 features
            deck_vector,                # card_vocab_size + 1 features
            [total_relics],             # 1 feature
            [total_deck_size]           # 1 feature
        ])

        # 5. Execute matrix scoring
        X_infer = np.array([full_row], dtype=np.float32)
        probabilities = bst.predict(X_infer)[0] # Extract the 4-element class probability array

        # Find the highest scoring horizontal probability slot index
        best_slot = np.argmax(probabilities)
        chosen_option_id = final_slots[best_slot]

        # 6. Format terminal feedback logs back to the 3-class CLI framework
        if chosen_option_id == 0 or best_slot == 3:
            return f"❌ AI ADVICE: SKIP ALL CARDS (Slot {best_slot} Softmax Confidence: {probabilities[best_slot]:.4f})", 0
        else:
            chosen_name = ALL_VANILLA_CARDS[chosen_option_id]
            return f"✅ AI ADVICE: TAKE '{chosen_name.upper()}' (Slot {best_slot} | Probability: {probabilities[best_slot]:.4f})", chosen_option_id

if __name__ == "__main__":
    recommender = LiveSpireRecommender(model_dir=MODELS_DIR)

    # Create a typical Act 2 Defect environment state scenario
    mock_metrics = {
        "floor": 21,
        "character_class": 2.0, # DEFECT class encoding index
        "ascension_level": 20.0,
        "gold": 120.0,
        "hp_ratio": 0.65
    }
    
    # Simulate a mid-sized power scaling deck setup 
    mock_deck = [34, 34, 112, 112, 5, 6, 7] 
    mock_relics = [1, 14, 88] # Ingot / standard relic indices
    
    # Offer a highly impactful class card vs garbage starter variants
    mock_candidates = [169, 1, 2] # e.g., Defragment vs Basic Strike/Defend
    
    advice, choice = recommender.get_recommendation(mock_metrics, mock_deck, mock_relics, mock_candidates)
    print("\n--- [LIVE CLI ENGINE INTEGRATION TEST] ---")
    print(advice)
    print("------------------------------------------\n")
