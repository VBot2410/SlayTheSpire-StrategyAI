# Script to check live suggestions in RAM
from apps.live_spire_recommender import LiveSpireRecommender

# Initialize the inference engine
recommender = LiveSpireRecommender(
    model_path="spire_lgb_recommender.txt",
    card_vocab_size=350,
    relic_vocab_size=150
)

# Simulate a tight mid-game scenario on Floor 12
mock_metrics = {
    "floor": 12.0,
    "character_class": 0.0,      # Ironclad
    "ascension_level": 20.0,     # A20
    "gold": 145.0,
    "hp_ratio": 0.450            # Low health!
}

# The player already owns these card IDs in their deck array
# Look up your true CARD_INDICES to map real card tokens
mock_deck = [2, 2, 2, 3, 3, 3, 4, 18, 22]  # Baselines + e.g., Armaments and Whirlwind

# The player already owns these relic IDs
mock_relics = [1, 24]  # Burning Blood + e.g., Vajra

# The reward screen offers three cards on-screen
mock_offered_cards = [22, 65, 48]  # e.g., Whirlwind (Duplicate), Carnage, or Evolve

print("Querying live model for card recommendations...")
recommendation = recommender.get_recommendation(
    current_run_metrics=mock_metrics,
    active_deck_list=mock_deck,
    active_relics_list=mock_relics,
    card_candidates=mock_offered_cards
)

print(recommendation)
