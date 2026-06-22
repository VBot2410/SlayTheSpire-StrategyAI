import torch
import torch.nn as nn
from torch.utils.data import Dataset
import json

class SpireSequenceDataset(Dataset):
    def __init__(self, csv_path):
        # We read strings using a standard engine or load into pandas
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Parse our dense JSON list blocks back into standard Python integer lists
        relics = json.loads(row['relic_seq'])
        deck = json.loads(row['deck_seq'])
        
        return {
            "group_id": torch.tensor(row['group_id'], dtype=torch.long),
            "floor": torch.tensor(row['floor'], dtype=torch.float32),
            "character": torch.tensor(row['character_class'], dtype=torch.long),
            "ascension": torch.tensor(row['ascension_level'], dtype=torch.float32),
            "gold": torch.tensor(row['gold'], dtype=torch.float32),
            "hp_ratio": torch.tensor(row['hp_ratio'], dtype=torch.float32),
            "relic_tokens": torch.tensor(relics, dtype=torch.long),
            "deck_tokens": torch.tensor(deck, dtype=torch.long),
            "candidate_id": torch.tensor(row['candidate_card_id'], dtype=torch.long),
            "is_virtual_skip": torch.tensor(row['is_virtual_skip'], dtype=torch.float32),
            "target": torch.tensor(row['target'], dtype=torch.float32)
        }

class SpireRecommendationTransformer(nn.Module):
    def __init__(self, card_vocab_size, relic_vocab_size, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.d_model = d_model
        
        # 1. Embedding Layers (+1 accounts for our 0 padding token index)
        self.card_embedding = nn.Embedding(card_vocab_size + 1, d_model, padding_idx=0)
        self.relic_embedding = nn.Embedding(relic_vocab_size + 1, d_model, padding_idx=0)
        self.char_embedding = nn.Embedding(5, 16) # 4 base classes + 1 fallback
        
        # 2. Linear projection layer for handling baseline flat game state metrics
        self.meta_projection = nn.Linear(3 + 16, d_model) # floor, gold, hp_ratio + character emb
        
        # 3. Transformer Processing Backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, 
            dropout=0.1, batch_first=True, activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Cross-Attention: Lets the candidate card inspect the fused game-state context
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        
        # 5. Output Feed-Forward Predictor Network
        self.classifier = nn.Sequential(
            nn.Linear(d_model + 1, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, 1) # Outputs a raw logit score for ranking
        )
        
    def forward(self, batch):
        # Extract features
        deck = batch['deck_tokens']          # Shape: [Batch, MAX_DECK_LEN]
        relics = batch['relic_tokens']        # Shape: [Batch, MAX_RELIC_LEN]
        candidate = batch['candidate_id']    # Shape: [Batch]
        is_skip = batch['is_virtual_skip'].unsqueeze(-1) # Shape: [Batch, 1]
        
        # Embed item sequences
        deck_emb = self.card_embedding(deck)       # [Batch, MAX_DECK_LEN, d_model]
        relic_emb = self.relic_embedding(relics)   # [Batch, MAX_RELIC_LEN, d_model]
        cand_emb = self.card_embedding(candidate).unsqueeze(1) # [Batch, 1, d_model]
        
        # Process and project categorical/numerical baseline states
        char_emb = self.char_embedding(batch['character'])
        meta_features = torch.stack([batch['floor'], batch['gold'], batch['hp_ratio']], dim=-1)
        meta_combined = torch.cat([meta_features, char_emb], dim=-1)
        meta_emb = self.meta_projection(meta_combined).unsqueeze(1) # [Batch, 1, d_model]
        
        # FUSE everything into a single absolute global timeline token matrix sequence
        # Fused sequence shape: [Batch, 1 + 1 + MAX_DECK_LEN + MAX_RELIC_LEN, d_model]
        fused_context = torch.cat([meta_emb, cand_emb, deck_emb, relic_emb], dim=1)
        
        # Run combined tokens through self-attention layers
        fused_context = self.transformer_encoder(fused_context)
        
        # Run Cross-Attention: Query = Candidate, Key/Value = Fused State Context Block
        # This isolates exactly how the card choice interacts with the current deck build
        attn_output, _ = self.cross_attention(query=cand_emb, key=fused_context, value=fused_context)
        attn_output = attn_output.squeeze(1) # [Batch, d_model]
        
        # Concatenate the final attention vector with the explicit is_virtual_skip flag
        final_features = torch.cat([attn_output, is_skip], dim=-1)
        
        # Generate the raw continuous logit score
        return self.classifier(final_features).squeeze(-1)
