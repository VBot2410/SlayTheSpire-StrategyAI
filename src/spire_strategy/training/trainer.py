import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from spire_strategy.models import SpireSequenceDataset, SpireRecommendationTransformer
import spire_strategy.data.pipeline as pipeline

def train_recommender():
    # Configure variables matching indexing maps
    CARD_VOCAB_SIZE = len(pipeline.ALL_VANILLA_CARDS)  # Size of ALL_VANILLA_CARDS array
    RELIC_VOCAB_SIZE = len(pipeline.ALL_VANILLA_RELICS) # Size of ALL_VANILLA_RELICS array
    BATCH_SIZE = 512
    EPOCHS = 50
    
    # Check for hardware acceleration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute acceleration target device: {device}")
    
    # Load the full big dataset
    full_dataset = SpireSequenceDataset("slay_the_spire_transformer.csv")

    # Calculate split dimensions (e.g., 85% Train, 15% Validation)
    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Build two separate data loaders
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)
    
    # Initialize Network Modules
    model = SpireRecommendationTransformer(
        card_vocab_size=CARD_VOCAB_SIZE, 
        relic_vocab_size=RELIC_VOCAB_SIZE
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss() # Perfect for evaluating contrastive selection rows
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            # Send sub-arrays to active device safely
            for key in batch:
                batch[key] = batch[key].to(device)
                
            optimizer.zero_grad()
            
            # Predict logits
            predictions = model(batch)
            loss = criterion(predictions, batch['target'])
            
            # Backward pass optimization step
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 200 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] | Batch {batch_idx}/{len(train_loader)} | Current Loss: {loss.item():.4f}")
                
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"=== Epoch [{epoch+1}/{EPOCHS}] Completed. Normalized Average Loss: {avg_loss:.4f} ===")

        # =========================================================================
        # VALIDATION PHASE: COMPUTE TRUE TOP-1 CHOICE SELECTION ACCURACY
        # =========================================================================
        model.eval()
        
        # Dictionaries to track the highest-scoring option per unique screen group
        group_max_score = {}
        group_winning_target = {}
        
        with torch.no_grad():
            for val_batch in val_loader:
                # Send sub-arrays to compute acceleration device safely
                for key in val_batch:
                    if torch.is_tensor(val_batch[key]):
                        val_batch[key] = val_batch[key].to(device)
                
                # Predict raw logit preference values
                val_logits = model(val_batch)
                
                # Unpack tensors to CPU to safely build lookup maps
                g_ids = val_batch['group_id'].cpu().numpy() if 'group_id' in val_batch else val_batch['character'].cpu().numpy() # Fallback if no explicit string ID passed
                scores = val_logits.cpu().numpy()
                targets = val_batch['target'].cpu().numpy()
                
                # Evaluate rows continuously across mini-batch segments
                for idx in range(len(scores)):
                    g_id = g_ids[idx]
                    score = scores[idx]
                    is_pick = targets[idx]
                    
                    # Track the single highest output prediction score the model generated for this screen
                    if g_id not in group_max_score or score > group_max_score[g_id]:
                        group_max_score[g_id] = score
                        # Record if the model's preferred option was the actual chosen card
                        group_winning_target[g_id] = is_pick
                        
        # Calculate how often the model's top choice aligned with the human player's choice
        total_screens = len(group_winning_target)
        correct_selections = sum(1 for g in group_winning_target if group_winning_target[g] == 1)
        
        if total_screens > 0:
            top1_accuracy = (correct_selections / total_screens) * 100
            print(f"-> Validation Evaluation | Total Screens Checked: {total_screens} | Top-1 Recommender Accuracy: {top1_accuracy:.2f}%")
        else:
            print("-> Validation Evaluation | Warning: No complete decision groups detected in validation subset.")
        
    # Save the trained parameters
    torch.save(model.state_dict(), "spire_transformer_recommender.pt")
    print("Model parameters weights written successfully to disk!")

if __name__ == "__main__":
    train_recommender()
