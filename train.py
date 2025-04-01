# train.py (with resume functionality)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import functools
import argparse # For command-line arguments
import os       # Needed for checking file existence
import sys

# Import from custom modules
from model import MultiLevelPosTransformer # Using the cleaned version
from data_utils import (
    build_char_vocab, CharPosTaggingDataset, collate_fn,
    PAD_TOKEN, SPACE_TOKEN, UNK_TOKEN
)

# --- Configuration & Hyperparameters ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SEQ_LEN = 512
MAX_WORD_LEN = 30
MAX_WORDS = 256
D_MODEL = 128
NHEAD = 4
NUM_ENCODER_LAYERS = 2
DIM_FEEDFORWARD = 512
DROPOUT = 0.26
EPOCHS = 30
# Total desired epochs
LEARNING_RATE = 0.001
BATCH_SIZE = 16
CLIP_GRAD = 1.0
# --- Use a clear name for the final fair model ---
MODEL_SAVE_PATH = 'final_fair_model.pt'
DATA_DIR = "data"

# --- Training and Evaluation Functions ---
# (train_epoch and evaluate_epoch functions remain exactly the same)
def train_epoch(model, dataloader, optimizer, criterion, device, tag_pad_idx, clip_value):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    start_time = time.time()
    for i, batch in enumerate(dataloader):
        chars = batch['chars'].to(device)
        wwp = batch['within_word_pos'].to(device)
        wi = batch['word_indices'].to(device)
        targets = batch['targets'].to(device)
        pad_mask = batch['pad_mask'].to(device)
        optimizer.zero_grad()
        logits = model(chars, wwp, wi, pad_mask=pad_mask)
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()
        if clip_value is not None and clip_value > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()
        total_loss += loss.item()
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            non_pad_target_mask = (targets != tag_pad_idx)
            correct = (preds == targets) & non_pad_target_mask
            total_correct += correct.sum().item()
            total_samples += non_pad_target_mask.sum().item()
        print_interval = max(1, len(dataloader) // 5)
        if (i + 1) % print_interval == 0:
             elapsed = time.time() - start_time
             total_batches = len(dataloader) if len(dataloader) > 0 else 1
             print(f'  Batch {i+1}/{total_batches} | Loss: {loss.item():.4f} | Elapsed: {elapsed:.2f}s')
    avg_loss = total_loss / len(dataloader) if dataloader and len(dataloader) > 0 else total_loss
    accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0
    return avg_loss, accuracy

def evaluate_epoch(model, dataloader, criterion, device, tag_pad_idx):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            chars = batch['chars'].to(device)
            wwp = batch['within_word_pos'].to(device)
            wi = batch['word_indices'].to(device)
            targets = batch['targets'].to(device)
            pad_mask = batch['pad_mask'].to(device)
            logits = model(chars, wwp, wi, pad_mask=pad_mask)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            non_pad_target_mask = (targets != tag_pad_idx)
            correct = (preds == targets) & non_pad_target_mask
            total_correct += correct.sum().item()
            total_samples += non_pad_target_mask.sum().item()
    avg_loss = total_loss / len(dataloader) if dataloader and len(dataloader) > 0 else total_loss
    accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0
    return avg_loss, accuracy

# --- Main Execution ---
if __name__ == "__main__":
    # --- Add Argument Parser ---
    parser = argparse.ArgumentParser(description="Train the POS Tagger model.")
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from the checkpoint specified by MODEL_SAVE_PATH')
    # You could add more arguments here for EPOCHS, LR, BATCH_SIZE etc. if needed
    args = parser.parse_args()

    print(f"Using device: {DEVICE}")

    # --- 1. Prepare Data ---
    TRAIN_FILE = os.path.join(DATA_DIR, "train.txt")
    VALID_FILE = os.path.join(DATA_DIR, "valid.txt")
    def load_sentences(filepath): # Same load_sentences function
        if not os.path.exists(filepath): print(f"Error: Data file not found: {filepath}."); return None
        try:
            with open(filepath, 'r', encoding='utf-8') as f: sentences = [line.strip() for line in f if line.strip()]
            if not sentences: print(f"Warning: No sentences loaded from {filepath}.")
            else: print(f"Loaded {len(sentences)} sentences from {filepath}.")
            return sentences
        except Exception as e: print(f"Error loading sentences: {e}"); return None
    print("\nLoading data from files...")
    train_texts = load_sentences(TRAIN_FILE); valid_texts = load_sentences(VALID_FILE)
    if train_texts is None or valid_texts is None: print("Error loading data. Exiting."); sys.exit(1)
    if not train_texts: print("Error: Training data empty. Exiting."); sys.exit(1)
    if not valid_texts: print("Warning: Validation data empty.")

    # --- 2. Build Character Vocabulary ---
    print("\nBuilding character vocabulary...")
    char_to_idx, idx_to_char = build_char_vocab(train_texts)
    char_pad_idx = char_to_idx[PAD_TOKEN]; char_vocab_size = len(char_to_idx)

    # --- 3. Create Datasets & Vocabularies ---
    print("\nCreating training dataset...")
    train_dataset = CharPosTaggingDataset(train_texts, char_to_idx, max_seq_len=MAX_SEQ_LEN,
                                        max_word_len=MAX_WORD_LEN, max_words=MAX_WORDS)
    if len(train_dataset) == 0: print("Error: Training dataset empty after processing."); sys.exit(1)
    tag_to_idx, idx_to_tag, tag_vocab_size, tag_pad_idx = train_dataset.get_tag_vocab_info()
    print("\nCreating validation dataset...")
    valid_dataset = None
    if valid_texts:
        valid_dataset = CharPosTaggingDataset(valid_texts, char_to_idx, tag_to_idx=tag_to_idx,
                                              max_seq_len=MAX_SEQ_LEN, max_word_len=MAX_WORD_LEN, max_words=MAX_WORDS)
        if len(valid_dataset) == 0: print("Warning: Validation dataset empty."); valid_dataset = None
    print(f"\n--- Final Vocab Sizes ---"); print(f"Character Vocab Size: {char_vocab_size}"); print(f"Tag Vocab Size: {tag_vocab_size}")

    # --- 4. Instantiate Model ---
    print("\nInstantiating model...")
    model_init_args = { # Store args for saving checkpoint consistency
        'char_vocab_size': char_vocab_size, 'tag_vocab_size': tag_vocab_size,
        'd_model': D_MODEL, 'nhead': NHEAD, 'num_encoder_layers': NUM_ENCODER_LAYERS,
        'dim_feedforward': DIM_FEEDFORWARD, 'dropout': DROPOUT,
        'max_word_len': MAX_WORD_LEN, 'max_words': MAX_WORDS, 'max_seq_len': MAX_SEQ_LEN + 10
    }
    model = MultiLevelPosTransformer(**model_init_args).to(DEVICE)
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    # --- 5. Define Loss & Optimizer ---
    criterion = nn.CrossEntropyLoss(ignore_index=tag_pad_idx)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    # --- !!! CHECKPOINT LOADING LOGIC !!! ---
    start_epoch = 0
    best_valid_loss = float('inf')

    if args.resume:
        if os.path.exists(MODEL_SAVE_PATH):
            print(f"\n--- Resuming training from checkpoint: {MODEL_SAVE_PATH} ---")
            try:
                checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)

                # Check for potential architecture mismatch (optional but good)
                saved_args = checkpoint.get('model_args', {})
                # Simple check example (could be more thorough)
                if saved_args.get('d_model') != D_MODEL or \
                   saved_args.get('char_vocab_size') != char_vocab_size or \
                   saved_args.get('tag_vocab_size') != tag_vocab_size:
                     print("Warning: Model hyperparameters in checkpoint might differ from current config.")
                     print("Ensure you are resuming with a compatible model structure.")

                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1 # Start from the next epoch
                best_valid_loss = checkpoint['loss']
                print(f"Resuming from Epoch {start_epoch}. Best validation loss so far: {best_valid_loss:.3f}")

                # Ensure loaded vocabs match current data (should if resuming same run)
                if checkpoint.get('char_to_idx') != char_to_idx or checkpoint.get('tag_to_idx') != tag_to_idx:
                     print("Warning: Vocabularies in checkpoint differ from current data. This might indicate resuming an incompatible run.")

            except Exception as e:
                 print(f"Error loading checkpoint: {e}. Starting training from scratch.")
                 start_epoch = 0
                 best_valid_loss = float('inf')
        else:
            print(f"Warning: --resume flag set, but checkpoint '{MODEL_SAVE_PATH}' not found. Starting training from scratch.")
    else:
        print("\n--- Starting training from scratch ---")
    # --- End Checkpoint Loading Logic ---


    # --- 6. Create DataLoaders ---
    collate_partial = functools.partial(collate_fn, char_pad_idx=char_pad_idx, tag_pad_idx=tag_pad_idx)
    if not train_dataset: print("Cannot create DataLoader: Training dataset invalid."); sys.exit(1)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  collate_fn=collate_partial, num_workers=2, pin_memory=True if DEVICE.type == 'cuda' else False)
    valid_dataloader = None
    if valid_dataset and len(valid_dataset) > 0:
        valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                      collate_fn=collate_partial, num_workers=2, pin_memory=True if DEVICE.type == 'cuda' else False)
        print(f"Validation DataLoader created with {len(valid_dataset)} samples.")
    else: print("Validation dataset empty or invalid, skipping validation.")

    # --- 7. Training Loop ---
    print(f"\n--- Starting Training from Epoch {start_epoch+1} ---") # Use start_epoch
    if start_epoch >= EPOCHS:
         print("Training already completed for the specified number of epochs.")
    else:
        # --- Loop starts from start_epoch ---
        for epoch in range(start_epoch, EPOCHS):
            epoch_num = epoch + 1 # Use epoch_num for printing
            start_epoch_time = time.time()

            train_loss, train_acc = train_epoch(model, train_dataloader, optimizer, criterion, DEVICE, tag_pad_idx, CLIP_GRAD)

            valid_loss, valid_acc = float('inf'), 0.0
            if valid_dataloader:
                 valid_loss, valid_acc = evaluate_epoch(model, valid_dataloader, criterion, DEVICE, tag_pad_idx)

            end_epoch_time = time.time()
            epoch_mins, epoch_secs = divmod(end_epoch_time - start_epoch_time, 60)

            print(f'\nEpoch: {epoch_num:02}/{EPOCHS} | Time: {int(epoch_mins)}m {epoch_secs:.0f}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%')
            if valid_dataloader:
                 print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc:.2f}%')
            else:
                 print("\t (No validation data to evaluate)")

            if valid_dataloader:
                 # --- Use epoch_num (which is epoch + 1) or just epoch? Save 'epoch' for consistency ---
                 if valid_loss < best_valid_loss:
                      best_valid_loss = valid_loss
                      checkpoint = {
                         'epoch': epoch, # Save the index of the completed epoch
                         'model_state_dict': model.state_dict(),
                         'optimizer_state_dict': optimizer.state_dict(),
                         'loss': best_valid_loss,
                         'char_to_idx': char_to_idx,
                         'tag_to_idx': tag_to_idx,
                         'model_args': model_init_args
                      }
                      try: torch.save(checkpoint, MODEL_SAVE_PATH); print(f"  ** Validation loss improved. Model saved to {MODEL_SAVE_PATH} **")
                      except Exception as e: print(f"  ** Error saving model: {e} **")
            else:
                 if epoch == EPOCHS - 1:
                      print(f"No validation data. Saving final model state to {MODEL_SAVE_PATH}")
                      checkpoint = {
                         'epoch': epoch, 'model_state_dict': model.state_dict(),
                         'optimizer_state_dict': optimizer.state_dict(), 'loss': train_loss,
                         'char_to_idx': char_to_idx, 'tag_to_idx': tag_to_idx, 'model_args': model_init_args
                      }
                      try: torch.save(checkpoint, MODEL_SAVE_PATH)
                      except Exception as e: print(f"  ** Error saving final model: {e} **")

        print("\n--- Training Finished ---")

    # Final message
    if os.path.exists(MODEL_SAVE_PATH):
         print(f"\nModel saved to {MODEL_SAVE_PATH}")
         print("\nTo evaluate this model:")
         print(f"python evaluate.py --model_path {MODEL_SAVE_PATH}")
    else:
         print("\nModel was not saved.")