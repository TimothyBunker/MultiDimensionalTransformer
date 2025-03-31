# evaluate.py (Consistent with cleaned model.py and data_utils.py)
import torch
import torch.nn as nn
# import torch.optim as optim # Not needed for evaluation
from torch.utils.data import DataLoader
import time
import functools
import argparse
import os
import sys

# Import from custom modules
from model import MultiLevelPosTransformer # Using the cleaned version
from data_utils import (
    CharPosTaggingDataset, collate_fn, PAD_TOKEN, SPACE_TOKEN, UNK_TOKEN
)

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "data"
TEST_FILE = os.path.join(DATA_DIR, "test.txt")
# --- Update default model path ---
DEFAULT_MODEL_PATH = 'final_fair_model.pt'
BATCH_SIZE = 16 # Can often be larger for evaluation than training

# --- Helper Function to Load Sentences ---
def load_sentences(filepath):
    """Loads sentences from a text file, one sentence per line."""
    if not os.path.exists(filepath):
         print(f"Error: Data file not found at {filepath}.")
         return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]
        if not sentences:
             print(f"Warning: No sentences loaded from {filepath}. File might be empty.")
        else:
             print(f"Loaded {len(sentences)} sentences from {filepath}.")
        return sentences
    except Exception as e:
         print(f"Error loading sentences from {filepath}: {e}")
         return None

# --- Evaluation Function ---
# --- UPDATED: Removed pti handling ---
def evaluate_epoch(model, dataloader, criterion, device, tag_pad_idx):
    """Evaluates the model for one epoch."""
    model.eval() # Set model to evaluation mode
    total_loss = 0
    total_correct = 0
    total_samples = 0 # Count non-padded tokens
    print("Starting evaluation...")
    start_time = time.time()
    with torch.no_grad(): # IMPORTANT: Disable gradient calculation
        for i, batch in enumerate(dataloader):
            chars = batch['chars'].to(device)
            wwp = batch['within_word_pos'].to(device)
            wi = batch['word_indices'].to(device)
            # --- REMOVED loading 'pos_tag_indices' ---
            # pti = batch['pos_tag_indices'].to(device) # Input features from test set
            targets = batch['targets'].to(device) # Ground truth labels from test set
            pad_mask = batch['pad_mask'].to(device)

            # --- REMOVED 'pti' from model call ---
            logits = model(chars, wwp, wi, pad_mask=pad_mask)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            total_loss += loss.item()

            preds = logits.argmax(dim=-1)
            non_pad_target_mask = (targets != tag_pad_idx)
            correct = (preds == targets) & non_pad_target_mask
            total_correct += correct.sum().item()
            total_samples += non_pad_target_mask.sum().item()

            print_interval = max(1, len(dataloader) // 5)
            if (i + 1) % print_interval == 0:
                 print(f"  Evaluated batch {i+1}/{len(dataloader)}")

    end_time = time.time()
    print(f"Evaluation finished in {end_time - start_time:.2f} seconds.")
    avg_loss = total_loss / len(dataloader) if dataloader and len(dataloader) > 0 else total_loss
    accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0
    return avg_loss, accuracy

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained POS tagging model on the test set.")
    # --- UPDATED default model path ---
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Path to the saved model checkpoint (.pt file).")
    parser.add_argument("--test_file", type=str, default=TEST_FILE, help="Path to the test data file (one sentence per line).")
    parser.add_argument("--device", type=str, default="auto", help="Device to use ('cuda', 'cpu', or 'auto').")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for evaluation.")
    args = parser.parse_args()

    if args.device == "auto": device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else: device = torch.device(args.device)
    print(f"Using device: {device}")

    print(f"\nLoading checkpoint from {args.model_path}...")
    try: checkpoint = torch.load(args.model_path, map_location=device); print("Checkpoint loaded.")
    except Exception as e: print(f"Error loading checkpoint: {e}"); sys.exit(1)

    model_args = checkpoint.get('model_args'); char_to_idx = checkpoint.get('char_to_idx'); tag_to_idx = checkpoint.get('tag_to_idx')
    if not all([model_args, char_to_idx, tag_to_idx]): print("Error: Checkpoint missing required info."); sys.exit(1)

    char_pad_idx = char_to_idx.get(PAD_TOKEN); tag_pad_idx = tag_to_idx.get(PAD_TOKEN)
    if char_pad_idx is None or tag_pad_idx is None: print("Error: Pad index not found in vocabs."); sys.exit(1)
    print("Vocabs and model args extracted.")

    print("\nInstantiating model...")
    try: model = MultiLevelPosTransformer(**model_args).to(device); print("Model instantiated.")
    except Exception as e: print(f"Error instantiating model: {e}"); sys.exit(1)

    try: model.load_state_dict(checkpoint['model_state_dict']); print("Weights loaded.")
    except Exception as e: print(f"Error loading state_dict: {e}"); sys.exit(1)

    print("\nLoading test data...")
    test_texts = load_sentences(args.test_file)
    if test_texts is None or not test_texts: print("Error loading test data."); sys.exit(1)

    print("\nCreating test dataset...")
    # --- Use model_args for consistency ---
    max_seq_len = model_args.get('max_seq_len', 512+10) - 10
    max_word_len = model_args.get('max_word_len', 30)
    max_words = model_args.get('max_words', 300)
    test_dataset = CharPosTaggingDataset(test_texts, char_to_idx, tag_to_idx=tag_to_idx,
                                         max_seq_len=max_seq_len, max_word_len=max_word_len, max_words=max_words)
    if len(test_dataset) == 0: print("Error: Test dataset empty."); sys.exit(1)

    # --- NOTE: collate_fn no longer expects/handles pos_tag_indices ---
    collate_partial = functools.partial(collate_fn, char_pad_idx=char_pad_idx, tag_pad_idx=tag_pad_idx)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=collate_partial, num_workers=2,
                                 pin_memory=True if device.type == 'cuda' else False)
    print(f"Test DataLoader created with {len(test_dataset)} samples.")

    criterion = nn.CrossEntropyLoss(ignore_index=tag_pad_idx)

    # --- Run Evaluation ---
    test_loss, test_acc = evaluate_epoch(model, test_dataloader, criterion, device, tag_pad_idx)

    # --- Report Results ---
    print("\n--- Test Set Evaluation Results ---")
    print(f"\tModel: {args.model_path}")
    print(f"\tTest Loss: {test_loss:.3f}")
    print(f"\tTest Acc:  {test_acc:.2f}%")
    print("-----------------------------------")