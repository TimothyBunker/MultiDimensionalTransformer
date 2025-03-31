# analyze_errors.py (Consistent with cleaned model.py and data_utils.py)
import torch
# import torch.nn as nn # Not directly needed here, but model uses it
from torch.utils.data import DataLoader
import time
import functools
import argparse
import os
import sys
from collections import Counter

# Import from custom modules
from model import MultiLevelPosTransformer # Using the cleaned version
from data_utils import (
    CharPosTaggingDataset, collate_fn, preprocess_sentence,
    PAD_TOKEN, SPACE_TOKEN, UNK_TOKEN
)

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "data"
TEST_FILE = os.path.join(DATA_DIR, "test.txt")
# --- Update default model path ---
DEFAULT_MODEL_PATH = 'final_fair_model.pt'
BATCH_SIZE = 16

# --- Helper Function to Load Sentences ---
def load_sentences(filepath):
    if not os.path.exists(filepath):
         print(f"Error: Data file not found at {filepath}.")
         return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]
        if not sentences:
             print(f"Warning: No sentences loaded from {filepath}.")
        else:
             print(f"Loaded {len(sentences)} sentences from {filepath}.")
        return sentences
    except Exception as e:
         print(f"Error loading sentences from {filepath}: {e}")
         return None

# --- Error Analysis Function for a Batch ---
# --- UPDATED: Removed pti handling ---
def analyze_batch_errors(batch, model, device, char_to_idx, tag_to_idx, idx_to_char, idx_to_tag):
    """
    Analyzes a single batch to find prediction errors.
    (Assumes model no longer takes pos_tag_indices as input).
    """
    model.eval()
    errors_found = []

    chars = batch['chars'].to(device)
    wwp = batch['within_word_pos'].to(device)
    wi = batch['word_indices'].to(device)
    # --- REMOVED loading 'pos_tag_indices' ---
    # pti = batch['pos_tag_indices'].to(device)
    targets = batch['targets'].to(device) # Ground truth tags per char
    pad_mask = batch['pad_mask'].to(device)
    char_pad_idx = char_to_idx[PAD_TOKEN]; tag_pad_idx = tag_to_idx[PAD_TOKEN]
    pad_tag_str = PAD_TOKEN; space_tag_str = SPACE_TOKEN

    with torch.no_grad():
        # --- REMOVED 'pti' from model call ---
        logits = model(chars, wwp, wi, pad_mask=pad_mask)
        preds = logits.argmax(dim=-1).cpu()
        targets = targets.cpu()
        chars = chars.cpu()
        word_indices_batch = batch['word_indices'].cpu()

    for i in range(chars.size(0)): # Loop through batch dimension
        seq_len = (chars[i] != char_pad_idx).sum().item()
        if seq_len == 0: continue

        original_chars_list = chars[i][:seq_len].tolist()
        word_indices_list = word_indices_batch[i][:seq_len].tolist()
        predicted_tags_list = preds[i][:seq_len].tolist()
        true_tags_list = targets[i][:seq_len].tolist()

        current_word_idx = -1; current_word_chars = []
        current_word_pred_tags_indices = []; current_word_true_tags_indices = []
        processed_words = 0

        for k in range(seq_len): # Iterate through characters
            char_idx = original_chars_list[k]; pred_tag_idx = predicted_tags_list[k]
            true_tag_idx = true_tags_list[k]; word_idx = word_indices_list[k]
            char_str = idx_to_char.get(char_idx, "?"); pred_tag_str = idx_to_tag.get(pred_tag_idx, "?")
            true_tag_str = idx_to_tag.get(true_tag_idx, "?")

            if char_idx == char_pad_idx or pred_tag_idx == tag_pad_idx or true_tag_idx == tag_pad_idx: continue

            is_space_char = (char_str == SPACE_TOKEN)

            process_previous = False
            if is_space_char and current_word_pred_tags_indices:
                process_previous = True
            elif word_idx != current_word_idx and current_word_pred_tags_indices:
                 process_previous = True

            if process_previous:
                word_text = "".join(current_word_chars)
                pred_counts = Counter(current_word_pred_tags_indices)
                pred_most_common = [p_idx for p_idx, c in pred_counts.most_common() if idx_to_tag.get(p_idx) not in [pad_tag_str, space_tag_str]]
                final_pred_tag = idx_to_tag.get(pred_most_common[0], "?") if pred_most_common else idx_to_tag.get(current_word_pred_tags_indices[0], "?")
                true_counts = Counter(current_word_true_tags_indices)
                true_most_common = [t_idx for t_idx, c in true_counts.most_common() if idx_to_tag.get(t_idx) not in [pad_tag_str, space_tag_str]]
                final_true_tag = idx_to_tag.get(true_most_common[0], "?") if true_most_common else idx_to_tag.get(current_word_true_tags_indices[0], "?")
                if final_pred_tag != final_true_tag:
                    errors_found.append({'word': word_text, 'predicted': final_pred_tag, 'true': final_true_tag,
                                         'sentence_idx_in_batch': i, 'word_position': processed_words})
                processed_words += 1
                current_word_chars = []; current_word_pred_tags_indices = []; current_word_true_tags_indices = []

            if is_space_char:
                 current_word_idx = -1
                 continue

            if word_idx != current_word_idx:
                current_word_idx = word_idx
                current_word_chars = [char_str]
                current_word_pred_tags_indices = [pred_tag_idx]
                current_word_true_tags_indices = [true_tag_idx]
            else:
                current_word_chars.append(char_str)
                current_word_pred_tags_indices.append(pred_tag_idx)
                current_word_true_tags_indices.append(true_tag_idx)

        if current_word_pred_tags_indices: # Process last word
            word_text = "".join(current_word_chars)
            pred_counts = Counter(current_word_pred_tags_indices)
            pred_most_common = [p_idx for p_idx, count in pred_counts.most_common() if idx_to_tag.get(p_idx) not in [pad_tag_str, space_tag_str]]
            final_pred_tag = idx_to_tag.get(pred_most_common[0], "?") if pred_most_common else idx_to_tag.get(current_word_pred_tags_indices[0], "?")
            true_counts = Counter(current_word_true_tags_indices)
            true_most_common = [t_idx for t_idx, count in true_counts.most_common() if idx_to_tag.get(t_idx) not in [pad_tag_str, space_tag_str]]
            final_true_tag = idx_to_tag.get(true_most_common[0], "?") if true_most_common else idx_to_tag.get(current_word_true_tags_indices[0], "?")
            if final_pred_tag != final_true_tag:
                errors_found.append({'word': word_text, 'predicted': final_pred_tag, 'true': final_true_tag,
                                     'sentence_idx_in_batch': i, 'word_position': processed_words})
    return errors_found

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze prediction errors of a trained POS tagging model.")
    # --- UPDATED default model path ---
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Path to the saved model checkpoint.")
    parser.add_argument("--test_file", type=str, default=TEST_FILE, help="Path to the test data file.")
    parser.add_argument("--device", type=str, default="auto", help="Device: 'cuda', 'cpu', 'auto'.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size.")
    parser.add_argument("--max_errors", type=int, default=50, help="Max errors to print (0 for all).")
    args = parser.parse_args()

    if args.device == "auto": device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else: device = torch.device(args.device)
    print(f"Using device: {device}")

    print(f"\nLoading checkpoint from {args.model_path}...")
    try: checkpoint = torch.load(args.model_path, map_location=device); print("Checkpoint loaded.")
    except Exception as e: print(f"Error loading checkpoint: {e}"); sys.exit(1)

    model_args = checkpoint.get('model_args'); char_to_idx = checkpoint.get('char_to_idx'); tag_to_idx = checkpoint.get('tag_to_idx')
    if not all([model_args, char_to_idx, tag_to_idx]): print("Error: Checkpoint missing info."); sys.exit(1)
    idx_to_char = {v: k for k, v in char_to_idx.items()}; idx_to_tag = {v: k for k, v in tag_to_idx.items()}
    char_pad_idx = char_to_idx.get(PAD_TOKEN); tag_pad_idx = tag_to_idx.get(PAD_TOKEN)
    if char_pad_idx is None or tag_pad_idx is None: print("Error: Pad index not found."); sys.exit(1)
    print("Vocabs and model args extracted.")

    print("\nInstantiating model...")
    try: model = MultiLevelPosTransformer(**model_args).to(device); model.load_state_dict(checkpoint['model_state_dict']); print("Model ready.")
    except Exception as e: print(f"Error loading model: {e}"); sys.exit(1)

    print("\nLoading test data...")
    test_texts = load_sentences(args.test_file)
    if test_texts is None or not test_texts: print("Error loading test data."); sys.exit(1)

    print("\nCreating test dataset for analysis...")
    max_seq_len = model_args.get('max_seq_len', 128+10) - 10; max_word_len = model_args.get('max_word_len', 25); max_words = model_args.get('max_words', 30)
    test_dataset = CharPosTaggingDataset(test_texts, char_to_idx, tag_to_idx=tag_to_idx,
                                         max_seq_len=max_seq_len, max_word_len=max_word_len, max_words=max_words)
    if len(test_dataset) == 0: print("Error: Test dataset empty."); sys.exit(1)

    # --- NOTE: collate_fn no longer expects/handles pos_tag_indices ---
    collate_partial = functools.partial(collate_fn, char_pad_idx=char_pad_idx, tag_pad_idx=tag_pad_idx)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=collate_partial, num_workers=0) # Using 0 workers often simpler for debugging/analysis
    print(f"Test DataLoader created with {len(test_dataset)} samples.")

    print("\n--- Starting Error Analysis ---")
    all_errors = []; total_processed_sentences = 0; start_time = time.time()
    for batch_num, batch in enumerate(test_dataloader):
        # --- Pass updated arguments to analyze_batch_errors ---
        batch_errors = analyze_batch_errors(batch, model, device, char_to_idx, tag_to_idx, idx_to_char, idx_to_tag)
        start_idx = batch_num * args.batch_size
        for error in batch_errors: error['sentence_idx_in_dataset'] = start_idx + error['sentence_idx_in_batch']
        all_errors.extend(batch_errors)
        total_processed_sentences += batch['chars'].size(0)
        if (batch_num + 1) % (max(1, len(test_dataloader) // 10)) == 0: print(f"  Processed batch {batch_num+1}/{len(test_dataloader)}")
    end_time = time.time(); print(f"Analysis finished in {end_time - start_time:.2f} seconds.")

    total_errors = len(all_errors)
    print(f"\n--- Error Analysis Report ---")
    print(f"Model: {args.model_path}")
    print(f"Total sentences analyzed: {len(test_dataset)}")
    print(f"Total errors found (word-level): {total_errors}")

    errors_to_show = args.max_errors if args.max_errors > 0 else total_errors
    if total_errors > 0:
        print(f"\nShowing details for up to {errors_to_show} errors:")
        print("-" * 40)
        for i, error in enumerate(all_errors):
            if i >= errors_to_show: print(f"... (omitting {total_errors - errors_to_show} more errors)"); break
            original_sentence_text = test_texts[error['sentence_idx_in_dataset']]
            print(f"Error {i+1}:")
            print(f"  Sentence (Index {error['sentence_idx_in_dataset']}): {original_sentence_text}")
            print(f"  Word (Position ~{error['word_position']}): '{error['word']}'")
            print(f"  Predicted: {error['predicted']}")
            print(f"  True:      {error['true']}")
            print("-" * 40)
        error_type_counts = Counter((err['true'], err['predicted']) for err in all_errors)
        print("\nError Type Counts (True -> Predicted):")
        for (true_tag, pred_tag), count in error_type_counts.most_common(): print(f"  {true_tag} -> {pred_tag} : {count}")
    else: print("\nNo errors found in the processed test set!")
    print("-----------------------------")