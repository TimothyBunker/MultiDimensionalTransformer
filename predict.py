# predict.py (Consistent with cleaned model.py and data_utils.py)
import torch
import argparse
import os
import sys
from collections import Counter

# Import necessary components from other files
from model import MultiLevelPosTransformer # Using the cleaned version
# --- Need CharPosTaggingDataset just to get access to preprocess_sentence logic? ---
# --- Or import preprocess_sentence directly if it doesn't rely on Dataset internals ---
from data_utils import preprocess_sentence, SPACE_TOKEN, PAD_TOKEN, UNK_TOKEN

# --- Configuration ---
# Update default model path
DEFAULT_MODEL_PATH = 'final_fair_model.pt'

# --- Prediction Function ---
# --- UPDATED: Removed pti handling ---
def predict_sentence_tags(sentence: str,
                          model: torch.nn.Module,
                          char_to_idx: dict,
                          tag_to_idx: dict,
                          device: torch.device,
                          max_seq_len: int = 128,
                          max_word_len: int = 25,
                          max_words: int = 30) -> list:
    """
    Performs inference on a single sentence and returns word-level tags.
    (Assumes model no longer takes pos_tag_indices as input).
    """
    model.eval() # Set model to evaluation mode

    # Preprocess the input sentence
    # preprocess_sentence now returns a dict without 'pos_tag_indices'
    processed = preprocess_sentence(sentence, char_to_idx, tag_to_idx,
                                    max_seq_len, max_word_len, max_words)
    if not processed:
        print(f"Warning: Sentence couldn't be processed: '{sentence}'")
        return None

    # Add batch dimension and move tensors to the specified device
    chars = processed['chars'].unsqueeze(0).to(device)
    wwp = processed['within_word_pos'].unsqueeze(0).to(device)
    wi = processed['word_indices'].unsqueeze(0).to(device)
    # --- REMOVED loading 'pos_tag_indices' ---
    # pti = processed['pos_tag_indices'].unsqueeze(0).to(device)
    pad_mask = torch.zeros(chars.shape, dtype=torch.bool).to(device) # No padding for single sentence

    # Perform inference
    with torch.no_grad():
        # --- REMOVED 'pti' from model call ---
        logits = model(chars, wwp, wi, pad_mask=pad_mask)

    # Get predictions per character
    preds = logits.argmax(dim=-1).squeeze(0).cpu() # Move predictions to CPU [seq_len]

    # --- Post-processing: Aggregate character tags to word tags ---
    idx_to_char = {v: k for k, v in char_to_idx.items()}
    idx_to_tag = {v: k for k, v in tag_to_idx.items()}
    char_pad_idx = char_to_idx.get(PAD_TOKEN) # Need this for length check
    tag_pad_idx = tag_to_idx.get(PAD_TOKEN) # Need this for ignoring pad tags
    pad_tag_str = PAD_TOKEN
    space_tag_str = SPACE_TOKEN

    word_level_tags = []
    current_word_idx = -1
    current_word_chars = []
    current_word_pred_tags_indices = []

    # Use processed['chars'] directly for iteration length
    seq_len = len(processed['chars'])
    original_chars_list = processed['chars'].tolist()
    word_indices_list = processed['word_indices'].tolist()
    predicted_tags_list = preds[:seq_len].tolist() # Ensure preds list matches original length

    for k in range(seq_len):
        char_idx = original_chars_list[k]; pred_tag_idx = predicted_tags_list[k]
        word_idx = word_indices_list[k]
        char_str = idx_to_char.get(char_idx, "?"); pred_tag_str = idx_to_tag.get(pred_tag_idx, "?")

        if char_idx == char_pad_idx or pred_tag_idx == tag_pad_idx: continue

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
            word_level_tags.append((word_text, final_pred_tag))
            current_word_chars = []; current_word_pred_tags_indices = []

        if is_space_char:
             current_word_idx = -1
             continue

        if word_idx != current_word_idx:
            current_word_idx = word_idx
            current_word_chars = [char_str]
            current_word_pred_tags_indices = [pred_tag_idx]
        else:
            current_word_chars.append(char_str)
            current_word_pred_tags_indices.append(pred_tag_idx)

    if current_word_pred_tags_indices: # Process last word
        word_text = "".join(current_word_chars)
        pred_counts = Counter(current_word_pred_tags_indices)
        pred_most_common = [p_idx for p_idx, count in pred_counts.most_common() if idx_to_tag.get(p_idx) not in [pad_tag_str, space_tag_str]]
        final_pred_tag = idx_to_tag.get(pred_most_common[0], "?") if pred_most_common else idx_to_tag.get(current_word_pred_tags_indices[0], "?")
        word_level_tags.append((word_text, final_pred_tag))

    return word_level_tags


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict POS tags for a sentence using a trained model.")
    parser.add_argument("sentence", type=str, help="The sentence to tag.")
    # --- UPDATED default model path ---
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Path to the saved model checkpoint (.pt file).")
    parser.add_argument("--device", type=str, default="auto", help="Device to use ('cuda', 'cpu', or 'auto').")
    args = parser.parse_args()

    if args.device == "auto": device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else: device = torch.device(args.device)
    print(f"Using device: {device}")

    print(f"\nLoading checkpoint from {args.model_path}...")
    try: checkpoint = torch.load(args.model_path, map_location=device); print("Checkpoint loaded.")
    except Exception as e: print(f"Error loading checkpoint: {e}"); sys.exit(1)

    model_args = checkpoint.get('model_args'); char_to_idx = checkpoint.get('char_to_idx'); tag_to_idx = checkpoint.get('tag_to_idx')
    if not all([model_args, char_to_idx, tag_to_idx]): print("Error: Checkpoint missing required info."); sys.exit(1)
    print("Vocabs and model args extracted.")

    print("\nInstantiating model...")
    try: model = MultiLevelPosTransformer(**model_args).to(device); model.load_state_dict(checkpoint['model_state_dict']); print("Model ready.")
    except Exception as e: print(f"Error loading model: {e}"); sys.exit(1)

    # --- Use model_args from checkpoint for preprocessing limits ---
    max_seq_len = model_args.get('max_seq_len', 128+10) - 10
    max_word_len = model_args.get('max_word_len', 25)
    max_words = model_args.get('max_words', 30)
    print(f"Using preprocessing limits from checkpoint: max_seq_len={max_seq_len}, max_word_len={max_word_len}, max_words={max_words}")


    # Perform prediction
    predicted_word_tags = predict_sentence_tags(
        args.sentence,
        model,
        char_to_idx,
        tag_to_idx,
        device,
        max_seq_len=max_seq_len, # Use limits from checkpoint
        max_word_len=max_word_len,
        max_words=max_words
    )

    if predicted_word_tags:
        print("\nPrediction:")
        print(f"Input: {args.sentence}")
        print("Tagged Output:")
        output_str = " ".join([f"{word}/{tag}" for word, tag in predicted_word_tags])
        print(f"  {output_str}")
        print("\n")