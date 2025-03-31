# data_utils.py
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import spacy
from collections import Counter

# --- Constants ---
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SPACE_TOKEN = "<space>" # Explicit token for spaces between words

# --- spaCy Loading ---
# Load once when the module is imported. Consider error handling for production.
try:
    nlp = spacy.load("en_core_web_sm")
    print("spaCy model 'en_core_web_sm' loaded successfully for data_utils.")
except OSError:
    print("Error loading spaCy model 'en_core_web_sm' in data_utils.")
    print("Please run: python -m spacy download en_core_web_sm")
    nlp = None # Set to None to indicate failure

# --- Vocabulary Builders ---
# (build_char_vocab and build_pos_tag_vocab remain unchanged)
def build_char_vocab(texts, min_freq=1):
    """Builds character vocabulary from a list of texts."""
    if not texts: return {PAD_TOKEN: 0, UNK_TOKEN: 1, SPACE_TOKEN: 2}, {}
    char_counts = Counter()
    for text in texts:
        char_counts.update(list(text))
    char_to_idx = {PAD_TOKEN: 0, UNK_TOKEN: 1, SPACE_TOKEN: 2}
    idx = len(char_to_idx)
    for char, count in char_counts.items():
        if count >= min_freq and char not in char_to_idx:
            char_to_idx[char] = idx
            idx += 1
    idx_to_char = {v: k for k, v in char_to_idx.items()}
    print(f"Built character vocabulary with {len(char_to_idx)} tokens.")
    return char_to_idx, idx_to_char

def build_pos_tag_vocab(spacy_docs):
    """Builds POS tag vocabulary from a list of spaCy Doc objects."""
    if not spacy_docs or nlp is None: return {PAD_TOKEN: 0, SPACE_TOKEN: 1}, {}
    tag_counts = Counter()
    for doc in spacy_docs:
        for token in doc:
            tag = token.pos_
            if token.is_punct: tag = 'PUNCT'
            tag_counts.update([tag])
    tag_to_idx = {PAD_TOKEN: 0, SPACE_TOKEN: 1}
    idx = len(tag_to_idx)
    for tag, _ in tag_counts.items():
        if tag not in tag_to_idx:
            tag_to_idx[tag] = idx
            idx += 1
    idx_to_tag = {v: k for k, v in tag_to_idx.items()}
    print(f"Built POS tag vocabulary with {len(tag_to_idx)} tags: {list(tag_to_idx.keys())}")
    return tag_to_idx, idx_to_tag

# --- Preprocessing Function ---

def preprocess_sentence(sentence: str, char_to_idx: dict, tag_to_idx: dict,
                        max_seq_len: int, max_word_len: int, max_words: int):
    """Preprocesses a single sentence string into index sequences."""
    if nlp is None:
        raise RuntimeError("spaCy model not loaded. Cannot preprocess.")
    doc = nlp(sentence)
    # Basic validation
    if len(doc) == 0 or len(doc) > max_words: return None

    unk_char_idx = char_to_idx[UNK_TOKEN]
    space_char_idx = char_to_idx[SPACE_TOKEN]
    space_tag_idx = tag_to_idx.get(SPACE_TOKEN, tag_to_idx[PAD_TOKEN])
    pad_tag_idx = tag_to_idx[PAD_TOKEN]

    char_indices, within_word_pos, word_indices = [], [], []
    # --- REMOVED pos_tag_indices list ---
    # pos_tag_indices = []
    target_pos_indices = [] # Keep targets!
    current_char_idx_global = 0
    processed_words = 0

    for token in doc:
        # Handle spaces
        num_spaces = token.idx - current_char_idx_global
        if num_spaces > 0:
            for _ in range(num_spaces):
                if len(char_indices) >= max_seq_len: break
                char_indices.append(space_char_idx)
                within_word_pos.append(0)
                word_indices.append(processed_words)
                # --- REMOVED appending to pos_tag_indices ---
                # pos_tag_indices.append(space_tag_idx)
                target_pos_indices.append(space_tag_idx) # Target is still SPACE
            if len(char_indices) >= max_seq_len: break

        # Handle token
        word_str = token.text
        tag = token.pos_
        if token.is_punct: tag = 'PUNCT'
        tag_id = tag_to_idx.get(tag, pad_tag_idx)
        current_word_idx_clamped = min(processed_words, max_words - 1)

        for j, char in enumerate(word_str):
            if len(char_indices) >= max_seq_len: break
            char_within_pos_clamped = min(j, max_word_len - 1)
            char_indices.append(char_to_idx.get(char, unk_char_idx))
            within_word_pos.append(char_within_pos_clamped)
            word_indices.append(current_word_idx_clamped)
            # --- REMOVED appending to pos_tag_indices ---
            # pos_tag_indices.append(tag_id)
            target_pos_indices.append(tag_id) # Keep appending target
        if len(char_indices) >= max_seq_len: break

        current_char_idx_global = token.idx + len(token.text)
        processed_words += 1

    if not char_indices: return None

    # Convert lists to tensors
    return {
        "chars": torch.tensor(char_indices, dtype=torch.long),
        "within_word_pos": torch.tensor(within_word_pos, dtype=torch.long),
        "word_indices": torch.tensor(word_indices, dtype=torch.long),
        # --- REMOVED pos_tag_indices from returned dict ---
        # "pos_tag_indices": torch.tensor(pos_tag_indices, dtype=torch.long),
        "targets": torch.tensor(target_pos_indices, dtype=torch.long),
    }


# --- Dataset Class ---
# (No changes needed in __init__, __len__, __getitem__, get_tag_vocab_info
# as they rely on the output of preprocess_sentence)
class CharPosTaggingDataset(Dataset):
    """PyTorch Dataset for character-level POS tagging."""
    def __init__(self, texts, char_to_idx, tag_to_idx=None,
                 max_seq_len=128, max_word_len=25, max_words=30):
        super().__init__()
        if nlp is None:
            raise RuntimeError("spaCy model not loaded. Cannot initialize Dataset.")

        print(f"Initializing dataset with {len(texts)} sentences...")
        self.max_seq_len = max_seq_len
        self.char_to_idx = char_to_idx

        print("Processing texts with spaCy..."); spacy_docs = list(nlp.pipe(texts)); print("Finished spaCy processing.")

        if tag_to_idx is None:
            print("Building POS tag vocabulary for this dataset split...")
            self.tag_to_idx, self.idx_to_tag = build_pos_tag_vocab(spacy_docs)
        else:
            print("Using provided POS tag vocabulary.")
            self.tag_to_idx = tag_to_idx
            self.idx_to_tag = {v: k for k, v in tag_to_idx.items()}

        self.tag_vocab_size = len(self.tag_to_idx)
        self.tag_pad_idx = self.tag_to_idx[PAD_TOKEN]

        self.processed_data = []
        print("Preprocessing sentences for the dataset...")
        num_skipped_length, num_skipped_empty = 0, 0
        for i, doc in enumerate(spacy_docs):
            processed = preprocess_sentence(texts[i], self.char_to_idx, self.tag_to_idx,
                                            max_seq_len, max_word_len, max_words)
            if processed:
                self.processed_data.append(processed)
            else:
                 # Use doc length before potential truncation in preprocess_sentence
                 if len(doc) > max_words: num_skipped_length += 1
                 else: num_skipped_empty += 1 # Includes empty docs

        print(f"Finished preprocessing. Kept {len(self.processed_data)} examples.")
        if num_skipped_length > 0: print(f"Skipped {num_skipped_length} sentences due to exceeding max_words ({max_words}).")
        if num_skipped_empty > 0: print(f"Skipped {num_skipped_empty} sentences due to being empty or other issues.")

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        return self.processed_data[idx]

    def get_tag_vocab_info(self):
        """Helper to get vocab info after initialization."""
        return self.tag_to_idx, self.idx_to_tag, self.tag_vocab_size, self.tag_pad_idx


# --- Collate Function ---

def collate_fn(batch, char_pad_idx, tag_pad_idx):
    """Collates a batch of processed data points, padding them dynamically."""
    if not batch: return {}

    # Separate the sequences from the batch
    # --- Get keys dynamically, excluding the removed 'pos_tag_indices' ---
    keys = batch[0].keys() # Get keys from the first item
    data = {k: [item[k] for item in batch] for k in keys}

    # Pad all sequences to the max length *in this batch*
    padded_data = {}
    for k, sequences in data.items():
        if k == "chars":
            pad_value = char_pad_idx
        elif k in ["within_word_pos", "word_indices"]:
            pad_value = 0 # Pad positions/indices with 0
        # --- REMOVED specific handling for 'pos_tag_indices' ---
        elif k == "targets": # Only targets use tag_pad_idx now
            pad_value = tag_pad_idx
        else:
             # This case should ideally not be hit if preprocess_sentence is correct
             raise ValueError(f"Unexpected key '{k}' in batch item during collation.")

        padded_data[k] = pad_sequence(sequences, batch_first=True, padding_value=pad_value)

    # Create the padding mask (True where characters are padded)
    padded_data["pad_mask"] = (padded_data["chars"] == char_pad_idx)

    # --- REMOVED 'pos_tag_indices' from the final returned dictionary ---
    return padded_data