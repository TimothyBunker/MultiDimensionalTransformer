# model.py (Cleaned version - No Input POS Tag Feature)
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Injects positional information into the sequence embeddings."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # Shape: [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Input tensor shape [seq_len, batch_size, d_model]
        Returns:
            Output tensor shape [seq_len, batch_size, d_model]
        """
        # Add positional encoding and apply dropout
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class MultiLevelPosTransformer(nn.Module):
    """
    Transformer Encoder for character-level tasks integrating multiple
    positional embeddings (Within-Word Position, Word Index).
    (Input POS Tag feature has been removed).

    Args:
        char_vocab_size (int): Size of the character vocabulary.
        tag_vocab_size (int): Size of the POS tag vocabulary (output dimension).
        d_model (int): Dimension of embeddings and hidden states.
        nhead (int): Number of attention heads in encoder layers.
        num_encoder_layers (int): Number of stacked Transformer encoder layers.
        dim_feedforward (int): Dimension of the feed-forward network in encoder layers.
        dropout (float): Dropout probability.
        max_word_len (int): Maximum length of a word for within-word positional embedding.
        max_words (int): Maximum number of words in a sequence for word index embedding.
        max_seq_len (int): Maximum sequence length for absolute positional encoding.
    """
    def __init__(self,
                 char_vocab_size: int,
                 tag_vocab_size: int, # Still needed for output classifier
                 d_model: int,
                 nhead: int,
                 num_encoder_layers: int,
                 dim_feedforward: int,
                 dropout: float,
                 max_word_len: int,
                 max_words: int,
                 max_seq_len: int = 5000):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        self.d_model = d_model

        # --- Embedding Layers ---
        self.char_embedding = nn.Embedding(char_vocab_size, d_model)
        self.within_word_pos_embedding = nn.Embedding(max_word_len, d_model)
        self.word_index_embedding = nn.Embedding(max_words, d_model)
        # --- REMOVED self.pos_tag_feature_embedding ---

        # --- Positional Encoding ---
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_seq_len)

        # --- Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation=nn.ReLU()
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_encoder_layers
        )

        # --- Output Layer ---
        self.classifier = nn.Linear(d_model, tag_vocab_size) # Output dimension remains tag_vocab_size

        # --- Initialization ---
        self._init_weights()

    def _init_weights(self):
        """Initializes weights using a uniform distribution."""
        initrange = 0.1
        self.char_embedding.weight.data.uniform_(-initrange, initrange)
        self.within_word_pos_embedding.weight.data.uniform_(-initrange, initrange)
        self.word_index_embedding.weight.data.uniform_(-initrange, initrange)
        # --- REMOVED initialization for pos_tag_feature_embedding ---
        self.classifier.bias.data.zero_()
        self.classifier.weight.data.uniform_(-initrange, initrange)

    # --- UPDATED forward signature: removed pos_tag_indices ---
    def forward(self,
                chars: torch.Tensor,           # [batch_size, seq_len]
                within_word_pos: torch.Tensor, # [batch_size, seq_len]
                word_indices: torch.Tensor,    # [batch_size, seq_len]
                # pos_tag_indices: torch.Tensor, # REMOVED from arguments
                pad_mask: torch.Tensor = None  # [batch_size, seq_len] - True where padded
               ) -> torch.Tensor:             # [batch_size, seq_len, tag_vocab_size]
        """Forward pass of the model."""

        # 1. Get Embeddings & Scale Character Embeddings
        char_emb = self.char_embedding(chars) * math.sqrt(self.d_model)
        wwp_emb = self.within_word_pos_embedding(within_word_pos)
        wi_emb = self.word_index_embedding(word_indices)
        # --- REMOVED pti_emb calculation ---

        # 2. Combine Embeddings (Summation)
        # --- UPDATED combined_emb calculation: removed pti_emb ---
        # --- ABLATION COMMENTS REMOVED FOR FINAL VERSION ---
        combined_emb = char_emb + wwp_emb + wi_emb

        # 3. Add Absolute Positional Encoding
        combined_emb_permuted = combined_emb.permute(1, 0, 2)
        pos_encoded_emb_permuted = self.pos_encoder(combined_emb_permuted) # Includes dropout
        pos_encoded_emb = pos_encoded_emb_permuted.permute(1, 0, 2)
        src = pos_encoded_emb # Use output from PE

        # 4. Pass through Transformer Encoder
        memory = self.transformer_encoder(src, src_key_padding_mask=pad_mask)

        # 5. Classify each position
        logits = self.classifier(memory)

        return logits