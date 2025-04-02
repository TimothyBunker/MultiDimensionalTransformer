# Multi-Dimensional Character Transformer for Sequence Tagging

This project implements a Transformer-based model designed for character-level sequence tagging tasks, such as Part-of-Speech (POS) tagging or Named Entity Recognition (NER).

The core idea is to leverage not only the character identities but also multiple "dimensions" or types of positional information to enrich the input representation fed into the Transformer encoder.

## Key Features

*   **Character-Level Processing:** Operates directly on sequences of characters rather than pre-tokenized words.
*   **Multi-Dimensional Positional Information:** Integrates several types of positional signals:
    *   **Absolute Position:** Standard sinusoidal positional encoding based on the character's overall position in the sequence.
    *   **Within-Word Position:** An embedding representing the character's position *within* its word (e.g., 1st char, 2nd char).
    *   **Word Index:** An embedding representing the index of the word the character belongs to in the sequence (e.g., 1st word, 2nd word).
*   **Transformer Architecture:** Utilizes a standard Transformer Encoder stack for contextual representation learning.
*   **Universal Dependencies Data:** Includes a script (`download_ud.py`) to automatically download and preprocess data from specified Universal Dependencies (UD) treebanks.

## Project Structure

*   `model.py`: Contains the PyTorch implementation of the `MultiLevelPosTransformer` model and the `PositionalEncoding` module.
*   `download_ud.py`: A utility script to download specified Universal Dependencies treebanks (primarily English ones by default), extract sentences, and save them into `train.txt`, `valid.txt`, and `test.txt` files.
*   **(Requires Implementation)** `preprocess.py` / `dataset.py`: Scripts/modules needed to convert the raw sentences from `.txt` files into the tensor formats required by `model.py` (character indices, positional indices, padding masks).
*   **(Requires Implementation)** `train.py`: A script to handle model training, including data loading, optimizer setup, loss calculation, and evaluation loops.
*   **(Requires Implementation)** `evaluate.py` / `predict.py`: Scripts for evaluating a trained model or making predictions on new data.

## Model Architecture (`model.py`)

The `MultiLevelPosTransformer` performs the following steps:

1.  **Input:** Takes batches of sequences containing character indices, within-word position indices, and word indices.
2.  **Embeddings:** Looks up embeddings for characters, within-word positions, and word indices.
3.  **Combine & Scale:** Sums the three types of embeddings. Character embeddings are scaled by `sqrt(d_model)`.
4.  **Add Absolute Position:** Adds standard sinusoidal positional encodings.
5.  **Transformer Encoder:** Processes the sequence through multiple self-attention and feed-forward layers. Handles padding via masks.
6.  **Classifier:** A final linear layer predicts tag scores for each character position.

## Data Acquisition (`download_ud.py`)

The `download_ud.py` script automates fetching data:

1.  **Clones/Pulls UD Repositories:** Downloads specified UD treebanks (e.g., `English-EWT`, `English-GUM`) from GitHub using `git`. Updates existing repositories if they exist.
2.  **Parses CoNLL-U:** Reads the `.conllu` files for each specified treebank.
3.  **Extracts Sentences:** Extracts plain text sentences from the `FORM` column of the CoNLL-U files.
4.  **Combines Training Data:** Concatenates training sentences from *all* specified treebanks into `data/train.txt`.
5.  **Selects Primary Treebank:** Uses the validation (`dev`) and test sets from *only* the specified primary treebank (e.g., `English-EWT`) for `data/valid.txt` and `data/test.txt`.
6.  **Output:** Creates a `data` directory (or specified output directory) containing `train.txt`, `valid.txt`, and `test.txt`.

## Requirements

*   Python 3.x
*   PyTorch (>= 1.8 recommended)
*   Git (must be installed and available in your system's PATH for `download_ud.py`)

```bash
pip install torch # Or install via PyTorch website instructions: https://pytorch.org/
