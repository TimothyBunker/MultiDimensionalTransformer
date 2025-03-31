# download_ud.py (Final Fix for File Prefix)
import os
import subprocess
import argparse
import sys

# --- Configuration ---
DEFAULT_OUTPUT_DIR = "data"
DEFAULT_TREEBANKS = ["English-EWT", "English-GUM", "English-LinES", "English-ParTUT"]
DEFAULT_PRIMARY_TREEBANK = "English-EWT"
FORM_COL_IDX = 1

def parse_conllu_to_sentences(filepath):
    """Parses a CoNLL-U file and extracts sentences as plain text strings."""
    sentences = []
    current_sentence_tokens = []
    print(f"  Parsing {os.path.basename(filepath)}...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    if current_sentence_tokens:
                        sentences.append(" ".join(current_sentence_tokens))
                        current_sentence_tokens = []
                elif line.startswith('#'):
                    continue
                else:
                    fields = line.split('\t')
                    if len(fields) > FORM_COL_IDX and '-' not in fields[0]:
                         word_form = fields[FORM_COL_IDX]
                         current_sentence_tokens.append(word_form)
            if current_sentence_tokens:
                sentences.append(" ".join(current_sentence_tokens))
    except FileNotFoundError:
        # This error should be resolved now, but keep for safety
        print(f"    Error: File not found: {filepath}")
        # Optional: Add the CWD print here too if it persists
        # print(f"    Current Working Directory: {os.getcwd()}")
        return None
    except Exception as e:
        print(f"    Error parsing file {filepath}: {e}")
        return None
    print(f"    -> Found {len(sentences)} sentences.")
    return sentences

def clone_or_pull_repo(repo_url, repo_path):
    """Clones a repository if it doesn't exist, or pulls updates if it does."""
    # (This function remains the same as the previous working version)
    if not os.path.isdir(repo_path):
        print(f"Cloning repository from {repo_url} into {repo_path}...")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, repo_path],
                check=True, capture_output=True, text=True, timeout=300
            )
            print("  Repository cloned successfully.")
            return True
        except FileNotFoundError: print("\nError: 'git' command not found."); return False
        except subprocess.TimeoutExpired: print(f"\nError: git clone timed out for {repo_url}"); return False
        except subprocess.CalledProcessError as e: print(f"\nError cloning repo {repo_url}: {e}\nStderr: {e.stderr}"); return False
        except Exception as e: print(f"\nUnexpected error cloning: {e}"); return False
    else:
        print(f"Repository exists: {repo_path}. Checking updates...")
        try:
            subprocess.run(["git", "-C", repo_path, "fetch"], check=True, capture_output=True, text=True, timeout=120)
            status_output = subprocess.run(["git", "-C", repo_path, "status", "-uno"], check=True, capture_output=True, text=True, timeout=60).stdout
            if "Your branch is behind" in status_output:
                 print("  Pulling updates...")
                 subprocess.run(["git", "-C", repo_path, "pull"], check=True, capture_output=True, text=True, timeout=120)
                 print("  Pulled latest changes.")
            else: print("  Repository up-to-date.")
            return True
        except Exception as e: print(f"  Could not check/pull updates: {e}. Using local version."); return True

def download_and_process_multiple_ud(treebank_ids, primary_treebank_id, output_dir):
    """Downloads/updates multiple UD treebanks, combines training data."""
    print(f"--- Processing Multiple UD Treebanks ---")
    print(f"Treebanks: {', '.join(treebank_ids)}")
    print(f"Primary (valid/test): {primary_treebank_id}")
    print(f"Output dir: {output_dir}")

    if primary_treebank_id not in treebank_ids:
        print(f"\nError: Primary treebank '{primary_treebank_id}' not in list.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    all_train_sentences, primary_valid_sentences, primary_test_sentences = [], None, None

    for tb_id in treebank_ids:
        print(f"\n--- Processing {tb_id} ---")
        try:
             lang_part, code_part = tb_id.split('-', 1)
             repo_name = f"UD_{lang_part.capitalize()}-{code_part.upper()}"
             # --- *** CORRECTED FILE PREFIX LOGIC (FINAL) *** ---
             # Use the standard two-letter language code (usually)
             # For English, this is 'en'. For others, derive from lang_part if needed.
             # This assumes English is always 'en', adjust if processing non-English
             lang_code = "en" # Hardcode for English, could be derived if needed
             file_prefix = f"{lang_code}_{code_part.lower()}" # e.g., en_ewt, en_partut
             # --- *** END CORRECTION *** ---
             repo_url = f"https://github.com/UniversalDependencies/{repo_name}.git"
        except ValueError:
             print(f"Error: Invalid treebank ID format '{tb_id}'. Skipping.")
             continue

        repo_path = os.path.join(output_dir, repo_name)
        if not clone_or_pull_repo(repo_url, repo_path):
            print(f"Skipping {tb_id} due to repository issues.")
            continue

        # Process Training Data
        train_conllu_path = os.path.join(repo_path, f"{file_prefix}-ud-train.conllu")
        print(f"Processing training data ({os.path.basename(train_conllu_path)})...")
        train_sentences = parse_conllu_to_sentences(train_conllu_path)
        if train_sentences:
            all_train_sentences.extend(train_sentences)
            print(f"  Added {len(train_sentences)} training sentences from {tb_id}.")
        else: print(f"  Warning: Could not process training data for {tb_id}.")

        # Process Validation and Test Data (ONLY for Primary Treebank)
        if tb_id == primary_treebank_id:
            print(f"Processing validation/test data for primary treebank {tb_id}...")
            valid_conllu_path = os.path.join(repo_path, f"{file_prefix}-ud-dev.conllu")
            primary_valid_sentences = parse_conllu_to_sentences(valid_conllu_path)
            if primary_valid_sentences is None: print(f"  Warning: Could not process validation data for {tb_id}.")

            test_conllu_path = os.path.join(repo_path, f"{file_prefix}-ud-test.conllu")
            primary_test_sentences = parse_conllu_to_sentences(test_conllu_path)
            if primary_test_sentences is None: print(f"  Warning: Could not process test data for {tb_id}.")

    # Write Combined Output Files
    print("\n--- Writing Output Files ---")
    train_output_path = os.path.join(output_dir, "train.txt")
    if all_train_sentences:
        try:
            with open(train_output_path, 'w', encoding='utf-8') as f: [f.write(s + '\n') for s in all_train_sentences]
            print(f"Wrote {len(all_train_sentences)} training sentences to {train_output_path}")
        except Exception as e: print(f"Error writing {train_output_path}: {e}")
    else: print("Warning: No training sentences collected.")

    valid_output_path = os.path.join(output_dir, "valid.txt")
    if primary_valid_sentences:
         try:
             with open(valid_output_path, 'w', encoding='utf-8') as f: [f.write(s + '\n') for s in primary_valid_sentences]
             print(f"Wrote {len(primary_valid_sentences)} validation sentences to {valid_output_path}")
         except Exception as e: print(f"Error writing {valid_output_path}: {e}")
    else: print(f"Warning: No validation sentences found for '{primary_treebank_id}'.")

    test_output_path = os.path.join(output_dir, "test.txt")
    if primary_test_sentences:
         try:
             with open(test_output_path, 'w', encoding='utf-8') as f: [f.write(s + '\n') for s in primary_test_sentences]
             print(f"Wrote {len(primary_test_sentences)} test sentences to {test_output_path}")
         except Exception as e: print(f"Error writing {test_output_path}: {e}")
    else: print(f"Warning: No test sentences found for '{primary_treebank_id}'.")

    print("\n--- UD Data Processing Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download/process multiple UD datasets.")
    parser.add_argument("--treebanks", type=str, nargs='+', default=DEFAULT_TREEBANKS, help=f"List of treebanks (Lang-Code). Default: {' '.join(DEFAULT_TREEBANKS)}")
    parser.add_argument("--primary", type=str, default=DEFAULT_PRIMARY_TREEBANK, help=f"Primary treebank for valid/test. Default: {DEFAULT_PRIMARY_TREEBANK}")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}")
    args = parser.parse_args()
    download_and_process_multiple_ud(args.treebanks, args.primary, args.output_dir)