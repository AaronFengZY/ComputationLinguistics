import os
import json
import pickle
import numpy as np

# Define the path to the dataset
data_file_path = "/home/v-zhifeng/HPE/nanoGPT/lyric_data_for_CL_no_id.jsonl"

# Initialize lists to store lyrics from each split
train_lyrics = []
valid_lyrics = []
test_lyrics = []

# Read the dataset and split into train, valid, and test
with open(data_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        entry = json.loads(line)
        if entry["dataset_split"] == "train":
            train_lyrics.extend(entry["lyric"])
        elif entry["dataset_split"] == "valid":
            valid_lyrics.extend(entry["lyric"])
        elif entry["dataset_split"] == "test":
            test_lyrics.extend(entry["lyric"])

# Combine all splits into single strings
train_data = "\n".join(train_lyrics)
valid_data = "\n".join(valid_lyrics)
test_data = "\n".join(test_lyrics)

# Print the length of each dataset in characters
print(f"Train data length in characters: {len(train_data):,}")
print(f"Valid data length in characters: {len(valid_data):,}")
print(f"Test data length in characters:  {len(test_data):,}")

# ---------------------------------------------------------------------
# IMPORTANT FIX:
# Build the vocabulary from ALL data (train, val, test)
# so that no character is missing in stoi/itos
# ---------------------------------------------------------------------
all_data = train_data + valid_data + test_data
chars = sorted(list(set(all_data)))
vocab_size = len(chars)

print(f"Vocabulary size: {vocab_size:,}")
# print(f"All unique characters: {''.join(chars)}")

# Create mappings from characters to integers and vice versa
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s: str):
    """Encodes a string into a list of integers."""
    return [stoi[c] for c in s]

def decode(arr: list):
    """Decodes a list of integers back into a string."""
    return ''.join([itos[i] for i in arr])

# Encode each split
train_ids = encode(train_data)
valid_ids = encode(valid_data)
test_ids = encode(test_data)

# Print dataset token counts
print(f"Train dataset tokens: {len(train_ids):,}")
print(f"Valid dataset tokens: {len(valid_ids):,}")
print(f"Test dataset tokens:  {len(test_ids):,}")

# Convert to numpy arrays and save as binary files
train_ids = np.array(train_ids, dtype=np.uint16)
valid_ids = np.array(valid_ids, dtype=np.uint16)
test_ids = np.array(test_ids, dtype=np.uint16)

output_dir = os.path.dirname(data_file_path)
train_bin_path = os.path.join(output_dir, 'train.bin')
valid_bin_path = os.path.join(output_dir, 'val.bin')
test_bin_path = os.path.join(output_dir, 'test.bin')

train_ids.tofile(train_bin_path)
valid_ids.tofile(valid_bin_path)
test_ids.tofile(test_bin_path)

# Save meta data for encoding/decoding
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(output_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print("Data preparation complete.")
print(f"Saved: {train_bin_path}, {valid_bin_path}, {test_bin_path}, and meta.pkl.")
