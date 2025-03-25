import re
import pickle
import torch
from torch.nn.utils.rnn import pad_sequence

# Load dataset
file_path = "shakespeare.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text = file.read().lower()

# Clean text (keep only letters, spaces, and apostrophes)
text = re.sub(r"[^a-z\s']", "", text)
text = re.sub(r"\s+", " ", text).strip()

# Tokenization using built-in Python `split()`
tokens = text.split()

# Create word-index mappings
word_to_index = {word: idx + 1 for idx, word in enumerate(set(tokens))}
index_to_word = {idx: word for word, idx in word_to_index.items()}
vocab_size = len(word_to_index) + 1 

# Convert text to sequences in chunks
input_sequences = []
chunk_size = 5000 
for i in range(0, len(tokens) - chunk_size, chunk_size):
    seq = [word_to_index[word] for word in tokens[i: i + chunk_size]]
    input_sequences.append(seq)

# Pad sequences to uniform length
padded_sequences = [torch.tensor(seq) for seq in input_sequences]
padded_sequences = pad_sequence(padded_sequences, batch_first=True, padding_value=0)

# Split into X (predictors) and y (labels)
X = padded_sequences[:, :-1]
y = padded_sequences[:, -1]

# Save processed data
torch.save(X, "X.pt")
torch.save(y, "y.pt")
with open("word_to_index.pkl", "wb") as f:
    pickle.dump(word_to_index, f)
with open("index_to_word.pkl", "wb") as f:
    pickle.dump(index_to_word, f)

print("✅ Preprocessing complete! Data saved successfully.")