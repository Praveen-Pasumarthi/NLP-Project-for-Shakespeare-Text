import re
import numpy as np
import pickle

def preprocess_text(file_path, save_data=True):
    """
    Preprocess the dataset by cleaning, tokenizing, and generating input sequences.

    Args:
        file_path (str): Path to the text file containing the dataset.
        save_data (bool): Whether to save the processed data to files.

    Returns:
        tuple: X (predictors), y (labels), tokenizer (dict), max_sequence_length
    """
    # Step 1: Load the dataset
    with open(file_path, 'r') as file:
        text = file.read().lower()

    # Step 2: Clean the text (remove unwanted characters and extra spaces)
    cleaned_text = re.sub(r'[^a-z\s]', '', text)  # Keep only lowercase letters and spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()  # Replace multiple spaces with a single space

    # Step 3: Tokenize the text
    words = cleaned_text.split()
    tokenizer = {word: i + 1 for i, word in enumerate(set(words))}  # Create a word-to-index dictionary
    total_words = len(tokenizer) + 1

    # Step 4: Create input sequences
    tokenized_text = [tokenizer[word] for word in words]
    input_sequences = []
    for i in range(1, len(tokenized_text)):
        input_sequences.append(tokenized_text[:i + 1])

    # Step 5: Pad sequences
    max_sequence_length = max(len(seq) for seq in input_sequences)
    padded_sequences = np.zeros((len(input_sequences), max_sequence_length), dtype=int)
    for i, seq in enumerate(input_sequences):
        padded_sequences[i, -len(seq):] = seq

    # Split into predictors (X) and labels (y)
    X = padded_sequences[:, :-1]
    y = padded_sequences[:, -1]

    # Save data and tokenizer if needed
    if save_data:
        np.save('X.npy', X)
        np.save('y.npy', y)
        with open('tokenizer.pkl', 'wb') as tokenizer_file:
            pickle.dump(tokenizer, tokenizer_file)

    print("Data preprocessing complete!")
    print(f"Vocabulary size: {total_words}")
    print(f"Max sequence length: {max_sequence_length}")

    return X, y, tokenizer, max_sequence_length


if __name__ == "__main__":
    # Path to the provided dataset
    file_path = "shakespeare.txt"
    X, y, tokenizer, max_sequence_length = preprocess_text(file_path)

    # Verify data shapes
    print(f"Shape of X (predictors): {X.shape}")
    print(f"Shape of y (labels): {y.shape}")
