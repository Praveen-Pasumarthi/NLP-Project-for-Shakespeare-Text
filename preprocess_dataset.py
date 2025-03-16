import re
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

def preprocess_text(file_path, save_data=True):
    """
    Preprocess the dataset by cleaning, tokenizing, and generating input sequences.

    Args:
        file_path (str): Path to the text file containing the dataset.
        save_data (bool): Whether to save the processed data to files.

    Returns:
        tuple: X (predictors), y (labels), tokenizer, max_sequence_length
    """
    # Step 1: Load the dataset
    with open(file_path, 'r') as file:
        text = file.read().lower()

    # Step 2: Clean the text (remove unwanted characters and extra spaces)
    cleaned_text = re.sub(r'[^a-z\s]', '', text)  # Keep only lowercase letters and spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()  # Replace multiple spaces with a single space

    # Step 3: Tokenize the text
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([cleaned_text])

    # Total number of unique words
    total_words = len(tokenizer.word_index) + 1

    # Step 4: Create input sequences
    input_sequences = []
    for line in cleaned_text.split('.'):  # Split by sentences for meaningful sequences
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i + 1]
            input_sequences.append(n_gram_sequence)

    # Step 5: Pad sequences
    max_sequence_length = max([len(seq) for seq in input_sequences])
    input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')

    # Split into predictors (X) and labels (y)
    X = input_sequences[:, :-1]
    y = input_sequences[:, -1]

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
    file_path = "path_to_your_shakespeare.txt"  # Replace with the actual path to your dataset file
    X, y, tokenizer, max_sequence_length = preprocess_text(file_path)

    # Verify data shapes
    print(f"Shape of X (predictors): {X.shape}")
    print(f"Shape of y (labels): {y.shape}")
