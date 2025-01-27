import re
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def preprocess_text(file_path, save_tokenizer=True):
   
    with open(file_path, 'r') as file:
        text = file.read().lower()
    
    cleaned_text = re.sub(r'[^a-z\s]', '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([cleaned_text])

    input_sequences = []
    for line in cleaned_text.split('.'):  
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i + 1]
            input_sequences.append(n_gram_sequence)

    max_sequence_length = max([len(seq) for seq in input_sequences])
    input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')

    X = input_sequences[:, :-1]
    y = input_sequences[:, -1]
    y = np.array(tf.keras.utils.to_categorical(y, num_classes=len(tokenizer.word_index) + 1))

    if save_tokenizer:
        import pickle
        with open('tokenizer.pkl', 'wb') as tokenizer_file:
            pickle.dump(tokenizer, tokenizer_file)

    return X, y, tokenizer.word_index, max_sequence_length

if __name__ == "__main__":
    file_path = "path_to_your_shakespeare.txt"  
    X, y, word_index, max_sequence_length = preprocess_text(file_path)

    np.save('X.npy', X)
    np.save('y.npy', y)
    print("Data preprocessing complete!")
    print(f"Vocabulary size: {len(word_index) + 1}")
    print(f"Max sequence length: {max_sequence_length}")
