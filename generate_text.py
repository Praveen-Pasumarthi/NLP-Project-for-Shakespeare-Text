import torch
import pickle
from train_lstm import LSTMModel

# Load tokenizer mappings
with open("word_to_index.pkl", "rb") as f:
    word_to_index = pickle.load(f)
with open("index_to_word.pkl", "rb") as f:
    index_to_word = pickle.load(f)

# Load model
vocab_size = len(word_to_index) + 1
model = LSTMModel(vocab_size, 100, 150, vocab_size, 2)
model.load_state_dict(torch.load("shakespeare_lstm.pth"))
model.eval()

# Function to generate text
def generate_text(seed_text, next_words=20):
    words = seed_text.lower().split()
    for _ in range(next_words):
        token_list = [word_to_index.get(word, 0) for word in words]  # Convert words to indices
        token_list = torch.tensor(token_list).unsqueeze(0)

        with torch.no_grad():
            predicted = model(token_list)
            predicted_index = torch.argmax(predicted, dim=1).item()

        new_word = index_to_word.get(predicted_index, "")
        words.append(new_word)

    return " ".join(words)

# Example Usage
seed_text = "shall i compare thee"
generated_text = generate_text(seed_text, 20)
print(f"✨ Shakespearean Text:\n{generated_text}")