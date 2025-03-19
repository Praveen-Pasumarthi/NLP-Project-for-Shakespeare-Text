import streamlit as st
import torch
import pickle
from train_lstm import LSTMModel

# Load tokenizer mappings
with open("word_to_index.pkl", "rb") as f:
    word_to_index = pickle.load(f)
with open("index_to_word.pkl", "rb") as f:
    index_to_word = pickle.load(f)

# Load trained model
vocab_size = len(word_to_index) + 1
model = LSTMModel(vocab_size, 100, 150, vocab_size, 2)
model.load_state_dict(torch.load("shakespeare_lstm.pth"))
model.eval()

# Function to generate text
def generate_text(seed_text, next_words=20):
    words = seed_text.lower().split()
    for _ in range(next_words):
        token_list = [word_to_index.get(word, 0) for word in words]
        token_list = torch.tensor(token_list).unsqueeze(0)

        with torch.no_grad():
            predicted = model(token_list)
            predicted_index = torch.argmax(predicted, dim=1).item()

        new_word = index_to_word.get(predicted_index, "")
        words.append(new_word)

    return " ".join(words)

# Streamlit UI
st.title("🎭 Shakespearean Text Generator")
st.write("Enter a seed phrase and generate text in Shakespeare's style!")

# Input box for seed text
seed_text = st.text_input("Enter a seed phrase:", "Shall I compare thee")

# Slider to select number of words to generate
num_words = st.slider("Number of words to generate:", min_value=5, max_value=50, value=20)

# Generate button
if st.button("Generate"):
    generated_text = generate_text(seed_text, num_words)
    st.subheader("📜 Generated Shakespearean Text:")
    st.write(generated_text)