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

try:
    model.load_state_dict(torch.load("shakespeare_lstm.pth", map_location=torch.device("cpu")))
    model.eval()
except Exception as e:
    st.error(f"Error loading model: {e}")

# Function to generate Shakespearean-style text
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

# Dictionary for word replacement (Modern → Shakespearean)
word_map = {
    "you": "thou", "your": "thy", "are": "art", "have": "hath",
    "do": "doth", "does": "dost", "my": "mine", "is": "be",
    "yes": "aye", "no": "nay", "very": "verily", "good": "fair",
    "hello": "good morrow", "friend": "companion", "love": "affection",
    "man": "gentleman", "woman": "lady", "king": "sovereign",
}

# Function to convert modern English to Shakespearean-style
def convert_to_shakespeare(text):
    words = text.lower().split()
    converted = [word_map.get(word, word) for word in words]
    return " ".join(converted)

# Streamlit UI
st.title("🎭 Shakespearean Text Converter & Generator")

# User selects the mode
option = st.radio("Choose an option:", ["Generate Shakespearean Text", "Convert Modern to Shakespearean"])

if option == "Generate Shakespearean Text":
    st.write("Enter a seed phrase and generate Shakespearean-style text!")
    seed_text = st.text_input("Enter a seed phrase:", "Shall I compare thee")
    num_words = st.slider("Number of words to generate:", min_value=5, max_value=50, value=20)

    if st.button("Generate"):
        generated_text = generate_text(seed_text, num_words)
        st.subheader("📜 Generated Shakespearean Text:")
        st.write(generated_text)

elif option == "Convert Modern to Shakespearean":
    st.write("Enter modern English text and convert it into Shakespearean-style language!")
    user_text = st.text_area("Enter your text:", "Hello, how are you?")

    if st.button("Convert"):
        shakespeare_text = convert_to_shakespeare(user_text)
        st.subheader("📜 Shakespearean Translation:")
        st.write(shakespeare_text)