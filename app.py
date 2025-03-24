import streamlit as st

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
st.title("Modern English To Shakespearean Text Converter")
st.write("Enter modern English text and convert it into Shakespearean-style language!")

# Input box for modern text
user_text = st.text_area("Enter your text:", "Hello, how are you?")

# Convert button
if st.button("Convert"):
    shakespeare_text = convert_to_shakespeare(user_text)
    st.subheader("📜 Shakespearean Translation:")
    st.write(shakespeare_text)
