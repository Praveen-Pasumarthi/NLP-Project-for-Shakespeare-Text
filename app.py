import streamlit as st

# Dictionary for word replacement (Modern → Shakespearean)
word_map = {
    "you": "thou", "your": "thy", "yours": "thine", "are": "art", "have": "hath",
    "do": "doth", "does": "dost", "my": "mine", "is": "be", "was": "wast",
    "were": "wert", "yes": "aye", "no": "nay", "very": "verily", "good": "fair",
    "hello": "good morrow", "friend": "companion", "love": "affection",
    "man": "gentleman", "woman": "lady", "king": "sovereign", "queen": "majesty",
    "strong": "stout", "beautiful": "comely", "happy": "merry", "angry": "vexed",
    "sad": "forlorn", "rich": "opulent", "poor": "destitute", "smart": "learned",
    "stupid": "foolish", "food": "victuals", "drink": "ale", "house": "abode",
    "dangerous": "perilous", "truth": "verity", "lie": "falsehood",
    "quick": "swift", "slow": "sluggish", "now": "anon", "soon": "presently",
    "always": "evermore", "never": "ne'er", "before": "ere", "after": "henceforth",
    "why": "wherefore", "because": "for", "how": "how dost", "think": "reckon",
    "speak": "utter", "listen": "hark", "see": "behold", "give": "bestow",
    "ask": "beseech", "curse": "besmirch", "promise": "vow", "goodbye": "fare thee well",
    "come": "approach", "go": "depart", "wait": "tarry", "stop": "halt",
    "sleep": "slumber", "awake": "rouse", "fight": "duel", "win": "prevail",
    "lose": "succumb", "understand": "comprehend", "wish": "desire", "pray": "entreat",
    "hate": "abhor", "fear": "dread", "enemy": "adversary", "fool": "knave",
    "young": "youthful", "old": "ancient", "small": "petite", "big": "grand",
    "father": "sire", "mother": "madam", "child": "offspring", "boy": "lad",
    "girl": "lass", "son": "heir", "daughter": "maid", "brother": "kinsman",
    "sister": "kinswoman", "husband": "consort", "wife": "spouse"
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
