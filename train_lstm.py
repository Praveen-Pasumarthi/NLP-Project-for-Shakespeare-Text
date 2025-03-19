import torch
import torch.nn as nn
import torch.optim as optim
import pickle

# Load preprocessed data
X = torch.load("X.pt")
y = torch.load("y.pt")

# Load tokenizer mappings
with open("word_to_index.pkl", "rb") as f:
    word_to_index = pickle.load(f)
with open("index_to_word.pkl", "rb") as f:
    index_to_word = pickle.load(f)

vocab_size = len(word_to_index) + 1  # Total vocabulary size
embed_size = 100  # Word embedding size
hidden_size = 150  # LSTM hidden units
num_layers = 2  # Number of LSTM layers

# Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size, num_layers=2):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :])

# Initialize model, loss function, and optimizer
model = LSTMModel(vocab_size, embed_size, hidden_size, vocab_size, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train only if this script is run directly
if __name__ == "__main__":
    num_epochs = 30
    batch_size = 64

    for epoch in range(num_epochs):
        for i in range(0, len(X), batch_size):
            batch_X = X[i : i + batch_size]
            batch_y = y[i : i + batch_size]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

    # Save trained model
    torch.save(model.state_dict(), "shakespeare_lstm.pth")
    print("✅ Model training complete! Saved as `shakespeare_lstm.pth`")