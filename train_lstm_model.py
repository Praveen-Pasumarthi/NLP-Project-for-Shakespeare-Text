# Import necessary libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Load preprocessed data
X = np.load('X.npy')
y = np.load('y.npy')

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.long) 
y = torch.tensor(np.argmax(y, axis=1), dtype=torch.long)  

# Define a custom dataset
class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = TextDataset(X, y)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size, num_layers=2):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        output = self.fc(output[:, -1, :]) 
        return output

# Model parameters
vocab_size = X.max().item() + 1  
embed_size = 100
hidden_size = 150
output_size = vocab_size
num_layers = 2

model = LSTMModel(vocab_size, embed_size, hidden_size, output_size, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 30
model.train()

for epoch in range(num_epochs):
    epoch_loss = 0
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader):.4f}')

# Save the model
torch.save(model.state_dict(), 'shakespeare_lstm_model.pth')
print("Model training complete and saved!")
