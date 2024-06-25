import torch
import torch.nn as nn

# using RNN LSTM model to fit the audio data
class LSTMAudioModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=256, num_layers=2, dropout=0.2, num_classes=10):
        super(LSTMAudioModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # # reshape the input in the form of (batch_size, seq_length, feature)
        x = x.reshape(x.size(0), x.size(2), -1)    

        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # get the output and hidden state
        # x shape (batch_size, seq_length, feature)
        out, _ = self.lstm(x, (h0, c0))

        # only take the output from the final time step
        out = self.fc(out[:, -1, :])
        return out