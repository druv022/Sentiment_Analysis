
import torch
import torch.nn as nn

# Bidirectional recurrent neural network (many-to-one)
class BiRNN(nn.Module):
    def __init__(self,vocab_size, embedding_dim, hidden_size, num_layers,\
                                        num_classes, dropout, device,PAD):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, \
                    batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size*2, num_classes)  # 2 for bidirection
    
    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(self.device) # 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(self.device)
        
        embedded = self.embedding(x)
        # Forward propagate LSTM
        out,(hidden, _) = self.lstm(embedded, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        
        # Decode the hidden state of the last time step
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        out = self.fc(hidden)
        return out.squeeze(-1)
