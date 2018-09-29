import torch
import torch.nn as nn

# Bidirectional RNN
class BiRNN(nn.Module):
    def __init__(self,vocab_size, embedding_dim, hidden_size, num_layers, dropout, device, PAD, embed_required = True):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.dropout = dropout
        self.embed_required = embed_required
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = PAD)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, \
                batch_first=True, bidirectional=True, dropout=dropout)
        self.tanh = nn.Tanh()
    
    def forward(self, x):

        # Set initial states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(self.device) # 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(self.device)

        if self.embed_required:
            embedded = self.embedding(x.long())
        else:
            embedded = x

        # Forward propagate LSTM
        out, _ = self.lstm(embedded, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)

        return out

# Attention Block
class Attention(nn.Module):
    
    def __init__(self,embedding_dim, hidden_size):
        super(Attention, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        self.linear1 = nn.Linear(embedding_dim,hidden_size)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

        self.linear2 = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):

        x = self.linear1(x)
        x = self.tanh(x)
        x = self.linear2(x)
        x = self.softmax(x)

        return x

# Heirarchical Attention Network
class HAN(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, num_classes, dropout, device, PAD):
        super(HAN, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.dropout = dropout

        self.biRNN_word = BiRNN(vocab_size, embedding_dim, hidden_size, num_layers, dropout, device, PAD)
        self.attention_word = Attention(hidden_size*2, hidden_size*2)
        
        self.biRNN_sentence = BiRNN(hidden_size, hidden_size*2, hidden_size, num_layers, dropout, device, PAD, embed_required=False)
        self.attention_sentence = Attention(hidden_size*2, hidden_size*2)

        self.linear = nn.Linear(hidden_size*2, num_classes)

    def forward(self, x):

        sentences = []
        # for each document get the sentence embeddings
        # TODO: Remove this for loop
        for index_b, batch in enumerate(x):

            sentence_b = self.biRNN_word(batch)
            sentence = self.attention_word(sentence_b)
            sentence = sentence * sentence_b
            sentence = torch.sum(sentence,dim=1)
            sentences.append(sentence)

        doc_b = self.biRNN_sentence(torch.stack(sentences))
        doc = self.attention_sentence(doc_b)
        doc = doc * doc_b
        doc = torch.sum(doc,dim=1)

        out = self.linear(doc)

        return out