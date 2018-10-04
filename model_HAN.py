import torch
import torch.nn as nn

# Bidirectional RNN
class BiRNN(nn.Module):
    def __init__(self,vocab_size, embedding_dim, hidden_size, num_layers, dropout, device, PAD, embed_required = True):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        if num_layers > 1:
            self.dropout = dropout
        else:
            self.dropout = 0.0

        self.embed_required = embed_required
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = PAD)
        # self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, \
        #         batch_first=True, bidirectional=True, dropout=self.dropout)
        # self.init_weight(self.lstm.all_weights)
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers, \
                batch_first=True, bidirectional=True, dropout=self.dropout)
        self.init_weight(self.gru.all_weights)
        self.tanh = nn.Tanh()
    
    def init_weight(self, w_list):
        for w in w_list[0]:
            if len(w.shape) > 1:
                nn.init.xavier_normal_(w)
            else:
                nn.init.ones_(w)

    def forward(self, x):

        reset_dim = False
        if self.embed_required:
            embedded = self.embedding(x.long())
        else:
            embedded = x
        
        if len(embedded.shape) > 3:
            b_size, s_size, w_size, e_size = embedded.shape
            embedded = embedded.view(-1, w_size, e_size)
            reset_dim = True
        # Set initial states
        h0 = torch.zeros(self.num_layers*2, embedded.size(0), self.hidden_size).to(self.device) # 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, embedded.size(0), self.hidden_size).to(self.device)

        # Forward propagate LSTM
        # out, _ = self.lstm(embedded, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        # Forward propagate GRU
        out,_ = self.gru(embedded, h0)

        
        if reset_dim:
            out = out.view(b_size, s_size, w_size, -1)
        return out

# Attention Block
class Attention(nn.Module):
    
    def __init__(self,embedding_dim, hidden_size, batch_size, num_sentence, softmax_dim=-1, device='cuda:0'):
        super(Attention, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_sentence = num_sentence

        self.linear1 = nn.Linear(embedding_dim,hidden_size)
        nn.init.xavier_normal_(self.linear1.weight.data)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=softmax_dim)

        self.u_w = nn.Parameter(torch.Tensor(hidden_size,1)).to(device)
        nn.init.normal_(self.u_w)


    def forward(self, x):
        if len(x.shape) > 3:
            b_size, s_size, w_size, e_size = x.shape
            batch_u_w = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(self.u_w,0),0),0)
            batch_u_w = batch_u_w.repeat(b_size,s_size,w_size,1,1)
        elif len(x.shape) > 2:
            b_size, s_size, e_size = x.shape
            batch_u_w = torch.unsqueeze(torch.unsqueeze(self.u_w,0),0)
            batch_u_w = batch_u_w.repeat(b_size,s_size,1,1)
        else:
            print("ERROR: input size mismatch")

        x = self.linear1(x)
        x = self.tanh(x)
        x = x.unsqueeze(-1)
        x = torch.matmul(x.transpose(-2,-1),batch_u_w)
        x = x.squeeze(-1)
        x = self.softmax(x)

        return x

# Heirarchical Attention Network
class HAN(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, num_classes, dropout, device, PAD,  batch_size, num_sentence):
        super(HAN, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.dropout = dropout
        self.batch_size = batch_size
        self.num_sentence = num_sentence

        # Attention for word to sentence
        self.biRNN_word = BiRNN(vocab_size, embedding_dim, hidden_size, num_layers, dropout, device, PAD)
        self.attention_word = Attention(hidden_size*2, hidden_size*2, batch_size, num_sentence, softmax_dim=2, device=device)
        
        # Attention for sentence to Doc
        self.biRNN_sentence = BiRNN(hidden_size, hidden_size*2, hidden_size, num_layers, dropout, device, PAD, embed_required=False)
        self.attention_sentence = Attention(hidden_size*2, hidden_size*2, batch_size, num_sentence, softmax_dim=1, device=device)

        self.linear = nn.Linear(hidden_size*2, num_classes)
        nn.init.xavier_normal_(self.linear.weight.data)

    def forward(self, x):

        sentence_b = self.biRNN_word(x)
        sentence = self.attention_word(sentence_b)
        sentence = torch.mul(sentence,sentence_b)
        sentence = torch.sum(sentence,dim=2)

        doc_b = self.biRNN_sentence(sentence)
        doc = self.attention_sentence(doc_b)
        doc = doc * doc_b
        doc = torch.sum(doc,dim=1)

        out = self.linear(doc)

        return out.squeeze(-1)