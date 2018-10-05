import torch.nn as nn
import torch.nn.functional as F
import torch

class CNN(nn.Module):
    def __init__(self,weights_matrix):
        super(CNN, self).__init__()
        ##self.conv_2 = nn.Conv2d(1, 100, 2)
        self.embeddings=nn.Embedding(35759,100)
        ##self.embeddings.load_state_dict({'weight': weights_matrix})
        self.conv_3= nn.Conv2d(1, 100, 3)
        self.conv_4 = nn.Conv2d(1, 100, 4)
        self.conv_5 = nn.Conv2d(1, 100, 5)
        self.linear=nn.Linear(300,2)
        self.dropout = nn.Dropout(0.5)



    def forward(self, x):
        ##y_2 = (torch.tanh(self.conv_2(x)))
        ##y_2_max=self.find_max(y_2)
        embed=self.embeddings(x)
        input=embed.view(1,1,embed.shape[2],100)
        y_3 = (F.relu(self.conv_3(input)))
        y_3_max = self.find_max(y_3)
        y_4 = (F.relu(self.conv_4(input)))
        y_4_max = self.find_max(y_4)
        y_5 = (F.relu(self.conv_5(input)))
        y_5_max = self.find_max(y_5)
        hidden_tensor= torch.cat(( y_3_max,y_4_max,y_5_max), 1)
        hidden_tensor=hidden_tensor.cuda()
        output=self.linear(hidden_tensor)

        return output
    def find_max(self,y_2):
        max_pool=torch.nn.MaxPool2d((y_2.shape[2],y_2.shape[3]))
        y_2_max=max_pool(y_2)
        y_2_max=y_2_max.reshape(1,100)
        ##for i in range(0, 100):
        ##    y_2_max[0, i] = torch.max(y_2[0, i, :, :])
        return y_2_max


