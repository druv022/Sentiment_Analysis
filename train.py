from model import BiRNN
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from collections import defaultdict
import json
import numpy as np 
import pickle
import torch.nn.functional as F


def PadSequence(batch, PAD):

    sequences, labels = [],[] 
    for sent in batch:
        sequences.append(sent[0])
        labels.append(sent[1])

    max_length = max(map(len, sequences))
    sequences = [seq + [PAD] * (max_length - len(seq)) for seq in sequences]

    return np.array(sequences), np.array(labels)

def BatchIterator(data, batch_size, PAD):
    for i in range(0, len(data), batch_size):
        yield PadSequence(data[i:i+batch_size], PAD)

def binary_accuracy(predictions, target):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    predictions = torch.round(torch.sigmoid(predictions))
    correct = (predictions.float() == target.float()) #convert into float for division 
    acc = correct.sum()
    return acc.item()

def evaluate(model, data_test, config, PAD, device):
    
    model.eval()
    acc=0   
    with torch.no_grad():
        for inputs, labels in BatchIterator(data_test, config.batch_size, PAD):
            inputs, labels = torch.from_numpy(inputs).to(device),\
                             torch.from_numpy(labels).to(device)
            predictions = model(inputs)
            acc+=binary_accuracy(predictions, labels)
            

    print("Test Accuracy: {}".format(acc/len(data_test)))

    return
    
def train(config):  


    np.random.seed(42)
    torch.manual_seed(42)

    #====================================
    with open(config.w2i, "r") as infile:
            w2i = json.load(infile)

    #restore default property of the w2i    
    w2i = defaultdict(int, w2i)
    w2i = defaultdict(lambda: w2i["<unk>"], w2i)
    PAD = w2i["<pad>"]
    
    #====================================
    with open(config.trainDataset, "rb") as infile:
        data_train = pickle.load(infile)
    
    with open(config.testDataset, "rb") as infile:
       data_test = pickle.load(infile)
    
    #===========Device configuration=============
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model = BiRNN(len(w2i),config.embedding_dim, config.num_hidden, 
                    config.num_layers, config.output_dim, config.dropout, device, PAD).to(device)
    
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    
    #=================train=================== 
    for epochs in range(config.train_epochs):
        
        np.random.shuffle(data_train)
        updates=0
        epoch_loss = 0
        
        for inputs, labels in BatchIterator(data_train, config.batch_size, PAD):
            
            updates += 1
            
            inputs, labels = torch.from_numpy(inputs).to(device), torch.from_numpy(labels).to(device)
            # print(inputs.size(), labels.size())

            model.zero_grad()
            predictions = model(inputs)
            loss = criterion(predictions, labels.float())
            
            loss.backward()
            optimizer.step()
            epoch_loss+=loss.item()
            
           
        print ('Epoch {}, training loss: {:.4f}' 
                   .format(epochs+1,epoch_loss/updates))

        #evaluate on the test set remember it's a test set 
        evaluate(model, data_test, config, PAD, device)
        model.train()
        #TODO
        # pretrianed emmbeddings
        #add pads in embedding
        #get h from last inputs





if __name__ == "__main__":
    

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--embedding_dim', type=int, default=100, help='Length of embedding dimension')
    parser.add_argument('--output_dim', type=int, default=1, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=256, help='Number of hidden units in the model')
    parser.add_argument('--num_layers', type=int, default=2, help = "Number of layers")
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--w2i', type=str, default="data/w2i.json", help="word to index mapping")
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--train_epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--trainDataset', type=str, default="data/trainDataset.pickle", help='pickle file for train')
    parser.add_argument('--testDataset', type=str, default="data/testDataset.pickle", help='pickle file for train')
    parser.add_argument('--dropout', type=float, default=0.5, help='regularizer')
    
    config = parser.parse_args()

    # Train the model
    train(config)