import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from collections import defaultdict
import json
import numpy as np 
import pickle
import torch.nn.functional as F
from model_HAN import HAN
from model_biRNN import BiRNN
import os
import shutil
import matplotlib.pyplot as plt
import sklearn.metrics as skl
from random import shuffle

def padSequence(batch, PAD):

    sequences, labels = [],[] 
    for sent in batch:
        sequences.append(sent[0])
        labels.append(sent[1])

    max_length = max(map(len, sequences))
    sequences = [seq + [PAD] * (max_length - len(seq)) for seq in sequences]

    return np.array(sequences), np.array(labels)

def padSeq_hierarchical(batch, PAD):

    batch_size = len(batch)

    sents, labels = np.zeros((batch_size,config.num_sentences, config.num_words),dtype=int), []
    for index_d,doc in enumerate(batch):
        for index_s,sent in enumerate(doc[0]):
            sent = np.asarray(sent,dtype=int)
            if config.num_words > len(sent):
                sent = np.hstack((sent, np.array([PAD] * (config.num_words - len(sent)),dtype=int)))

            if index_s < config.num_sentences:
                sents[index_d,index_s,:] = sent[:config.num_words]

        labels.append(doc[1])

    return np.array(sents), np.array(labels)



def batchIterator(data, batch_size, PAD):
    for i in range(0, len(data), batch_size):
        if config.sentences:
            yield padSeq_hierarchical(data[i:i+batch_size], PAD)
        else:
            yield padSequence(data[i:i+batch_size], PAD)

def binary_accuracy(predictions, target):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    predictions = torch.round(torch.sigmoid(predictions))
    correct = (predictions.float() == target.float()) #convert into float for division 
    acc = correct.sum()
    return acc.item()

def roc_metric(model_pred, labels, file_name="ROC_data.pkl"):
    pred = torch.sigmoid(model_pred)

    fpr, tpr, _ = skl.roc_curve(labels,pred,pos_label = 1)
    auc = skl.roc_auc_score(labels,pred)

    with open(file_name,'wb') as f:
        pickle.dump([fpr,tpr,auc],f,protocol=2)

    plt.plot(fpr, tpr,'r')
    plt.grid()
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC")
    plt.show()


def evaluate(model, data_test, config, PAD, device, roc=False):
    
    model.eval()
    acc=0
    pred_list = []
    label_list = []

    with torch.no_grad():

        for inputs, labels in batchIterator(data_test, config.batch_size, PAD):
            inputs, labels = torch.from_numpy(inputs).to(device),\
                             torch.from_numpy(labels).to(device)
            predictions = model(inputs)
            acc+=binary_accuracy(predictions, labels)
            
            pred_list.append(predictions)
            label_list.append(labels)

    print("Test Accuracy: {}".format(acc/len(data_test)))

    if roc:
        predictions = torch.stack(pred_list[0:-1],dim=1)
        predictions = predictions.view(predictions.numel())
        predictions = torch.cat((predictions, pred_list[-1]))
        labels = torch.stack(label_list[0:-1],dim=1)
        labels = labels.view(labels.numel())
        labels = torch.cat((labels, label_list[-1]))
        roc_metric(predictions,labels)

    return acc/len(data_test)

# save checkpoint
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
    
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

    with open(config.valDataset, "rb") as infile:
        data_val = pickle.load(infile)

    if config.pretrained:
        if os.path.exists(config.pretrained):
            with open(config.pretrained,'rb') as f:
                pretrained = pickle.load(f)

    
    #===========Device configuration=============
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    if config.model == 'HAN':
        model = HAN(len(w2i),config.embedding_dim, config.num_hidden, 
                    config.num_layers, config.output_dim, config.dropout, device, PAD, config.batch_size, config.num_sentences).to(device)
    else:
        model = BiRNN(len(w2i),config.embedding_dim, config.num_hidden, 
                    config.num_layers, config.output_dim, config.dropout, device, PAD).to(device)

    if config.pretrained:
        model.biRNN_word.embedding.weight.data = torch.Tensor(pretrained).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    criterion = nn.BCEWithLogitsLoss()
    
    # Learning rate scheduler
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.learning_rate_step, gamma=config.learning_rate_decay)

    if config.resume:
        if os.path.isfile(config.resume):
            print("Loading checkpoint '{}'".format(config.resume))
            checkpoint = torch.load(config.resume)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            print("Checkpoint loaded '{}'".format(config.resume))
    
    best_accuracy = 0.0
    loss_app = []
    acc_app = []
    
    #=================train=================== 
    for epochs in range(config.train_epochs):
        
        np.random.shuffle(data_train)
        updates=0
        epoch_loss = 0
        
        for inputs, labels in batchIterator(data_train, config.batch_size, PAD):
            
            updates += 1
            
            inputs, labels = torch.from_numpy(inputs).to(device), torch.from_numpy(labels).to(device)
            # print(inputs.size(), labels.size())
            
            model.train()
            model.zero_grad()
            predictions = model(inputs)
            loss = criterion(predictions, labels.float())
            
            loss.backward()
            optimizer.step()
            epoch_loss+=loss.item()
            
            if updates % 10 == 0:
                print ('Epoch {}, training loss: {:.4f}' 
                    .format(epochs+1,epoch_loss/updates))
            
            loss_app.append(epoch_loss/updates)

        # lr_scheduler.step()
        #evaluate on the test set remember it's a test set
        print("Training set accuracy:")
        accuracy = evaluate(model, data_train, config, PAD, device)
        print("Validation set accuracy:")
        accuracy = evaluate(model, data_val, config, PAD, device)

        acc_app.append(accuracy)

        is_best = accuracy > best_accuracy
        if is_best:
            best_accuracy = accuracy
        
            save_checkpoint({
                'epoch': epochs + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                # 'lr_scheduler':lr_scheduler.state_dict(),
                'accuracy': accuracy
            }, is_best)
            # Test accuracy evaluated on the best model:
            print("Test set accuracy:")
            accuracy = evaluate(model, data_test, config, PAD, device)


if __name__ == "__main__":
    

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--embedding_dim', type=int, default=300, help='Length of embedding dimension')
    parser.add_argument('--output_dim', type=int, default=1, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=150, help='Number of hidden units in the model')
    parser.add_argument('--num_layers', type=int, default=2, help = "Number of layers")
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')# default: 0.001
    parser.add_argument('--dropout', type=float, default=0.5, help='regularizer')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--train_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--trainDataset', type=str, default="data/train_100_s.pickle", help='pickle file for train') # Default: data/trainDataset.pickle
    parser.add_argument('--testDataset', type=str, default="data/test_100_s.pickle", help='pickle file for test') # Default: data/testDataset.pickle
    parser.add_argument('--valDataset', type=str, default="data/val_100_s.pickle", help='pickle file for validation') # Default: data/testDataset.pickle
    parser.add_argument('--w2i', type=str, default="data/w2i_100_s.json", help="word to index mapping") # Default: data/w2i.json
    parser.add_argument('--sentences', action='store_false', help='Process data hierarchically, doc=>sentences=>words') # Default: action=store_true
    parser.add_argument('--num_words', type=int, default=40, help='Maximum number of words in a sentence')
    parser.add_argument('--num_sentences', type=int, default=20, help='Maximum number of sentences in a review')
    parser.add_argument('--model', type=str, choices=['BiRNN','HAN'], default='HAN', help='Model for training, BiRNN(default) or HAN')
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction') #0.96
    parser.add_argument('--learning_rate_step', type=int, default=1, help='Learning rate decay epoch')
    parser.add_argument('--resume', type=str, default='checkpoint.pth.tar', help="Path to latest checkpoint")
    parser.add_argument('--pretrained', type=str, default='/media/druv022/Data1/NLP_data/glove.6B/pretrained_300d', help="Path to latest checkpoint") #default: None
    
    config = parser.parse_args()

    # Train the model
    train(config)
