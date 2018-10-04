import torch.optim as optim
import torch.nn as nn
from CNN_Classifier import CNN
import torch
import pickle
import json
import numpy as np
from random import shuffle



current_index=0
le=0

def load_glove_vector():
    glove_word_vector = pickle.load(
        open("pretrained_100d", "rb"))
    return glove_word_vector

def save_model(model):
    torch.save(model,"model_100_g.pkl")
def load_model():
    model=torch.load("model_100_g.pkl")
    return model
def accuracy(prediction,targets):
    accur = 0.0
    ##print(targets.shape)
    targets=targets.numpy()
    predictions_index = np.argmax(prediction, axis=1)
    ##print(predictions_index,targets)
    value_arr = predictions_index - targets
    matches = (value_arr == 0).sum()
    accur = matches / len(targets)

    return accur

def read_word_vector():
    word_vector_dict = pickle.load(open("word_vectors.pkl", "rb"))
    return word_vector_dict

def create_word_matrix(word_vector_dict,i2w_dict):
    word_matrix=torch.zeros((35759, 300))
    for key in i2w_dict:
        word_matrix[int(key),:]=torch.from_numpy(word_vector_dict[i2w_dict[key]])
    return word_matrix

def read_comments():
    train_data_set_list = pickle.load(open("/home/vik1/Downloads/subj/dl_nlp/Sentiment_Analysis/data/trainDataset.pickle", "rb"))
    return train_data_set_list

def i2w():
    json_data=open("data/i2w.json")
    json1_str = json_data.read()
    i2w_dict=json.loads(json1_str)
    return i2w_dict

def create_train_and_validation(lis):
    le=len(lis)
    train_le=int(0.8*le)
    training_list=lis[0:train_le]
    validation_list=lis[train_le:le]
    return training_list,validation_list
def train():
    word_vector_dict=read_word_vector()
    train_data_set_list=read_comments()
    shuffle(train_data_set_list)
    train_data_set_list,validation_list=create_train_and_validation(train_data_set_list)
    i2w_dict=i2w()
    word_matrix=load_glove_vector()
    ##word_matrix=create_word_matrix(word_vector_dict,i2w_dict)
    model=CNN(word_matrix)
    ##model=load_model()
    model=model.cuda()
    softmax=nn.Softmax(dim=1)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adagrad(model.parameters())
    batch_size=200
    acc=0
    index=0
    tot_loss=0
    for epoch in range(0,18):
        index=0
        shuffle(train_data_set_list)
        for data in train_data_set_list:
            index += 1
            optimizer.zero_grad()
            input_data = torch.zeros(1, 1,len(data[0]), 1,dtype=torch.long)
            input_data = input_data.cuda()
            count=0
            for i in range(0, len(data[0])):
                if(data[0][i]=="."):
                    count+=1
                if(count<=2):
                  input_data[0,0,i,0]=data[0][i]
            target = torch.zeros(1, dtype=torch.long)
            target[0] = data[1]
            target = target.cuda()
            ##input_data = torch.autograd.Variable(input_data)
            ##print(input_data)
            model_output = model.forward(input_data)
            loss = criterion(model_output, target)
            loss.backward()
            optimizer.step()
            ##print(input_data)
            softmax_output = softmax(model_output)
            softmax_output = softmax_output.cpu()
            target = target.cpu()
            softmax_output = softmax_output.detach().numpy()
            acc += accuracy(softmax_output, target)
            tot_loss += loss.item()
            if (index % 200 == 0):
                print(epoch,index,tot_loss, acc / 200)
                acc = 0
                tot_loss = 0
        validate(validation_list,model,softmax)
        save_model(model)


def test():
    softmax=nn.Softmax(dim=1)
    word_vector_dict = read_word_vector()
    train_data_set_list = read_comments()
    ##shuffle(train_data_set_list)
    ##train_data_set_list, validation_list = create_train_and_validation(train_data_set_list)
    ##i2w_dict = i2w()
    ##word_matrix = create_word_matrix(word_vector_dict, i2w_dict)
    ##model=CNN(word_matrix)
    model = load_model()
    index = 0
    acc = 0
    y_true=[]
    y_prob=[]
    for data in train_data_set_list:
        lis=[x for x in data[0] if x != 35759]
        print(index)
        index += 1
        input_data = torch.zeros(1, 1, len(data[0]), 1, dtype=torch.long)
        input_data = input_data.cuda()
        count = 0
        for i in range(0, len(lis)):
            if (data[0][i] == "."):
                count += 1
            if (count <= 2):
                input_data[0, 0, i, 0] = lis[i]
        if (len(lis) > 1):
            target = torch.zeros(1, dtype=torch.long)
            target[0] = data[1]
            target = target.cuda()
            ##input_data = torch.autograd.Variable(input_data)
            model_output = model.forward(input_data)
            ##print(input_data)
            softmax_output = softmax(model_output)
            softmax_output = softmax_output.cpu()
            target = target.cpu()
            softmax_output = softmax_output.detach().numpy()
            y_true.append(target.item())
            y_prob.append(softmax_output[0][1])
            acc += accuracy(softmax_output, target)

    acc = acc / len(train_data_set_list)
    print("Test Accuracy " + str(acc))
    print(y_true)
    print(y_prob)



def validate(validation_list,model,softmax):
    index=0
    acc=0
    for data in validation_list:
        index += 1
        input_data = torch.zeros(1, 1, len(data[0]), 1, dtype=torch.long)
        input_data = input_data.cuda()
        count = 0
        for i in range(0, len(data[0])):
            if (data[0][i] == "."):
                count += 1
            if (count <= 2):
                input_data[0, 0, i, 0] = data[0][i]
        target = torch.zeros(1, dtype=torch.long)
        target[0] = data[1]
        target = target.cuda()
        ##input_data = torch.autograd.Variable(input_data)
        ##print(input_data)
        model_output = model.forward(input_data)
        ##print(input_data)
        softmax_output = softmax(model_output)
        softmax_output = softmax_output.cpu()
        target = target.cpu()
        softmax_output = softmax_output.detach().numpy()
        acc += accuracy(softmax_output, target)
    acc=acc/len(validation_list)
    print("Validation Accuracy "+str(acc))

##test()
train()
##load_glove_vector()


