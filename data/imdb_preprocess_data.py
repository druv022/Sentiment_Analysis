import os
import csv 
import numpy as np 
from collections import Counter, defaultdict
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import PorterStemmer
import argparse
import pickle
import json
from textblob import TextBlob
import re

np.random.seed(7)

train_path = ""
test_path  =  ""


parser = argparse.ArgumentParser()
parser.add_argument(
        '--check_vocab', type=str, help = 'saved vocab.txt', default=None)



args = parser.parse_args()

# save vocab file
def save_list(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


# load doc into memory
def load_doc(filename):
    
    # open the file as read only
    file = open(filename, 'r')
    
    # read all text
    text = file.read()
    
    # close the file
    file.close()
    return text


def clean_words(string,TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", "\'s", string)
    string = re.sub(r"\'ve", "\'ve", string)
    string = re.sub(r"n\'t", "n\'t", string)
    string = re.sub(r"\'re", "\'re", string)
    string = re.sub(r"\'d", "\'d", string)
    string = re.sub(r"\'ll", "\'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\<br /><br />", " ", string)
    return string.strip() if TREC else string.strip().lower()

# turn a doc into clean tokens
def clean_doc(doc):

    # stemming 
    ps = PorterStemmer()
    
    # split into tokens by white space
    tokens = doc.split()

    # remove punctuation from each token
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    
    # filter out stop words
    ##stop_words = set(stopwords.words('english'))
    ##tokens = [ps.stem(w.lower()) for w in tokens if not w in stop_words]
    
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens


class TokenizeCorpus():
    
    def __init__(self, vocab, corpus_folder=None):
        self.directory = corpus_folder
        self.vocab = vocab 
      
    # load doc and add to vocab
    def add_doc_to_vocab(self, filename):
        
        # load doc
        doc = load_doc(filename)
        """
        str_arr=doc.split(".")
        new_str=""
        for i in range(0,len(str_arr)):
            str=TextBlob(str_arr[i])
            temp=str.correct()
            temp=temp.string
            temp=clean_words(temp)
            temp=temp.replace("\\","")
            temp = temp.replace("(", "")
            temp = temp.replace(")", "")
            print(temp)
            new_str+=temp+"."
        if(new_str!=doc):
            ##print(doc)
            print(new_str)
            doc=new_str
        """
        
        # clean doc
        tokens = clean_doc(doc)
        
        # update counts
        self.vocab.update(tokens)
        
    # load all docs in a directory
    def process_docs(self):

        # walk through all files in the folder
        for filename in os.listdir(self.directory):
            
            # skip files that do not have the right extension
            if not filename.endswith(".txt"):
                continue
            
            # create the full path of the file to open
            path = self.directory + '/' + filename
            
            # add doc to vocab
            self.add_doc_to_vocab(path)
        




class ProcessTrainData:

    def __init__(self, vocabulary, data_folder=""):

        self.vocab = vocabulary
        self.data_type = ""
        self.save_name = "trainDataset"
        self.directory = [os.path.join(data_folder, self.data_type,"pos"),os.path.join(data_folder, self.data_type,"neg")]
        self.w2i  = defaultdict(lambda: len(self.w2i))
        self.i2w = defaultdict(lambda: len(self.i2w))
        self.PAD = self.w2i["<pad>"]
        self.lines, self.ratings = [],[] 
        self.process_docs()
        self.UNK = self.w2i["<unk>"]
        self.w2i = defaultdict(lambda: self.UNK, self.w2i)
        self.save_data() 

    # load doc, clean and return line of tokens
    def doc_to_line(self,filename):
        # load the doc
        doc = load_doc(filename)
        # clean doc
        tokens = clean_doc(doc)
        # filter by vocab
        clean_token =[]
        for w in tokens:
            if w in self.vocab:
                clean_token.append(self.w2i[w])
                self.i2w[self.w2i[w]] = w
       
        return clean_token
    
    # load all docs in a directory
    def process_docs(self):
       
        # walk through all files in the folder(pos, neg)
        for dataset in self.directory:
            for filename in os.listdir(dataset):
                # skip files that do not have the right extension
                if not filename.endswith(".txt"):
                    continue
                # create the full path of the file to open
                path = dataset + '/' + filename
                # load and clean the doc
                line = self.doc_to_line(path)
                # add to list
                self.lines.append(line)

                # check for the ratings. 1 +ve, 0 -ve
                if dataset.split("/")[-1]=="pos":
                    self.ratings.append(1)
                else:
                    self.ratings.append(0)
        
    # save train data
    def save_data(self):
            Dataset = list(zip(self.lines, self.ratings))
            np.random.shuffle(Dataset)

            with open(self.save_name+".pickle", "wb") as outfile:
                pickle.dump(Dataset, outfile) 
            
            with open("w2i.josn","w") as outfile:
                json.dump(self.w2i, outfile, indent=4)

            with open("i2w.json", "w") as outfile:
                json.dump(self.i2w, outfile, indent=4)

# class ProcessTestData:
#     def __init__():
class ProcessTestData(ProcessTrainData):

    def __init__(self, w2i_json, data_folder=""):
        with open(w2i_json, "r") as infile:
            self.w2i = json.load(infile)

        # restore default property of the w2i
        self.w2i = defaultdict(int, self.w2i)
        self.w2i = defaultdict(lambda: self.w2i["<unk>"], self.w2i)
        self.data_type = ""
        self.save_name = "testDataset"
        self.directory = [os.path.join(data_folder, self.data_type, "pos"), \
                          os.path.join(data_folder, self.data_type, "neg")]

        self.lines, self.ratings = [], []
        self.process_docs()
        self.save_data()

    # load doc, clean and return line of tokens
    def doc_to_line(self, filename):
        # load the doc
        doc = load_doc(filename)
        # clean doc
        tokens = clean_doc(doc)
        # filter by vocab
        clean_token = [self.w2i[w] for w in tokens]

        return clean_token

    # save train data
    def save_data(self):
        Dataset = list(zip(self.lines, self.ratings))
        np.random.shuffle(Dataset)

        with open(self.save_name + ".pickle", "wb") as outfile:
            pickle.dump(Dataset, outfile)


        




        # add all docs to vocab
"""
vocab = Counter()
if not args.check_vocab:
    print("---------No vocab previously saved creating new one---------")
    print()
    vocab = TokenizeCorpus(vocab, "pos")
    vocab.process_docs()
    vocab = TokenizeCorpus(vocab.vocab, "neg")
    vocab.process_docs()
    
    
    # keep tokens with > 5 occurrence
    min_occurane = 5
    tokens = [k for k,c in vocab.vocab.items() if c >= min_occurane]
    print(len(tokens))

    # save tokens to a vocabulary file
    save_list(tokens, 'vocab.txt')

# load vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)

Data = ProcessTrainData(vocab)

"""
json_file  = 'w2i.josn'
testData = ProcessTestData(json_file)