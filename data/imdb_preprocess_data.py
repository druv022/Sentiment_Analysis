import os
import csv 
import numpy as np 
from collections import Counter, defaultdict
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import argparse
import pickle
import json
import shutil


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

# turn a doc into clean tokens
def clean_doc(doc, sent_flag = False):

    # stemming 
    ps = PorterStemmer()
    
    if sent_flag:
        # split into tokens by sentence
        tokens = sent_tokenize(doc)
    else:
        # split into tokens by white space
        tokens = word_tokenize(doc)

    # remove punctuation from each token
    if not args.keep_punc:
        table = str.maketrans('', '', punctuation)
        tokens = [w.translate(table) for w in tokens]
    
    if sent_flag:
        tokens_sent = []
        for sentc in tokens:
            sentc = word_tokenize(sentc)
            tokens_s = [word for word in sentc if word.isalpha()]
            
            # filter out stop words
            if not args.keep_stopw:
                stop_words = set(stopwords.words('english'))
                tokens_s = [ps.stem(w.lower()) for w in tokens_s if not w in stop_words]
            
            # filter out short tokens
            tokens_s = [word for word in tokens_s if len(word) > 1]
            tokens_sent.append(tokens_s)

        tokens = tokens_sent
    else:
        # remove remaining tokens that are not alphabetic
        tokens = [word for word in tokens if word.isalpha()]
        
        # filter out stop words
        if args.keep_stopw:
            stop_words = set(stopwords.words('english'))
            tokens = [ps.stem(w.lower()) for w in tokens if not w in stop_words]
        
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

    def __init__(self, vocabulary, data_folder, train_size, d_type="train"):

        self.vocab = vocabulary
        self.data_folder = data_folder
        self.save_name = "train_"+str(train_size)
        self.train_size = train_size
        self.directory = [os.path.join(data_folder, "pos"),os.path.join(data_folder, "neg")]
        self.w2i  = defaultdict(lambda: len(self.w2i))
        self.i2w = defaultdict(lambda: len(self.i2w))
        self.PAD = self.w2i["<pad>"]
        self.lines, self.ratings = [],[]
        self.process_docs()
        self.UNK = self.w2i["<unk>"]
        self.i2w[self.UNK] = "<unk>"
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

    def doc_to_sentence(self, filename):
        # load the doc
        doc = load_doc(filename)
        # clean doc and 
        tokens = clean_doc(doc, sent_flag=True)
        # filter by vocal
        clean_token = []
        for sentc in tokens:
            tokens_sent = []
            for w in sentc:
                if w in self.vocab:
                    tokens_sent.append(self.w2i[w])
                    self.i2w[self.w2i[w]] = w

            clean_token.append(tokens_sent)

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
                # load , clean the doc and add to list
                if args.sentences:
                    sentences = self.doc_to_sentence(path)
                    self.lines.append(sentences)
                else:
                    line = self.doc_to_line(path)
                    self.lines.append(line)

                # check for the ratings. 1 +ve, 0 -ve
                if dataset.split("/")[-1]=="pos":
                    self.ratings.append(1)
                else:
                    self.ratings.append(0)

    # save train data
    def save_data(self):
        Dataset = list(zip(self.lines ,self.ratings))
        np.random.shuffle(Dataset)

        if args.sentences:
            add_name = '_s'
        else:
            add_name = ''

        with open(self.save_name+add_name+".pickle", "wb") as outfile:
            pickle.dump(Dataset, outfile) 
        
        with open("w2i_"+str(self.train_size)+add_name+".json","w") as outfile:
            json.dump(self.w2i, outfile, indent=4)

        with open("i2w_"+str(self.train_size)+add_name+".json", "w") as outfile:
            json.dump(self.i2w, outfile, indent=4)

# -------------------------------Test Data----------------------------------------------
class ProcessTestData(ProcessTrainData):

    def __init__(self,w2i_json, data_folder, train_size, d_type="testDataset"):
        
        with open(w2i_json, "r") as infile:
            self.w2i = json.load(infile)

        #restore default property of the w2i    
        self.w2i = defaultdict(int, self.w2i)
        self.w2i = defaultdict(lambda: self.w2i["<unk>"], self.w2i)
        self.data_folder = data_folder
        self.save_name = d_type+"_"+str(train_size)
        self.train_size = train_size

        self.directory = [os.path.join(data_folder, "pos"),\
                          os.path.join(data_folder, "neg")]

        self.lines, self.ratings = [],[]
        self.process_docs()
        self.save_data()
        
     # load doc, clean and return line of tokens
    def doc_to_line(self,filename):
        # load the doc
        doc = load_doc(filename)
        # clean doc
        tokens = clean_doc(doc)
        # filter by vocab
        clean_token =[self.w2i[w] for w in tokens]
               
        return clean_token

    def doc_to_sentence(self, filename):
        # load the doc
        doc = load_doc(filename)
        # clean doc and 
        tokens = clean_doc(doc, sent_flag=True)
        # filter by vocal
        clean_token = []
        for sentc in tokens:
            tokens_sent = []
            for w in sentc:
                tokens_sent.append(self.w2i[w])

            clean_token.append(tokens_sent)
        
        return clean_token

    # save train data
    def save_data(self):
            Dataset = list(zip(self.lines, self.ratings))
            np.random.shuffle(Dataset)

            if args.sentences:
                self.save_name = self.save_name + '_s'

            with open(self.save_name+".pickle", "wb") as outfile:
                pickle.dump(Dataset, outfile)    

        


if __name__ == "__main__":

    np.random.seed(7)

    data_folder = "/media/druv022/Data1/git/DL_NLP_Project/IMDB_data/aclImdb"
    train_path = "/media/druv022/Data1/git/DL_NLP_Project/IMDB_data/aclImdb/train"
    dev_path =   "/media/druv022/Data1/git/DL_NLP_Project/IMDB_data/aclImdb/dev"
    test_path = "/media/druv022/Data1/git/DL_NLP_Project/IMDB_data/aclImdb/test"

    #=================================================================================
    parser = argparse.ArgumentParser()

    parser.add_argument(
            '--check_vocab', type=str, help = 'saved vocab.txt', default=None)
    parser.add_argument('--keep_punc', action='store_true', help='Keep punctuation in sentences') # Default: action=store_true
    parser.add_argument('--sentences', action='store_false', help='Get word tokens per sentence in a document') # Default: action=store_true
    parser.add_argument('--keep_stopw', action='store_true', help='Keep stop words')

    parser.add_argument(
        '--train_size', type=int, help = 'training size to be considered in %', default =100)
    """
    The given training data contains 12500 negative and postive reviews each. Firstly, we split our data into 80:20 ratio as training and dev Set.
    Now our training set is 80% of the total given training data. In order to study the effect varying data sizes we will consider chunks( in %)
    of new trianing data (80% of original training data). 
    """
    args = parser.parse_args()

    # first check if there is a folder for train in corresponding to that size
    train_new_path = os.path.join(train_path +"_"+ str(args.train_size))

    if not os.path.isdir(train_new_path):
        
        print("Creating the training data with given size")
        print()
        os.makedirs(train_new_path)

        # copy [train_size]% of the data in this new path under positive and negative reviews
        directory = [os.path.join(train_path, "pos"), os.path.join(train_path, "neg")]

        for folder in directory:
                
            files  = os.listdir(folder)
            np.random.shuffle(files)

            no_files = int(len(files)*(args.train_size/100))
            train_new_files = files[:no_files]

            path = os.path.join(train_new_path, folder.split("/")[-1])

            if not os.path.isdir(path):
                os.makedirs(path)

            for f_ in train_new_files:
                shutil.copy(os.path.join(folder, f_), path)
    else:
        
        print("Data already exist with under: {}".format(train_new_path))


    # check if vocabulary exist for given data
    check_vocab = os.path.join(train_new_path+"_.txt")

    if not os.path.isfile(check_vocab):
        print("---------No vocab previously saved creating new one---------")
        print()
        # add all docs to vocab
        vocab = Counter()
        vocab = TokenizeCorpus(vocab, os.path.join(train_new_path, "pos"))
        vocab.process_docs()
        vocab = TokenizeCorpus(vocab.vocab, os.path.join(train_new_path, "neg"))
        vocab.process_docs()
        
        
        # keep tokens with > 5 occurrence
        min_occurane = 5
        tokens = [k for k,c in vocab.vocab.items() if c >= min_occurane]
        print(len(tokens))

        # save tokens to a vocabulary file
        save_list(tokens, check_vocab)
    else:

        print("vocab already exit")

    # # load vocabulary
    vocab_filename = check_vocab
    vocab = load_doc(vocab_filename)
    vocab = vocab.split()
    vocab = set(vocab)

    trainData = ProcessTrainData(vocab, train_new_path, args.train_size)
    if args.sentences:
        add_name = 's'
    else:
        add_name = ''

    json_file  = "w2i_"+ str(args.train_size)+ "_"+ add_name+".json"
    testData = ProcessTestData(json_file, test_path, args.train_size,'test')
    devData = ProcessTestData(json_file, dev_path, args.train_size,'val')
    # #==============================================================================