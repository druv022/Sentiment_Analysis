import gensim
import numpy as np
from autocorrect import spell
from textblob import TextBlob
import pickle
import re
model=None
model = gensim.models.KeyedVectors.load_word2vec_format('/home/vik1/Downloads/subj/dl_nlp/GoogleNews-vectors-negative300.bin/data', binary=True)
##stat_dict={"filmmak":"director","appropri":"appropriate","unfortun":"unforunate","moviemak":"director","extravag":"extravagant"}
def clean_words(string,TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()

def create_vector():
    rand_vector=np.random.uniform(-0.25,0.25,(1,300))
    lis = []
    with open("data/vocab.txt") as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    ##for i in range(0,len(content)):
    ##    content[i]=clean_words(content[i],False)
    dict={}
    x=0
    y=0
    le=len(content)
    index=0
    for data in content:
        print(le,index)
        index+=1
        if data in model:
            x=x+1
            dict[data]=np.asarray(model[data])
        else:
            dict[data]=np.random.randn(1,300)
            y=y+1
            """
            data_1=spell(data)
            if(data_1 not in model):
               b=TextBlob(data)
               b=b.correct()
               print(data,b)
               if(b not in model):
                  lis.append(data)
                  y=y+1
                  dict[data] = rand_vector
               else:
                   x=x+1
            else:
                x=x+1
            """
    print(len(content),x,y)
    print(lis)
    filename = 'word_vectors.pkl'
    pickle.dump(dict, open(filename, 'wb'))
create_vector()



