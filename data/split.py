"""
"""

import os
import numpy as np
import shutil

dev_path = "/media/druv022/Data1/git/DL_NLP_Project/IMDB_data/aclImdb/dev"
train_path = "/media/druv022/Data1/git/DL_NLP_Project/IMDB_data/aclImdb/train"

directory = [os.path.join(train_path, "pos"), os.path.join(train_path, "neg")]

if not os.path.isdir(dev_path):
    os.makedirs(dev_path)

split_size = 0.20 # 20% validation size

for folder in directory:
    files  = os.listdir(folder)
    np.random.shuffle(files)

    no_files = int(len(files)*split_size)
    dev_files = files[:no_files]

    path = os.path.join(dev_path, folder.split("/")[-1])
    
    if not os.path.isdir(path):
        os.makedirs(path)
    for f_ in dev_files:
        shutil.move(os.path.join(folder, f_), path)


