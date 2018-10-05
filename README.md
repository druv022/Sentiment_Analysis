# Bi-LSTM Model for Sentiment analysis

## Dataset
We have uploaded the data folder consisting training, testing and validation data (all are pickle files) as well as or vocab json namely w2i_100.json here https://drive.google.com/drive/folders/1UcNlWvyjgBlYb6ZanT-KRAtHj6o2TacL?usp=sharing. 

Regarding teh data folder in the brach it contains pre-processing scripts for the given IMDB data, wg=hich basically creates the pickle file whcih is shared above.

## Training

This repository contains code for Bi-LSTM based nueral model for sentiment classification.

To run the code for Training:

python train.py --num_layers 1 --embedding_dim 100 --learning_rate 0.001 --trainDataset path_for_train_pickle ----testDataset path_for_test_pickle ----devDataset path_for_dev_pickle

Models perfroms better with single layer of bi-LSTM. So, I would suggest to use num_layers to be fixed. To use the model to learn from pre-trained embedding one could run:

python train.py --num_layers 1 --embedding_dim 200 --learning_rate 0.001 --pre_train path of pretrained embeddings --trainDataset path_for_train_pickle ----testDataset path_for_test_pickle ----devDataset path_for_dev_pickle
  
We used Glove embeddings that too with only 200 dimensions. Embeddings can be downloaded from https://drive.google.com/file/d/1FYyCcQqdcmg6UXUyQ7Hi20UtP4K638iH/view?usp=sharing and the specied path can be given under "path of pretrained embeddings". Running train.py with the given conguration will produce the resutls as mentioned in our report.

