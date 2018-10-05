# Bi-LSTM Model for Sentiment analysis

This repository contains code for Bi-LSTM based nueral model for sentiment classification.

To run the code for Training:

python train.py --num_layers 1 --embedding_dim 100 --learning_rate 0.001. 

Models perfroms better with single layer of bi-LSTM. So, I would suggest to use num_layers to be fixed. To use the model to learn from pre-trained embedding one could run:

python train.py --num_layers 1 --embedding_dim 200 --learning_rate 0.001 --pre_train <path of pretrained embeddings>
  
We used Glove embeddings that too with only 200 dimensions. Embeddings can be downloaded from        and the specied path can be given under <path of pretrained embeddings>. Running train.py with given conguration will produece the resutls as mentioned in our report.

