# Hierarchical Attention Network Model for Sentiment analysis

## Dataset
The *data* folder contains pre-processing scripts for the IMDB dataset and the required files.

## Training

This repository contains code for **Hierarchical Attention Network** based nueral model for sentiment classification.

To run the code for Training:
```

python train.py --num_layers 2 --embedding_dim 300 --learning_rate 0.001
```

To use the model to learn from pre-trained embedding one could also run:

```
python train.py --num_layers 2 --embedding_dim 200 --learning_rate 0.001 --pretrained "path_of_pretrained_embeddings"
```
  
We used Glove embeddings with only 300 dimensions. Embeddings can be downloaded from [here](https://drive.google.com/file/d/1eMaFKiSIrZ9wZ9GvvrIWKliP4tXYAKBt/view?usp=sharing) and the specified path can be given under *"path of pretrained embeddings"*. Running *train.py* with the given conguration will produce the resutls as mentioned in our report.



*Note: Default settings below should produce the reported result.* 

Parameter | Default value
------------|---------------
embedding_dim|300
output_dim | 1
num_hidden | 150
num_layers | 2
batch_size | 64
learning_rate | 0.001
dropout | 0.2
num_words | 60
num_sentences | 30
