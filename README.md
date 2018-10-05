# CNN Model for Sentiment analysis

## Dataset
We have uploaded the data folder consisting training, testing  (all are pickle files) as well as or vocab json namely w2i.josn

The *data* folder contains pre-processing scripts for the IMDB dataset, which basically creates the pickle files.

## Training

This repository contains code for CNN based nueral model for sentiment classification.

To run the code for Training:

Please call the train() function in train_cnn.py . It automtaically saves model after every epoch , to change the file name please change the file_name in save_model 

For training on glove data please uncomment the line in CNN_classifier.py self.embeddings.load_state_dict({'weight': weights_matrix})

For Testing
Please call the test() function in train_cnn.py . First it loads the model from the file path specified in load_model and prints accuracy at the end 

  
We used Glove embeddings that too with only 200 dimensions. Embeddings can be downloaded from [here](https://drive.google.com/file/d/1FYyCcQqdcmg6UXUyQ7Hi20UtP4K638iH/view?usp=sharing) 
