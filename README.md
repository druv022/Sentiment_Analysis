# Sentiment_Analysis

## Dataset

The dataset of the movie reviews is given by the Stanford AI group. http://ai.stanford.edu/%7Eamaas/data/sentiment/

Note : I did not upload the aclImdb folder to git, as it is a huge file, but my code refers to this folder in order to retrieve the data. Please keep the aclImdb folder in the same directory as the model_selection.ipynb file.

## Model selection

First I load the train and test data with positive and negative labels. 

Then, create a train and a test file with positive reviews labeled as 1 and negative reviews labeled as 0. 

The data are vectorized using a scikit learn library. The models that were used for training, are:

1. Logistic Regression
2. Naive Bayes
3. Random Forest
4. Gradient Boosted Trees

Then I created an ensemble model of those 4, that can be used as baseline. Training accuracy is 95% and validation accuracy is 85%.