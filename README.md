# Bag-of-Words

Sentiment Analysis for Movie reviews is to be done using the given IMDB data.
The model is built using tensorflow.
The data is the imdb movie data which was downloaded.
The train and test data both had 25,000 records each.
Firstly,a virtual tensorflow environment was created from command prompt.
Jupyter notebook  in Anaconda was used to do the task.
#Libraries used:
tensorflow library was used using import tensflow as tf command.
Pandas library was used to read the data;import pandas as pd.
Numpy library was  usedfor numerical operations.
keras API was used and many keras library were used 
Tokenizer was used to convert words into tokens
pad sequences was used for padding and truncating.
#import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from scipy.spatial.distance import cdist
from tensorflow.python.keras.layers import Dense,GRU,Embedding
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import pandas as pd
Train data was read into python environment
#train = pd.read_csv("labeledTrainData.tsv", header=0, \
                    delimiter="\t", quoting=3)
#train
#test = pd.raed_csv("testData.tsv",header=0.\delimiter="\t",quting=3)
Most frequent 10000 words were used.
Tokenizer was used for tokenizing words into tokens

#num_words=10000
#tokenizer = Tokenizer(num_words=num_words)
#tokenizer.word_index
the tokenizer gave every word a unique index for identification.
train_tokens = tokenizer.texts_to_sequences(train["review"])
test_tokens = tokenizer.texts_to_sequences(test["review"])
num_tokens = [len(tokens) for tokens in train_tokens + test_tokens]
np.mean(num_tokens)
np.max(num_tokens)
the maximum nuber of tokens was 2209 and mean number of tokens was 221.
to save the memory the tokens with average length of mean + 2 std deviation were taken into consideration.
This covered almost 95% of all data.
max_tokens = np.mean(num_tokens) + 2* np.std(num_tokens)
max_tokens
np.sum(num_tokens < max_tokens)/ len(num_tokens)
#Padding and Truncating were used to get bthe average token length.
# A customised function was built to convert the tokens back into words.

def tokens_to_string(tokens):
    #Map from tokens back to words
    words = [inverse_map[token] for token in tokens if token !=0]
    #Concatenate all words
    text = " ".join(words)
    return text
    
# A Recurrent neural network was built 
embedding vector of size 8 was taken.
An embedding vector converts tokens into real-valued vectors which are fed into the neural network.

Gated recurrent units were used with each GRU layer output acting as input for the next layer.
The last layed gave only one dense layer as the output.
The dense layer gives the output between 0 and 1 which is used as the classification output.
The keras model is trained on the train data and tested oon the test data.
    



