from keras.layers.core import Activation, Dense, Dropout, SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from urllib.request import urlopen
from urllib.error import URLError
import collections
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os

url ='https://storage.googleapis.com/kaggle-competitions-data/kaggle/2558/training.txt?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1524220521&Signature=TPjIb75%2F%2FqR%2BOED07%2F5lmm0%2BhdQpF4krnee%2BItbalHM9sCrDgwovGG2Rk3htjyy3z3zTRnOiGNMFIwWpLyjq2gfSWzt9c0y6UfUtnjBszRofADfbTuurI3jQzhQuBHVne2GpC%2FwW2NxMgcxXNu%2Fx3akApb6ZNsWToK1i797ge1xFAv%2BJKlIjBYO%2FfCyvzB%2FZmstw2oGiu0KqbrrrPJoi15plFHH%2BL2zqrXGCHXuLoTB5Y%2BhyFfx6wbQwGDmnqkWVL4BE1HerqkyiCq1RQiFMyN%2Fo5I4a8qe5M17CFy0HBdAApOWIhhzl%2BUuS%2FkNAlBUUxX1qaUv%2Fm3aWBZH4G02xKA%3D%3D'
maxlen = 0
word_freqs = collections.Counter()
num_recs = 0

try:
    ftrain = urlopen(url).read()
    
    for line in ftrain:
        #if type(line) is str:
        label, sentence = line.strip().split("\t")
        words = nltk.word_tokenize(sentence.lower())
        if len(words) > maxlen:
            maxlen = len(words)
        for word in words:
            word_freqs[word] += 1
        num_recs += 1
except AttributeError as e:
    print(str(e))
    pass
except URLError as e:
    print(str(e))
    pass
except:
    pass

    
print(maxlen)
print(len(word_freqs))


                
