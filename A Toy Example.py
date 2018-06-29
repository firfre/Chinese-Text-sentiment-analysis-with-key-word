#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 15:24:25 2018

@author: maorongrao
"""


#A Toy Example
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import jieba  
import re
import pickle
from keras import backend as K
from keras import initializers, regularizers, constraints
from keras.engine import network
from bs4 import BeautifulSoup
import sys
import os
from keras.preprocessing import sequence
from keras.models import Sequential,Model,load_model
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding, LSTM, GRU,  Input, Flatten, merge
from keras.engine.topology import Layer, InputSpec
from keras.layers import *
from pprint import pprint
from gensim import corpora
from sklearn.cross_validation import train_test_split
from collections import defaultdict
from numpy.random import randint
from numpy import argmax
from keras.models import model_from_json
from keras.callbacks import History 


#pickle load data:
    with open('TOYData.pickle', 'rb') as f:
        data = pickle.load(f)   
    
    data.keys()
    x_tt=data['x_train']  
    y_tt=data['y_train']
    x_vt=data['x_val']
    y_vt=data['y_val']
    TOYword2index=data['word2index']
    TOYindex2word=data['index2word']
    TOYembedding_matrix=data['embedding_matrix']
        
#Model:      
    TOYmodel=BuildModel(maxlen=6, EDIM=300, n_units=64, embedding_matrix=TOYembedding_matrix)
  
  
# train
    #train-val size
   
    b = 50
    e = 1

    TOYmodel.fit(x_tt,y_tt,
              batch_size=b, epochs=e, 
              validation_data=(x_vt, y_vt))
    
# Predict (1:poisve; 0:negative)
    x_test_toy=x_vt[-10:] 
    
    kw = 2   # key word number
    
    pred = PredKey(x_test_toy,TOYmodel,kw=2)
    print(pred)
    
    
    
    
    
    
    
    
    
