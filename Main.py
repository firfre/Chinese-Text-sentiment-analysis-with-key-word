#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 15:24:22 2018

@author: maorongrao
"""
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

if __name__ == '__main__':
   
    inFile1='pos.txt'
    inFile2='neg.txt'
    outFile1='pos2.txt'
    outFile2='neg2.txt'
    maxlen=6
    maxlen_q=0.7 #quantile of segments' length
    t_split=0.7  #trainning-validation split ratio
    kw = 2   # key word number
    wordvector='wordvector.txt'
    EDIM=300
    testFile='testFile.txt'
    maxvacab=10000
    n_units=64

    #Check Embedding_Index.pickle exist: if Embedding_Index.pickle not exists:
    
    #embedding_index=GetEmbedIndex(wordvector='wordvector.txt')
    
    
    #0. drop NA
   
    #1.Get Training Data
    
   x,y, x_train,y_train,x_val,y_val,word2index,index2word,embedding_matrix = PreDoc(
           inFile1='pos.txt',inFile2='neg.txt',maxlen=6, maxlen_q=0.7, t_split=0.7)
    
    
    
    #2. Construct Model 
    
   MYmodel=BuildModel(maxlen=6, EDIM=300, n_units=64, embedding_matrix=embedding_matrix)
  
    
    #3. train
    
   # load weights (optional)
   #Mymodel.load_weights('Mymodel_weights.h5')
    
   MYmodel.fit(x_train, y_train, 
              batch_size=256, epochs=10, 
              validation_data=(x_val, y_val))
   
   
   
   
    #4. validation
    #score, acc = model.evaluate(x_test, y_test, batch_size=100)
    #print("score: %.3f, accuracy: %.3f" % (score, acc))

   # 5. Sentiment Prediction Example(1:pos, 0:neg)
   ###load or create a textfile example
   #f=open('testFile.txt','w')
   #f.write('啊，我超喜欢的，好温暖啊～～～')
   #f.write('\n')
   #f.write('这什么鬼，态度差，物流慢。')
   #f.close()
   
   ## test sentiment result with key word
   testFile='testFile.txt'
   #clean raw testFile
   x_test = PreTest(testFile,maxlen=6)
   #prediction with keywords
   pred = PredKey(x_test,MYmodel,kw=2)
   print(pred)
   
   
   
   #save Clean Data and parameters for later(optional)
   
   saveFile='Data.pickle'
   SaveData(saveFile)
         
        
   # save weights for transfer study (optional)
   MYmodel.save_weights('MYmodel_weights.h5')
  
   
   
   
   
