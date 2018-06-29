#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 15:24:25 2018

@author: maorongrao
"""


#A Toy Example


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
    TOYmodel=BuildModel(maxlen=6, EDIM=300, nunits=64, embedding_matrix=TOYembedding_matrix)
  
  
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
    
    
    
    
    
    
    
    
    