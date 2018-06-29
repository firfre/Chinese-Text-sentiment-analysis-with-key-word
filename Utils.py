#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 15:24:23 2018

@author: maorongrao
"""

#Get Inbedding Index:


#If Embedding_Index.pickle not exists or nedd to get new pre trained word vector:
def GetEmbedIndex(wordvector='wordvector.txt'):
    embedding_index = {}
    f = open( wordvector)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs
    f.close()
    #save embeddings_index for future use
    Embedding_Index={'embeddings_index':embedding_index}
    with open('Embedding_Index.pickle','wb') as f:
        pickle.dump(Embedding_Index,f,pickle.HIGHEST_PROTOCOL)
    print('Embedding_Index has saved, found %s word vectors.' % len(embedding_index))
    
    return embedding_index



 def SaveData(saveFile): 
       Data = {
              'x': x,
              'y': y,
              'word2index':word2index,
              'index2word':index2word,
              'x_train':x_train,
              'x_val':x_val,
              'y_train':y_train,
              'y_val': y_val,
              'embedding_matrix':embedding_matrix
              }
       with open(saveFile, 'wb') as f:
           pickle.dump(Data, f, pickle.HIGHEST_PROTOCOL)
    

           
    

    
        
        
    
   