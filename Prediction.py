#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 13:47:53 2018

@author: maorongrao
"""
 ## clean raw test text file 

def PreTest(testFile,maxlen=6):
    
    f = open(testFile, 'r', encoding='utf-8')   
    line = f.readlines()
    f.close()
   
    test = pd.DataFrame(line, columns=['review'])
    test['seg']=test['review'].apply(lambda s: CleanText(s))
    test['seg2']=test['seg'].apply(lambda x:x.split())
    def Seg2Index(s, maxlen):
          s = [i for i in s if i in set(word2index.index)]  
          s = s[:maxlen] + ['']*max(0, maxlen-len(s))  
          return list(word2index[s]) 
      
    x_test=test['seg2'].apply(lambda s: Seg2Index(s, maxlen))  
    
    x_test = np.array(list(x_test))  
    print('x_test ready')
    return x_test


 #6. Predict

def PredKey(x_test,MYmodel,kw=2):
    y_prob=MYmodel.predict(x_test)
    y_pred=y_prob.argmax(axis=-1)
    #get layer out
    get_layer_output = K.function([MYmodel.layers[0].input, K.learning_phase()], [MYmodel.layers[2].output])
    get_layer_output2 = K.function([MYmodel.layers[0].input, K.learning_phase()], [MYmodel.layers[3].output])
    out = get_layer_output([x_test, 0])[0] 
    out2 = get_layer_output2([x_test, 0])[0]
    weight=[np.exp(np.tanh(np.dot(out[i], out2[i])))/np.sum(np.exp(np.tanh(np.dot(out[i], out2[i])))) for i in range(len(x_test))]
    TopKeyIndex = [x_test[i][np.argpartition(weight[i],-kw)[-kw:]] for i in range(len(x_test))]
    TopKey=[[index2word.get(TopKeyIndex[i][j]) for j in range(kw)]for i in range(len(x_test))]
    pred=pd.DataFrame(list(zip(y_pred,TopKey)))
    return pred
     
         
    