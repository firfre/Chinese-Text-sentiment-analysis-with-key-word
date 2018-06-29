#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 15:36:11 2018

@author: maorongrao
"""


####1.delete 'na'

   # f = open(inFile1,"r")
   # lines = f.readlines()
    #f.close()
   # f = open(inFile1,"w")
   # for line in lines:
    #   if line!="na"+"\n":
     #       f.write(line)
      #     f.close()    
      
    


#######2.
def CleanText(s):
    
    stoplist = {}.fromkeys([line.strip() for line in open('stopword.txt', 'r', encoding='utf-8')])
   # 1. keep chinese words and english words
    #FilterPattern = re.compile('[^\u4E00-\u9FD5A-Za-z]+')
    FilterPattern = re.compile('[^\u4E00-\u9FD5]+')
    b = FilterPattern.sub('', s)
    #b = BeautifulSoup(s, "lxml")
    #b= b.text
    # 2. jieba cut
    #seg_list = jieba.cut(b,cut_all=True)
    seg_list = jieba.cut_for_search(b)
    # 3. remove stopword
    segs = []
    for seg in seg_list:
        if seg.lstrip() is not None and seg.lstrip() not in stoplist:
            segs.append(seg)
            
    return " ".join(segs)
    



 ##########3.   
def PreDoc(inFile1, inFile2, maxlen=6, maxlen_q=0.7, t_split=0.7):
    print('loading data')
    f1 = open(inFile1, 'r', encoding='utf-8')
    f2 = open(inFile2, 'r', encoding='utf-8')
    line1 = f1.readlines()
    line2 = f2.readlines()
    f1.close()
    f2.close()
    pos = pd.DataFrame(line1, columns=['review'])
    neg = pd.DataFrame(line2, columns=['review'])
    pos['label']=int(1)
    neg['label']=int(0)
    train=pos.append(neg,ignore_index=True) 
    #train1=train[:1000]
    #train2=train[-1000:]
    #train=train1.append(train2,ignore_index=True)
    train['seg']=train['review'].apply(lambda s: CleanText(s))
    print('clean data done')
    cut_word=[]
   
    for line in train['seg']:
        cut_word.extend(line.split())
        
    word2index = pd.Series(cut_word).value_counts()  
    
    #word2index = word2index[:min(len(word2index),maxVacab)]
    
    word2index[:] = range(1, len(word2index)+1)  
    word2index[''] = 0  
    word_set = set(word2index.index)  
    index2word = {value: key for key, value in word2index.items()} 
    
    train['seg2']=train['seg'].apply(lambda x:x.split())
    lens = list(map(len, train['seg2']))
    maxlen_s = pd.DataFrame(lens).quantile(maxlen_q) 
    maxlen_s=int(maxlen_s[0]+1)
    #print('for quantile = %d suggest maxlen = %d', [%maxlen_q ,%maxlen_s])
    plt.hist(lens,bins=30,range=[0,30])
    plt.show() 
    print('start padding')
    def Seg2Index(s, maxlen):
           s = [i for i in s if i in word_set]  
           s = s[:maxlen] + ['']*max(0, maxlen-len(s))  
           return list(word2index[s]) 
      
    X=train['seg2'].apply(lambda s: Seg2Index(s, maxlen))  
    Y=train['label'] 
    
    print('prepare training sample')
    idx = np.arange(len(X))  
    np.random.shuffle(idx)
    x = X.loc[idx]  
    y = Y.loc[idx]  
    x = np.array(list(x))  
    y = np.array(list(y)) 
    y=to_categorical(y)
    #y=to_categorical(y) 
    train_num = int(t_split * len(x))
    x_train = x[:train_num]
    y_train = y[:train_num]
    x_val = x[train_num:]
    y_val =  y[train_num:]
    #print('saving training data to current folder...')
    #pickle load embedding-index
    with open('Embedding_Index.pickle', 'rb') as f:
        embedding_index= pickle.load(f)  
    embedding_index = embedding_index['embeddings_index']
    print('embedding_index loaded')
    #initialized embedding matrix
    embedding_matrix = np.random.random((len(word2index.index)+1 , 300))
    for word in word2index.index:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word2index[word]] = embedding_vector
    print('done')
    return x, y, x_train, y_train, x_val, y_val, word2index, index2word, embedding_matrix
    

   
     
        
   
    

    
    
    
    
    
    
    
    
    
    
    
    
         

