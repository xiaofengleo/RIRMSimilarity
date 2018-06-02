# -*- coding: utf-8 -*-
"""
compute sentence similarity usning Doc2Vec
Xiaofeng Liao,
13 May, 2018
This script is most based on this post:
https://medium.com/@klintcho/doc2vec-tutorial-using-gensim-ab3ac03d3a1
"""
import gensim
from gensim.models.doc2vec import LabeledSentence
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn import metrics
import pylab as pl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


LabeledSentence = gensim.models.doc2vec.LabeledSentence
#sentence = LabeledSentence(words=[u'some', u'words', u'here'], tags=[u'SENT_1'])

from os import listdir
from os.path import isfile, join

class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
          #print(idx)
          #print(doc)
          words=doc.split()
          tags = [self.labels_list[idx]]
          #print('in LabeledLineSentence words')
          #print(words)
          #print('in LabeledLineSentence tag')
          #print(tags)
          yield LabeledSentence(words=doc.split(),tags=[self.labels_list[idx]])


#now create a list that contains the name of all the text file in your data #folder
docLabels = []
docLabels = [f for f in listdir("inputfile/") if f.endswith('.txt')]
print(docLabels)
#create a list data that stores the content of all text files in order of their names in docLabels
data = []
for doc in docLabels:
  data.append(open('inputfile/' + doc,encoding='utf-8').read())
#print('data')
#print(data)  


it = LabeledLineSentence(data, docLabels)
it.doc_list.sort
print('doc_list len:',len(it.doc_list))
print('labels_list len:',len(it.labels_list))
model = gensim.models.Doc2Vec(size=300, window=10, min_count=10, workers=11,alpha=0.025, min_alpha=0.025)
model.build_vocab(it)
#print('model vocabulary')
#print(model.wv.vocab)
#print()
for epoch in range(10):
    model.train(it,total_examples=model.corpus_count,epochs=model.iter)
    model.alpha -= 0.002            # decrease the learning rate
    model.min_alpha = model.alpha       # fix the learning rate, no deca
#    model.train(it)
model.save("doc2vec.model")
#loading the model
d2v_model = gensim.models.doc2vec.Doc2Vec.load('doc2vec.model')
#start testing
#printing the vector of document at index 1 in docLabels
docvec = d2v_model.docvecs[1]
#print('vocabulary')
#print(d2v_model.wv.vocab)
#similar_doc = d2v_model.docvecs.most_similar(1)
for doc in docLabels:
  print(doc)
  similar_doc = d2v_model.docvecs.most_similar(doc)
  docvec = d2v_model.docvecs[doc]
  similar_words = d2v_model.wv.most_similar(positive=[docvec],topn=30)
  print(similar_doc)
  print(similar_words)
'''
print('RMConceptDefiniton')
similar_doc = d2v_model.docvecs.most_similar('RMConceptDefiniton.txt')
print(similar_doc)
print('D5.2.part4.txt')
similar_doc = d2v_model.docvecs.most_similar('D5.2.part4.txt')
print(similar_doc)
print('D5.2.part6.txt')
similar_doc = d2v_model.docvecs.most_similar('D5.2.part6.txt')
print(similar_doc)
'''
#print(len(d2v_model.wv.vocab))
#print(d2v_model.wv.vocab)
#print(len(d2v_model.docvecs))

#print(docvec)
#print(model.most_similar("D3.3.txt"))
#######################################
#test
