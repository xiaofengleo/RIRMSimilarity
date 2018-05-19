# -*- coding: utf-8 -*-
"""
compute sentence similarity usning Doc2Vec
Xiaofeng Liao,
13 May, 2018

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


#LabeledSentence = gensim.models.doc2vec.LabeledSentence
#sentence = LabeledSentence(words=[u'some', u'words', u'here'], tags=[u'SENT_1'])

from os import listdir
from os.path import isfile, join

'''
docLabels = []
docLabels = [f for f in listdir("inputfile/") if f.endswith('.txt')]
print(docLabels)
data = []
for doc in docLabels:
    data.append(open("inputfile/" + doc, 'r'))
    
'''
'''
class LabeledLineSentence(object):
    def __init__(self,  filename):
        self.filename = filename
    def __iter__(self):
        for uid, line in enumerate(open(filename)):
            yield LabeledSentence(words=line.split(), labels=['SENT_%s' % uid])
'''

#now create a list that contains the name of all the text file in your data #folder
docLabels = []
docLabels = [f for f in listdir("inputfile/") if f.endswith('.txt')]
#create a list data that stores the content of all text files in order of their names in docLabels
data = []
for doc in docLabels:
  data.append(open('inputfile/' + doc,encoding='utf-8').read())
  
class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
              yield gensim.models.doc2vec.LabeledSentence(doc,    
[self.labels_list[idx]])


it = LabeledLineSentence(data, docLabels)
it.doc_list.sort
#print(it.doc_list)
#print(it.labels_list)
model = gensim.models.Doc2Vec(size=300, window=10, min_count=5, workers=11,alpha=0.025, min_alpha=0.025)
model.build_vocab(it)
for epoch in range(10):
    model.train(it,total_examples=model.corpus_count,epochs=model.iter)
    model.alpha -= 0.002            # decrease the learning rate
    model.min_alpha = model.alpha       # fix the learning rate, no deca
    #model.train(it)
model.save("doc2vec.model")
#loading the model
d2v_model = gensim.models.doc2vec.Doc2Vec.load('doc2vec.model')
#start testing
#printing the vector of document at index 1 in docLabels
docvec = d2v_model.docvecs[1]
#similar_doc = d2v_model.docvecs.most_similar(1)
for doc in docLabels:
  print(doc)
  similar_doc = d2v_model.docvecs.most_similar(doc)
  print(similar_doc)
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
'''
for i in range(10):
    doc_id = np.random.randint(d2v_model.docvecs.count)  # pick random doc; re-run cell for more examples
    doc = d2v_model.docvecs[doc_id]
    print(doc)
#    print('doc %d length %d' % (doc_id, len(doc.words)))
#    if len(doc.words) < 100:
#        continue
    # pick random 50-word slice of document
    #slice50 = doc.words[np.random.randint(len(doc.words)-50):][:50]
    #slice_inferred = model.infer_vector(slice50)
    similars = d2v_model.docvecs.most_similar(positive=[doc],topn=100000)
    print(similars)
    #rank_of_source = [sim_id for sim_id, sim in similars].index(doc_id)
    #print('%s: %s' % (model, rank_of_source))
'''
