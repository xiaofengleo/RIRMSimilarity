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


#LabeledSentence = gensim.models.doc2vec.LabeledSentence
#sentence = LabeledSentence(words=[u'some', u'words', u'here'], tags=[u'SENT_1'])

from os import listdir
from os.path import isfile, join
docLabels = []
docLabels = [f for f in listdir("myDirPath") if f.endswith('.txt')]

data = []
for doc in docLabels:
    data.append(open(“myDirPath/” + doc, ‘r’)



class LabeledLineSentence(object):
    def __init__(self, filename):
        self.filename = filename
    def __iter__(self):
        for uid, line in enumerate(open(filename)):
            yield LabeledSentence(words=line.split(), labels=[‘SENT_%s’ % uid])


it = LabeledLineSentence(data, docLabels)

model = gensim.models.Doc2Vec(size=300, window=10, min_count=5, workers=11,alpha=0.025, min_alpha=0.025)

model.build_vocab(it)

for epoch in range(10):
    model.train(it)
    model.alpha -= 0.002            # decrease the learning rate
    model.min_alpha = model.alpha       # fix the learning rate, no deca
    model.train(it)

model.save("doc2vec.model")
print model.most_similar("documentFileNameInYourDataFolder")

##########################################################
#plot by using tSNE
def display_closestwords_tsnescatterplot(model, word):    
    arr = np.empty((0,10), dtype='f')
    word_labels = [word]
    # get close words
    close_words = model.similar_by_word(word)    
    # add the vector for each of the closest words to the array
    arr = np.append(arr, np.array([model[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = model[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)        
    # find tsne coords for 2 dimensions
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)
    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    # display scatter plot
    plt.scatter(x_coords, y_coords)
    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
    plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
    plt.show()


def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    #plt.show()
    plt.savefig('whole.png')


text_file_name = 'RMVocabulary.txt'
sentences = gensim.models.word2vec.LineSentence(text_file_name)
print(sentences)
simple_model = gensim.models.Word2Vec(sentences, size=10, window=5,min_count=1, workers=2)
#SV_object/Actor
#words1 = ['CapacityManager', 'DataCollector', 'DataConsumer', 'DataCurator', 'DataOriginator',
#          'DataProvider', 'DataPublisher','MeasurementModelDesigner', 'Measurer', 'MetadataHarvester',
#          'Observer', 'PIDManager','ProcessingEnvironmentPlanner', 'SemanticCurator', 'SemanticMediator',
#          'StorageAdministrator', 'Technician']
#words1 = simple_model.most_similar('CapacityManager')
words1 = [pairs[0] for pairs in simple_model.most_similar('CapacityManager')]
words1.append('CapacityManager')
#User
#words2 = ['User', 'Citizen', 'CitizenScientist', 'Consultant', 'DecisionMaker', 'Educator', 'Engineer',
#          'Investor', 'Journalist','PolicyMaker', 'Researcher', 'Scientist', 'EnvironmentalScientist',
#          'Technologist', 'Trainer']
#words2 = simple_model.most_similar('User')
words2 = [pairs[0] for pairs in simple_model.most_similar('User')]
words2.append('User')
#operation
#words3 = ['BrokeredDataExport', 'BrokeredDataImport', 'DataStaging', 'RawDataCollection', 'CV_SimpleOperation',
#          'AcquireIdentifier','AnnotateData', 'AuthoriseAction', 'CalibrateInstrument', 'ConfigureController',
#          'CoordinateProcess', 'ExportMetadata', 'InvokeResource', 'PrepareDataTransfer', 'ProcessRequest',
#          'QueryCatalogues', 'QueryData', 'QueryResource','RequestData', 'ResolveIdentifier', 'RetrieveData',
#'TranslateRequest', 'UpdateCatalogues', 'UpdateModel', 'UpdateRecord', 'UpdateRegistry']
#words3 = simple_model.most_similar('BrokeredDataExport')
words3 = [pairs[0] for pairs in simple_model.most_similar('BrokeredDataExport')]
words3.append('BrokeredDataExport')
#SV_object/Actor
#words1 = ['capacitymanager', 'datacollector', 'dataconsumer', 'datacurator', 'dataoriginator',
#         'dataprovider', 'datapublisher', 'measurementmodeldesigner', 'measurer', 'metadataharvester',
#         'observer', 'PIDmanager', 'processingenvironmentplanner', 'semanticcurator', 'semanticmediator',
#         'storageadministrator', 'technician']
#CV_simpleoperation
words4 = ['acquireidentifier', 'annotatedata', 'authoriseaction', 'calibrateinstrument',
         'configurecontroller', 'coordinateprocess', 'exportmetadata', 'preparedatatransfer',
         'processrequest', 'querydata', 'queryresource', 'requestdata',
         'resolveidentifier', 'retrievedata', 'translaterequest',
         'updatemodel' ,'updaterecord', 'updateregistry']
#user
words5 = ['user', 'citizen','citizenscientist', 'consultant', 'decisionmaker', 'educator', 'engineer',
          'investor', 'journalist','policymaker', 'researcher', 'scientist', 'technologist', 'trainer']
#CV_object
words6 = ['bindingobject', 'datatransporter', 'dataexporter', 'dataimporter', 'datastager',
'rawdatacollector', 'persistentobject', 'backendobject', 'brokerobject','databroker', 'semanticbroker',
'componentobject', 'datastorecontroller','instrumentcontroller', 'processcontroller', 'externalresource',
'presentationobject', 'sciencegateway', 'virtuallaboratory','experimentallaboratory', 'fieldlaboratory',
'semanticlaboratory', 'serviceobject', 'AAAIservice', 'acquisitionservice',
'annotationservice', 'catalogueservice', 'coordinationservice', 'datatransferservice', 'PIDservice']
print('CapacityManager')
print(simple_model.most_similar('CapacityManager'))
print('User')
print(simple_model.most_similar('User'))
print('BrokeredDataExport')
print(simple_model.most_similar('BrokeredDataExport'))

tsne_plot(simple_model)

