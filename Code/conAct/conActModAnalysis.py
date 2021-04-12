#!/usr/bin/env python3

import os
import natsort
import pickle

import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel

import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pyLDAvis.gensim

from nltk.corpus import wordnet as wn
import nltk
nltk.download('wordnet')

modelList = []
for file in natsort.os_sorted(os.listdir("../../Data/conAct/")):
    if file.startswith("conActModel"):
        model = file
        openmodel = open("../../Data/conAct/"+model, 'rb')
        model = pickle.load(openmodel)
        modelList.append(model)

cvList = []
for file in natsort.os_sorted(os.listdir("../../Data/conAct/")):
    if file.startswith("conActCv"):
        cv = file
        opencv = open("../../Data/conAct/"+cv, 'rb')
        cv = pickle.load(opencv)
        cvList.append(cv)

# plot graph for coherence values
limit=51;start=2;step=1
x = range(start,limit,step)
f = plt.figure()
plt.plot(x, cvList)
plt.xlabel("Number of topics")
plt.ylabel("C_v score")
plt.show()
f.savefig("../../Data/conAct/coherencePlot.png", dpi=500)
# 16

for idx, topic in modelList[13].print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

with open("../../Data/conAct/conActModel16.pkl", "rb") as f:
    model16 = pickle.load(f)
    f.close()

corpus = corpora.MmCorpus("../../Data/conAct/conActBoWCorpus.mm")

dictionary = corpora.Dictionary.load("../../Data/conAct/conActDict.dict")

pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(modelList[14], corpus, dictionary, sort_topics=False)
pyLDAvis.show(vis)
