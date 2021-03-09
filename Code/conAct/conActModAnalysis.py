#!/usr/bin/env python3

import os
import natsort
import pickle

import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel

import matplotlib.pyplot as plt
from wordcloud import WordCloud

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
limit=151;start=2;step=1
x = range(start,limit,step)
f = plt.figure()
plt.plot(x, cvList) 
plt.xlabel("Number of topics")
plt.ylabel("Coherence value")
plt.show()
f.savefig("../../Data/conAct/coherencePlot.png", dpi=500)
# 17*

for idx, topic in modelList[15].print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

