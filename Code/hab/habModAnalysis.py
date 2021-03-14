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
for file in natsort.os_sorted(os.listdir("../../Data/hab/")):
    if file.startswith("habModel"):
        model = file
        openmodel = open("../../Data/hab/"+model, 'rb')
        model = pickle.load(openmodel)
        modelList.append(model)

cvList = []
for file in natsort.os_sorted(os.listdir("../../Data/hab/")):
    if file.startswith("habCv"):
        cv = file
        opencv = open("../../Data/hab/"+cv, 'rb')
        cv = pickle.load(opencv)
        cvList.append(cv)

# plot graph for coherence values
limit=101;start=2;step=1
x = range(start,limit,step)
f = plt.figure()
plt.plot(x, cvList) # peaks at 13*, 15, 18
plt.xlabel("Number of topics")
plt.ylabel("C_v score")
plt.show()
f.savefig("../../Data/hab/coherencePlot.png", dpi=500)
# 16, 17*, 18

for idx, topic in modelList[16].print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

