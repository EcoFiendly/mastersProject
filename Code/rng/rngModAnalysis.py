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
for file in natsort.os_sorted(os.listdir("../../Data/rng/")):
    if file.startswith("rngModel"):
        model = file
        openmodel = open("../../Data/rng/"+model, 'rb')
        model = pickle.load(openmodel)
        modelList.append(model)

cvList = []
for file in natsort.os_sorted(os.listdir("../../Data/rng/")):
    if file.startswith("rngCv"):
        cv = file
        opencv = open("../../Data/rng/"+cv, 'rb')
        cv = pickle.load(opencv)
        cvList.append(cv)

# plot graph for coherence values
limit=101;start=2;step=1
x = range(start,limit,step)
f = plt.figure()
plt.plot(x, cvList) 
plt.xlabel("Number of topics")
plt.ylabel("C_v score")
plt.show()
f.savefig("../../Data/rng/coherencePlot.png", dpi=500)
# 40, 43* (very close)

for idx, topic in modelList[41].print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))
