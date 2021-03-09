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
for file in natsort.os_sorted(os.listdir("../../Data/comb/")):
    if file.startswith("combModel"):
        model = file
        openmodel = open("../../Data/comb/"+model, 'rb')
        model = pickle.load(openmodel)
        modelList.append(model)

cvList = []
for file in natsort.os_sorted(os.listdir("../../Data/comb/")):
    if file.startswith("combCv"):
        cv = file
        opencv = open("../../Data/comb/"+cv, 'rb')
        cv = pickle.load(opencv)
        cvList.append(cv)

# plot graph for coherence values
limit=151;start=2;step=1
x = range(start,limit,step)
f = plt.figure()
plt.plot(x, cvList)
plt.xlabel("Number of topics")
plt.ylabel("coherence value")
plt.show()
f.savefig("../../Data/comb/coherencePlot.png", dpi=500)
# 24, 29*

for idx, topic in modelList[27].print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

