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
for file in natsort.os_sorted(os.listdir("../../Data/use/")):
    if file.startswith("useModel"):
        model = file
        openmodel = open("../../Data/use/"+model, 'rb')
        model = pickle.load(openmodel)
        modelList.append(model)

cvList = []
for file in natsort.os_sorted(os.listdir("../../Data/use/")):
    if file.startswith("useCv"):
        cv = file
        opencv = open("../../Data/use/"+cv, 'rb')
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
f.savefig("../../Data/use/coherencePlot.png", dpi=500)
# 9, 14*

for idx, topic in modelList[12].print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

