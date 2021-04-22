#!/usr/bin/env python3

import os
import natsort
import pickle
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
import pyLDAvis.gensim

model_list = []
for file in natsort.os_sorted(os.listdir("../../Data/comb_global/")):
    if file.startswith("model_"):
        model = file
        open_model = open("../../Data/comb_global/"+model, 'rb')
        model = pickle.load(open_model)
        model_list.append(model)

cv_list = []
for file in natsort.os_sorted(os.listdir("../../Data/comb_global/")):
    if file.startswith("cv_"):
        cv = file
        open_cv = open("../../Data/comb_global/"+cv, 'rb')
        cv = pickle.load(open_cv)
        cv_list.append(cv)

# plot graph for coherence values
limit=101;start=2;step=1
x = range(start,limit,step)
f = plt.figure()
plt.plot(x, cv_list)
plt.xlabel("Number of topics")
plt.ylabel("C_v score")
plt.show()
f.savefig("../../Data/comb_global/coherencePlot.png", dpi=500)
# 10, 17*, 22

for idx, topic in model_list[15].print_topics(-1, num_words=8):
    print('Topic: {} \nWords: {}'.format(idx, topic))

corpus = corpora.MmCorpus("../../Data/comb_global/bow_corpus.mm")

dic = corpora.Dictionary.load("../../Data/comb_global/dic.dict")

pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(model_list[20], corpus, dic, sort_topics=False)
pyLDAvis.show(vis)
pyLDAvis.save_html(vis, "../../Data/comb_global/vis_22.html")

