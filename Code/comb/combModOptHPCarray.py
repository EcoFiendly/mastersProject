#!/usr/bin/env python3

import pickle
import os

# gensim
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel

# get working directory
home = os.path.expanduser('~')

# load tokens
with open(home+"/Data/comb/tokens_2.pkl", "rb") as f:
   tokens_2 = pickle.load(f)
   f.close()

# load dictionary
dic = corpora.Dictionary.load(home+"/Data/comb/dic.dict")

# load corpus
corpus = corpora.MmCorpus(home+"/Data/comb/bow_corpus.mm")

iter = int(os.getenv("PBS_ARRAY_INDEX")) # read job number from cluster

model = gensim.models.LdaMulticore(corpus = corpus, num_topics = iter, id2word = dic, passes = 10, workers = 32, random_state = 95)
cv_coh = CoherenceModel(model = model, corpus = corpus, texts = tokens_2, coherence = 'c_v', processes = 32)
cv = cv_coh.get_coherence()

with open(home+"/Data/comb/model_"+str(iter)+".pkl", "wb") as f:
    pickle.dump(model, f)
    f.close()

with open(home+"/Data/comb/cv_"+str(iter)+".pkl", "wb") as f:
    pickle.dump(cv, f)
    f.close()
